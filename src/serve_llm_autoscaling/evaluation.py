"""Score a continuous run's autoscaling: SLO attainment against replica cost.

Everything derives from ``plot_data.json`` and the per-request records, so a
score can be regenerated without the cluster. Attainment is reported two
ways, because they answer different questions: per request (how many turns
felt slow; dominated by the tail) and per rolling window (how much of the
time the mean latency met the target; what a mean-latency policy controls).
"""
from __future__ import annotations

import collections
import math
from bisect import bisect_left, bisect_right
from collections import defaultdict
from typing import Any

from .windows import Request, attainment, mean, percentile

SCHEMA_VERSION = 1

# Without a configured warmup, a run without a ramp settles within this.
DEFAULT_WARMUP_S = 120.0


def scoring_periods(experiment: dict[str, Any], lifecycle: dict[str, Any]) -> dict[str, Any]:
    """The whole recorded run, and the steady-state load it is scored on."""
    markers = lifecycle["markers"]
    load_end = markers["load_end_s"]
    warmup = experiment.get("warmup_s")
    if warmup is None:
        warmup = markers.get("ramp_end_s") or min(DEFAULT_WARMUP_S, load_end / 2)
    end = max((p["end_s"] for p in lifecycle["periods"]), default=load_end)
    return {
        "steady": {"start_s": min(warmup, load_end), "end_s": load_end},
        "whole_run": {"start_s": 0.0, "end_s": end},
    }


# --- SLO attainment -----------------------------------------------------------


def _latency_summary(
    values: list[float], slo_ms: float | None, slo_percentile: float
) -> dict[str, Any]:
    """Distribution plus the SLO verdict: the configured percentile within target.

    ``slo_met`` is equivalent to ``slo_attainment >= slo_percentile``.
    """
    ordered = sorted(values)
    at_percentile = percentile(ordered, slo_percentile)
    return {
        "sample_count": len(ordered),
        "mean_ms": mean(ordered),
        "p50_ms": percentile(ordered, 0.50),
        "p90_ms": percentile(ordered, 0.90),
        "p95_ms": percentile(ordered, 0.95),
        "p99_ms": percentile(ordered, 0.99),
        "slo_attainment": attainment(ordered, slo_ms),
        "slo_percentile_ms": at_percentile,
        "slo_met": (
            at_percentile <= slo_ms if slo_ms is not None and at_percentile is not None else None
        ),
    }


def score_requests(
    requests: list[Request], t0: int, period: dict[str, float],
    ttft_slo_ms: float | None, itl_slo_ms: float | None, slo_percentile: float = 0.95,
) -> dict[str, Any]:
    """Per-request attainment for requests that started in ``period``."""
    started = [
        r for r in requests
        if period["start_s"] <= (r.start_ns - t0) / 1e9 < period["end_s"]
    ]
    ok = [r for r in started if not r.failed]
    both = [r for r in ok if r.ttft_ms is not None and r.itl_ms is not None]
    both_within = None
    if both and ttft_slo_ms is not None and itl_slo_ms is not None:
        both_within = sum(
            r.ttft_ms <= ttft_slo_ms and r.itl_ms <= itl_slo_ms for r in both
        ) / len(both)
    ttft = _latency_summary(
        [r.ttft_ms for r in ok if r.ttft_ms is not None], ttft_slo_ms, slo_percentile
    )
    itl = _latency_summary(
        [r.itl_ms for r in ok if r.itl_ms is not None], itl_slo_ms, slo_percentile
    )
    verdicts = [v for v in (ttft["slo_met"], itl["slo_met"]) if v is not None]
    # A cold prompt (under half of it cached) costs seconds of prefill that
    # no replica count shortens; the warm TTFT is what autoscaling can move.
    by_cache = {
        name: _latency_summary(
            [r.ttft_ms for r in ok if r.cold is cold and r.ttft_ms is not None],
            ttft_slo_ms, slo_percentile,
        )
        for name, cold in (("cold", True), ("warm", False))
    }
    return {
        "requests": len(started),
        "failed_requests": len(started) - len(ok),
        "ttft": ttft,
        "itl": itl,
        "ttft_by_cache": by_cache,
        "both_slo_attainment": both_within,
        "slo_percentile": slo_percentile,
        "slo_met": all(verdicts) if verdicts else None,
    }


def score_windows(
    windows: list[dict[str, Any]], period: dict[str, float], window_s: float,
    ttft_slo_ms: float | None, itl_slo_ms: float | None,
) -> dict[str, Any]:
    """The fraction of time the rolling mean latencies met their targets.

    Each request window's rolling mean covers the ``tail_window_s`` before
    its end; windows ending in ``period`` with no samples are idle and skipped.
    """
    rows = [
        w for w in windows if period["start_s"] < w["window_end_s"] <= period["end_s"]
    ]
    # (verdict, seconds the row stands for) per row with samples.
    verdicts: dict[str, list[tuple[bool, float]]] = {}
    for name, slo in (("ttft", ttft_slo_ms), ("itl", itl_slo_ms)):
        if slo is not None:
            verdicts[name] = [
                (w[f"rolling_mean_{name}_ms"] <= slo, w["window_end_s"] - w["window_start_s"])
                for w in rows if w.get(f"rolling_{name}_sample_count")
            ]
    if len(verdicts) == 2:
        verdicts["both"] = [
            (w["rolling_mean_ttft_ms"] <= ttft_slo_ms and w["rolling_mean_itl_ms"] <= itl_slo_ms,
             w["window_end_s"] - w["window_start_s"])
            for w in rows
            if w.get("rolling_ttft_sample_count") and w.get("rolling_itl_sample_count")
        ]
    result: dict[str, Any] = {"rolling_window_s": window_s, "windows": len(rows)}
    for name in ("ttft", "itl", "both"):
        values = verdicts.get(name)
        result[f"{name}_within_slo"] = (
            sum(ok for ok, _ in values) / len(values) if values else None
        )
        result[f"{name}_violation_s"] = (
            sum(seconds for ok, seconds in values if not ok) if values else None
        )
    return result


# --- Cost -----------------------------------------------------------------------


def replica_usage(
    status: list[dict[str, Any]], period: dict[str, float], gpus_per_replica: int = 1
) -> dict[str, Any]:
    """Replica-seconds in ``period``, integrating the status step function.

    A failed status sample holds the previous counts; time before the first
    good sample is not covered. GPU time counts starting replicas too: a
    replica loading weights holds its GPUs without serving.
    """
    samples = [s for s in status if s.get("error") is None]
    start, end = period["start_s"], period["end_s"]
    running_s = starting_s = 0.0
    covered_s = 0.0
    peak = 0
    for sample, following in zip(samples, samples[1:] + [None]):
        t = max(sample["relative_time_s"], start)
        until = end if following is None else min(following["relative_time_s"], end)
        if until <= t:
            continue
        running_s += sample["running_replicas"] * (until - t)
        starting_s += sample["starting_replicas"] * (until - t)
        covered_s += until - t
        peak = max(peak, sample["running_replicas"])
    return {
        "covered_s": covered_s,
        "running_replica_seconds": running_s,
        "starting_replica_seconds": starting_s,
        "mean_running_replicas": running_s / covered_s if covered_s else None,
        "max_running_replicas": peak if covered_s else None,
        "gpu_hours": (running_s + starting_s) * gpus_per_replica / 3600,
    }


# --- Scaling decisions ----------------------------------------------------------


def _step_lookup(points: list[dict[str, Any]], key: str = "value"):
    """A function giving the latest value at or before a time, or None."""
    times = [p["relative_time_s"] for p in points]

    def at(time_s: float) -> Any:
        i = bisect_right(times, time_s) - 1
        return points[i][key] if i >= 0 else None

    return at


def scale_events(
    series: dict[str, Any], status: list[dict[str, Any]], sla_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    """Every change of the autoscaler's desired replica count.

    ``running_matched_at_s`` is when the running count first equalled the new
    target, before any later decision; its lag is reaction plus startup time.
    """
    key = "desired_replicas" if "desired_replicas" in series else "target_replicas"
    points = (series.get(key) or {}).get("points") or []
    good = [s for s in status if s.get("error") is None]
    status_times = [s["relative_time_s"] for s in good]
    reason_at = _step_lookup(sla_events, "reason")
    recommended_at = _step_lookup((series.get("sla_recommended_replicas") or {}).get("points") or [])

    events = []
    for previous, point in zip(points, points[1:]):
        if point["value"] == previous["value"]:
            continue
        events.append({
            "time_s": point["relative_time_s"],
            "from_replicas": previous["value"],
            "to_replicas": point["value"],
            "direction": "up" if point["value"] > previous["value"] else "down",
            "sla_reason": reason_at(point["relative_time_s"]),
            "sla_recommended_replicas": recommended_at(point["relative_time_s"]),
        })
    for event, following in zip(events, events[1:] + [None]):
        lo = bisect_left(status_times, event["time_s"])
        hi = len(good) if following is None else bisect_left(status_times, following["time_s"])
        matched = next(
            (s["relative_time_s"] for s in good[lo:hi]
             if s["running_replicas"] == event["to_replicas"]),
            None,
        )
        event["running_matched_at_s"] = matched
        event["lag_s"] = None if matched is None else matched - event["time_s"]
    return events


def scale_summary(events: list[dict[str, Any]]) -> dict[str, Any]:
    ups = [e for e in events if e["direction"] == "up"]
    lags = [e["lag_s"] for e in ups if e["lag_s"] is not None]
    return {
        "decisions": len(events),
        "upscales": len(ups),
        "downscales": len(events) - len(ups),
        "first_upscale_at_s": ups[0]["time_s"] if ups else None,
        "mean_upscale_lag_s": mean(lags),
        "max_upscale_lag_s": max(lags, default=None),
    }


def latency_by_replica_count(
    requests: list[Request], t0: int, status: list[dict[str, Any]], period: dict[str, float]
) -> list[dict[str, Any]]:
    """Latency of steady-state requests grouped by running replicas at start."""
    good = [s for s in status if s.get("error") is None]
    running_at = _step_lookup(good, "running_replicas")
    groups: dict[int, list[Request]] = defaultdict(list)
    for r in requests:
        start_s = (r.start_ns - t0) / 1e9
        if r.failed or not period["start_s"] <= start_s < period["end_s"]:
            continue
        replicas = running_at(start_s)
        if replicas is not None:
            groups[replicas].append(r)
    rows = []
    for replicas in sorted(groups):
        ttft = sorted(r.ttft_ms for r in groups[replicas] if r.ttft_ms is not None)
        itl = sorted(r.itl_ms for r in groups[replicas] if r.itl_ms is not None)
        rows.append({
            "running_replicas": replicas,
            "requests": len(groups[replicas]),
            "mean_ttft_ms": mean(ttft), "p90_ttft_ms": percentile(ttft, 0.90),
            "mean_itl_ms": mean(itl), "p90_itl_ms": percentile(itl, 0.90),
        })
    return rows


# --- SLA policy -------------------------------------------------------------------


def _error_summary(pairs: list[tuple[float, float]]) -> dict[str, Any]:
    """Predicted-minus-actual errors; MAPE is over non-zero actuals."""
    diffs = [p - a for p, a in pairs]
    relative = [abs(p - a) / a for p, a in pairs if a > 0]
    return {
        "samples": len(pairs),
        "bias_ms": mean(diffs),
        "mae_ms": mean([abs(d) for d in diffs]),
        "mape": mean(relative),
    }


def prediction_error(
    series: dict[str, Any], status: list[dict[str, Any]], windows: list[dict[str, Any]]
) -> dict[str, Any]:
    """How the policy's predictions compare with what was measured.

    Predictions are for the recommended replica count, so only scrapes where
    that equals the running count (and the model is ready) are comparable.
    The server mean is vLLM's, the policy's own input; the client mean is the
    rolling mean of the request window containing the scrape.
    """
    def values(key: str) -> dict[float, float]:
        return {p["relative_time_s"]: p["value"] for p in (series.get(key) or {}).get("points") or []}

    ready = values("sla_model_ready")
    recommended = values("sla_recommended_replicas")
    running_at = _step_lookup([s for s in status if s.get("error") is None], "running_replicas")
    window_ends = [w["window_end_s"] for w in windows]

    def client_mean(name: str, time_s: float) -> float | None:
        i = bisect_left(window_ends, time_s)
        if i >= len(windows) or windows[i]["window_start_s"] > time_s:
            return None
        if not windows[i].get(f"rolling_{name}_sample_count"):
            return None
        return windows[i][f"rolling_mean_{name}_ms"]

    result: dict[str, Any] = {}
    for name in ("ttft", "itl"):
        predicted = values(f"sla_predicted_{name}_ms")
        server = values(f"vllm_mean_{name}_ms")
        comparable = [
            t for t in predicted
            if ready.get(t, 0) >= 1 and recommended.get(t) == running_at(t)
        ]
        against_server = [(predicted[t], server[t]) for t in comparable if t in server]
        against_client = [
            (predicted[t], c) for t in comparable
            if (c := client_mean(name, t)) is not None
        ]
        client_vs_server = [
            (c, server[t]) for t in server if (c := client_mean(name, t)) is not None
        ]
        result[name] = {
            "predicted_vs_server": _error_summary(against_server),
            "predicted_vs_client": _error_summary(against_client),
            # What users saw minus what the policy saw: router and proxy queueing.
            "client_vs_server": _error_summary(client_vs_server),
        }
    return result


def sla_evaluation(plot_data: dict[str, Any], horizon_s: float) -> dict[str, Any] | None:
    series = plot_data["serve_metric_series"]
    events = plot_data["sla_policy_events"]
    availability = plot_data["availability"]
    if not (availability["sla_policy_metrics"] or availability["sla_policy_events"]):
        return None

    ready_points = (series.get("sla_model_ready") or {}).get("points") or []
    rows_points = (series.get("sla_training_rows") or {}).get("points") or []
    timeline = []
    time_in_reason: dict[str, float] = defaultdict(float)
    for event, following in zip(events, events[1:] + [None]):
        end = horizon_s if following is None else following["relative_time_s"]
        timeline.append({
            "start_s": event["relative_time_s"], "end_s": end, "reason": event["reason"],
            "recommended_replicas": event["recommended_replicas"],
        })
        time_in_reason[event["reason"]] += max(0.0, min(end, horizon_s) - max(event["relative_time_s"], 0.0))
    return {
        "advisory": next((e["advisory"] for e in events), None),
        "model_ready_at_s": next((p["relative_time_s"] for p in ready_points if p["value"] >= 1), None),
        "first_capacity_decision_at_s": next(
            (e["relative_time_s"] for e in events if e["reason"] == "capacity"), None
        ),
        "final_training_rows": rows_points[-1]["value"] if rows_points else None,
        "reason_timeline": timeline,
        "time_in_reason_s": dict(time_in_reason),
        "prediction": prediction_error(series, plot_data["serve_status_samples"], plot_data["request_windows"]),
    }


# --- Token policy -----------------------------------------------------------------


def spearman(pairs: list[tuple[float, float]]) -> float | None:
    """Spearman rank correlation of (x, y) pairs; None below three points."""
    if len(pairs) < 3:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(len(values)), key=lambda i: values[i])
        result = [0.0] * len(values)
        i = 0
        while i < len(order):
            j = i
            while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
                j += 1
            for k in range(i, j + 1):
                result[order[k]] = (i + j) / 2 + 1
            i = j + 1
        return result

    xs = ranks([p[0] for p in pairs])
    ys = ranks([p[1] for p in pairs])
    mx, my = mean(xs), mean(ys)
    cov = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    vx = sum((x - mx) ** 2 for x in xs)
    vy = sum((y - my) ** 2 for y in ys)
    return cov / (vx * vy) ** 0.5 if vx and vy else None


def _quartile_table(pairs: list[tuple[float, float]]) -> list[dict[str, Any]]:
    """Median y per quartile of x: the shape of the relation, not just its rank."""
    ordered = sorted(pairs)
    rows = []
    for q in range(4):
        chunk = ordered[q * len(ordered) // 4:(q + 1) * len(ordered) // 4]
        if chunk:
            ys = sorted(y for _, y in chunk)
            rows.append({"quartile": q + 1, "x_min": chunk[0][0], "x_max": chunk[-1][0],
                         "samples": len(chunk), "median_y": percentile(ys, 0.5),
                         "p90_y": percentile(ys, 0.9)})
    return rows


def excess_ttft(requests: list[Request]) -> dict[int, float]:
    """Per request index, TTFT above the run's own queue-free floor.

    The floor for a request is the 5th percentile TTFT of the requests in
    its factor-of-two bucket of uncached prompt tokens: the cost of the
    prefill itself. What remains is waiting, which is what scaling can cut.
    """
    def bucket(r: Request) -> int | None:
        if r.input_tokens is None or r.cached_tokens is None:
            return None
        uncached = r.input_tokens - r.cached_tokens
        return 0 if uncached < 1 else int(math.log2(uncached))

    by_bucket: dict[int, list[float]] = defaultdict(list)
    for r in requests:
        if not r.failed and r.ttft_ms is not None and (b := bucket(r)) is not None:
            by_bucket[b].append(r.ttft_ms)
    floors = {b: percentile(sorted(v), 0.05) for b, v in by_bucket.items()}
    return {
        i: r.ttft_ms - floors[b]
        for i, r in enumerate(requests)
        if not r.failed and r.ttft_ms is not None and (b := bucket(r)) is not None
    }


def in_flight_prefill(requests: list[Request], t0: int) -> dict[int, float]:
    """Per request index, the uncached tokens of other requests still in
    prefill (started, no first token yet) when this one started.

    This is what the policy's prefill backlog measures, rebuilt from the
    request records, so it is free of the metric path's export latency.
    """
    spans = sorted(
        ((r.start_ns - t0) / 1e9, (r.start_ns - t0) / 1e9 + r.ttft_ms / 1000,
         r.input_tokens - r.cached_tokens, i)
        for i, r in enumerate(requests)
        if not r.failed and r.ttft_ms is not None
        and r.input_tokens is not None and r.cached_tokens is not None
    )
    result = {}
    for start, _, _, i in spans:
        result[i] = sum(u for s, e, u, j in spans if j != i and s <= start < e)
    return result


def token_policy_evaluation(
    plot_data: dict[str, Any], requests: list[Request], events: list[dict[str, Any]],
    period: dict[str, float],
) -> dict[str, Any] | None:
    """Which term decided each scale-up, and how well each input predicts latency.

    Signal strength is judged per request: the policy's input per running
    replica at the moment the request started against the latency the
    request then saw. Prefill backlog is held against warm-request TTFT (a
    cold prompt's own prefill is not queueing) and against every request's
    TTFT in excess of its prefill floor; active decodes against ITL. The
    cross pairs show whether the signals tell the two latencies apart.
    """
    series = plot_data["serve_metric_series"]
    if "token_prefill_backlog" not in series:
        return None
    t0 = plot_data["time_origin"]["profiling_start_ns"]
    status = [s for s in plot_data["serve_status_samples"] if s.get("error") is None]
    running_at = _step_lookup(status, "running_replicas")
    lookups = {
        key: _step_lookup(series[key]["points"]) if key in series else (lambda t: None)
        for key in ("token_prefill_backlog", "token_active_decodes",
                    "token_prefill_replicas", "token_decode_replicas")
    }

    for event in events:
        prefill = lookups["token_prefill_replicas"](event["time_s"])
        decode = lookups["token_decode_replicas"](event["time_s"])
        event["prefill_term_replicas"] = prefill
        event["decode_term_replicas"] = decode
        if prefill is None or decode is None:
            event["binding_term"] = None
        elif event["direction"] == "down":
            event["binding_term"] = "both" if prefill == decode else ("prefill" if prefill > decode else "decode")
        else:
            event["binding_term"] = (
                "both" if prefill == decode else ("prefill" if prefill > decode else "decode")
            )

    pairs: dict[str, list[tuple[float, float]]] = {
        "prefill_backlog_vs_warm_ttft": [], "prefill_backlog_vs_excess_ttft": [],
        "in_flight_prefill_vs_warm_ttft": [], "in_flight_prefill_vs_excess_ttft": [],
        "decodes_vs_itl": [], "prefill_backlog_vs_itl": [], "decodes_vs_warm_ttft": [],
        "decodes_vs_excess_ttft": [],
    }
    excess = excess_ttft(requests)
    in_flight = in_flight_prefill(requests, t0)
    for i, r in enumerate(requests):
        start_s = (r.start_ns - t0) / 1e9
        if r.failed or not period["start_s"] <= start_s < period["end_s"]:
            continue
        replicas = running_at(start_s)
        backlog = lookups["token_prefill_backlog"](start_s)
        decodes = lookups["token_active_decodes"](start_s)
        if not replicas or backlog is None or decodes is None:
            continue
        backlog, decodes = backlog / replicas, decodes / replicas
        truth = in_flight.get(i)
        if r.ttft_ms is not None and r.cold is False:
            pairs["prefill_backlog_vs_warm_ttft"].append((backlog, r.ttft_ms))
            pairs["decodes_vs_warm_ttft"].append((decodes, r.ttft_ms))
            if truth is not None:
                pairs["in_flight_prefill_vs_warm_ttft"].append((truth / replicas, r.ttft_ms))
        if i in excess:
            pairs["prefill_backlog_vs_excess_ttft"].append((backlog, excess[i]))
            pairs["decodes_vs_excess_ttft"].append((decodes, excess[i]))
            if truth is not None:
                pairs["in_flight_prefill_vs_excess_ttft"].append((truth / replicas, excess[i]))
        if r.itl_ms is not None:
            pairs["decodes_vs_itl"].append((decodes, r.itl_ms))
            pairs["prefill_backlog_vs_itl"].append((backlog, r.itl_ms))
    return {
        "scale_up_binding_terms": dict(collections.Counter(
            e["binding_term"] for e in events if e["direction"] == "up"
        )),
        "signal_strength": {
            name: {"samples": len(p), "spearman": spearman(p), "quartiles": _quartile_table(p)}
            for name, p in pairs.items()
        },
    }


# --- TTFT decomposition -----------------------------------------------------------


def ttft_decomposition(
    requests: list[Request], t0: int, events: list[dict[str, Any]], period: dict[str, float],
    slo_percentile: float,
) -> dict[str, Any]:
    """Split the scored TTFT percentile into the prompt's own prefill and waiting.

    The floor is each request's bucket floor (see ``excess_ttft``); what
    remains is waiting. ``instant_startup_ms`` replaces the TTFT of requests
    that started while a replica was starting (between a scale-up decision
    and the running count matching it) with their floor: the percentile an
    autoscaler with zero startup lag could have reached. The rest of the
    waiting happens with no replica on the way, which more replicas do not
    touch.
    """
    excess = excess_ttft(requests)
    lags = [
        (e["time_s"], e["running_matched_at_s"]) for e in events
        if e["direction"] == "up" and e.get("running_matched_at_s") is not None
    ]
    actual, floors, ideal, in_lag = [], [], [], 0
    for i, r in enumerate(requests):
        start_s = (r.start_ns - t0) / 1e9
        if i not in excess or not period["start_s"] <= start_s < period["end_s"]:
            continue
        floor = r.ttft_ms - excess[i]
        starting = any(a <= start_s < b for a, b in lags)
        in_lag += starting
        actual.append(r.ttft_ms)
        floors.append(floor)
        ideal.append(floor if starting else r.ttft_ms)
    q = slo_percentile
    at = lambda v: percentile(sorted(v), q)  # noqa: E731
    return {
        "samples": len(actual),
        "requests_during_startup": in_lag,
        "percentile": q,
        "actual_ms": at(actual),
        "own_prefill_floor_ms": at(floors),
        "waiting_ms": (at(actual) - at(floors)) if actual else None,
        "instant_startup_ms": at(ideal),
    }


# --- Entry point ------------------------------------------------------------------


def evaluate_run(plot_data: dict[str, Any], requests: list[Request]) -> dict[str, Any]:
    experiment = plot_data["experiment"]
    t0 = plot_data["time_origin"]["profiling_start_ns"]
    status = plot_data["serve_status_samples"]
    series = plot_data["serve_metric_series"]
    windows = plot_data["request_windows"]
    ttft_slo, itl_slo = experiment["ttft_slo_ms"], experiment["itl_slo_ms"]
    slo_percentile = experiment.get("slo_percentile", 0.95)
    gpus = experiment.get("gpus_per_replica", 1)
    periods = scoring_periods(experiment, plot_data["lifecycle"])
    events = scale_events(series, status, plot_data["sla_policy_events"])
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": experiment["name"],
        "autoscaling_policy": experiment["autoscaling_policy"],
        "autoscaling_policy_kwargs": experiment["autoscaling_policy_kwargs"],
        "slo": {"ttft_ms": ttft_slo, "itl_ms": itl_slo, "percentile": slo_percentile},
        "periods": periods,
        "slo_attainment": {
            name: {
                "requests": score_requests(
                    requests, t0, period, ttft_slo, itl_slo, slo_percentile
                ),
                "windows": score_windows(windows, period, experiment["tail_window_s"], ttft_slo, itl_slo),
            }
            for name, period in periods.items()
        },
        "cost": {
            **{name: replica_usage(status, period, gpus) for name, period in periods.items()},
            **{p["name"]: replica_usage(status, p, gpus) for p in plot_data["lifecycle"]["periods"]},
        },
        "scale_events": events,
        "scaling": scale_summary(events),
        "ttft_decomposition": ttft_decomposition(
            requests, t0, events, periods["steady"], slo_percentile
        ),
        "latency_by_replica_count": latency_by_replica_count(
            requests, t0, status, periods["steady"]
        ),
        "sla_policy": sla_evaluation(plot_data, periods["whole_run"]["end_s"]),
        "token_policy": token_policy_evaluation(plot_data, requests, events, periods["steady"]),
    }
