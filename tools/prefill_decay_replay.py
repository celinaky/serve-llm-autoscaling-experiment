#!/usr/bin/env python3
"""Replay oldest-first linear prefill decay, independently on each engine.

The replay uses engine-measured uncached prompt lengths because historical
per-request router estimates were not logged. It starts at engine queue entry,
spends a shared tokens/s budget oldest-first, and clears residual work at the
observed first token (or termination). Continuous linear decay is sampled every
second for plots/CSV; lookback means integrate the continuous curve exactly.
The recorded replica placements and completion times are held fixed: this is
an accounting replay, not a prediction of a deployment with another policy.

Example:
  python tools/prefill_decay_replay.py runs/<run> --rate 10000
"""

import argparse
import csv
import json
from collections import OrderedDict, defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


class Curve:
    """Right-continuous piecewise-linear gauge with exact integration."""

    def __init__(self):
        self.points = []

    def put(self, time, value, slope=0.0):
        point = (float(time), max(0.0, float(value)), float(slope))
        if self.points and time == self.points[-1][0]:
            self.points[-1] = point
        else:
            if self.points and time < self.points[-1][0]:
                raise ValueError("Curve points must be ordered")
            self.points.append(point)

    def finish(self):
        if not self.points:
            self.put(0, 0)
        self.t, self.v, self.s = np.array(self.points).T
        dt = np.diff(self.t)
        self.area = np.r_[0, np.cumsum(self.v[:-1] * dt + self.s[:-1] * dt**2 / 2)]
        return self

    def value(self, times):
        times = np.asarray(times)
        indices = np.searchsorted(self.t, times, side="right") - 1
        i = np.maximum(indices, 0)
        return np.where(indices < 0, 0, np.maximum(0, self.v[i] + self.s[i] * (times - self.t[i])))

    def integral(self, times):
        times = np.asarray(times)
        indices = np.searchsorted(self.t, times, side="right") - 1
        i = np.maximum(indices, 0)
        dt = times - self.t[i]
        return np.where(indices < 0, 0, self.area[i] + self.v[i] * dt + self.s[i] * dt**2 / 2)

    def mean(self, times, window):
        times = np.asarray(times)
        return (self.integral(times) - self.integral(times - window)) / window


def step_curve(deltas):
    changes = defaultdict(float)
    for time, delta in deltas:
        changes[time] += delta
    curve, value = Curve(), 0.0
    for time, delta in sorted(changes.items()):
        value += delta
        if value < -1e-5:
            raise ValueError(f"Negative work {value} at {time}")
        curve.put(time, value)
    if abs(value) > 1e-5:
        raise ValueError(f"Work did not return to zero: {value}")
    return curve.finish()


def replay_fifo(requests, rate):
    """One engine's capacity; no idle credit and no future completions used to decay."""
    if rate <= 0:
        raise ValueError("rate must be positive")
    events = []
    for index, request in enumerate(requests):
        events.append((request["queued"], 1, index))
        events.append((request["terminal"], 0, index))
    events.sort()
    pending = OrderedDict()
    curve, diagnostics = Curve(), {}
    now = events[0][0] if events else 0.0
    total = 0.0
    for time, kind, index in events:
        while pending and time > now:
            head, remaining = next(iter(pending.items()))
            exhausted_at = now + remaining / rate
            if exhausted_at <= time:
                total -= remaining
                now = exhausted_at
                pending.pop(head)
                diagnostics[head]["estimated_zero_at"] = now
                curve.put(now, total, -rate if pending else 0)
            else:
                consumed = rate * (time - now)
                pending[head] -= consumed
                total -= consumed
                now = time
        now = time
        if kind:
            remaining = requests[index]["uncached"]
            diagnostics[index] = {"estimated_zero_at": None, "residual_at_terminal": 0.0}
            if remaining > 0:
                pending[index] = float(remaining)
                total += remaining
        else:
            remaining = pending.pop(index, 0.0)
            total -= remaining
            diagnostics[index]["residual_at_terminal"] = remaining
        curve.put(now, total, -rate if pending else 0)
    if pending or abs(total) > 1e-5:
        raise ValueError("FIFO replay leaked work")
    return curve.finish(), diagnostics


def read_requests(run, traces, output, origin):
    engines = sorted({row["engine"] for row in csv.DictReader((run / "analysis/load_requests.csv").open())})
    paths = [path for engine in engines for path in sorted(traces.glob(f"{engine}.*.jsonl"))]
    if any(not list(traces.glob(f"{engine}.*.jsonl")) for engine in engines):
        raise ValueError("Missing a trace for an engine listed in load_requests.csv")
    signature = [(str(p), p.stat().st_size, p.stat().st_mtime_ns) for p in paths]
    cache = output / "requests.json"
    if cache.exists():
        saved = json.loads(cache.read_text())
        if saved["signature"] == [list(s) for s in signature] and saved["origin"] == origin:
            return saved["requests"], saved["coverage"]
    requests, coverage = [], {"trace_files": [str(p) for p in paths], "queued": 0, "excluded": []}
    for engine in engines:
        queued, admitted, first, finished, completed = {}, {}, {}, {}, {}
        chunks = defaultdict(list)
        preemptions = 0
        for path in sorted(traces.glob(f"{engine}.*.jsonl")):
            print(f"Reading {path.name}", flush=True)
            for line in path.open():
                event = json.loads(line)
                kind = event["event"]
                if kind == "batch_scheduled":
                    for r in event["requests"]:
                        if r["phase"] == "prefill":
                            chunks[r["request_id"]].append((event["batch_index"], r["scheduled_tokens"]))
                elif kind == "batch_completed":
                    completed[event["batch_index"]] = event["wall_time_s"] - origin
                elif kind == "request_queued":
                    queued[event["request_id"]] = event
                elif kind == "request_admitted":
                    admitted[event["request_id"]] = event
                elif kind == "request_first_token":
                    first[event["request_id"]] = event
                elif kind == "request_finished":
                    finished[event["request_id"]] = event
                elif kind == "request_preempted":
                    preemptions += 1
        if preemptions:
            raise ValueError("This replay isolates initial prefill; recompute needs separate accounting")
        coverage["queued"] += len(queued)
        for rid, q in queued.items():
            admission = admitted.get(rid)
            terminal = first.get(rid, finished.get(rid))
            if admission is None or terminal is None:
                coverage["excluded"].append({"request_id": rid, "reason": "missing admission or terminal"})
                continue
            own = admission["prompt_tokens"] - admission["cached_prefix_tokens"]
            if q["wall_time_s"] >= terminal["wall_time_s"]:
                raise ValueError("Request lifetime must be positive")
            # Cap credits at the uncached prompt. Decode or speculative work must
            # not reduce outstanding prompt tokens below zero.
            remaining = own
            credits = []
            for batch, scheduled in chunks[rid]:
                if batch not in completed:
                    raise ValueError(f"Missing batch completion: {engine} {batch}")
                if first.get(rid) and batch > first[rid]["batch_index"]:
                    continue
                credit = min(remaining, scheduled)
                if credit:
                    credits.append([completed[batch], credit])
                    remaining -= credit
            requests.append({
                "request_id": rid, "engine": engine,
                "queued": q["wall_time_s"] - origin,
                "admitted": admission["wall_time_s"] - origin,
                "terminal": terminal["wall_time_s"] - origin,
                "uncached": own, "prompt": admission["prompt_tokens"],
                "chunks": credits, "terminal_residual": remaining,
                "first_token": rid in first,
            })
    coverage["included"] = len(requests)
    coverage["preemptions"] = 0
    coverage["terminal_residual_tokens"] = sum(r["terminal_residual"] for r in requests)
    cache.write_text(json.dumps({"signature": signature, "origin": origin, "requests": requests, "coverage": coverage}))
    return requests, coverage


def published_mean(run, origin, target):
    points = {}
    for line in (run / "telemetry/serve_metrics.jsonl").open():
        record = json.loads(line)
        time = record["scrape_started_at_ns"] / 1e9 - origin
        values = [sample["value"] * target for sample in record.get("samples", [])
                  if sample["name"] == "ray_serve_autoscaling_policy_metric"
                  and sample["labels"].get("metric") == "prefill_required_replicas"
                  and "LLMServer" in sample["labels"].get("deployment", "")]
        if values:
            points[time] = max(values)
    points = sorted(points.items())
    return np.array([p[0] for p in points]), np.array([p[1] for p in points])


def save_plot(figure, path):
    figure.savefig(path.with_suffix(".png"), dpi=170, bbox_inches="tight")
    figure.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(figure)


def policy_overlay_plot(run, output, origin, end, window, rate, target, grid, mean, observed_t, observed_v, evaluation):
    """Original policy-vs-engine presentation with one additional decay line."""
    resolved = yaml.safe_load((run / "resolved.yaml").read_text())
    autoscaling = resolved["deployment"]["autoscaling"]
    curve = resolved["benchmark"]["workload"].get("concurrency_series", [])
    surface, ink, secondary, grid_color = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
    purple, green, orange = "#6250d6", "#1baf7a", "#e08b24"
    phase_tint = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
    visible = (observed_t >= 0) & (observed_t <= end)
    observed_t, observed_v = observed_t[visible], observed_v[visible]
    peak = max(np.max(observed_v, initial=0), np.max(mean["engine_actual"]), np.max(mean["fifo_decay"])) / 1000
    ycap = max(min(peak * 1.05, 12 * target / 1000), 1.5 * target / 1000)
    figure, axis = plt.subplots(figsize=(14, 5.8), dpi=150)
    figure.patch.set_facecolor(surface)
    axis.set_facecolor(surface)
    for name in ("top", "right"):
        axis.spines[name].set_visible(False)
    for name in ("left", "bottom"):
        axis.spines[name].set_color(grid_color)
    axis.tick_params(colors=secondary, labelsize=8)
    axis.grid(axis="x", visible=False)
    axis.grid(axis="y", color=grid_color, alpha=1, lw=.5)
    axis.set_axisbelow(True)
    for start, finish in zip(curve, curve[1:]):
        phase = "ramp up" if finish["sessions"] > start["sessions"] else "ramp down" if finish["sessions"] < start["sessions"] else "hold"
        lo, hi = max(0, start["time_s"]), min(end, finish["time_s"])
        if hi > lo:
            axis.axvspan(lo, hi, color=phase_tint[phase], zorder=0, lw=0)
            axis.text((lo + hi) / 2, ycap * 1.09, f"{phase}: {start['sessions']}→{finish['sessions']} sessions", ha="center", va="top", fontsize=8, color=secondary)
    for event in evaluation["scale_events"]:
        time = event["time_s"]
        if not 0 <= time <= end:
            continue
        axis.axvline(time, color=ink, lw=.8, alpha=.6)
        ready = event.get("running_matched_at_s")
        if ready is not None and ready > time:
            axis.axvspan(time, min(ready, end), color=secondary, alpha=.08, lw=0, zorder=.5)
            axis.axvline(min(ready, end), color=secondary, lw=.8, ls="--", alpha=.6)
        axis.text(time + 4, ycap * 1.02, f"{int(event['from_replicas'])}→{int(event['to_replicas'])}", ha="left", va="bottom", fontsize=8, color=ink, fontweight="bold")
    axis.plot(observed_t, observed_v / 1000, color=purple, lw=1.8)
    axis.plot(grid, mean["engine_actual"] / 1000, color=green, lw=1.8)
    axis.plot(grid, mean["fifo_decay"] / 1000, color=orange, lw=1.8)
    axis.axhline(target / 1000, color=secondary, lw=1, ls=":")
    # Retain the original plot's observed upscale-delay annotations.
    asks = []
    for line in (run / "telemetry/serve_metrics.jsonl").open():
        record = json.loads(line)
        time = record["scrape_started_at_ns"] / 1e9 - origin
        if record.get("error") or not 0 <= time <= end:
            continue
        values = {}
        for sample in record.get("samples", []):
            if "LLMServer" not in sample["labels"].get("deployment", ""):
                continue
            if sample["name"] == "ray_serve_autoscaling_policy_metric":
                values[sample["labels"]["metric"]] = sample["value"]
            elif sample["name"] == "ray_serve_autoscaling_desired_replicas":
                values["desired"] = sample["value"]
        if "required_replicas" in values and "desired" in values:
            asks.append((time, values["required_replicas"], values["desired"]))
    spans, current = [], None
    for time, required, desired in asks:
        if required > desired:
            if current is None:
                current = [time, time]
            current[1] = time
        elif current is not None:
            spans.append(current)
            current = None
    if current is not None:
        spans.append(current)
    up_times = [event["time_s"] for event in evaluation["scale_events"] if event["direction"] == "up"]
    gap = asks[1][0] - asks[0][0] if len(asks) > 1 else 2
    timer_pass, timer_reset = "#c9480f", "#e8b89a"
    for lo, hi in spans:
        passed = [time for time in up_times if hi <= time <= hi + 2 * gap + 1]
        stop = passed[0] if passed else hi
        axis.axvspan(lo, stop, ymin=0, ymax=.035, color=timer_pass if passed else timer_reset, lw=0, zorder=2)
        if not passed and hi - lo >= 8:
            axis.text((lo + hi) / 2, -ycap * .055, f"{hi - lo:.0f}s", ha="center", va="top", fontsize=6.5, color=timer_pass)
    axis.set_ylim(-ycap * .06, ycap * 1.12)
    axis.set_xlim(0, end)
    axis.set_xlabel("seconds since profiling start", color=secondary)
    axis.set_ylabel("total outstanding uncached prefill tokens, all replicas (K)", color=secondary)
    handles = [
        Line2D([], [], color=purple, lw=1.8, label=f"policy: {window:g}s mean of the recorded router gauge"),
        Line2D([], [], color=green, lw=1.8, label=f"engine: {window:g}s mean of remaining uncached prefill (traces)"),
        Line2D([], [], color=orange, lw=1.8, label=f"decay: {window:g}s mean, oldest first at {rate/1000:g}K tokens/s per engine"),
        Line2D([], [], color=secondary, lw=1, ls=":", label=f"target {target/1000:g}K per replica"),
        Line2D([], [], color=ink, lw=.8, alpha=.6, label="scale decision (replicas before→after)"),
        Line2D([], [], color=secondary, lw=.8, ls="--", alpha=.6, label="new replica running (shaded: startup)"),
        Patch(color=timer_pass, label=f"upscale-delay timer ran and passed ({autoscaling['upscale_delay_s']:g}s)"),
        Patch(color=timer_reset, label="upscale-delay timer reset before the delay elapsed"),
    ]
    figure.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(.5, -.15))
    settings = f"policy: ceil({window:g}s mean / {target/1000:g}K) · Serve upscale delay {autoscaling['upscale_delay_s']:g}s · downscale delay {autoscaling['downscale_delay_s']:g}s · replicas {autoscaling['min_replicas']}-{autoscaling['max_replicas']} · metrics every {autoscaling['metrics_interval_s']:g}s"
    figure.text(.5, -.19, settings, ha="center", va="top", fontsize=8.5, color=secondary)
    figure.text(.5, -.23, "Orange replay uses engine uncached lengths at queue entry; first token clears residual. Scaling markers show the recorded run.", ha="center", va="top", fontsize=8, color=secondary)
    figure.suptitle(f"{evaluation['experiment']} · router vs engine, with oldest-first linear decay", x=.01, ha="left", fontsize=10, color=ink)
    save_plot(figure, output / "policy_vs_engine_with_decay")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--rate", type=float, default=10000)
    parser.add_argument("--lookback", type=float, default=60)
    parser.add_argument("--traces", type=Path, default=Path("/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"))
    parser.add_argument("--out-dir", type=Path)
    args = parser.parse_args()
    if args.rate <= 0 or args.lookback <= 0:
        parser.error("rate and lookback must be positive")
    run = args.run
    output = args.out_dir or run / "analysis" / f"prefill_decay_fifo_{args.rate:g}tps"
    output.mkdir(parents=True, exist_ok=True)
    origin = json.loads((run / "analysis/plot_data.json").read_text())["time_origin"]["profiling_start_ns"] / 1e9
    evaluation = json.loads((run / "analysis/autoscaling_evaluation.json").read_text())
    end = evaluation["periods"]["steady"]["end_s"]
    target = (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica", 20000)
    requests, coverage = read_requests(run, args.traces, output, origin)
    if not requests:
        raise ValueError("No complete requests")
    baseline = step_curve([(t, d) for r in requests for t, d in [(r["queued"], r["uncached"]), (r["terminal"], -r["uncached"])]])
    deltas = []
    for r in requests:
        deltas.append((r["queued"], r["uncached"]))
        deltas.extend((time, -tokens) for time, tokens in r["chunks"])
        if r["terminal_residual"]:
            deltas.append((r["terminal"], -r["terminal_residual"]))
    actual = step_curve(deltas)
    by_engine = defaultdict(list)
    for r in requests:
        by_engine[r["engine"]].append(r)
    fifo, records = {}, []
    for engine, engine_requests in by_engine.items():
        fifo[engine], diagnostic = replay_fifo(engine_requests, args.rate)
        for i, r in enumerate(engine_requests):
            records.append({k: v for k, v in r.items() if k != "chunks"} | diagnostic[i])
    grid = np.arange(0, end + 0.001, 1.0)
    raw = {"no_decay": baseline.value(grid), "fifo_decay": sum(c.value(grid) for c in fifo.values()), "engine_actual": actual.value(grid)}
    mean = {"no_decay": baseline.mean(grid, args.lookback), "fifo_decay": sum(c.mean(grid, args.lookback) for c in fifo.values()), "engine_actual": actual.mean(grid, args.lookback)}
    observed_t, observed_v = published_mean(run, origin, target)
    mask = (grid >= args.lookback) & (mean["engine_actual"] > 5000)
    summary = {
        "run": str(run), "rate_tokens_per_second_per_engine": args.rate,
        "decay": "continuous linear, oldest first, no banking idle capacity; sampled every 1 s",
        "initial_counts": "engine uncached tokens known retrospectively; original router counts unavailable",
        "lifetime": "engine queue entry to observed first token (or termination)",
        "lookback_seconds": args.lookback, "load_end_seconds": end,
        "coverage": coverage, "metrics": {},
        "load_phase_raw_mean_tokens": {
            "no_decay": float((baseline.integral(end) - baseline.integral(0)) / end),
            "fifo_decay": float(sum(c.integral(end) - c.integral(0) for c in fifo.values()) / end),
            "engine_actual": float((actual.integral(end) - actual.integral(0)) / end),
        },
    }
    for key in ("no_decay", "fifo_decay"):
        difference = mean[key][mask] - mean["engine_actual"][mask]
        ratio = mean[key][mask] / mean["engine_actual"][mask]
        summary["metrics"][key] = {
            "median_ratio_lookback_mean_loaded": float(np.median(ratio)),
            "p10_ratio": float(np.quantile(ratio, .1)), "p90_ratio": float(np.quantile(ratio, .9)),
            "mean_absolute_error_tokens_lookback_mean_loaded": float(np.mean(np.abs(difference))),
            "bias_tokens_lookback_mean_loaded": float(np.mean(difference)),
            "fraction_loaded_mean_below_engine": float(np.mean(difference < 0)),
            "fraction_loaded_mean_over_10k_below_engine": float(np.mean(difference < -10000)),
        }
    summary["metric_filter"] = "one-second samples, t >= lookback and actual lookback mean > 5000 tokens"
    premature = [r for r in records if r["uncached"] > 0 and r["estimated_zero_at"] is not None and r["estimated_zero_at"] < r["terminal"] - 1e-6]
    summary["requests_estimated_zero_before_first_token"] = len(premature)
    summary["requests_at_least_10k_estimated_zero_before_first_token"] = sum(r["uncached"] >= 10000 for r in premature)
    summary["maximum_early_zero_seconds"] = max((r["terminal"] - r["estimated_zero_at"] for r in premature), default=0)
    summary["sum_residual_cleared_at_terminal_tokens"] = sum(r["residual_at_terminal"] for r in records)
    summary["final_estimated_tokens"] = float(sum(c.value(max(r["terminal"] for r in requests) + 1) for c in fifo.values()))
    columns = {"time_s": grid} | {f"{k}_tokens": v for k, v in raw.items()} | {f"{k}_mean_tokens": v for k, v in mean.items()}
    with (output / "timeseries.csv").open("w") as stream:
        writer = csv.writer(stream)
        writer.writerow(columns)
        writer.writerows(zip(*columns.values()))
    with (output / "request_replay.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": .2})
    policy_overlay_plot(run, output, origin, end, args.lookback, args.rate, target, grid, mean, observed_t, observed_v, evaluation)
    colors = {"no_decay": "#7860b6", "fifo_decay": "#e08b24", "engine_actual": "#16896d"}
    labels = {"no_decay": "No decay replay (same initial counts)", "fifo_decay": f"Oldest first: {args.rate/1000:g}K tokens/s per engine", "engine_actual": "Actual remaining work (completed chunks)"}
    figure, axes = plt.subplots(2, 1, figsize=(15, 8.5), sharex=True)
    for key in raw:
        axes[0].plot(grid, mean[key] / 1000, color=colors[key], lw=1.7, label=labels[key])
        axes[1].plot(grid, raw[key] / 1000, color=colors[key], lw=1, alpha=.9)
    axes[0].plot(observed_t, observed_v / 1000, color="#353535", lw=1.2, ls="--", alpha=.7, label="Recorded router policy mean (different initial estimates + export lag)")
    axes[0].axhline(target / 1000, color="#888888", lw=.8, ls=":")
    for axis in axes:
        axis.set_ylabel("Outstanding prefill (K tokens)")
        axis.set_ylim(bottom=0)
        axis.set_xlim(0, end)
        for event in evaluation.get("scale_events", []):
            axis.axvline(event["time_s"], color="#777777", lw=.8, alpha=.5)
    axes[0].set_title(f"{args.lookback:g}s trailing time-weighted means (vertical lines: recorded scaling decisions)", loc="left")
    axes[0].legend(loc="upper left", fontsize=8)
    axes[1].set_title("Instantaneous signals, sampled every second", loc="left")
    axes[1].set_xlabel("Seconds since profiling start")
    figure.suptitle(f"B200 trace replay · oldest-prefill-first linear decay at {args.rate/1000:g}K tokens/s per engine", fontsize=14, x=.07, ha="left")
    figure.text(.07, .012, "Replay starts with engine uncached lengths at queue entry; first token clears residual. Placements/completions fixed. No counterfactual scaling simulated.", fontsize=9)
    figure.tight_layout(rect=(0, .04, 1, .95))
    save_plot(figure, output / "comparison")
    # Inspect the engine/window with the most overlapping positive prefill work.
    scored = []
    for engine, group in by_engine.items():
        instantaneous = step_curve([(t, d) for r in group for t, d in [(r["queued"], r["uncached"]), (r["terminal"], -r["uncached"])]])
        overlap = step_curve([(t, d) for r in group if r["uncached"] > 0 for t, d in [(r["queued"], 1), (r["terminal"], -1)]])
        score = instantaneous.value(grid) * np.maximum(overlap.value(grid) - 1, 0)
        index = int(np.argmax(score))
        scored.append((score[index], engine, grid[index]))
    _, selected, center = max(scored)
    lo, hi = max(0, center - 15), min(end, center + 35)
    zoom_grid = np.arange(lo, hi, .05)
    group = by_engine[selected]
    instant = step_curve([(t, d) for r in group for t, d in [(r["queued"], r["uncached"]), (r["terminal"], -r["uncached"])]])
    actual_deltas = [(r["queued"], r["uncached"]) for r in group]
    actual_deltas += [(t, -n) for r in group for t, n in r["chunks"]]
    actual_deltas += [(r["terminal"], -r["terminal_residual"]) for r in group if r["terminal_residual"]]
    engine_actual = step_curve(actual_deltas)
    zoom_mean = {"no_decay": instant.value(zoom_grid), "fifo_decay": fifo[selected].value(zoom_grid), "engine_actual": engine_actual.value(zoom_grid)}
    figure, axes = plt.subplots(2, 1, figsize=(14, 7.5), sharex=True)
    for key, values in zoom_mean.items():
        axes[0].plot(zoom_grid, values / 1000, color=colors[key], label=labels[key], lw=1.6)
    axes[0].legend(fontsize=8)
    axes[0].set_ylabel("Work on engine (K tokens)")
    axes[0].set_title("One engine: aggregate outstanding work", loc="left")
    selected_requests = sorted([r for r in group if r["queued"] <= hi and r["terminal"] >= lo and r["uncached"] >= 10000], key=lambda r: r["queued"])
    # Reconstruct individual FIFO curves by applying the same shared budget.
    individual = individual_curves(group, args.rate)
    for i, r in enumerate(selected_requests):
        curve = individual[r["request_id"]]
        values = curve.value(zoom_grid)
        axes[1].plot(zoom_grid, values / 1000, lw=1.4, label=f"{r['request_id'].removeprefix('chatcmpl-')[:6]}: {r['uncached']/1000:.0f}K initial")
        axes[1].axvline(r["terminal"], color="#aaaaaa", lw=.6, ls=":")
    axes[1].set_title("Individual estimated contributions (dotted lines: observed first tokens)", loc="left")
    axes[1].set_ylabel("Remaining per request (K tokens)")
    axes[1].set_xlabel("Seconds since profiling start")
    axes[1].legend(fontsize=8, ncol=3)
    for axis in axes:
        axis.set_xlim(lo, hi)
        axis.set_ylim(bottom=0)
    figure.suptitle(f"Overlapping prefills · {selected} · shared {args.rate/1000:g}K tokens/s budget", x=.07, ha="left")
    figure.tight_layout(rect=(0, 0, 1, .95))
    save_plot(figure, output / "overlap_zoom")
    summary["overlap_zoom"] = {"engine": selected, "start_s": lo, "end_s": hi}
    (output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "coverage"}, indent=2))
    print(f"Artifacts: {output.resolve()}")


def individual_curves(requests, rate):
    """Build individual contributions for an explanatory zoom, not for metrics."""
    events = sorted([(r["queued"], 1, i) for i, r in enumerate(requests)] + [(r["terminal"], 0, i) for i, r in enumerate(requests)])
    curves = {i: Curve() for i in range(len(requests))}
    pending = OrderedDict()
    now = events[0][0]
    for time, kind, index in events:
        while pending and time > now:
            head, remaining = next(iter(pending.items()))
            exhausted_at = now + remaining / rate
            if exhausted_at <= time:
                now = exhausted_at
                curves[head].put(now, 0)
                pending.pop(head)
                if pending:
                    next_head, next_remaining = next(iter(pending.items()))
                    curves[next_head].put(now, next_remaining, -rate)
            else:
                pending[head] -= rate * (time - now)
                now = time
        now = time
        if kind:
            remaining = requests[index]["uncached"]
            if remaining:
                pending[index] = remaining
            curves[index].put(now, remaining, -rate if next(iter(pending), None) == index else 0)
        else:
            head = next(iter(pending), None)
            pending.pop(index, None)
            curves[index].put(now, 0)
            if head == index and pending:
                next_head, next_remaining = next(iter(pending.items()))
                curves[next_head].put(now, next_remaining, -rate)
    return {requests[i]["request_id"]: c.finish() for i, c in curves.items()}


if __name__ == "__main__":
    main()
