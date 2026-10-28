"""Render the autoscaling timeline from ``analysis/plot_data.json`` alone."""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


PERIOD_COLORS = {"drain": "tab:orange", "idle": "tab:green"}
PERIOD_LABELS = {"drain": "AIPerf draining", "idle": "idle (no requests)"}
# SLA policy states, from fallback through learned decisions.
REASON_COLORS = {
    "metrics_unavailable": "tab:gray",
    "model_not_ready": "tab:brown",
    "scaling_in_progress": "tab:olive",
    "unsupported_workload": "tab:orange",
    "no_supported_candidate_meets_slo": "tab:red",
    "capacity": "tab:green",
    "safe_scale_down": "tab:cyan",
    "observed_slo_violation": "tab:purple",
}


def primary_points(
    points: list[dict[str, Any]], key: str, *, step: bool = False
) -> tuple[list[float], list[float]]:
    """Points on the profiling clock (t >= 0).

    Pre-profile samples are excluded; a step series instead carries its last
    pre-profile value to t=0 so the state at the start is still visible.
    """
    xs: list[float] = []
    ys: list[float] = []
    carried = None
    for p in points:
        value = p.get(key)
        if value is None:
            continue
        if p["relative_time_s"] < 0:
            carried = value
            continue
        if step and carried is not None and not xs and p["relative_time_s"] > 0:
            xs.append(0.0)
            ys.append(carried)
        xs.append(p["relative_time_s"])
        ys.append(value)
    return xs, ys


def _nan(value: Any) -> float:
    return math.nan if value is None else value



def _setting(value):
    """A configured number, or a note that Serve's default applies."""
    return "serve default" if value is None else f"{value:g}"

def _title(experiment: dict[str, Any]) -> str:
    policy = experiment.get("autoscaling_policy", "default")
    if "advisory" in (experiment.get("autoscaling_policy_kwargs") or {}):
        policy += f" (advisory={experiment['autoscaling_policy_kwargs']['advisory']})"
    return (
        f"{experiment['name']} · {experiment['model_id']}\n"
        f"replicas {_setting(experiment.get('min_replicas'))}-"
        f"{_setting(experiment.get('max_replicas'))} "
        f"(initial {_setting(experiment.get('initial_replicas'))}) · "
        f"autoscaling {policy} · "
        f"target_ongoing_requests={_setting(experiment.get('target_ongoing_requests'))} · "
        f"routing {experiment.get('routing_policy') or 'serve default'}"
        f"{' (direct streaming)' if experiment.get('direct_streaming') else ''}\n"
        f"window {experiment['window_s']:g}s · "
        f"status every {experiment['status_interval_s']:g}s · "
        f"metrics every {experiment['metrics_interval_s']:g}s"
    )


def _series(metrics: dict[str, Any], key: str, **kwargs: Any) -> tuple[list[float], list[float]]:
    return primary_points(metrics[key]["points"], "value", **kwargs) if key in metrics else ([], [])


def _latency_panel(
    ax: Any, windows: list[dict[str, Any]], name: str, label: str,
    slo: float | None, tail_window_s: float,
) -> None:
    """Percentiles by window, rolling mean and p99, the SLO and its violations.

    Windows whose rolling mean exceeds the SLO are shaded: that is the
    condition a mean-latency policy is meant to prevent.
    """
    mids = [(w["window_start_s"] + w["window_end_s"]) / 2 for w in windows]
    ends = [w["window_end_s"] for w in windows]
    for q in ("p50", "p90", "p99"):
        ax.plot(mids, [_nan(w[f"{q}_{name}_ms"]) for w in windows], marker=".", ms=4,
                label=f"{label} {q}")
    if windows and f"rolling_mean_{name}_ms" in windows[0]:
        ax.plot(ends, [_nan(w[f"rolling_mean_{name}_ms"]) for w in windows], "k-", lw=2,
                label=f"rolling {tail_window_s:g}s mean")
        ax.plot(ends, [_nan(w[f"rolling_p99_{name}_ms"]) for w in windows], "k-", lw=0.8,
                alpha=0.6, label=f"rolling {tail_window_s:g}s p99")
    values = [w[k] for w in windows for k in (f"p50_{name}_ms", f"p99_{name}_ms") if w[k]]
    if values and max(values) / max(min(values), 1e-9) >= 20:
        ax.set_yscale("log")
    if slo:
        ax.axhline(slo, color="red", ls=":", label=f"SLO {slo:g}ms (mean)")
        for w in windows:
            rolling = w.get(f"rolling_mean_{name}_ms")
            if rolling is not None and rolling > slo:
                ax.axvspan(w["window_start_s"], w["window_end_s"], color="red", alpha=0.12, lw=0)
    ax.set_ylabel(f"{label} (ms)")
    ax.set_title(f"{label} (by request start)", loc="left", fontsize=10)


def _sla_panel(ax: Any, data: dict[str, Any]) -> dict[Any, Any]:
    """What the policy predicted against what was measured, with its state."""
    metrics = data["serve_metric_series"]
    windows = data["request_windows"]
    ends = [w["window_end_s"] for w in windows]
    right = ax.twinx()
    for axis, name, label, style in ((ax, "ttft", "TTFT", "-"), (right, "itl", "ITL", "--")):
        xs, ys = _series(metrics, f"sla_predicted_{name}_ms")
        axis.plot(xs, ys, style, color="tab:purple", lw=1.5, label=f"{label} predicted")
        xs, ys = _series(metrics, f"vllm_mean_{name}_ms")
        axis.plot(xs, ys, style, color="tab:blue", lw=1.5, label=f"{label} server mean")
        if windows and f"rolling_mean_{name}_ms" in windows[0]:
            axis.plot(ends, [_nan(w[f"rolling_mean_{name}_ms"]) for w in windows], style,
                      color="tab:gray", lw=1, label=f"{label} client mean")
    ax.set_ylabel("TTFT (ms)")
    right.set_ylabel("ITL (ms)")

    xs, ys = _series(metrics, "sla_model_ready")
    ready = next((x for x, y in zip(xs, ys) if y >= 1), None)
    if ready is not None:
        ax.axvline(ready, color="green", lw=1.2, label=f"model ready {ready:.0f}s")
    # The policy's state as bands: transitions can come seconds apart, so
    # each state is labelled once in the legend rather than at every change.
    events = data["sla_policy_events"]
    end = max([w["window_end_s"] for w in windows] + [e["relative_time_s"] for e in events] + [0.0])
    labelled = set()
    for event, following in zip(events, events[1:] + [None]):
        start = max(event["relative_time_s"], 0.0)
        stop = end if following is None else following["relative_time_s"]
        if stop <= start:
            continue
        reason = event["reason"]
        ax.axvspan(start, stop, color=REASON_COLORS.get(reason, "tab:gray"), alpha=0.18, lw=0,
                   label=None if reason in labelled else reason)
        labelled.add(reason)
    ax.set_title("SLA policy", loc="left", fontsize=10)
    return {ax: right}


def _token_panel(ax: Any, data: dict[str, Any]) -> dict[Any, Any]:
    """The token policy's inputs beside the replicas each term asked for."""
    metrics = data["serve_metric_series"]
    right = ax.twinx()
    xs, ys = _series(metrics, "token_prefill_backlog")
    ax.plot(xs, [y / 1000 for y in ys], color="tab:blue", lw=1.5, label="outstanding prefill (K tokens)")
    xs, ys = _series(metrics, "token_active_decodes")
    right.plot(xs, ys, color="tab:orange", lw=1.5, label="active decode requests")
    for key, label, style in (("token_prefill_replicas", "prefill term wants", {"color": "tab:blue"}),
                              ("token_decode_replicas", "decode term wants", {"color": "tab:orange"})):
        xs, ys = _series(metrics, key, step=True)
        right.step(xs, ys, where="post", ls=":", lw=1.2, label=label, **style)
    ax.set_ylabel("K tokens")
    right.set_ylabel("requests / replicas")
    ax.set_ylim(bottom=0)
    right.set_ylim(bottom=0)
    ax.set_title("Token policy inputs", loc="left", fontsize=10)
    return {ax: right}


def plot_timeline(plot_data_path: Path, out_path: Path) -> None:
    data = json.loads(Path(plot_data_path).read_text())
    experiment = data["experiment"]
    windows = data["request_windows"]
    status = data["serve_status_samples"]
    metrics = data["serve_metric_series"]
    availability = data.get("availability") or {}
    with_sla = availability.get("sla_policy_metrics") or availability.get("sla_policy_events")
    with_token = availability.get("token_policy_metrics")

    panels = 5 + bool(with_sla) + bool(with_token)
    fig, axes = plt.subplots(panels, 1, figsize=(14, 3.6 * panels), sharex=True)
    traffic, ttft, itl, pressure, replicas = axes[:5]
    twins = {}

    edges = [w["window_start_s"] for w in windows[:1]] + [w["window_end_s"] for w in windows]
    mids = [(w["window_start_s"] + w["window_end_s"]) / 2 for w in windows]
    ends = [w["window_end_s"] for w in windows]

    # Panel 1: traffic. Observed request rates are on the left axis; AgentX
    # configures session-tree concurrency, drawn on its own right axis.
    load = data["configured_load"]
    load_xs = [p["time_s"] for p in load["points"]]
    load_ys = [p["value"] for p in load["points"]]
    if load["dimension"] == "request_rate":
        traffic.plot(load_xs, load_ys, "k--", lw=1.5, label=load["label"])
    else:
        sessions = twins[traffic] = traffic.twinx()
        sessions.plot(load_xs, load_ys, "k--", lw=1.5, label=load["label"])
        live = [w.get("live_session_trees_at_end") for w in windows]
        if any(v is not None for v in live):
            # From the records only: a tree counts from its first request start
            # to its last request end, so sessions idle in a long think gap, or
            # waiting on a turn that never came before load ended, read as gone.
            # The AIPerf log's "live sessions" count at each limit change is exact.
            sessions.plot(ends, [_nan(v) for v in live], color="tab:purple", lw=1.2,
                          label="session trees with requests (records)")
        sessions.set_ylim(bottom=0, top=max(load_ys + [v or 0 for v in live]) * 1.1)
        sessions.set_ylabel(load["unit"])
    for key, label, lw in (("offered_rps", "offered", 3), ("started_rps", "started", 1.5),
                           ("successful_completed_rps", "completed (ok)", 1.5)):
        if windows:
            traffic.stairs([w[key] for w in windows], edges, label=label, lw=lw)
    if any(w["failed_completed_rps"] for w in windows):
        traffic.stairs([w["failed_completed_rps"] for w in windows], edges,
                       label="completed (failed)", lw=1.5, color="red")
    if any(w.get("offered_subagent_rps") for w in windows):
        traffic.stairs([w["offered_root_rps"] for w in windows], edges,
                       label="offered (root agents)", lw=1, ls=":")
        traffic.stairs([w["offered_subagent_rps"] for w in windows], edges,
                       label="offered (subagents)", lw=1, ls=":")
    traffic.set_ylabel("requests / s")
    traffic.set_title("Traffic", loc="left", fontsize=10)

    # Panels 2-3: user experience. Empty windows plot as gaps, not points.
    tail = experiment["tail_window_s"]
    _latency_panel(ttft, windows, "ttft", "TTFT", experiment.get("ttft_slo_ms"), tail)
    _latency_panel(itl, windows, "itl", "ITL", experiment.get("itl_slo_ms"), tail)

    # Panel 4: pressure. Missing Serve signals are omitted, never drawn as zero.
    pressure.plot(ends, [w["in_flight_at_end"] for w in windows], label="client in-flight")
    for key, label in (("ongoing_requests", "Serve ongoing (replicas)"),
                       ("handle_ongoing_requests", "Serve ongoing (handles)"),
                       ("router_queue", "router queue len"),
                       ("queued_queries", "handle queued queries"),
                       ("vllm_running_requests", "vLLM running"),
                       ("vllm_waiting_requests", "vLLM waiting")):
        if key in metrics:
            xs, ys = _series(metrics, key)
            pressure.plot(xs, ys, lw=1, label=label)
    pressure.set_ylabel("requests")
    queue = twins[pressure] = pressure.twinx()
    queue.plot(mids, [_nan(w["p99_client_queue_ms"]) for w in windows], color="gray",
               ls="--", lw=1, label="client queue p99")
    queue.set_ylabel("client queue delay (ms)")
    pressure.set_title("Pressure", loc="left", fontsize=10)

    # Panel 5: autoscaling, as step functions at the original sample times.
    # Unavailable status samples have no counts and break the line.
    observed = [experiment.get("max_replicas") or 1, 1]
    for key, label, style in (("target_replicas", "target", {"lw": 2.5}),
                              ("desired_replicas", "desired", {"lw": 1, "ls": "--"}),
                              ("sla_recommended_replicas", "SLA recommended",
                               {"lw": 1, "ls": ":", "color": "purple"})):
        if key in metrics:
            xs, ys = _series(metrics, key, step=True)
            replicas.step(xs, ys, where="post", label=label, **style)
            observed += ys
    for key, label in (("running_replicas", "running"), ("starting_replicas", "starting"),
                       ("stopping_replicas", "stopping")):
        xs, ys = primary_points(
            [s if s.get("error") is None else {**s, key: math.nan} for s in status],
            key, step=True,
        )
        if any(not math.isnan(y) for y in ys):
            replicas.step(xs, ys, where="post", label=label, lw=1.5)
            observed += [y for y in ys if not math.isnan(y)]
    if not replicas.lines:
        replicas.text(0.5, 0.5, "no replica telemetry", transform=replicas.transAxes,
                      ha="center", va="center", color="gray")
    replicas.set_ylim(bottom=0, top=max(observed) + 0.5)
    replicas.yaxis.get_major_locator().set_params(integer=True)
    replicas.set_ylabel("replicas")
    replicas.set_title("Autoscaling", loc="left", fontsize=10)

    # Policy panels: the SLA policy's predictions, readiness and state
    # transitions; the token policy's inputs and per-term demand.
    extra = iter(axes[5:])
    if with_sla:
        twins.update(_sla_panel(next(extra), data))
    if with_token:
        twins.update(_token_panel(next(extra), data))

    # Draining still has requests in flight; idle has none, so it shows
    # recovery and downscaling on their own.
    lifecycle = data.get("lifecycle") or {}
    boundaries = {v for k, v in (lifecycle.get("markers") or {}).items()
                  if v is not None and k != "ramp_end_s"}
    for ax in axes:
        for marker in load["markers"]:
            ax.axvline(marker, color="gray", alpha=0.3, lw=0.8)
        for boundary in boundaries:
            ax.axvline(boundary, color="black", alpha=0.5, lw=0.8, ls=":")
        for period in lifecycle.get("periods") or []:
            if period["name"] in PERIOD_COLORS:
                ax.axvspan(period["start_s"], period["end_s"], color=PERIOD_COLORS[period["name"]],
                           alpha=0.08, lw=0,
                           label=PERIOD_LABELS[period["name"]] if ax is traffic else None)
        ax.grid(alpha=0.2)
    for ax in axes:
        handles, labels = ax.get_legend_handles_labels()
        if ax in twins:
            extra_handles, extra_labels = twins[ax].get_legend_handles_labels()
            handles, labels = handles + extra_handles, labels + extra_labels
        if handles:
            ax.legend(handles, labels, loc="upper right", fontsize=8)

    end = max(ends + [s["relative_time_s"] for s in status] + [1.0]
              + [p["end_s"] for p in lifecycle.get("periods") or []])
    axes[-1].set_xlim(0, end)
    axes[-1].set_xlabel("seconds since profiling start")
    fig.suptitle(_title(experiment), fontsize=11)
    if data["warnings"]:
        fig.text(0.01, 0.005, "\n".join(data["warnings"][:6]), fontsize=7, color="dimgray",
                 va="bottom")
    fig.tight_layout(rect=(0, 0.02 + 0.012 * min(len(data["warnings"]), 6), 1, 0.96))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
