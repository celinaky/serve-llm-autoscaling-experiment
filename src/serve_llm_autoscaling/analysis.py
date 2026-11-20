"""Offline analysis of an AgentX session-curve run directory.

Everything here reads saved artifacts only, so it runs without Ray or a GPU
and can be repeated with different windows after the experiment.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

from .artifacts import write_json
from .config import (
    WORKLOAD_DIR,
    AnalysisConfig,
    ExperimentConfig,
    WorkloadConfig,
    load_saved_config,
)
from .evaluation import evaluate_run
from .telemetry import normalize_metric_name
from .windows import (
    configured_series_concurrency_at,
    profiling_requests,
    request_timeseries,
)

# Written into plot_data.json and concurrency_qps.json.
SCHEMA_VERSION = 4

# key -> (normalized sample name, aggregation, identity labels[, label match])
#
# A fourth element restricts the series to samples whose labels equal the
# given values: the controller publishes every policy metric under one
# family, serve_autoscaling_policy_metric, told apart by its metric label.
#
# A series is identified by application, deployment and the identity labels:
# the handle for router metrics, nothing more for controller gauges. Copies of
# one series exported by several nodes are deduplicated; samples missing an
# identity label are excluded, since they cannot be told apart.
METRIC_SERIES = {
    "desired_replicas": ("serve_autoscaling_desired_replicas", "sum", ()),
    "target_replicas": ("serve_autoscaling_target_replicas", "sum", ()),
    # The token policy publishes its raw prefill backlog snapshot and the
    # lookback-mean prefill required replicas. The decode series are kept so
    # runs from the two-term policy still load.
    "token_prefill_backlog": (
        "serve_autoscaling_policy_metric", "sum", (), {"metric": "outstanding_effective_prefill_tokens"}
    ),
    "token_active_decodes": (
        "serve_autoscaling_policy_metric", "sum", (), {"metric": "active_decode_requests"}
    ),
    "token_prefill_replicas": (
        "serve_autoscaling_policy_metric", "sum", (), {"metric": "prefill_required_replicas"}
    ),
    "token_decode_replicas": (
        "serve_autoscaling_policy_metric", "sum", (), {"metric": "decode_required_replicas"}
    ),
    # The look-back mean of outstanding uncached prefill tokens each KV router
    # reports, one series per router (see ``router_report``). The token policy
    # scales on the max across routers; metric_series divides it by the
    # target to rebuild token_prefill_replicas when the policy metric gauge
    # is absent.
    "token_router_prefill_mean": (
        "serve_autoscaling_router_metric", "max", ("handle",),
        {"metric": "outstanding_effective_prefill_tokens"},
    ),
}


ROUTER_PREFILL_METRIC = "outstanding_effective_prefill_tokens"


def router_report(sample: dict[str, Any]) -> dict[str, Any]:
    """Present the router's reported prefill load in one shape across Ray versions.

    Current Ray exports ``serve_autoscaling_replica_metric`` on the LLMRouter
    deployment, one series per router replica, with the server it is bound to
    in the metric label (``outstanding_effective_prefill_tokens:<server>``).
    Earlier branches exported ``serve_autoscaling_router_metric`` on the
    server deployment, one series per handle. A current sample is returned
    in the earlier shape; anything else is returned unchanged.
    """
    labels = sample.get("labels") or {}
    metric = labels.get("metric", "")
    if (normalize_metric_name(sample["name"]) != "serve_autoscaling_replica_metric"
            or not metric.startswith(ROUTER_PREFILL_METRIC + ":")):
        return sample
    converted = {
        **labels,
        "deployment": metric[len(ROUTER_PREFILL_METRIC) + 1:],
        "metric": ROUTER_PREFILL_METRIC,
    }
    if "replica" in labels:  # without it the series cannot be told apart
        converted["handle"] = labels["replica"]
    return {**sample, "name": "ray_serve_autoscaling_router_metric", "labels": converted}


def read_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open() as fh:
        for line in fh:
            try:
                record = json.loads(line)
            except ValueError:
                continue  # e.g. a line truncated by an interrupted run
            if isinstance(record, dict):
                yield record


def load_run_config(root: Path) -> ExperimentConfig:
    return load_saved_config(root / "resolved.yaml")


def workload_artifact_dir(root: Path, config: ExperimentConfig) -> Path:
    """The run's raw AIPerf artifact directory."""
    return root / "benchmark" / WORKLOAD_DIR


# --- Lifecycle ----------------------------------------------------------------


def read_lifecycle(root: Path) -> dict[str, Any] | None:
    try:
        return json.loads((root / "benchmark" / "lifecycle.json").read_text())
    except (OSError, ValueError):
        return None  # runs recorded before lifecycle.json, or interrupted ones


def lifecycle_horizon_s(lifecycle: dict[str, Any] | None) -> float | None:
    """Seconds from profiling start to the end of the recorded run."""
    if not lifecycle or lifecycle.get("profiling_start_ns") is None:
        return None
    end = lifecycle.get("observation_end_ns") or lifecycle.get("aiperf_exited_ns")
    return None if end is None else (end - lifecycle["profiling_start_ns"]) / 1e9


def lifecycle_periods(
    config: ExperimentConfig,
    t0: int,
    profiling_end_ns: int | None,
    lifecycle: dict[str, Any] | None,
) -> dict[str, Any]:
    """Load, drain and idle periods on the profiling clock.

    Draining ends when AIPerf closes the profiling phase; from then on no
    request is in flight, whether AIPerf is still exporting or has exited and
    the harness is observing recovery.
    """
    benchmark = config.benchmark

    def rel(ns: int | None) -> float | None:
        return None if ns is None else (ns - t0) / 1e9

    lifecycle = lifecycle or {}
    load_end = benchmark.duration_s
    aiperf_exit = rel(lifecycle.get("aiperf_exited_ns"))
    drain_end = rel(profiling_end_ns)
    if drain_end is None or drain_end < load_end:
        drain_end = aiperf_exit
    observation_end = rel(lifecycle.get("observation_end_ns"))
    markers = {
        "ramp_end_s": None,
        "load_end_s": load_end,
        "drain_end_s": drain_end,
        "aiperf_exit_s": aiperf_exit,
        "observation_end_s": observation_end,
    }
    periods = [{"name": "load", "start_s": 0.0, "end_s": load_end}]
    if drain_end is not None and drain_end > load_end:
        periods.append({"name": "drain", "start_s": load_end, "end_s": drain_end})
    idle_end = observation_end or aiperf_exit
    if drain_end is not None and idle_end is not None and idle_end > drain_end:
        periods.append({"name": "idle", "start_s": drain_end, "end_s": idle_end})
    return {
        "source": "lifecycle.json" if lifecycle else "config",
        "post_load_observation_s": benchmark.post_load_observation_s,
        "observation_skipped": lifecycle.get("observation_skipped"),
        "markers": markers,
        "periods": periods,
    }


# --- Configured load ----------------------------------------------------------


def configured_load(config: ExperimentConfig, artifact_dir: Path) -> dict[str, Any]:
    """The load the harness asked for, tagged with what it measures.

    This is session-tree concurrency, not a request rate: the request rate
    it produces is an observed outcome.
    """
    benchmark = config.benchmark
    workload = benchmark.workload
    # The curve's control points, held at the last value until load ends.
    curve = workload.concurrency_series
    points = [{"time_s": p.time_s, "value": float(p.sessions)} for p in curve]
    if benchmark.duration_s > curve[-1].time_s:
        points.append({"time_s": benchmark.duration_s, "value": float(curve[-1].sessions)})
    return {"dimension": "session_concurrency", "unit": "session trees",
            "label": "configured session concurrency", "points": points,
            "markers": [p.time_s for p in curve[1:]]}


def concurrency_qps(
    workload: WorkloadConfig, timeseries: dict[str, Any], duration_s: float
) -> dict[str, Any]:
    """Configured session concurrency beside the request rates it generated.

    Concurrency is the configured curve at each window's midpoint. The live session count can
    briefly differ because of scheduling and tree turnover. No sessions are
    configured once load issuance ends at ``duration_s``.
    """
    curve = [(p.time_s, float(p.sessions)) for p in workload.concurrency_series]
    rows = []
    for w in timeseries["windows"]:
        mid = (w["window_start_s"] + w["window_end_s"]) / 2
        if mid >= duration_s:
            sessions = 0.0
        else:
            sessions = configured_series_concurrency_at(mid, curve)
        row = {
            "window_start_s": w["window_start_s"],
            "window_end_s": w["window_end_s"],
            "configured_session_concurrency": sessions,
            "offered_qps": w["offered_rps"],
            "started_qps": w["started_rps"],
            "successful_completed_qps": w["successful_completed_rps"],
            "failed_completed_qps": w["failed_completed_rps"],
            "in_flight_at_end": w["in_flight_at_end"],
            **({"live_session_trees_at_end": w["live_session_trees_at_end"]}
               if "live_session_trees_at_end" in w else {}),
            # A diagnostic only: it varies with trace mix, think time,
            # latency and subagent activity.
            "requests_per_second_per_configured_session": (
                w["offered_rps"] / sessions if sessions else None
            ),
        }
        if "offered_root_rps" in w:
            row["offered_root_qps"] = w["offered_root_rps"]
            row["offered_subagent_qps"] = w["offered_subagent_rps"]
        rows.append(row)
    result: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "target_concurrency": workload.target_concurrency,
        "concurrency_series": [p.model_dump() for p in workload.concurrency_series],
    }
    return {
        **result,
        "window_s": timeseries["window_s"],
        "summary": timeseries["summary"],
        "windows": rows,
    }


# --- Serve status -----------------------------------------------------------


def relevant_deployments(names: set[str]) -> set[str]:
    """The LLM server deployments; the ingress does not run the model."""
    llm = {name for name in names if name.startswith("LLMServer")}
    return llm or set(names)


STARTING_STATES = ("STARTING", "RECOVERING")


def status_samples(
    records: list[dict[str, Any]], deployments: set[str], t0: int
) -> list[dict[str, Any]]:
    """Replica counts per status sample; unavailable samples carry only an error.

    An absent application or deployment is a gap, never zero replicas.
    """
    samples = []
    for record in records:
        sample = {
            "relative_time_s": (record["timestamp_ns"] - t0) / 1e9,
            "timestamp_ns": record["timestamp_ns"],
            "error": record.get("error"),
        }
        chosen = [
            d for name, d in (record.get("deployments") or {}).items() if name in deployments
        ]
        if not sample["error"] and not chosen:
            sample["error"] = "no matching deployment in Serve status"
        if not sample["error"]:
            states: dict[str, int] = defaultdict(int)
            for d in chosen:
                # ``replicas_by_state`` is the key in runs recorded before serve.status().
                for state, n in (d.get("replica_states") or d.get("replicas_by_state") or {}).items():
                    states[state] += n
            sample.update({
                "application_status": record.get("application_status"),
                "live_replicas": sum(n for state, n in states.items() if state != "STOPPED"),
                "running_replicas": states.get("RUNNING", 0),
                "starting_replicas": sum(states.get(state, 0) for state in STARTING_STATES),
                "stopping_replicas": states.get("STOPPING", 0),
                "replicas_by_state": dict(states),
            })
        samples.append(sample)
    return samples


# --- Prometheus aggregation -------------------------------------------------


def aggregate_metric(
    ticks: list[tuple[int, list[tuple[str | None, dict[str, Any]]]]],
    aggregation: str,
    identity: tuple[str, ...],
    t0: int,
    warnings: list[str],
    metric: str,
) -> list[dict[str, Any]]:
    """Aggregate one metric per scrape tick.

    ``ticks`` holds (timestamp_ns, [(node_id, sample), ...]) in time order.
    Ticks without samples produce no point, so gaps are not zeros. A tick in
    which copies of one series disagree is omitted rather than guessed.
    """
    points = []
    unidentified = conflicting = 0
    for timestamp_ns, samples in ticks:
        series: dict[tuple, set[float]] = defaultdict(set)
        for _, sample in samples:
            labels = sample["labels"]
            if sample.get("value") is None:
                continue
            if any(key not in labels for key in identity):
                unidentified += 1
                continue
            key = (labels.get("application"), labels.get("deployment"),
                   *(labels[k] for k in identity))
            series[key].add(sample["value"])
        if any(len(values) > 1 for values in series.values()):
            conflicting += 1
            continue
        current = {key: values.pop() for key, values in series.items()}
        if not current:
            continue
        if aggregation == "sum":
            value = sum(current.values())
        elif aggregation == "max":
            value = max(current.values())
        else:
            raise ValueError(f"unknown aggregation {aggregation!r}")
        points.append({
            "relative_time_s": (timestamp_ns - t0) / 1e9,
            "value": value,
            "series_count": len(current),
        })
    if unidentified:
        warnings.append(
            f"{metric}: excluded {unidentified} samples missing identity labels "
            f"{list(identity)}"
        )
    if conflicting:
        warnings.append(
            f"{metric}: omitted {conflicting} scrapes where copies of one series disagreed"
        )
    return points


def metric_series(
    records: list[dict[str, Any]],
    application: str,
    deployments: set[str],
    interval_s: float,
    t0: int,
    warnings: list[str],
    target_prefill_tokens_per_replica: float | None = None,
) -> dict[str, dict[str, Any]]:
    ticks: dict[Any, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record.get("error") or record.get("timestamp_ns") is None:
            continue
        tick = record.get("tick")
        if tick is None:
            tick = round(record["timestamp_ns"] / 1e9 / interval_s)
        ticks[tick].append(record)

    def attributed(labels: dict[str, str]) -> bool:
        return (
            labels.get("application") == application
            and labels.get("deployment") in deployments
        )

    unlabeled: dict[str, int] = defaultdict(int)
    # Other recorded families (older runs kept many more) are ignored.
    by_metric: dict[str, list] = {spec[0]: [] for spec in METRIC_SERIES.values()}
    for tick in sorted(ticks):
        tick_records = ticks[tick]
        timestamp_ns = max(r["timestamp_ns"] for r in tick_records)
        selected: dict[str, list] = defaultdict(list)
        for record in tick_records:
            for sample in record.get("samples") or []:
                sample = router_report(sample)
                name = normalize_metric_name(sample["name"])
                if name not in by_metric:
                    continue
                labels = sample.get("labels") or {}
                if "application" not in labels or "deployment" not in labels:
                    unlabeled[name] += 1  # cannot be attributed to the deployment
                    continue
                elif not attributed(labels):
                    continue
                selected[name].append((record.get("node_id"), sample))
        for name in by_metric:
            by_metric[name].append((timestamp_ns, selected.get(name, [])))

    for name in sorted(unlabeled):
        warnings.append(
            f"{name}: excluded {unlabeled[name]} samples without application/deployment labels"
        )
    result = {}
    for key, (name, aggregation, identity, *rest) in METRIC_SERIES.items():
        match = rest[0] if rest else {}
        ticks_for_key = by_metric[name]
        if match:
            ticks_for_key = [
                (ts, [(node, s) for node, s in samples
                      if all(s["labels"].get(k) == v for k, v in match.items())])
                for ts, samples in ticks_for_key
            ]
        label = name if not match else f"{name}{{{', '.join(f'{k}={v}' for k, v in match.items())}}}"
        points = aggregate_metric(ticks_for_key, aggregation, identity, t0, warnings, label)
        if points:
            result[key] = {"metric": name, "aggregation": aggregation, "points": points}
    # Runs without the policy metric gauge: the policy's prefill required
    # replicas is the controller's max per-handle look-back mean over the target.
    mean = result.get("token_router_prefill_mean")
    if "token_prefill_replicas" not in result and mean and target_prefill_tokens_per_replica:
        result["token_prefill_replicas"] = {
            "metric": f"{mean['metric']} / target_prefill_tokens_per_replica",
            "aggregation": "max",
            "points": [
                {**p, "value": p["value"] / target_prefill_tokens_per_replica}
                for p in mean["points"]
            ],
        }
    return result


# --- Plot data --------------------------------------------------------------


def build_plot_data(
    root: Path,
    config: ExperimentConfig,
    analysis: AnalysisConfig,
    timeseries: dict[str, Any],
    lifecycle: dict[str, Any] | None = None,
) -> dict[str, Any]:
    artifact_dir = workload_artifact_dir(root, config)
    warnings: list[str] = []
    t0 = timeseries["profiling_start_ns"]
    if t0 is None:
        raise ValueError("no profiling records or phase manifest; cannot align timestamps")
    if timeseries["time_origin_source"] != "phase_manifest":
        warnings.append("phase_manifest.json unavailable; time origin is the first profiling credit")
    if timeseries["malformed_record_count"]:
        warnings.append(
            f"{timeseries['malformed_record_count']} malformed AIPerf records were skipped"
        )

    application = config.deployment.application_name
    status_path = root / "telemetry" / "serve_status.jsonl"
    metrics_path = root / "telemetry" / "serve_metrics.jsonl"
    status_records = list(read_jsonl(status_path)) if status_path.exists() else []
    metric_records = list(read_jsonl(metrics_path)) if metrics_path.exists() else []

    names = {n for r in status_records for n in (r.get("deployments") or {})}
    names |= {
        s["labels"]["deployment"]
        for r in metric_records for s in r.get("samples") or []
        if s.get("labels", {}).get("application") == application
        and "deployment" in s.get("labels", {})
    }
    deployments = relevant_deployments(names)

    status = status_samples(status_records, deployments, t0)
    if not status_path.exists():
        warnings.append("Serve status telemetry was not collected; replica counts are missing")
    elif not any(s.get("error") is None for s in status):
        warnings.append("every Serve status sample failed; replica counts are missing")
    elif errors := sum(1 for s in status if s.get("error")):
        warnings.append(f"{errors} of {len(status)} Serve status samples failed")

    if not metrics_path.exists():
        warnings.append("Prometheus metrics were not collected; Serve-side signals are omitted")
    elif metric_records:
        failed = sum(1 for record in metric_records if record.get("error"))
        if failed > 0.1 * len(metric_records):
            warnings.append(
                f"{failed} of {len(metric_records)} metrics scrapes failed: the policy "
                "signal and other Serve-side series have gaps (see the error field in "
                "telemetry/serve_metrics.jsonl; a slow metrics page on a long-lived "
                "cluster is the usual cause)"
            )
    series = metric_series(
        metric_records, application, deployments, analysis.metrics_interval_s, t0, warnings,
        target_prefill_tokens_per_replica=config.deployment.autoscaling.policy.kwargs.get(
            "target_prefill_tokens_per_replica"
        ),
    )
    if metrics_path.exists() and "desired_replicas" not in series:
        warnings.append(
            f"{METRIC_SERIES['desired_replicas'][0]} was not exported; scale events "
            "fall back to the target replica count"
        )
    autoscaling = config.deployment.autoscaling
    if autoscaling.policy.name == "token" and "token_prefill_replicas" not in series:
        warnings.append("token policy gauges were not exported; its evaluation is omitted")

    manifest_end = None
    try:
        phases = json.loads((artifact_dir / "phase_manifest.json").read_text())["phases"]
        manifest_end = next(
            p.get("end_ns") for p in phases if p.get("phase_kind") == "profiling"
        )
    except (OSError, ValueError, KeyError, StopIteration):
        pass

    periods = lifecycle_periods(config, t0, manifest_end, lifecycle)
    if periods["observation_skipped"]:
        warnings.append(f"post-load observation skipped: {periods['observation_skipped']}")
    return {
        "schema_version": SCHEMA_VERSION,
        "experiment": {
            "name": config.name,
            "model_id": config.deployment.model_id,
            "application": application,
            "deployments": sorted(deployments),
            "min_replicas": autoscaling.min_replicas,
            "initial_replicas": autoscaling.initial_replicas,
            "max_replicas": autoscaling.max_replicas,
            "target_ongoing_requests": autoscaling.target_ongoing_requests,
            "upscale_delay_s": autoscaling.upscale_delay_s,
            "downscale_delay_s": autoscaling.downscale_delay_s,
            "duration_s": config.benchmark.duration_s,
            "grace_period_s": config.benchmark.grace_period_s,
            "post_load_observation_s": config.benchmark.post_load_observation_s,
            "autoscaling_policy": autoscaling.policy.name,
            "autoscaling_policy_kwargs": autoscaling.policy.kwargs,
            "routing_policy": config.deployment.routing.policy,
            "direct_streaming": config.deployment.routing.direct_streaming,
            "window_s": analysis.window_s,
            "tail_window_s": analysis.tail_window_s,
            "status_interval_s": analysis.status_interval_s,
            "metrics_interval_s": analysis.metrics_interval_s,
            "ttft_slo_ms": analysis.ttft_slo_ms,
            "itl_slo_ms": analysis.itl_slo_ms,
            "slo_percentile": analysis.slo_percentile,
            "warmup_s": analysis.warmup_s,
            "gpus_per_replica": config.deployment.engine.tensor_parallel_size or 1,
            "mode": "agentx_concurrency_ramp",
        },
        "time_origin": {
            "profiling_start_ns": t0,
            "source": timeseries["time_origin_source"],
            "profiling_end_s": (manifest_end - t0) / 1e9 if manifest_end else None,
        },
        "configured_load": configured_load(config, artifact_dir),
        "lifecycle": periods,
        "request_windows": timeseries["windows"],
        "serve_status_samples": status,
        "serve_metric_series": series,
        "availability": {
            "replica_status": any(s.get("error") is None for s in status),
            "desired_replicas": "desired_replicas" in series,
            "target_replicas_metric": "target_replicas" in series,
            "token_policy_metrics": "token_prefill_replicas" in series,
        },
        "warnings": warnings,
    }


def analyze_run(
    root: Path,
    *,
    window_s: float | None = None,
    tail_window_s: float | None = None,
    plot: bool | None = None,
    traces_dir: Path | None = None,
) -> dict[str, Any]:
    """Regenerate every derived analysis artifact from a run's raw files."""
    config = load_run_config(root)
    overrides = {k: v for k, v in (("window_s", window_s), ("tail_window_s", tail_window_s))
                 if v is not None}
    analysis = AnalysisConfig.model_validate({**config.analysis.model_dump(), **overrides})
    out = root / "analysis"

    lifecycle = read_lifecycle(root)
    artifact_dir = workload_artifact_dir(root, config)
    timeseries = request_timeseries(
        artifact_dir,
        window_s=analysis.window_s,
        tail_window_s=analysis.tail_window_s,
        ttft_slo_ms=analysis.ttft_slo_ms,
        itl_slo_ms=analysis.itl_slo_ms,
        duration_s=config.benchmark.duration_s,
        horizon_s=lifecycle_horizon_s(lifecycle),
    )
    write_json(out / "request_timeseries.json", timeseries)
    write_json(out / "concurrency_qps.json", concurrency_qps(
        config.benchmark.workload, timeseries, config.benchmark.duration_s
    ))
    plot_data = build_plot_data(root, config, analysis, timeseries, lifecycle)
    write_json(out / "plot_data.json", plot_data)
    requests, _ = profiling_requests(artifact_dir)
    write_json(out / "autoscaling_evaluation.json", evaluate_run(plot_data, requests))

    # ``plot_error`` is always None: plot failures are reported per plot in
    # ``diagnostic_plots``. The key remains for the CLI and saved manifests.
    result: dict[str, Any] = {"warnings": plot_data["warnings"], "plot_error": None}
    if plot is None or plot:
        from .diagnostics.pipeline import generate_diagnostics

        diagnostics = generate_diagnostics(root, config, traces_dir=traces_dir)
        result["diagnostic_plots"] = diagnostics
        result["warnings"].extend(diagnostics["warnings"])
    return result
