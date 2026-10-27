"""Offline analysis of a continuous (request-rate-series or AgentX) run directory.

Everything here reads saved artifacts only, so it runs without Ray or a GPU
and can be repeated with different windows after the experiment.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterator

import yaml

from .artifacts import write_json
from .config import (
    CONTINUOUS_MODE_DIRS,
    AgentXWorkloadConfig,
    AnalysisConfig,
    ExperimentConfig,
)
from .evaluation import evaluate_run
from .telemetry import SELECTED_METRICS, normalize_metric_name
from .windows import (
    configured_concurrency_at,
    configured_series_concurrency_at,
    profiling_requests,
    request_timeseries,
    slo_calibration,
)

# 2: ``request_rate_curve`` became the dimension-tagged ``configured_load``.
# 3: added ``lifecycle`` (load, drain and idle observation periods).
# 4: added SLA policy and vLLM series, ``sla_policy_events`` and the ITL SLO.
SCHEMA_VERSION = 4

# key -> (normalized sample name, aggregation, identity labels)
#
# A series is identified by application, deployment and the identity labels:
# the replica for replica metrics, the handle or router actor for router
# metrics, nothing more for controller gauges. Copies of one series exported by
# several nodes are deduplicated; samples missing an identity label are
# excluded, since they cannot be told apart. vLLM metrics carry no deployment
# labels; they are attributed through their ReplicaId instead.
METRIC_SERIES = {
    "ongoing_requests": ("serve_replica_processing_queries", "sum", ("replica",)),
    "handle_ongoing_requests": (
        "serve_num_ongoing_requests_at_replicas", "sum", ("handle", "actor_id")
    ),
    "router_queue": (
        "serve_request_router_queue_len", "sum", ("actor_id", "handle_source", "replica_id")
    ),
    "queued_queries": ("serve_deployment_queued_queries", "sum", ("handle", "actor_id")),
    "healthy_replicas": ("serve_deployment_replica_healthy", "healthy", ("replica",)),
    "desired_replicas": ("serve_autoscaling_desired_replicas", "sum", ()),
    "target_replicas": ("serve_autoscaling_target_replicas", "sum", ()),
    "autoscaling_total_requests": ("serve_autoscaling_total_requests", "sum", ()),
    "target_ongoing_requests": ("serve_autoscaling_target_ongoing_requests", "sum", ()),
    "replica_startups": ("serve_replica_startup_latency_ms_count", "counter_delta", ()),
    "sla_recommended_replicas": ("serve_sla_autoscaling_recommended_replicas", "sum", ()),
    "sla_model_ready": ("serve_sla_autoscaling_model_ready", "sum", ()),
    "sla_training_rows": ("serve_sla_autoscaling_training_rows", "sum", ()),
    "sla_training_buckets": ("serve_sla_autoscaling_training_buckets", "sum", ()),
    "sla_predicted_ttft_ms": ("serve_sla_autoscaling_predicted_ttft_ms", "sum", ()),
    "sla_predicted_itl_ms": ("serve_sla_autoscaling_predicted_itl_ms", "sum", ()),
    "token_prefill_backlog": ("serve_token_autoscaling_outstanding_prefill_tokens", "sum", ()),
    "token_active_decodes": ("serve_token_autoscaling_active_decode_requests", "sum", ()),
    "token_prefill_replicas": ("serve_token_autoscaling_prefill_required_replicas", "sum", ()),
    "token_decode_replicas": ("serve_token_autoscaling_decode_required_replicas", "sum", ()),
    "token_desired_replicas": ("serve_token_autoscaling_desired_replicas", "sum", ()),
    "vllm_running_requests": ("vllm_num_requests_running", "sum", ("ReplicaId",)),
    "vllm_waiting_requests": ("vllm_num_requests_waiting", "sum", ("ReplicaId",)),
}

# key -> (numerator counter, denominator counter, scale). Each point is the
# ratio of the counters' increases over the preceding ``tail_window_s``,
# summed across replicas: the means the SLA policy reads from Prometheus.
WINDOWED_RATIOS = {
    "vllm_mean_ttft_ms": (
        "vllm_time_to_first_token_seconds_sum",
        "vllm_time_to_first_token_seconds_count", 1000.0,
    ),
    "vllm_mean_itl_ms": (
        "vllm_request_time_per_output_token_seconds_sum",
        "vllm_request_time_per_output_token_seconds_count", 1000.0,
    ),
    "vllm_prefix_cache_hit_rate": (
        "vllm_prefix_cache_hits_total", "vllm_prefix_cache_queries_total", 1.0,
    ),
}


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
    with (root / "resolved.yaml").open() as fh:
        return ExperimentConfig.model_validate(yaml.safe_load(fh))


def workload_artifact_dir(root: Path, config: ExperimentConfig) -> Path:
    """The raw AIPerf artifact directory of a continuous run."""
    mode = config.benchmark.mode
    if mode not in CONTINUOUS_MODE_DIRS:
        raise ValueError(f"mode {mode!r} is not a continuous workload; nothing to analyze")
    return root / "benchmark" / CONTINUOUS_MODE_DIRS[mode]


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
    workload = benchmark.workload

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
        "ramp_end_s": (
            workload.concurrency_ramp_duration_s
            if isinstance(workload, AgentXWorkloadConfig) else None
        ),
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

    For AgentX this is session-tree concurrency, not a request rate: the
    request rate it produces is an observed outcome.
    """
    benchmark = config.benchmark
    workload = benchmark.workload
    if isinstance(workload, AgentXWorkloadConfig) and benchmark.mode == "agentx_fixed_concurrency":
        # Held at the target from the profiling start until load issuance ends.
        target = float(workload.target_concurrency)
        return {"dimension": "session_concurrency", "unit": "session trees",
                "label": "configured session concurrency",
                "points": [{"time_s": 0.0, "value": target},
                           {"time_s": benchmark.duration_s, "value": target},
                           {"time_s": benchmark.duration_s, "value": 0.0}],
                "markers": []}
    if isinstance(workload, AgentXWorkloadConfig) and workload.concurrency_series is not None:
        # The curve's control points, held at the last value until load ends.
        curve = workload.concurrency_series
        points = [{"time_s": p.time_s, "value": float(p.sessions)} for p in curve]
        if benchmark.duration_s > curve[-1].time_s:
            points.append({"time_s": benchmark.duration_s, "value": float(curve[-1].sessions)})
        return {"dimension": "session_concurrency", "unit": "session trees",
                "label": "configured session concurrency", "points": points,
                "markers": [p.time_s for p in curve[1:]]}
    if isinstance(workload, AgentXWorkloadConfig):
        ramp = workload.concurrency_ramp_duration_s
        points = [{"time_s": 0.0, "value": 1.0},
                  {"time_s": ramp, "value": float(workload.target_concurrency)}]
        if benchmark.duration_s > ramp:
            points.append({"time_s": benchmark.duration_s,
                           "value": float(workload.target_concurrency)})
        return {"dimension": "session_concurrency", "unit": "session trees",
                "label": "configured session concurrency", "points": points,
                "markers": [ramp]}
    try:
        curve = json.loads((artifact_dir / "rate_series.json").read_text())["points"]
    except (OSError, ValueError, KeyError):
        curve = [p.model_dump() for p in benchmark.rate_series or []]
    return {"dimension": "request_rate", "unit": "requests/s", "label": "configured QPS",
            "points": [{"time_s": p["time_s"], "value": p["qps"]} for p in curve],
            "markers": [p["time_s"] for p in curve[1:]]}


def concurrency_qps(
    workload: AgentXWorkloadConfig, timeseries: dict[str, Any], duration_s: float
) -> dict[str, Any]:
    """Configured session concurrency beside the request rates it generated.

    Concurrency is the configured load at each window's midpoint: the ramp,
    or the target throughout for fixed concurrency. The live session count can
    briefly differ because of scheduling and tree turnover. No sessions are
    configured once load issuance ends at ``duration_s``.
    """
    ramp = workload.concurrency_ramp_duration_s
    curve = (
        [(p.time_s, float(p.sessions)) for p in workload.concurrency_series]
        if workload.concurrency_series is not None else None
    )
    rows = []
    for w in timeseries["windows"]:
        mid = (w["window_start_s"] + w["window_end_s"]) / 2
        if mid >= duration_s:
            sessions = 0.0
        elif curve is not None:
            sessions = configured_series_concurrency_at(mid, curve)
        elif ramp is None:
            sessions = float(workload.target_concurrency)
        else:
            sessions = configured_concurrency_at(mid, workload.target_concurrency, ramp)
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
    }
    if workload.concurrency_series is not None:
        result["concurrency_series"] = [p.model_dump() for p in workload.concurrency_series]
    elif ramp is not None:
        result["ramp_duration_s"] = ramp
    return {
        **result,
        "window_s": timeseries["window_s"],
        "summary": timeseries["summary"],
        "windows": rows,
    }


# --- Metrics inventory ------------------------------------------------------


def build_metrics_inventory(metrics_path: Path) -> dict[str, Any]:
    metrics: dict[str, dict[str, Any]] = {}
    other: set[str] = set()
    endpoints: set[str] = set()
    scrapes = errors = 0
    durations: list[float] = []
    sizes: list[int] = []
    overran: set[Any] = set()
    if metrics_path.exists():
        for record in read_jsonl(metrics_path):
            scrapes += 1
            if record.get("endpoint"):
                endpoints.add(record["endpoint"])
            if record.get("error"):
                errors += 1
            if record.get("scrape_duration_s") is not None:
                durations.append(record["scrape_duration_s"])
            if record.get("bytes") is not None:
                sizes.append(record["bytes"])
            if record.get("cycle_overran"):
                overran.add(record.get("tick"))
            other.update(record.get("available_serve_metrics") or [])
            ts = record.get("timestamp_ns")
            for sample in record.get("samples") or []:
                entry = metrics.setdefault(sample["normalized_name"], {
                    "original_names": set(), "sample_names": set(), "types": set(),
                    "labels": set(), "source_nodes": set(), "sample_count": 0,
                    "first_timestamp_ns": ts, "last_timestamp_ns": ts,
                })
                family = sample["name"]
                if sample.get("type") in ("histogram", "summary", "counter"):
                    family = family.removesuffix("_bucket").removesuffix("_count")
                    family = family.removesuffix("_sum").removesuffix("_total")
                entry["original_names"].add(family)
                entry["sample_names"].add(sample["name"])
                entry["types"].add(sample.get("type"))
                entry["labels"].update(sample.get("labels") or {})
                entry["source_nodes"].add(record.get("node_id"))
                entry["sample_count"] += 1
                entry["first_timestamp_ns"] = min(entry["first_timestamp_ns"], ts)
                entry["last_timestamp_ns"] = max(entry["last_timestamp_ns"], ts)
    for entry in metrics.values():
        for key, value in entry.items():
            if isinstance(value, set):
                entry[key] = sorted(v for v in value if v is not None)
    return {
        "schema_version": SCHEMA_VERSION,
        "collected": metrics_path.exists(),
        "scrape_records": scrapes,
        "scrape_errors": errors,
        "endpoints": sorted(endpoints),
        # Collector overhead, to check that scraping did not perturb the run.
        "max_scrape_duration_s": max(durations, default=None),
        "max_scrape_bytes": max(sizes, default=None),
        "overran_cycles": len(overran),
        "metrics": dict(sorted(metrics.items())),
        "missing_recommended_metrics": [m for m in SELECTED_METRICS if m not in metrics],
        "other_available_serve_metrics": sorted(other - set(SELECTED_METRICS)),
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


def _label_key(labels: dict[str, str]) -> tuple:
    return tuple(sorted(labels.items()))


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
    previous: dict[tuple, float] = {}
    unidentified = conflicting = 0
    for timestamp_ns, samples in ticks:
        series: dict[tuple, set[float]] = defaultdict(set)
        for node_id, sample in samples:
            labels = sample["labels"]
            if sample.get("value") is None:
                continue
            if any(key not in labels for key in identity):
                unidentified += 1
                continue
            key = (labels.get("application"), labels.get("deployment"),
                   *(labels[k] for k in identity))
            if aggregation == "counter_delta":
                # Counters are node-local; deltas are taken per node series.
                key = (node_id, _label_key(labels))
            series[key].add(sample["value"])
        if any(len(values) > 1 for values in series.values()):
            conflicting += 1
            continue
        current = {key: values.pop() for key, values in series.items()}
        value: float
        if aggregation == "counter_delta":
            # A drop means the counter reset.
            delta, seen = 0.0, False
            for key, v in current.items():
                if key in previous:
                    delta += v - previous[key] if v >= previous[key] else v
                    seen = True
                previous[key] = v
            if not seen:
                continue
            value = delta
        elif not current:
            continue
        elif aggregation == "sum":
            value = sum(current.values())
        elif aggregation == "healthy":
            value = sum(1 for v in current.values() if v >= 1)
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


def _replica_values(
    samples: list[tuple[str | None, dict[str, Any]]]
) -> dict[str, float]:
    """One value per ReplicaId; a replica whose copies disagree is dropped."""
    values: dict[str, set[float]] = defaultdict(set)
    for _, sample in samples:
        replica = sample["labels"].get("ReplicaId")
        if replica and sample.get("value") is not None:
            values[replica].add(sample["value"])
    return {replica: v.pop() for replica, v in values.items() if len(v) == 1}


def windowed_ratio(
    numerator_ticks: list[tuple[int, list[tuple[str | None, dict[str, Any]]]]],
    denominator_ticks: list[tuple[int, list[tuple[str | None, dict[str, Any]]]]],
    window_s: float,
    t0: int,
    scale: float = 1.0,
) -> list[dict[str, Any]]:
    """Per tick, the ratio of two counters' increases over the trailing window.

    Increases are taken per replica against its latest sample at or before
    the window start, or its first sample if it appeared inside the window,
    then summed over the replicas present at the tick. Ticks with no
    denominator increase produce no point.
    """
    history: dict[str, list[tuple[int, float, float]]] = defaultdict(list)
    points = []
    window_ns = round(window_s * 1e9)
    for (timestamp_ns, numerators), (_, denominators) in zip(numerator_ticks, denominator_ticks):
        current_num = _replica_values(numerators)
        current_den = _replica_values(denominators)
        total_num = total_den = 0.0
        for replica in current_num.keys() & current_den.keys():
            samples = history[replica]
            baseline = samples[0] if samples else None
            for sample in reversed(samples):
                if sample[0] <= timestamp_ns - window_ns:
                    baseline = sample
                    break
            samples.append((timestamp_ns, current_num[replica], current_den[replica]))
            if baseline is None:
                continue
            total_num += current_num[replica] - baseline[1]
            total_den += current_den[replica] - baseline[2]
        if total_den > 0:
            points.append({
                "relative_time_s": (timestamp_ns - t0) / 1e9,
                "value": total_num / total_den * scale,
                "series_count": len(current_num.keys() & current_den.keys()),
            })
    return points


def metric_series(
    records: list[dict[str, Any]],
    application: str,
    deployments: set[str],
    interval_s: float,
    t0: int,
    warnings: list[str],
    window_s: float = 30.0,
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

    # vLLM samples belong to the deployment when their ReplicaId is one of its
    # replicas; without any replica-labeled Serve sample, every one is kept.
    replica_ids = {
        s["labels"]["replica"]
        for r in records for s in r.get("samples") or []
        if "replica" in (s.get("labels") or {}) and attributed(s["labels"])
    }

    unlabeled: dict[str, int] = defaultdict(int)
    wanted = {name for name, _, _ in METRIC_SERIES.values()}
    wanted |= {name for pair in WINDOWED_RATIOS.values() for name in pair[:2]}
    by_metric: dict[str, list] = {name: [] for name in wanted}
    for tick in sorted(ticks):
        tick_records = ticks[tick]
        timestamp_ns = max(r["timestamp_ns"] for r in tick_records)
        selected: dict[str, list] = defaultdict(list)
        for record in tick_records:
            for sample in record.get("samples") or []:
                name = normalize_metric_name(sample["name"])
                if name not in by_metric:
                    continue
                labels = sample.get("labels") or {}
                if name.startswith("vllm_"):
                    if replica_ids and labels.get("ReplicaId") not in replica_ids:
                        continue
                elif "application" not in labels or "deployment" not in labels:
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
    for key, (name, aggregation, identity) in METRIC_SERIES.items():
        points = aggregate_metric(by_metric[name], aggregation, identity, t0, warnings, name)
        if points:
            result[key] = {"metric": name, "aggregation": aggregation, "points": points}
    for key, (numerator, denominator, scale) in WINDOWED_RATIOS.items():
        points = windowed_ratio(by_metric[numerator], by_metric[denominator], window_s, t0, scale)
        if points:
            result[key] = {"metric": f"{numerator} / {denominator}",
                           "aggregation": f"windowed_ratio_{window_s:g}s", "points": points}
    return result


# --- SLA policy events ------------------------------------------------------


def read_sla_policy_events(root: Path, application: str, t0: int) -> list[dict[str, Any]]:
    """The SLA policy's logged state transitions, on the profiling clock."""
    path = root / "telemetry" / "sla_policy_events.jsonl"
    if not path.exists():
        return []
    events = [
        {
            "relative_time_s": (record["logged_at_ns"] - t0) / 1e9,
            "reason": record["reason"],
            "recommended_replicas": record["recommended_replicas"],
            "advisory": record["advisory"],
        }
        for record in read_jsonl(path)
        if not record.get("error") and application in record.get("deployment", "")
    ]
    return sorted(events, key=lambda e: e["relative_time_s"])


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
        warnings.append("Serve status telemetry was not collected; replica panel is empty")
    elif not any(s.get("error") is None for s in status):
        warnings.append("every Serve status sample failed; replica panel is empty")
    elif errors := sum(1 for s in status if s.get("error")):
        warnings.append(f"{errors} of {len(status)} Serve status samples failed")

    if not metrics_path.exists():
        warnings.append("Prometheus metrics were not collected; Serve-side signals are omitted")
    series = metric_series(
        metric_records, application, deployments, analysis.metrics_interval_s, t0, warnings,
        window_s=analysis.tail_window_s,
    )
    if metrics_path.exists():
        for key in ("ongoing_requests", "router_queue", "desired_replicas"):
            if key not in series:
                warnings.append(f"{METRIC_SERIES[key][0]} was not exported; omitted from the plot")
    autoscaling = config.deployment.autoscaling
    sla_events = read_sla_policy_events(root, application, t0)
    if autoscaling.policy.name == "token" and "token_prefill_backlog" not in series:
        warnings.append("token policy gauges were not exported; policy panel is empty")
    if autoscaling.policy.name == "sla":
        if "sla_recommended_replicas" not in series:
            warnings.append("SLA policy gauges were not exported; policy panel is empty")
        if not sla_events:
            warnings.append("no SLA policy state transitions were found in the controller log")

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
            "gpus_per_replica": config.deployment.engine.tensor_parallel_size,
            "mode": config.benchmark.mode,
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
        "sla_policy_events": sla_events,
        "availability": {
            "replica_status": any(s.get("error") is None for s in status),
            "router_queue": "router_queue" in series,
            "ongoing_requests": "ongoing_requests" in series or "handle_ongoing_requests" in series,
            "desired_replicas": "desired_replicas" in series,
            "target_replicas_metric": "target_replicas" in series,
            "healthy_replicas": "healthy_replicas" in series,
            "sla_policy_metrics": "sla_recommended_replicas" in series,
            "token_policy_metrics": "token_prefill_backlog" in series,
            "sla_policy_events": bool(sla_events),
            "vllm_latency": "vllm_mean_ttft_ms" in series,
        },
        "warnings": warnings,
    }


def analyze_run(
    root: Path,
    *,
    window_s: float | None = None,
    tail_window_s: float | None = None,
    plot: bool | None = None,
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
    if isinstance(config.benchmark.workload, AgentXWorkloadConfig):
        write_json(out / "concurrency_qps.json", concurrency_qps(
            config.benchmark.workload, timeseries, config.benchmark.duration_s
        ))
    if config.benchmark.mode == "agentx_fixed_concurrency":
        write_json(out / "slo_calibration.json",
                   slo_calibration(workload_artifact_dir(root, config)))
    write_json(
        out / "metrics_inventory.json",
        build_metrics_inventory(root / "telemetry" / "serve_metrics.jsonl"),
    )
    plot_data = build_plot_data(root, config, analysis, timeseries, lifecycle)
    write_json(out / "plot_data.json", plot_data)
    requests, _ = profiling_requests(artifact_dir)
    write_json(out / "autoscaling_evaluation.json", evaluate_run(plot_data, requests))

    result: dict[str, Any] = {"warnings": plot_data["warnings"], "plot_error": None}
    if analysis.generate_plots if plot is None else plot:
        try:
            from .plotting import plot_timeline

            plot_timeline(out / "plot_data.json", out / "autoscaling_timeline.png")
        except Exception as exc:  # raw and derived data stay valid without the figure
            result["plot_error"] = f"{type(exc).__name__}: {exc}"
    return result
