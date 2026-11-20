import json
from pathlib import Path

import pytest

from serve_llm_autoscaling.analysis import (
    aggregate_metric,
    analyze_run,
    build_plot_data,
    load_run_config,
    metric_series,
    router_report,
    workload_artifact_dir,
)
from serve_llm_autoscaling.artifacts import write_json
from serve_llm_autoscaling.cli import main
from conftest import build_config, series, short_config
from serve_llm_autoscaling.config import WORKLOAD_DIR, write_config
from serve_llm_autoscaling.windows import request_timeseries

S = 1_000_000_000
T0 = 1_790_000_000 * S  # an epoch timestamp, as AIPerf and time.time_ns() use
APP = "ttft-benchmark"
LLM = "LLMServer:Qwen--Qwen3-0_6B-FP8"
INGRESS = "OpenAiIngress"


def _request(credit_s, end_s, ttft=100.0):
    return {"metadata": {"credit_issued_ns": T0 + int(credit_s * S),
                         "request_start_ns": T0 + int(credit_s * S),
                         "request_end_ns": T0 + int(end_s * S),
                         "benchmark_phase": "profiling"},
            "metrics": {"time_to_first_token": {"value": ttft}}}


def _status(t_s, states, error=None):
    if error:
        return {"timestamp_ns": T0 + int(t_s * S), "application": APP,
                "application_status": None, "deployments": {}, "available": False,
                "error": error}
    deployment = {"status": "HEALTHY", "status_trigger": "AUTOSCALING",
                  "replica_states": {s: states.count(s) for s in set(states)}}
    ingress = {**deployment, "replica_states": {"RUNNING": 1}}
    return {"timestamp_ns": T0 + int(t_s * S), "application": APP,
            "application_status": "RUNNING", "available": True, "error": None,
            "deployments": {LLM: deployment, INGRESS: ingress}}


def _sample(name, value, **labels):
    normalized = name.removeprefix("ray_")
    for suffix in ("_count", "_sum", "_bucket"):
        if normalized.startswith("serve_replica_startup_latency_ms"):
            normalized = normalized.removesuffix(suffix)
    return {"name": name, "normalized_name": normalized, "type": "gauge",
            "labels": labels, "value": value}


def _scrape(t_s, tick, samples, node="n1", error=None):
    return {"timestamp_ns": T0 + int(t_s * S), "tick": tick, "node_id": node,
            "node_ip": "10.0.0.1", "endpoint": f"http://{node}:8085/metrics",
            "samples": [] if error else samples, "error": error}


def _jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


def make_run(root: Path, *, status=True, metrics=True) -> Path:
    """A 300s run (1 -> 4 -> 1 sessions) with status and Prometheus telemetry."""
    config = build_config(
        {"concurrency_series": series((0, 1), (60, 4), (180, 4), (300, 1)),
         "unsafe_override": True},
        deployment={"model_id": "Qwen/Qwen3-0.6B-FP8"}, duration_s=300, grace_period_s=120,
    )
    root.mkdir(parents=True, exist_ok=True)
    write_config(config, root / "resolved.yaml")
    artifacts = root / "benchmark" / WORKLOAD_DIR
    artifacts.mkdir(parents=True)
    write_json(artifacts / "phase_manifest.json", {"phases": [
        {"phase_kind": "warmup", "start_ns": T0 - 30 * S, "end_ns": T0},
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + 300 * S},
    ]})
    _jsonl(artifacts / "profile_export.jsonl",
           [_request(t, t + 0.5, ttft=100 + t) for t in range(0, 300, 2)])
    if status:
        _jsonl(root / "telemetry" / "serve_status.jsonl", [
            _status(-10, ["RUNNING"]),
            _status(2, ["RUNNING"]),
            _status(3, [], error="ConnectionError: controller"),
            _status(90, ["RUNNING", "STARTING", "STARTING"]),
            _status(150, ["RUNNING", "RUNNING", "RUNNING", "STOPPING"]),
        ])
    if metrics:
        labels = {"application": APP, "deployment": LLM}
        _jsonl(root / "telemetry" / "serve_metrics.jsonl", [
            _scrape(t, i, [
                # Families that older runs recorded and analysis no longer reads.
                _sample("ray_serve_replica_processing_queries", 4, replica="r0", **labels),
                _sample("ray_vllm_num_requests_running", 2, ReplicaId="r0"),
                _sample("ray_serve_autoscaling_desired_replicas", 1 + (t >= 90) * 2, **labels),
                _sample("ray_serve_autoscaling_desired_replicas", 9,
                        application=APP, deployment=INGRESS),
                _sample("ray_serve_autoscaling_target_replicas", 1 + (t >= 90) * 2, **labels),
            ])
            for i, t in enumerate(range(-5, 200, 5))
        ] + [_scrape(200, 99, [], node="n2", error="TimeoutError: slow node")])
    return root


def test_plot_data_aligns_epoch_timestamps(tmp_path: Path):
    root = make_run(tmp_path / "run")
    analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["schema_version"] == 4
    assert data["time_origin"] == {"profiling_start_ns": T0, "source": "phase_manifest",
                                   "profiling_end_s": 300.0}
    times = [s["relative_time_s"] for s in data["serve_status_samples"]]
    assert times == [-10.0, 2.0, 3.0, 90.0, 150.0]
    points = data["serve_metric_series"]["desired_replicas"]["points"]
    assert points[0]["relative_time_s"] == -5.0 and points[1]["relative_time_s"] == 0.0
    load = data["configured_load"]
    assert (load["dimension"], load["unit"]) == ("session_concurrency", "session trees")
    assert load["points"][1] == {"time_s": 60.0, "value": 4.0}
    assert load["markers"] == [60.0, 180.0, 300.0]
    assert data["request_windows"][0]["ttft_sample_count"] == 3
    assert data["experiment"]["deployments"] == [LLM]


def test_replica_counts_remain_step_data(tmp_path: Path):
    root = make_run(tmp_path / "run")
    analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    samples = data["serve_status_samples"]
    # Kept at their own timestamps, not averaged into request windows.
    assert len(samples) == 5
    upscaling = samples[3]
    # The LLM deployment only, not the ingress.
    assert (upscaling["running_replicas"], upscaling["starting_replicas"]) == (1, 2)
    assert upscaling["live_replicas"] == 3
    assert samples[4]["stopping_replicas"] == 1 and samples[4]["live_replicas"] == 4
    assert samples[2]["error"] and "running_replicas" not in samples[2]
    assert "1 of 5 Serve status samples failed" in data["warnings"]


def test_missing_application_is_a_gap_not_zero(tmp_path: Path):
    from serve_llm_autoscaling.analysis import status_samples

    missing = {"timestamp_ns": T0, "application": APP, "application_status": None,
               "deployments": {}, "available": False,
               "error": "configured application not present"}
    other = {**_status(1, ["RUNNING"]), "deployments": {INGRESS: {"replica_states": {}}}}
    samples = status_samples([missing, other], {LLM}, T0)
    assert all(s["error"] and "running_replicas" not in s for s in samples)
    assert samples[1]["error"] == "no matching deployment in Serve status"


def test_old_status_records_are_read():
    from serve_llm_autoscaling.analysis import status_samples

    old = {**_status(0, []), "deployments": {LLM: {"replicas_by_state": {"RUNNING": 2}}}}
    assert status_samples([old], {LLM}, T0)[0]["running_replicas"] == 2


def test_metrics_are_filtered_to_the_llm_deployment(tmp_path: Path):
    root = make_run(tmp_path / "run")
    result = analyze_run(root, plot=False)
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    series = data["serve_metric_series"]
    # The ingress sample is excluded; unread families produce no series or warning.
    assert set(series) == {"desired_replicas", "target_replicas"}
    assert [p["value"] for p in series["desired_replicas"]["points"]][18:20] == [1, 3]
    assert data["availability"] == {"replica_status": True, "desired_replicas": True,
                                    "target_replicas_metric": True,
                                    "token_policy_metrics": False}
    assert result["warnings"] == ["1 of 5 Serve status samples failed"]


def test_duplicate_controller_metrics_are_not_summed():
    labels = {"application": APP, "deployment": LLM}
    samples = [("n1", _sample("serve_autoscaling_desired_replicas", 3, **labels)),
               ("n2", _sample("serve_autoscaling_desired_replicas", 3, **labels))]
    warnings = []
    points = aggregate_metric([(T0, samples)], "sum", (), T0, warnings, "desired")
    assert points == [{"relative_time_s": 0.0, "value": 3, "series_count": 1}]
    assert warnings == []


def test_conflicting_controller_metrics_are_omitted():
    labels = {"application": APP, "deployment": LLM}
    conflicting = [("n1", _sample("serve_autoscaling_desired_replicas", 3, **labels)),
                   ("n2", _sample("serve_autoscaling_desired_replicas", 2, **labels))]
    agreeing = [("n1", _sample("serve_autoscaling_desired_replicas", 4, **labels))]
    warnings = []
    points = aggregate_metric([(T0, conflicting), (T0 + S, agreeing)], "sum", (), T0,
                              warnings, "desired")
    assert points == [{"relative_time_s": 1.0, "value": 4, "series_count": 1}]
    assert "omitted 1 scrapes" in warnings[0]


def test_additive_metrics_without_identity_are_excluded():
    labels = {"application": APP, "deployment": LLM}
    samples = [("n1", _sample("serve_autoscaling_router_metric", 4, **labels)),
               ("n2", _sample("serve_autoscaling_router_metric", 4, **labels))]
    warnings = []
    assert aggregate_metric([(T0, samples)], "sum", ("handle",), T0, warnings, "q") == []
    assert "missing identity labels ['handle']" in warnings[0]


def test_per_handle_aggregation():
    labels = {"application": APP, "deployment": LLM}
    samples = [("n1", _sample("serve_autoscaling_router_metric", 4, handle="a", **labels)),
               ("n2", _sample("serve_autoscaling_router_metric", 5, handle="b", **labels)),
               ("n1", _sample("serve_autoscaling_router_metric", None, handle="c", **labels))]
    ticks = [(T0, samples), (T0 + S, [])]
    points = aggregate_metric(ticks, "sum", ("handle",), T0, [], "router")
    assert points == [{"relative_time_s": 0.0, "value": 9, "series_count": 2}]  # no zero gap
    assert aggregate_metric(ticks, "max", ("handle",), T0, [], "router")[0]["value"] == 5


def test_unlabeled_series_are_excluded_with_warning():
    records = [_scrape(0, 0, [_sample("ray_serve_autoscaling_desired_replicas", 2)])]
    warnings = []
    assert metric_series(records, APP, {LLM}, 1, T0, warnings) == {}
    assert "excluded 1 samples without application/deployment labels" in warnings[0]


def test_analysis_with_all_telemetry(tmp_path: Path):
    root = make_run(tmp_path / "run")
    result = analyze_run(root)
    assert result["plot_error"] is None
    # No scheduler traces: the diagnostic plots are skipped, the gallery still written.
    assert result["diagnostic_plots"]["status"] == "skipped"
    assert sorted(p.name for p in (root / "analysis").iterdir() if p.is_file()) == [
        "autoscaling_evaluation.json", "concurrency_qps.json", "diagnostic_plots.json",
        "index.html", "plot_data.json", "request_timeseries.json",
        "scheduler_trace_inputs.json",
    ]


def test_analysis_without_prometheus(tmp_path: Path):
    root = make_run(tmp_path / "run", metrics=False)
    result = analyze_run(root)
    assert any("Prometheus metrics were not collected" in w for w in result["warnings"])
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["serve_metric_series"] == {}
    assert data["availability"]["replica_status"] is True
    evaluation = json.loads((root / "analysis" / "autoscaling_evaluation.json").read_text())
    assert evaluation["scale_events"] == []


def test_analysis_with_only_status_telemetry(tmp_path: Path):
    root = make_run(tmp_path / "run", metrics=False)
    _jsonl(root / "telemetry" / "serve_metrics.jsonl",
           [_scrape(0, 0, []), _scrape(1, 1, [])])
    result = analyze_run(root)
    # Missing Serve metrics are warnings, not zero-valued series.
    assert any("serve_autoscaling_desired_replicas was not exported" in w
               for w in result["warnings"])
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["serve_metric_series"] == {}
    assert data["availability"]["desired_replicas"] is False


def test_analysis_without_any_telemetry(tmp_path: Path):
    root = make_run(tmp_path / "run", status=False, metrics=False)
    result = analyze_run(root)
    assert any("Serve status telemetry was not collected" in w for w in result["warnings"])


def test_analyze_command_regenerates_with_new_window(tmp_path: Path, capsys):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--window-s", "10", "--tail-window-s", "60"]) == 0
    timeseries = json.loads((root / "analysis" / "request_timeseries.json").read_text())
    assert timeseries["window_s"] == 10 and timeseries["tail_window_s"] == 60
    assert timeseries["windows"][1]["window_start_s"] == 10
    data = json.loads((root / "analysis" / "plot_data.json").read_text())
    assert data["experiment"]["window_s"] == 10
    assert (root / "analysis" / "index.html").exists()


def test_analyze_command_no_plot(tmp_path: Path):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--no-plot"]) == 0
    assert (root / "analysis" / "plot_data.json").exists()
    assert not (root / "analysis" / "index.html").exists()


def test_analyze_command_rejects_short_tail_window(tmp_path: Path, capsys):
    root = make_run(tmp_path / "run")
    assert main(["analyze", str(root), "--window-s", "60"]) == 1
    assert "tail_window_s must be >= window_s" in capsys.readouterr().err


def test_old_runs_without_analysis_config_load(tmp_path: Path):
    root = make_run(tmp_path / "run")
    resolved = (root / "resolved.yaml").read_text()
    (root / "resolved.yaml").write_text(resolved.split("analysis:")[0])
    assert load_run_config(root).analysis.window_s == 5


def test_build_plot_data_requires_time_origin(tmp_path: Path):
    root = make_run(tmp_path / "run")
    artifacts = root / "benchmark" / WORKLOAD_DIR
    (artifacts / "phase_manifest.json").unlink()
    (artifacts / "profile_export.jsonl").write_text("")
    config = load_run_config(root)
    with pytest.raises(ValueError, match="cannot align"):
        build_plot_data(root, config, config.analysis, request_timeseries(artifacts))


# --- Session concurrency ----------------------------------------------------


def _agentx_request(credit_s, end_s, depth):
    record = _request(credit_s, end_s)
    record["metadata"]["agent_depth"] = depth
    return record


def make_agentx_run(root: Path, config) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    write_config(config, root / "resolved.yaml")
    ramp = root / "benchmark" / WORKLOAD_DIR
    ramp.mkdir(parents=True)
    duration = config.benchmark.duration_s
    write_json(ramp / "phase_manifest.json", {"phases": [
        {"phase_kind": "warmup", "start_ns": T0 - 30 * S, "end_ns": T0},
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + int(duration) * S},
    ]})
    records = [_agentx_request(t, t + 3, 0) for t in range(0, int(duration), 4)]
    records += [_agentx_request(t + 1, t + 2, 1) for t in range(0, int(duration), 8)]
    warmup = _agentx_request(-10, -9, 0)
    warmup["metadata"]["benchmark_phase"] = "warmup"
    _jsonl(ramp / "profile_export.jsonl", records + [warmup])
    _jsonl(root / "telemetry" / "serve_status.jsonl",
           [_status(0, ["RUNNING"]), _status(100, ["RUNNING", "STARTING"])])
    return root


@pytest.fixture
def agentx_run(tmp_path: Path):
    return make_agentx_run(tmp_path / "run", short_config())


def test_workload_artifact_dir(agentx_config, tmp_path: Path):
    assert workload_artifact_dir(tmp_path, agentx_config()) == (
        tmp_path / "benchmark" / "agentx-concurrency-ramp"
    )


def test_agentx_analysis_reads_ramp_artifacts(agentx_run: Path):
    analyze_run(agentx_run, plot=False)
    timeseries = json.loads((agentx_run / "analysis" / "request_timeseries.json").read_text())
    assert timeseries["time_origin_source"] == "phase_manifest"
    assert timeseries["windows"][0]["offered_requests"] == 3  # 2 root + 1 subagent
    assert timeseries["summary"]["request_count"] == 75  # warmup excluded


def test_agentx_plot_data_uses_session_concurrency(agentx_run: Path):
    analyze_run(agentx_run, plot=False)
    data = json.loads((agentx_run / "analysis" / "plot_data.json").read_text())
    load = data["configured_load"]
    assert load["dimension"] == "session_concurrency"
    assert load["unit"] == "session trees"
    assert load["label"] == "configured session concurrency"
    assert "QPS" not in load["label"] and "qps" not in json.dumps(load)
    assert load["points"] == [{"time_s": 0.0, "value": 1.0},
                              {"time_s": 120.0, "value": 4.0},
                              {"time_s": 200.0, "value": 4.0}]
    assert load["markers"] == [120.0]
    assert data["experiment"]["mode"] == "agentx_concurrency_ramp"


def test_agentx_concurrency_qps_artifact(agentx_run: Path):
    analyze_run(agentx_run, plot=False)
    result = json.loads((agentx_run / "analysis" / "concurrency_qps.json").read_text())
    assert result["target_concurrency"] == 4
    assert result["concurrency_series"] == [{"time_s": 0.0, "sessions": 1},
                                            {"time_s": 120.0, "sessions": 4}]
    first, *_ = windows = result["windows"]
    # Midpoint 2.5s of a 1 -> 4 ramp over 120s.
    assert first["configured_session_concurrency"] == pytest.approx(1 + 3 * 2.5 / 120)
    assert first["offered_qps"] == pytest.approx(0.6)
    assert (first["offered_root_qps"], first["offered_subagent_qps"]) == (0.4, 0.2)
    assert first["requests_per_second_per_configured_session"] == pytest.approx(
        0.6 / first["configured_session_concurrency"]
    )
    assert set(first) >= {"window_start_s", "window_end_s", "started_qps",
                          "successful_completed_qps", "failed_completed_qps",
                          "in_flight_at_end"}
    assert windows[30]["configured_session_concurrency"] == 4.0  # after the ramp
    assert result["summary"]["mean_offered_qps"] == pytest.approx(75 / 200)


# --- Token policy series ----------------------------------------------------


def test_policy_metric_gauge_is_split_by_its_metric_label():
    labels = {"application": APP, "deployment": LLM}
    records = [
        _scrape(t, i, [
            _sample("ray_serve_autoscaling_policy_metric", 1.0 + t, metric="prefill_required_replicas", **labels),
            _sample("ray_serve_autoscaling_policy_metric", 2.0, metric="decode_required_replicas", **labels),
            _sample("ray_serve_autoscaling_policy_metric", 3.0, metric="required_replicas", **labels),
            _sample("ray_serve_autoscaling_policy_metric", 9.0, metric="required_replicas",
                    application="other", deployment=LLM),
        ])
        for i, t in enumerate(range(0, 11, 5))
    ]
    warnings = []
    series = metric_series(records, APP, {LLM}, 5, T0, warnings)
    assert [p["value"] for p in series["token_prefill_replicas"]["points"]] == [1.0, 6.0, 11.0]
    assert [p["value"] for p in series["token_decode_replicas"]["points"]] == [2.0, 2.0, 2.0]
    assert set(series) == {"token_prefill_replicas", "token_decode_replicas"}
    assert "token_prefill_backlog" not in series
    assert not warnings


def test_router_metric_gauge_rebuilds_token_prefill_replicas():
    labels = {"application": APP, "deployment": LLM, "metric": "outstanding_effective_prefill_tokens"}
    records = [
        _scrape(t, i, [
            _sample("ray_serve_autoscaling_router_metric", 1000.0 * t, handle="h1", **labels),
            _sample("ray_serve_autoscaling_router_metric", 4000.0, handle="h2", **labels),
            _sample("ray_serve_autoscaling_router_metric", 99999.0, handle="h1",
                    application=APP, deployment=LLM, metric="other"),
        ])
        for i, t in enumerate(range(0, 11, 5))
    ]
    warnings = []
    series = metric_series(
        records, APP, {LLM}, 5, T0, warnings, target_prefill_tokens_per_replica=2000
    )
    assert [p["value"] for p in series["token_router_prefill_mean"]["points"]] == [
        4000.0, 5000.0, 10000.0
    ]
    assert [p["value"] for p in series["token_prefill_replicas"]["points"]] == [2.0, 2.5, 5.0]
    assert "token_prefill_backlog" not in series
    assert not warnings

    series = metric_series(records, APP, {LLM}, 5, T0, [])
    assert "token_prefill_replicas" not in series


ROUTER_PREFILL = "outstanding_effective_prefill_tokens"


def _report(value, replica, server=LLM, **labels):
    """The router replica's reported prefill load, as current Ray exports it."""
    labels = {"application": APP, "deployment": "LLMRouter", "replica": replica,
              "metric": f"{ROUTER_PREFILL}:{server}", **labels}
    return _sample("ray_serve_autoscaling_replica_metric", value, **labels)


def test_router_report_presents_the_replica_gauge_as_the_router_gauge():
    sample = _report(4000.0, "r1")
    assert router_report(sample) == {
        "name": "ray_serve_autoscaling_router_metric",
        "normalized_name": sample["normalized_name"],
        "type": "gauge",
        "value": 4000.0,
        # The server the router is bound to, with the router replica as the handle.
        "labels": {"application": APP, "deployment": LLM, "replica": "r1", "handle": "r1",
                   "metric": ROUTER_PREFILL},
    }
    # The recorded sample is not modified.
    assert sample == _report(4000.0, "r1")
    # The exporter's ray_ prefix is optional.
    bare = {**sample, "name": "serve_autoscaling_replica_metric"}
    assert router_report(bare)["labels"]["deployment"] == LLM


@pytest.mark.parametrize("sample", [
    # The earlier per-handle gauge is already in the reported shape.
    _sample("ray_serve_autoscaling_router_metric", 4.0, application=APP, deployment=LLM,
            handle="h1", metric=ROUTER_PREFILL),
    # Other custom metrics a replica reports, on the router or the server.
    _sample("ray_serve_autoscaling_replica_metric", 4.0, application=APP,
            deployment="LLMRouter", replica="r1", metric="queued_requests:" + LLM),
    _sample("ray_serve_autoscaling_replica_metric", 4.0, application=APP,
            deployment=LLM, replica="r1", metric=ROUTER_PREFILL),
    _sample("ray_serve_autoscaling_replica_metric", 4.0, application=APP,
            deployment="LLMRouter", replica="r1", metric=ROUTER_PREFILL + "_total:" + LLM),
    # The same metric name under another family, and samples without labels.
    _sample("ray_serve_autoscaling_policy_metric", 4.0, application=APP, deployment=LLM,
            metric=ROUTER_PREFILL),
    _sample("ray_serve_autoscaling_desired_replicas", 2.0, application=APP, deployment=LLM),
    _sample("ray_serve_autoscaling_replica_metric", 4.0),
    {"name": "ray_serve_autoscaling_replica_metric", "value": 4.0},
])
def test_router_report_passes_other_samples_through(sample):
    assert router_report(sample) is sample


def test_replica_metric_gauge_gives_the_max_router_prefill_mean():
    records = [
        _scrape(t, i, [
            _report(1000.0 * t, "r1"),
            _report(4000.0, "r2"),
            # A NaN gap in one router's report is not a value.
            _report(None, "r3"),
            # The same metric for a server this analysis does not cover, for
            # another application, and another metric of the same router.
            _report(99999.0, "r1", server="LLMServer:other-model"),
            _report(88888.0, "r4", application="other-app"),
            _sample("ray_serve_autoscaling_replica_metric", 77777.0, application=APP,
                    deployment="LLMRouter", replica="r1", metric="queued_requests:" + LLM),
        ])
        for i, t in enumerate(range(0, 11, 5))
    ]
    warnings = []
    series = metric_series(
        records, APP, {LLM}, 5, T0, warnings, target_prefill_tokens_per_replica=2000
    )
    assert series["token_router_prefill_mean"] == {
        "metric": "serve_autoscaling_router_metric",
        "aggregation": "max",
        "points": [
            {"relative_time_s": 0.0, "value": 4000.0, "series_count": 2},
            {"relative_time_s": 5.0, "value": 5000.0, "series_count": 2},
            {"relative_time_s": 10.0, "value": 10000.0, "series_count": 2},
        ],
    }
    assert [p["value"] for p in series["token_prefill_replicas"]["points"]] == [2.0, 2.5, 5.0]
    assert set(series) == {"token_router_prefill_mean", "token_prefill_replicas"}
    assert not warnings


def test_replica_metric_gauge_with_only_gaps_gives_no_point():
    records = [
        _scrape(0, 0, [_report(None, "r1"), _report(None, "r2")]),
        _scrape(5, 1, [_report(None, "r1"), _report(3000.0, "r2")]),
    ]
    series = metric_series(records, APP, {LLM}, 5, T0, [], 2000)
    assert series["token_router_prefill_mean"]["points"] == [
        {"relative_time_s": 5.0, "value": 3000.0, "series_count": 1},
    ]
    # Only gaps, or only another server's report: no series at all.
    assert metric_series(records[:1], APP, {LLM}, 5, T0, [], 2000) == {}
    other = [_scrape(0, 0, [_report(3000.0, "r1", server="LLMServer:other-model")])]
    assert metric_series(other, APP, {LLM}, 5, T0, [], 2000) == {}


def test_replica_metric_copies_from_several_nodes_are_one_series():
    records = [
        _scrape(0, 0, [_report(3000.0, "r1")], node="n1"),
        _scrape(0, 0, [_report(3000.0, "r1"), _report(1000.0, "r2")], node="n2"),
    ]
    (point,) = metric_series(records, APP, {LLM}, 5, T0, [])["token_router_prefill_mean"]["points"]
    assert (point["value"], point["series_count"]) == (3000.0, 2)


def test_policy_metric_gauge_takes_precedence_over_the_router_report():
    records = [_scrape(0, 0, [
        _report(8000.0, "r1"),
        _sample("ray_serve_autoscaling_policy_metric", 3.0, application=APP, deployment=LLM,
                metric="prefill_required_replicas"),
    ])]
    series = metric_series(records, APP, {LLM}, 5, T0, [], 2000)
    assert series["token_prefill_replicas"]["metric"] == "serve_autoscaling_policy_metric"
    assert [p["value"] for p in series["token_prefill_replicas"]["points"]] == [3.0]
    assert [p["value"] for p in series["token_router_prefill_mean"]["points"]] == [8000.0]


# --- A curve that rises, falls and holds -------------------------------------


@pytest.fixture
def agentx_curve_run(tmp_path: Path):
    config = short_config({"concurrency_series": series((0, 1), (100, 4), (150, 2))})
    return make_agentx_run(tmp_path / "run", config)


def test_agentx_curve_configured_load_and_markers(agentx_curve_run: Path):
    analyze_run(agentx_curve_run, plot=False)
    data = json.loads((agentx_curve_run / "analysis" / "plot_data.json").read_text())
    load = data["configured_load"]
    assert load["dimension"] == "session_concurrency"
    assert load["points"] == [{"time_s": 0.0, "value": 1.0}, {"time_s": 100.0, "value": 4.0},
                              {"time_s": 150.0, "value": 2.0}, {"time_s": 200.0, "value": 2.0}]
    assert load["markers"] == [100.0, 150.0]
    assert data["lifecycle"]["markers"]["ramp_end_s"] is None


def test_agentx_curve_concurrency_qps_follows_curve(agentx_curve_run: Path):
    analyze_run(agentx_curve_run, plot=False)
    result = json.loads((agentx_curve_run / "analysis" / "concurrency_qps.json").read_text())
    assert result["concurrency_series"] == [
        {"time_s": 0.0, "sessions": 1}, {"time_s": 100.0, "sessions": 4},
        {"time_s": 150.0, "sessions": 2},
    ]
    windows = result["windows"]
    assert windows[0]["configured_session_concurrency"] == pytest.approx(1.075)  # mid 2.5s
    assert windows[20]["configured_session_concurrency"] == pytest.approx(3.9)   # mid 102.5s
    assert windows[35]["configured_session_concurrency"] == 2.0                  # mid 177.5s
    assert windows[-1]["configured_session_concurrency"] == 2.0                  # mid 197.5s
