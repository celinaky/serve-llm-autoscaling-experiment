import json
from pathlib import Path

import pytest

from serve_llm_autoscaling.analysis import analyze_run
from serve_llm_autoscaling.artifacts import write_json
from serve_llm_autoscaling.config import CONTINUOUS_MODE_DIRS, ExperimentConfig, write_config
from serve_llm_autoscaling.evaluation import (
    evaluate_run,
    latency_by_replica_count,
    prediction_error,
    replica_usage,
    scale_events,
    scale_summary,
    score_requests,
    score_windows,
    scoring_periods,
    sla_evaluation,
)
from serve_llm_autoscaling.windows import Request

S = 1_000_000_000
T0 = 1_790_000_000 * S
APP = "ttft-benchmark"
LLM = "LLMServer:Qwen--Qwen3-0_6B-FP8"


def _request(start_s, ttft, itl=None, failed=False):
    ns = T0 + int(start_s * S)
    return Request(ns, ns, ns + S, failed, ttft, itl, 0)


def _status(t, running, starting=0, error=None):
    sample = {"relative_time_s": t, "timestamp_ns": T0 + int(t * S), "error": error}
    if error is None:
        sample.update(running_replicas=running, starting_replicas=starting, stopping_replicas=0)
    return sample


def _points(*pairs):
    return {"points": [{"relative_time_s": t, "value": v} for t, v in pairs]}


def _window(start, ttft=None, itl=None, n=1):
    return {
        "window_start_s": start, "window_end_s": start + 5,
        "rolling_ttft_sample_count": n if ttft is not None else 0,
        "rolling_mean_ttft_ms": ttft,
        "rolling_itl_sample_count": n if itl is not None else 0,
        "rolling_mean_itl_ms": itl,
    }


LIFECYCLE = {
    "markers": {"ramp_end_s": 60.0, "load_end_s": 200.0, "drain_end_s": 220.0,
                "aiperf_exit_s": 222.0, "observation_end_s": 300.0},
    "periods": [{"name": "load", "start_s": 0.0, "end_s": 200.0},
                {"name": "drain", "start_s": 200.0, "end_s": 220.0},
                {"name": "idle", "start_s": 220.0, "end_s": 300.0}],
}


def test_scoring_periods_default_to_the_ramp():
    periods = scoring_periods({"warmup_s": None}, LIFECYCLE)
    assert periods == {"steady": {"start_s": 60.0, "end_s": 200.0},
                       "whole_run": {"start_s": 0.0, "end_s": 300.0}}
    assert scoring_periods({"warmup_s": 30}, LIFECYCLE)["steady"]["start_s"] == 30
    no_ramp = {**LIFECYCLE, "markers": {**LIFECYCLE["markers"], "ramp_end_s": None}}
    assert scoring_periods({"warmup_s": None}, no_ramp)["steady"]["start_s"] == 100.0


def test_score_requests_counts_attainment_per_request():
    requests = [_request(10, 100, 20), _request(20, 4000, 20), _request(30, 500, 80),
                _request(40, 9000, 90, failed=True), _request(250, 100, 10)]
    score = score_requests(requests, T0, {"start_s": 0, "end_s": 200}, 3000, 50)
    assert (score["requests"], score["failed_requests"]) == (4, 1)
    assert score["ttft"]["sample_count"] == 3
    assert score["ttft"]["slo_attainment"] == pytest.approx(2 / 3)
    assert score["itl"]["slo_attainment"] == pytest.approx(2 / 3)
    assert score["both_slo_attainment"] == pytest.approx(1 / 3)
    assert score["ttft"]["mean_ms"] == pytest.approx(4600 / 3)
    # p95 of TTFT [100, 500, 4000] is 3650 and of ITL [20, 20, 80] is 74: both fail.
    assert score["ttft"]["slo_percentile_ms"] == pytest.approx(3650)
    assert (score["ttft"]["slo_met"], score["itl"]["slo_met"], score["slo_met"]) == (False, False, False)
    lenient = score_requests(requests, T0, {"start_s": 0, "end_s": 200}, 3000, 50, slo_percentile=0.5)
    assert (lenient["ttft"]["slo_met"], lenient["slo_met"]) == (True, True)
    unscored = score_requests(requests, T0, {"start_s": 0, "end_s": 200}, None, None)
    assert (unscored["ttft"]["slo_attainment"], unscored["slo_met"]) == (None, None)


def test_score_windows_is_the_share_of_time_within_slo():
    windows = [_window(0, 1000, 20), _window(5, 4000, 20), _window(10, 2000, 60),
               _window(15, None, None), _window(20, 1000, 10)]
    score = score_windows(windows, {"start_s": 0, "end_s": 20}, 30, 3000, 50)
    assert score["windows"] == 4  # the window ending at 25s is outside the period
    assert score["ttft_within_slo"] == pytest.approx(2 / 3)  # the idle window is skipped
    assert score["ttft_violation_s"] == 5  # one 5s request window over the target
    assert score["itl_within_slo"] == pytest.approx(2 / 3)
    assert score["both_within_slo"] == pytest.approx(1 / 3)
    assert score_windows(windows, {"start_s": 0, "end_s": 20}, 30, 3000, None)["both_within_slo"] is None


def test_replica_usage_integrates_running_replicas():
    status = [_status(-5, 1), _status(50, 1, starting=2), _status(60, 0, error="boom"),
              _status(100, 3), _status(250, 2)]
    usage = replica_usage(status, {"start_s": 0, "end_s": 200})
    # 1 replica for 100s (the failed sample holds the count), then 3 for 100s.
    assert usage["running_replica_seconds"] == pytest.approx(400)
    assert usage["starting_replica_seconds"] == pytest.approx(100)
    assert usage["mean_running_replicas"] == pytest.approx(2)
    assert (usage["max_running_replicas"], usage["covered_s"]) == (3, 200)
    assert usage["gpu_hours"] == pytest.approx(500 / 3600)
    assert replica_usage(status, {"start_s": 0, "end_s": 200}, gpus_per_replica=2)["gpu_hours"] == pytest.approx(1000 / 3600)
    assert replica_usage(status[3:], {"start_s": 0, "end_s": 50})["max_running_replicas"] is None


STATUS = [_status(0, 1), _status(60, 1, starting=2), _status(160, 3), _status(400, 3),
          _status(420, 2)]
SERIES = {
    "desired_replicas": _points((0, 1), (58, 1), (60, 3), (100, 3), (400, 2)),
    "sla_recommended_replicas": _points((0, 1), (59, 3), (399, 2)),
}
EVENTS = [
    {"relative_time_s": -1, "reason": "model_not_ready", "recommended_replicas": 1.0, "advisory": False},
    {"relative_time_s": 59, "reason": "observed_slo_violation", "recommended_replicas": 3.0, "advisory": False},
    {"relative_time_s": 300, "reason": "capacity", "recommended_replicas": 2.0, "advisory": False},
]


def test_scale_events_record_lag_and_policy_state():
    events = scale_events(SERIES, STATUS, EVENTS)
    assert [(e["time_s"], e["from_replicas"], e["to_replicas"], e["direction"]) for e in events] == [
        (60, 1, 3, "up"), (400, 3, 2, "down"),
    ]
    assert (events[0]["running_matched_at_s"], events[0]["lag_s"]) == (160, 100)
    assert (events[0]["sla_reason"], events[0]["sla_recommended_replicas"]) == (
        "observed_slo_violation", 3,
    )
    assert (events[1]["lag_s"], events[1]["sla_reason"]) == (20, "capacity")
    assert scale_summary(events) == {
        "decisions": 2, "upscales": 1, "downscales": 1, "first_upscale_at_s": 60,
        "mean_upscale_lag_s": 100, "max_upscale_lag_s": 100,
    }


def test_scale_events_without_a_matching_status_sample():
    series = {"target_replicas": _points((0, 1), (60, 4))}
    (event,) = scale_events(series, STATUS, [])
    assert event["running_matched_at_s"] is None and event["lag_s"] is None
    assert event["sla_reason"] is None
    assert scale_summary([])["first_upscale_at_s"] is None


def test_latency_by_replica_count():
    requests = [_request(10, 100, 10), _request(170, 300, 30), _request(180, 500, 50),
                _request(420, 900, 90, failed=True)]
    rows = latency_by_replica_count(requests, T0, STATUS, {"start_s": 0, "end_s": 500})
    assert [(r["running_replicas"], r["requests"], r["mean_ttft_ms"], r["mean_itl_ms"])
            for r in rows] == [(1, 1, 100, 10), (3, 2, 400, 40)]


def test_prediction_error_compares_matching_scrapes():
    series = {
        "sla_model_ready": _points((0, 0), (170, 1), (180, 1), (190, 1)),
        "sla_recommended_replicas": _points((0, 1), (170, 3), (180, 4), (190, 3)),
        "sla_predicted_ttft_ms": _points((0, 500), (170, 1000), (180, 1000), (190, 1200)),
        "sla_predicted_itl_ms": _points((170, 30), (190, 30)),
        "vllm_mean_ttft_ms": _points((0, 1000), (170, 1100), (180, 1100), (190, 1000)),
        "vllm_mean_itl_ms": _points((170, 25), (190, 35)),
    }
    windows = [_window(165, 1300, 40), _window(170, 1300, 40), _window(185, 1500, 40)]
    error = prediction_error(series, STATUS, windows)
    ttft = error["ttft"]
    # t=0 is before the model is ready and t=180 recommends more replicas than
    # are running, so only t=170 and t=190 are comparable.
    assert ttft["predicted_vs_server"] == {
        "samples": 2, "bias_ms": pytest.approx(50), "mae_ms": pytest.approx(150),
        "mape": pytest.approx((100 / 1100 + 200 / 1000) / 2),
    }
    assert ttft["predicted_vs_client"]["samples"] == 2
    assert ttft["predicted_vs_client"]["bias_ms"] == pytest.approx(((1000 - 1300) + (1200 - 1500)) / 2)
    # Independent of the model; t=180 falls in no request window.
    assert ttft["client_vs_server"]["samples"] == 2
    assert error["itl"]["predicted_vs_server"]["bias_ms"] == pytest.approx(0)
    assert prediction_error({}, STATUS, windows)["ttft"]["predicted_vs_server"]["samples"] == 0


def _plot_data(**overrides):
    return {
        "experiment": {"name": "run", "autoscaling_policy": "sla",
                       "autoscaling_policy_kwargs": {"ttft_ms": 3000, "itl_ms": 50},
                       "ttft_slo_ms": 3000, "itl_slo_ms": 50, "warmup_s": None,
                       "tail_window_s": 30},
        "time_origin": {"profiling_start_ns": T0},
        "lifecycle": LIFECYCLE,
        "request_windows": [_window(t, 1000, 20) for t in range(0, 300, 5)],
        "serve_status_samples": STATUS,
        "serve_metric_series": {**SERIES, "sla_model_ready": _points((0, 0), (150, 1))},
        "sla_policy_events": EVENTS,
        "availability": {"sla_policy_metrics": True, "sla_policy_events": True},
        **overrides,
    }


def test_sla_evaluation_reports_readiness_and_state_timeline():
    sla = sla_evaluation(_plot_data(), 300)
    assert (sla["model_ready_at_s"], sla["first_capacity_decision_at_s"]) == (150, 300)
    assert sla["advisory"] is False
    assert [(s["start_s"], s["end_s"], s["reason"]) for s in sla["reason_timeline"]] == [
        (-1, 59, "model_not_ready"), (59, 300, "observed_slo_violation"), (300, 300, "capacity"),
    ]
    assert sla["time_in_reason_s"] == {"model_not_ready": 59, "observed_slo_violation": 241,
                                       "capacity": 0}
    assert sla_evaluation(_plot_data(availability={"sla_policy_metrics": False,
                                                   "sla_policy_events": False}), 300) is None


def test_evaluate_run_assembles_the_scorecard():
    requests = [_request(t, 1000 + t, 20) for t in range(0, 200, 10)]
    evaluation = evaluate_run(_plot_data(), requests)
    assert evaluation["autoscaling_policy"] == "sla"
    assert evaluation["slo"] == {"ttft_ms": 3000, "itl_ms": 50, "percentile": 0.95}
    assert evaluation["periods"]["steady"] == {"start_s": 60.0, "end_s": 200.0}
    steady = evaluation["slo_attainment"]["steady"]
    assert steady["requests"]["requests"] == 14
    assert steady["requests"]["ttft"]["slo_met"] and steady["requests"]["slo_met"]
    # Running replica-seconds plus the two replicas that spent 100s starting.
    assert evaluation["cost"]["load"]["gpu_hours"] == pytest.approx((160 + 40 * 3 + 200) / 3600)
    assert steady["windows"]["ttft_within_slo"] == 1.0
    assert set(evaluation["cost"]) == {"steady", "whole_run", "load", "drain", "idle"}
    assert evaluation["cost"]["load"]["running_replica_seconds"] == pytest.approx(160 + 40 * 3)
    assert evaluation["scaling"]["upscales"] == 1
    assert evaluation["sla_policy"]["model_ready_at_s"] == 150
    assert [r["running_replicas"] for r in evaluation["latency_by_replica_count"]] == [1, 3]


# --- End to end from a run directory ------------------------------------------


def _sla_config(agentx_config) -> ExperimentConfig:
    config = agentx_config(
        {"concurrency_ramp_duration_s": 60, "target_concurrency": 4, "unsafe_override": True},
        duration_s=200, grace_period_s=30,
    )
    raw = config.resolved_dict()
    raw["deployment"]["autoscaling"]["policy"] = {
        "name": "sla", "kwargs": {"ttft_ms": 3000, "itl_ms": 50, "advisory": False},
    }
    return ExperimentConfig.model_validate(raw)


def _record(t, ttft, itl):
    ns = T0 + t * S
    return {"metadata": {"credit_issued_ns": ns, "request_start_ns": ns, "request_end_ns": ns + S,
                         "benchmark_phase": "profiling", "agent_depth": 0},
            "metrics": {"time_to_first_token": {"value": ttft},
                        "inter_token_latency": {"value": itl}}}


def _metric(name, value, **labels):
    normalized = name.removeprefix("ray_")
    for suffix in ("_sum", "_count", "_total"):
        normalized = normalized.removesuffix(suffix)
    return {"name": name, "normalized_name": normalized, "type": "gauge",
            "labels": labels, "value": value}


def _jsonl(path: Path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in records))


@pytest.fixture
def sla_run(agentx_config, tmp_path: Path) -> Path:
    config = _sla_config(agentx_config)
    root = tmp_path / "run"
    root.mkdir()
    write_config(config, root / "resolved.yaml")
    artifacts = root / "benchmark" / CONTINUOUS_MODE_DIRS[config.benchmark.mode]
    artifacts.mkdir(parents=True)
    write_json(artifacts / "phase_manifest.json", {"phases": [
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + 210 * S},
    ]})
    _jsonl(artifacts / "profile_export.jsonl",
           [_record(t, 500 + 20 * t, 20) for t in range(0, 200, 4)])
    _jsonl(root / "telemetry" / "serve_status.jsonl", [
        {"timestamp_ns": T0 + t * S, "application": APP, "application_status": "RUNNING",
         "available": True, "error": None,
         "deployments": {LLM: {"status": "HEALTHY", "status_trigger": "AUTOSCALING",
                               "replica_states": {"RUNNING": 1 if t < 150 else 2}}}}
        for t in range(-5, 210, 5)
    ])
    labels = {"application": APP, "deployment": LLM}
    _jsonl(root / "telemetry" / "serve_metrics.jsonl", [
        {"timestamp_ns": T0 + t * S, "tick": i, "node_id": "n1", "node_ip": "10.0.0.1",
         "endpoint": "http://n1:8085/metrics", "error": None, "samples": [
             _metric("ray_serve_replica_processing_queries", 4, replica="r0", **labels),
             _metric("ray_serve_autoscaling_desired_replicas", 1 if t < 60 else 2, **labels),
             _metric("ray_serve_sla_autoscaling_recommended_replicas", 1 if t < 60 else 2, **labels),
             _metric("ray_serve_sla_autoscaling_model_ready", float(t >= 100), **labels),
             _metric("ray_serve_sla_autoscaling_training_rows", float(t), **labels),
             _metric("ray_serve_sla_autoscaling_predicted_ttft_ms", 900.0, **labels),
             _metric("ray_serve_sla_autoscaling_predicted_itl_ms", 25.0, **labels),
             _metric("ray_vllm_time_to_first_token_seconds_sum", 0.8 * max(t, 0), ReplicaId="r0"),
             _metric("ray_vllm_time_to_first_token_seconds_count", float(max(t, 0)), ReplicaId="r0"),
             _metric("ray_vllm_request_time_per_output_token_seconds_sum", 0.02 * max(t, 0), ReplicaId="r0"),
             _metric("ray_vllm_request_time_per_output_token_seconds_count", float(max(t, 0)), ReplicaId="r0"),
             _metric("ray_vllm_num_requests_waiting", 3.0, ReplicaId="r0"),
         ]}
        for i, t in enumerate(range(-5, 210, 5))
    ])
    _jsonl(root / "telemetry" / "sla_policy_events.jsonl", [
        {"timestamp_ns": T0, "error": None, "logged_at_ns": T0 - 20 * S,
         "deployment": f"Deployment(name='{LLM}', app='{APP}')", "reason": "model_not_ready",
         "recommended_replicas": 1.0, "advisory": False},
        {"timestamp_ns": T0 + 100 * S, "error": None, "logged_at_ns": T0 + 100 * S,
         "deployment": f"Deployment(name='{LLM}', app='{APP}')", "reason": "capacity",
         "recommended_replicas": 2.0, "advisory": False},
    ])
    return root


def test_analyze_run_writes_the_evaluation_and_sla_panel(sla_run: Path):
    result = analyze_run(sla_run)
    assert result["plot_error"] is None
    assert (sla_run / "analysis" / "autoscaling_timeline.png").exists()
    plot_data = json.loads((sla_run / "analysis" / "plot_data.json").read_text())
    assert plot_data["availability"]["sla_policy_metrics"]
    assert plot_data["availability"]["vllm_latency"]
    assert plot_data["sla_policy_events"][1]["relative_time_s"] == 100.0
    series = plot_data["serve_metric_series"]
    assert series["vllm_mean_ttft_ms"]["points"][-1]["value"] == pytest.approx(800)
    assert series["vllm_mean_itl_ms"]["points"][-1]["value"] == pytest.approx(20)
    assert series["vllm_waiting_requests"]["points"][0]["value"] == 3

    evaluation = json.loads((sla_run / "analysis" / "autoscaling_evaluation.json").read_text())
    assert evaluation["slo"] == {"ttft_ms": 3000.0, "itl_ms": 50.0, "percentile": 0.95}
    assert evaluation["periods"]["steady"] == {"start_s": 60.0, "end_s": 200.0}
    steady = evaluation["slo_attainment"]["steady"]
    assert steady["requests"]["itl"]["slo_attainment"] == 1.0
    assert 0 < steady["requests"]["ttft"]["slo_attainment"] < 1
    assert steady["windows"]["rolling_window_s"] == 30
    assert evaluation["scale_events"][0]["sla_reason"] == "model_not_ready"
    assert evaluation["scale_events"][0]["lag_s"] == 90.0
    sla = evaluation["sla_policy"]
    assert (sla["model_ready_at_s"], sla["first_capacity_decision_at_s"]) == (100.0, 100.0)
    assert sla["prediction"]["ttft"]["predicted_vs_server"]["bias_ms"] == pytest.approx(100)
    assert sla["prediction"]["itl"]["predicted_vs_server"]["bias_ms"] == pytest.approx(5)
    assert sla["prediction"]["ttft"]["client_vs_server"]["samples"] > 0
