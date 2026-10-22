import json
import threading
from datetime import datetime
from enum import Enum
from pathlib import Path

import pytest

from serve_llm_autoscaling.config import load_config
from serve_llm_autoscaling.telemetry import (
    TelemetrySession,
    discover_endpoints,
    fetch_serve_status,
    parse_metrics,
    parse_sla_policy_event,
    prometheus_collector,
    serve_status_collector,
    serve_status_sample,
    sla_policy_log_collector,
)

APP = "ttft-benchmark"
LLM = "LLMServer:Qwen--Qwen3-0_6B-FP8"


def serve_status(states=None):
    return {"applications": {APP: {
        "status": "RUNNING",
        "deployments": {LLM: {
            "status": "UPSCALING",
            "status_trigger": "AUTOSCALING",
            "replica_states": {"RUNNING": 1, "STARTING": 1} if states is None else states,
        }},
    }}}


def _read(path: Path):
    return [json.loads(line) for line in path.read_text().splitlines()]


def _wait_for_records(collector, count, timeout=5):
    event = threading.Event()
    original = collector._write

    def write(record):
        original(record)
        if collector.records_written >= count:
            event.set()

    collector._write = write
    collector.start()
    assert event.wait(timeout)
    collector.stop(timeout)


def test_status_sample_records_replica_states():
    sample = serve_status_sample(serve_status(), APP, 123)
    assert sample["timestamp_ns"] == 123 and sample["error"] is None
    assert sample["available"] is True
    assert sample["application_status"] == "RUNNING"
    assert sample["deployments"][LLM] == {
        "status": "UPSCALING", "status_trigger": "AUTOSCALING",
        "replica_states": {"RUNNING": 1, "STARTING": 1},
    }


def test_status_sample_records_enum_values():
    class State(str, Enum):
        RUNNING = "RUNNING"

    status = serve_status(states={State.RUNNING: 2})
    status["applications"][APP]["deployments"][LLM]["status"] = State.RUNNING
    deployment = serve_status_sample(status, APP, 1)["deployments"][LLM]
    assert type(deployment["status"]) is str
    assert [type(k) for k in deployment["replica_states"]] == [str]


def test_status_sample_missing_application_is_unavailable():
    sample = serve_status_sample({"applications": {}}, APP, 1)
    assert sample["available"] is False
    assert sample["error"] == "configured application not present"
    assert sample["deployments"] == {}


def test_fetch_serve_status_uses_public_api(monkeypatch):
    from ray import serve
    from ray.serve._private.common import ReplicaState
    from ray.serve.schema import (
        ApplicationStatusOverview, DeploymentStatusOverview, ServeStatus,
    )

    status = ServeStatus(applications={APP: ApplicationStatusOverview(
        status="RUNNING", message="", last_deployed_time_s=0,
        deployments={LLM: DeploymentStatusOverview(
            status="HEALTHY", status_trigger="AUTOSCALING", message="",
            replica_states={ReplicaState.RUNNING: 2, ReplicaState.STARTING: 1},
        )},
    )})
    monkeypatch.setattr(serve, "status", lambda: status)
    sample = serve_status_sample(fetch_serve_status(), APP, 1)
    assert json.loads(json.dumps(sample)) == sample
    assert sample["deployments"][LLM]["replica_states"] == {"RUNNING": 2, "STARTING": 1}


def test_status_collector_recovers_after_error(tmp_path: Path):
    calls = []

    def fetch():
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("controller unavailable")
        return serve_status()

    path = tmp_path / "serve_status.jsonl"
    _wait_for_records(serve_status_collector(path, APP, 0.01, fetch), 2)
    first, second = _read(path)[:2]
    assert first["error"] == "ConnectionError: controller unavailable"
    assert first["deployments"] == {} and first["timestamp_ns"] > 0
    assert second["error"] is None
    assert second["deployments"][LLM]["replica_states"] == {"RUNNING": 1, "STARTING": 1}


def _session(tmp_path, **overrides):
    config = load_config("experiments/step_rate_autoscaling.yaml")
    config.analysis.status_interval_s = config.analysis.metrics_interval_s = 0.01
    kwargs = {
        "status_fetch": serve_status,
        "discover": lambda: [{"NodeID": "n1", "NodeManagerAddress": "10.0.0.1",
                              "MetricsExportPort": 8085, "Alive": True}],
        "fetch_metrics": lambda url: "ray_serve_replica_processing_queries 1\n",
    }
    return TelemetrySession(config, tmp_path, **{**kwargs, **overrides})


def test_session_stops_and_flushes_after_exception(tmp_path: Path):
    session = _session(tmp_path)
    with pytest.raises(RuntimeError, match="aiperf failed"):
        with session:
            threading.Event().wait(0.1)
            raise RuntimeError("aiperf failed")
    assert all(not c._thread.is_alive() for c in session.collectors)
    assert all(c._fh.closed for c in session.collectors)
    assert session.collectors[1]._sample._pool._shutdown  # scrape threads released
    status = _read(tmp_path / "telemetry" / "serve_status.jsonl")
    metrics = _read(tmp_path / "telemetry" / "serve_metrics.jsonl")
    assert len(status) == session.summary()["serve_status.jsonl"]["records"] > 0
    assert metrics and metrics[0]["samples"][0]["value"] == 1


def test_session_without_prometheus(tmp_path: Path):
    config = load_config("experiments/step_rate_autoscaling.yaml")
    config.analysis.prometheus_enabled = False
    session = TelemetrySession(config, tmp_path, status_fetch=serve_status)
    assert [Path(c.path).name for c in session.collectors] == ["serve_status.jsonl"]


def test_session_stops_started_collectors_when_a_start_fails(tmp_path: Path, monkeypatch):
    session = _session(tmp_path)
    status, metrics = session.collectors

    def fail():
        raise OSError("disk full")

    monkeypatch.setattr(metrics, "start", fail)
    with pytest.raises(OSError, match="disk full"):
        session.__enter__()
    assert not status._thread.is_alive() and status._fh.closed


def test_discover_endpoints():
    nodes = [
        {"NodeID": "a", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 8085,
         "Alive": True},
        {"NodeID": "a2", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 8085,
         "Alive": True},
        {"NodeID": "b", "NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 8085,
         "Alive": False},
        {"NodeID": "c", "NodeManagerAddress": "10.0.0.3", "Alive": True},
        {"NodeID": "d", "NodeManagerAddress": "10.0.0.4", "MetricsExportPort": 9000,
         "Alive": True},
    ]
    assert discover_endpoints(nodes) == [
        {"node_id": "a", "node_ip": "10.0.0.1", "endpoint": "http://10.0.0.1:8085/metrics"},
        {"node_id": "d", "node_ip": "10.0.0.4", "endpoint": "http://10.0.0.4:9000/metrics"},
    ]


EXPOSITION = """\
# HELP ray_serve_num_ongoing_requests_at_replicas Ongoing requests.
# TYPE ray_serve_num_ongoing_requests_at_replicas gauge
ray_serve_num_ongoing_requests_at_replicas{application="ttft-benchmark",deployment="LLMServer:x",handle="h1",actor_id="a1"} 4.0
# TYPE serve_autoscaling_desired_replicas gauge
serve_autoscaling_desired_replicas{application="ttft-benchmark",deployment="LLMServer:x"} 2.0
# TYPE ray_serve_http_request_latency_ms histogram
ray_serve_http_request_latency_ms_count{route="/"} 3.0
# TYPE ray_tasks gauge
ray_tasks{State="RUNNING"} 5.0
"""


def test_parse_metrics_selects_and_normalizes():
    samples, families = parse_metrics(EXPOSITION)
    assert [(s["name"], s["normalized_name"]) for s in samples] == [
        ("ray_serve_num_ongoing_requests_at_replicas",
         "serve_num_ongoing_requests_at_replicas"),
        ("serve_autoscaling_desired_replicas", "serve_autoscaling_desired_replicas"),
    ]
    assert samples[0]["labels"] == {"application": "ttft-benchmark",
                                    "deployment": "LLMServer:x", "handle": "h1",
                                    "actor_id": "a1"}
    assert samples[0]["value"] == 4 and samples[0]["type"] == "gauge"
    # Unselected Serve families are listed, non-Serve families are not.
    assert "serve_http_request_latency_ms" in families
    assert not any("tasks" in f for f in families)


def test_parse_metrics_without_serve_metrics():
    assert parse_metrics("# TYPE ray_tasks gauge\nray_tasks 1\n") == ([], [])


def test_prometheus_collector_records_node_failure(tmp_path: Path):
    nodes = [{"NodeID": "good", "NodeManagerAddress": "10.0.0.1", "MetricsExportPort": 1,
              "Alive": True},
             {"NodeID": "bad", "NodeManagerAddress": "10.0.0.2", "MetricsExportPort": 1,
              "Alive": True}]

    def fetch(url):
        if "10.0.0.2" in url:
            raise TimeoutError("scrape timed out")
        return EXPOSITION

    path = tmp_path / "serve_metrics.jsonl"
    _wait_for_records(prometheus_collector(path, 0.01, lambda: nodes, fetch), 4)
    records = _read(path)
    by_node = {r["node_id"]: r for r in records[:2]}
    assert by_node["bad"]["error"] == "TimeoutError: scrape timed out"
    assert by_node["bad"]["samples"] == []
    assert by_node["good"]["error"] is None
    good = by_node["good"]
    assert good["endpoint"] == "http://10.0.0.1:1/metrics"
    assert len(good["samples"]) == 2
    assert "available_serve_metrics" in good
    # Samples are aligned on scrape completion; the start bounds the uncertainty.
    assert good["scrape_started_at_ns"] <= good["scrape_finished_at_ns"] == good["timestamp_ns"]
    assert good["bytes"] == len(EXPOSITION)
    assert good["cycle_overran"] in (True, False) and good["cycle_duration_s"] >= 0
    later = [r for r in records[2:] if r["node_id"] == "good"]
    assert later and "available_serve_metrics" not in later[0]
    assert records[0]["tick"] == records[1]["tick"] == 0


def test_prometheus_collector_records_discovery_failure(tmp_path: Path):
    def discover():
        raise RuntimeError("ray not connected")

    path = tmp_path / "serve_metrics.jsonl"
    _wait_for_records(prometheus_collector(path, 0.01, discover, lambda url: ""), 1)
    record = _read(path)[0]
    assert record["error"] == "RuntimeError: ray not connected"
    assert record["endpoint"] is None and record["samples"] == []


def test_prometheus_collector_records_empty_discovery(tmp_path: Path):
    path = tmp_path / "serve_metrics.jsonl"
    _wait_for_records(prometheus_collector(path, 0.01, lambda: [], lambda url: ""), 2)
    records = _read(path)
    assert all(r["error"] == "RuntimeError: No live Ray metrics endpoints discovered"
               for r in records)


VLLM_EXPOSITION = """\
# TYPE ray_vllm_time_to_first_token_seconds histogram
ray_vllm_time_to_first_token_seconds_bucket{ReplicaId="r0",le="+Inf",model_name="m"} 3.0
ray_vllm_time_to_first_token_seconds_sum{ReplicaId="r0",model_name="m"} 1.5
ray_vllm_time_to_first_token_seconds_count{ReplicaId="r0",model_name="m"} 3.0
# TYPE ray_vllm_prefix_cache_hits_total counter
ray_vllm_prefix_cache_hits_total{ReplicaId="r0",model_name="m"} 7.0
# TYPE ray_vllm_num_requests_running gauge
ray_vllm_num_requests_running{ReplicaId="r0",model_name="m"} 2.0
# TYPE ray_vllm_gpu_cache_usage_perc gauge
ray_vllm_gpu_cache_usage_perc{ReplicaId="r0",model_name="m"} 0.5
# TYPE ray_serve_sla_autoscaling_predicted_ttft_ms gauge
ray_serve_sla_autoscaling_predicted_ttft_ms{application="ttft-benchmark",deployment="LLMServer:x"} 1200.0
"""


def test_parse_metrics_selects_vllm_and_sla_families():
    samples, families = parse_metrics(VLLM_EXPOSITION)
    assert [s["name"] for s in samples] == [
        "ray_vllm_time_to_first_token_seconds_bucket",
        "ray_vllm_time_to_first_token_seconds_sum",
        "ray_vllm_time_to_first_token_seconds_count",
        "ray_vllm_prefix_cache_hits_total",
        "ray_vllm_num_requests_running",
        "ray_serve_sla_autoscaling_predicted_ttft_ms",
    ]
    assert samples[1]["normalized_name"] == "vllm_time_to_first_token_seconds"
    assert samples[1]["labels"]["ReplicaId"] == "r0"
    # Unselected vLLM families are neither sampled nor listed as Serve metrics.
    assert families == ["serve_sla_autoscaling_predicted_ttft_ms"]


SLA_LOG_LINE = (
    "INFO 2026-10-20 12:34:56,789 controller 4242 -- SLA autoscaling policy for "
    "Deployment(name='LLMServer:x', app='ttft-benchmark') changed state to capacity "
    "(recommended=2.0, advisory=False).\n"
)


def test_parse_sla_policy_event():
    event = parse_sla_policy_event(SLA_LOG_LINE)
    assert event["deployment"] == "Deployment(name='LLMServer:x', app='ttft-benchmark')"
    assert (event["reason"], event["recommended_replicas"], event["advisory"]) == (
        "capacity", 2.0, False
    )
    expected = datetime(2026, 10, 20, 12, 34, 56, 789000).timestamp()
    assert event["logged_at_ns"] == round(expected * 1e9)
    assert parse_sla_policy_event("INFO 2026-10-20 12:34:56,789 controller -- hello\n") is None


def test_sla_policy_log_collector_tails_controller_logs(tmp_path: Path):
    logs = tmp_path / "serve"
    logs.mkdir()
    log = logs / "controller_4242.log"
    log.write_text("INFO 2026-10-20 12:34:50,000 controller 4242 -- started\n" + SLA_LOG_LINE)
    path = tmp_path / "sla_policy_events.jsonl"
    collector = sla_policy_log_collector(path, 0.01, lambda: logs)
    _wait_for_records(collector, 1)
    (first,) = _read(path)
    assert first["reason"] == "capacity" and first["error"] is None
    assert first["timestamp_ns"] > 0 and first["logged_at_ns"] < first["timestamp_ns"]
    # Later ticks read only what was appended since.
    with log.open("a") as fh:
        fh.write(SLA_LOG_LINE.replace("capacity", "safe_scale_down"))
    (second,) = collector._sample(1, 1)
    assert second["reason"] == "safe_scale_down"
    assert collector._sample(2, 2) == []


def test_session_collects_sla_log_only_for_sla_policy(tmp_path: Path):
    config = load_config("experiments/step_rate_autoscaling.yaml")
    raw = config.resolved_dict()
    raw["deployment"]["autoscaling"]["policy"] = {
        "name": "sla", "kwargs": {"ttft_ms": 3000, "itl_ms": 50},
    }
    sla = TelemetrySession(
        config.model_validate(raw), tmp_path / "sla", status_fetch=serve_status,
        logs_dir=lambda: tmp_path,
    )
    assert [Path(c.path).name for c in sla.collectors] == [
        "serve_status.jsonl", "serve_metrics.jsonl", "sla_policy_events.jsonl",
    ]
    default = TelemetrySession(config, tmp_path / "default", status_fetch=serve_status)
    assert [Path(c.path).name for c in default.collectors] == [
        "serve_status.jsonl", "serve_metrics.jsonl",
    ]
