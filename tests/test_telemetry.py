import json
import threading
from enum import Enum
from pathlib import Path

import pytest

from conftest import build_config
from serve_llm_autoscaling.telemetry import (
    TelemetrySession,
    discover_endpoints,
    fetch_serve_status,
    parse_metrics,
    prometheus_collector,
    serve_status_collector,
    serve_status_sample,
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
    config = build_config()
    config.analysis.status_interval_s = config.analysis.metrics_interval_s = 0.01
    kwargs = {
        "status_fetch": serve_status,
        "discover": lambda: [{"NodeID": "n1", "NodeManagerAddress": "10.0.0.1",
                              "MetricsExportPort": 8085, "Alive": True}],
        "fetch_metrics": lambda url: "ray_serve_autoscaling_desired_replicas 1\n",
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
# HELP ray_serve_autoscaling_replica_metric Replica metric.
# TYPE ray_serve_autoscaling_replica_metric gauge
ray_serve_autoscaling_replica_metric{application="ttft-benchmark",deployment="LLMRouter",replica="r1",metric="outstanding_effective_prefill_tokens:LLMServer:x"} 4.0
# TYPE ray_serve_autoscaling_router_metric gauge
ray_serve_autoscaling_router_metric{application="ttft-benchmark",deployment="LLMServer:x",handle="h1",metric="outstanding_effective_prefill_tokens"} 4.0
# TYPE serve_autoscaling_desired_replicas gauge
serve_autoscaling_desired_replicas{application="ttft-benchmark",deployment="LLMServer:x"} 2.0
# TYPE ray_serve_http_request_latency_ms histogram
ray_serve_http_request_latency_ms_count{route="/"} 3.0
# TYPE ray_vllm_num_requests_running gauge
ray_vllm_num_requests_running{ReplicaId="r0"} 2.0
# TYPE ray_tasks gauge
ray_tasks{State="RUNNING"} 5.0
"""


def test_parse_metrics_selects_and_normalizes():
    samples = parse_metrics(EXPOSITION)
    # Unselected Serve, vLLM and Ray families are not sampled; nor is the
    # per-handle router gauge of earlier Ray branches.
    assert [(s["name"], s["normalized_name"]) for s in samples] == [
        ("ray_serve_autoscaling_replica_metric", "serve_autoscaling_replica_metric"),
        ("serve_autoscaling_desired_replicas", "serve_autoscaling_desired_replicas"),
    ]
    assert samples[0]["labels"] == {
        "application": "ttft-benchmark", "deployment": "LLMRouter", "replica": "r1",
        "metric": "outstanding_effective_prefill_tokens:LLMServer:x",
    }
    assert samples[0]["value"] == 4 and samples[0]["type"] == "gauge"


def test_selected_metrics_are_the_families_the_analysis_reads():
    from serve_llm_autoscaling.analysis import METRIC_SERIES, router_report
    from serve_llm_autoscaling.telemetry import SELECTED_METRICS

    assert SELECTED_METRICS == (
        "serve_autoscaling_desired_replicas",
        "serve_autoscaling_target_replicas",
        "serve_autoscaling_policy_metric",
        "serve_autoscaling_replica_metric",
    )
    # Each analysed family is collected, the router's report as the replica gauge.
    (report,) = [s for s in parse_metrics(EXPOSITION)
                 if s["normalized_name"] == "serve_autoscaling_replica_metric"]
    assert router_report(report)["name"] == "ray_serve_autoscaling_router_metric"
    analysed = {spec[0] for spec in METRIC_SERIES.values()}
    assert analysed - {"serve_autoscaling_router_metric"} <= set(SELECTED_METRICS)


def test_parse_metrics_without_serve_metrics():
    assert parse_metrics("# TYPE ray_tasks gauge\nray_tasks 1\n") == []


@pytest.mark.parametrize("value, expected", [(12345, 12345), (0, 0), ("NaN", None)])
def test_parse_metrics_records_non_finite_values_as_null(value, expected):
    name = "ray_serve_autoscaling_policy_metric"
    (sample,) = parse_metrics(
        f'# TYPE {name} gauge\n'
        f'{name}{{application="app",deployment="LLMServer:x",metric="required_replicas"}} {value}\n'
    )
    assert sample["value"] == expected


def test_fetch_metrics_text_keeps_only_selected_lines(monkeypatch):
    from serve_llm_autoscaling import telemetry

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return None

        def raise_for_status(self):
            return None

        def iter_lines(self, chunk_size):
            return iter(EXPOSITION.encode().splitlines())

    monkeypatch.setattr(telemetry.requests, "get", lambda url, **kwargs: Response())
    kept = telemetry.fetch_metrics_text("http://node/metrics").splitlines()
    assert len(kept) == 5 and all("serve_autoscaling_" in line for line in kept)
    assert not any("serve_autoscaling_router_metric" in line for line in kept)
    assert len(parse_metrics("\n".join(kept) + "\n")) == 2


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
    # Samples are aligned on scrape completion; the start bounds the uncertainty.
    assert good["scrape_started_at_ns"] <= good["scrape_finished_at_ns"] == good["timestamp_ns"]
    assert good["bytes"] == len(EXPOSITION)
    assert good["cycle_overran"] in (True, False) and good["cycle_duration_s"] >= 0
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


def test_session_collects_status_and_metrics(tmp_path: Path):
    session = TelemetrySession(build_config(), tmp_path, status_fetch=serve_status)
    assert [Path(c.path).name for c in session.collectors] == [
        "serve_status.jsonl", "serve_metrics.jsonl",
    ]
