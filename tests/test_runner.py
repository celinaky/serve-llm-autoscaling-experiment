import json
import json
from pathlib import Path

import pytest

from conftest import TOKEN_POLICY, short_config
from serve_llm_autoscaling import runner
from serve_llm_autoscaling.backend import TRACING_SCHEDULER
from serve_llm_autoscaling.config import load_config

ROW = {"mode": "agentx_concurrency_ramp", "level": 4, "target_concurrency": 4,
       "request_count": 40, "error": None}


class Telemetry:
    """A telemetry stand-in that records when it was collecting."""

    def __init__(self):
        self.events = []

    def __enter__(self):
        self.events.append("start")
        return self

    def __exit__(self, *exc):
        self.events.append("stop")


def _read(path: Path):
    return json.loads(path.read_text())


def _events(root: Path):
    lines = (root / "deployment" / "events.jsonl").read_text().splitlines()
    return [json.loads(line)["event"] for line in lines]


# --- run_benchmark ----------------------------------------------------------


def test_run_benchmark_writes_summary_and_lifecycle(monkeypatch, tmp_path: Path):
    telemetry = Telemetry()
    runners = []

    def run(self):
        runners.append(self)
        telemetry.events.append("aiperf")
        return dict(ROW)

    monkeypatch.setattr(runner.AIPerfRunner, "run", run)
    monkeypatch.setattr(runner.time, "sleep", lambda s: pytest.fail("no observation configured"))
    config = short_config()
    assert runner.run_benchmark(config, tmp_path, telemetry) == ROW
    # Telemetry brackets AIPerf, which writes below <run>/benchmark.
    assert telemetry.events == ["start", "aiperf", "stop"]
    assert (runners[0].config, runners[0].benchmark_root) == (config, tmp_path / "benchmark")
    assert _read(tmp_path / "benchmark" / "sweep_summary.json") == [ROW]
    lifecycle = _read(tmp_path / "benchmark" / "lifecycle.json")
    assert lifecycle["aiperf_launched_ns"] <= lifecycle["aiperf_exited_ns"]
    assert lifecycle["post_load_observation_s"] == 0
    assert lifecycle["observation_end_ns"] is None
    # AIPerf wrote no phase manifest, so the load period is unknown.
    assert lifecycle["profiling_start_ns"] is None and lifecycle["load_end_ns"] is None


def test_run_benchmark_records_and_reraises_an_aiperf_failure(monkeypatch, tmp_path: Path):
    telemetry = Telemetry()

    def fail(self):
        raise RuntimeError("AIPerf exited 1; see x/stderr.log")

    monkeypatch.setattr(runner.AIPerfRunner, "run", fail)
    monkeypatch.setattr(runner.time, "sleep", lambda s: pytest.fail("observed a failed run"))
    config = short_config(post_load_observation_s=90)
    with pytest.raises(RuntimeError, match="AIPerf exited 1"):
        runner.run_benchmark(config, tmp_path, telemetry)
    assert telemetry.events == ["start", "stop"]
    assert _read(tmp_path / "benchmark" / "sweep_summary.json") == [{
        "target_concurrency": 4,
        "concurrency_series": [{"time_s": 0.0, "sessions": 1}, {"time_s": 120.0, "sessions": 4}],
        "error": "RuntimeError: AIPerf exited 1; see x/stderr.log",
    }]
    lifecycle = _read(tmp_path / "benchmark" / "lifecycle.json")
    assert lifecycle["aiperf_exited_ns"] is not None and lifecycle["observation_end_ns"] is None


def test_run_benchmark_stops_telemetry_on_interrupt(monkeypatch, tmp_path: Path):
    telemetry = Telemetry()

    def interrupt(self):
        raise KeyboardInterrupt

    monkeypatch.setattr(runner.AIPerfRunner, "run", interrupt)
    with pytest.raises(KeyboardInterrupt):
        runner.run_benchmark(short_config(), tmp_path, telemetry)
    # The collectors stopped, and the interrupted run is still recorded.
    assert telemetry.events == ["start", "stop"]
    (row,) = json.loads((tmp_path / "benchmark" / "sweep_summary.json").read_text())
    assert row["error"].startswith("KeyboardInterrupt")
    assert (tmp_path / "benchmark" / "lifecycle.json").is_file()


# --- Trace directory and compile caches -------------------------------------


def test_trace_directory_is_per_run(tmp_path: Path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    config = short_config()
    # By default inside the run directory, as an absolute path for the replicas.
    assert runner.trace_directory(config, Path("runs/run-a")) == (
        tmp_path / "runs/run-a/telemetry/scheduler-traces"
    )
    # A shared base keeps each run's traces apart.
    config.analysis.scheduler_trace_dir = tmp_path / "shared"
    first = runner.trace_directory(config, tmp_path / "runs/run-a")
    second = runner.trace_directory(config, tmp_path / "runs/run-b")
    assert (first, second) == (tmp_path / "shared/run-a", tmp_path / "shared/run-b")
    assert first.is_dir() and second.is_dir()
    assert runner.trace_directory(config, tmp_path / "runs/run-a") == first  # idempotent
    assert config.analysis.scheduler_trace_dir == tmp_path / "shared"


def test_compile_cache_devices(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(runner.Path, "home", lambda: tmp_path)
    assert runner.compile_cache_devices() == []
    cache = tmp_path / ".cache/vllm/torch_compile_cache/torch_aot_compile"
    for name in ("hash-a/device_1", "hash-a/device_0", "hash-b/device_0", "hash-b/other"):
        (cache / name).mkdir(parents=True)
    assert runner.compile_cache_devices() == ["device_0", "device_1"]


def _warmup_cluster(monkeypatch, *, gpus=8, running=None):
    """Fake the backend for ``warm_compile_caches``; returns what it deployed."""
    deployed = {}
    monkeypatch.setattr(runner, "check_routing_support", lambda config: {})
    monkeypatch.setattr(runner, "check_autoscaling_support", lambda config: {})
    monkeypatch.setattr(runner, "compile_cache_devices", lambda: ["device_0"])
    monkeypatch.setattr("ray.serve.context._set_global_client", lambda client: None)
    backend = runner.RayServeBackend
    monkeypatch.setattr(backend, "connect", lambda self: {"cluster_resources": {"GPU": gpus}})

    def deploy(self):
        deployed.update(config=self.config.model_copy(deep=True), replace=self.replace,
                        traces_exist=self.config.analysis.scheduler_trace_dir.is_dir())

    def status(self):
        count = self.config.deployment.autoscaling.max_replicas if running is None else running
        app = self.config.deployment.application_name
        return {"applications": {app: {"deployments": {
            "LLMServer:m": {"replica_states": {"RUNNING": count}}, "LLMRouter": {},
        }}}}

    monkeypatch.setattr(backend, "deploy", deploy)
    monkeypatch.setattr(backend, "status", status)
    monkeypatch.setattr(backend, "wait_healthy", lambda self: {"models": {"data": []}})
    monkeypatch.setattr(backend, "teardown",
                        lambda self: deployed.setdefault("torn_down", True))
    return deployed


def test_warm_compile_caches_starts_one_replica_per_gpu(monkeypatch):
    deployed = _warmup_cluster(monkeypatch)
    config = short_config(deployment={
        **TOKEN_POLICY, "engine": {"tensor_parallel_size": 2},
    })
    result = runner.warm_compile_caches(config, replace=True)
    assert result == {"replicas": 4, "running_replicas": {"LLMServer:m": 4},
                      "cached_devices": ["device_0"], "models": {"data": []}}
    warm = deployed["config"]
    scaling = warm.deployment.autoscaling
    assert (scaling.min_replicas, scaling.initial_replicas, scaling.max_replicas) == (4, 4, 4)
    assert warm.deployment.application_name == "ttft-benchmark-warmup"
    # The experiment's own engine and policy, traced into a scratch directory
    # so the compile-cache key matches the run's.
    assert warm.deployment.engine == config.deployment.engine
    assert warm.deployment.autoscaling.policy == config.deployment.autoscaling.policy
    assert deployed["traces_exist"] and not warm.analysis.scheduler_trace_dir.exists()
    assert deployed["replace"] and deployed["torn_down"]
    # The experiment's config is untouched.
    assert config.deployment.autoscaling.max_replicas == 4
    assert config.deployment.autoscaling.min_replicas is None
    assert config.deployment.application_name == "ttft-benchmark"
    assert config.analysis.scheduler_trace_dir is None


def test_warm_compile_caches_replica_count(monkeypatch):
    deployed = _warmup_cluster(monkeypatch, gpus=8)
    assert runner.warm_compile_caches(short_config(), replicas=3)["replicas"] == 3
    assert deployed["config"].deployment.autoscaling.max_replicas == 3
    _warmup_cluster(monkeypatch, gpus=0)
    with pytest.raises(RuntimeError, match="does not report any GPUs"):
        runner.warm_compile_caches(short_config())


def test_warm_compile_caches_requires_every_replica_and_tears_down(monkeypatch):
    deployed = _warmup_cluster(monkeypatch, gpus=2, running=1)
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    config = short_config(runtime={"readiness_timeout_s": 0.01})
    with pytest.raises(RuntimeError, match=r"expected 2 RUNNING replicas, got \{'LLMServer:m': 1\}"):
        runner.warm_compile_caches(config)
    assert deployed["torn_down"]


# --- environment_check ------------------------------------------------------


def test_environment_check_without_a_cluster(monkeypatch):
    monkeypatch.setattr(runner, "_aiperf_version", lambda source: f"aiperf from {source}")
    monkeypatch.setattr(runner.sys, "base_prefix", runner.sys.prefix)
    result = runner.environment_check(connect=False, aiperf_source="aiperf==9.9")
    assert result["aiperf_version"] == "aiperf from aiperf==9.9"
    assert result["ray_version"] and "cluster_resources" not in result
    # The driver must share the workers' interpreter.
    monkeypatch.setattr(runner.sys, "base_prefix", "/somewhere/else")
    with pytest.raises(RuntimeError, match="not a venv"):
        runner.environment_check(connect=False)


# --- run_experiment ---------------------------------------------------------


def test_experiment_deploys_collects_telemetry_and_analyzes(
    fake_cluster, experiment, monkeypatch
):
    config, path = experiment(post_load_observation_s=0)
    alive, analyzed = [], []

    def run(self):
        alive.append(fake_cluster.collectors_alive())
        return dict(ROW)

    monkeypatch.setattr(runner.AIPerfRunner, "run", run)
    monkeypatch.setattr(runner, "run_analysis",
                        lambda root: analyzed.append(root) or {"warnings": ["w"]})
    root = runner.run_experiment(config, path)

    assert root.parent == config.runtime.results_dir and root.name.endswith("-agentx-ramp")
    assert alive == [[True, True]]  # collectors ran during AIPerf
    assert fake_cluster.collectors_alive() == [False, False]
    assert analyzed == [root]
    assert fake_cluster.calls == ["deploy", "wait_healthy", "teardown"]
    manifest = _read(root / "manifest.json")
    assert manifest["status"] == "succeeded" and manifest["failure_stage"] is None
    assert manifest["analysis"] == {"warnings": ["w"]}
    assert set(manifest["telemetry"]) == {"serve_status.jsonl", "serve_metrics.jsonl"}
    assert manifest["routing"] == {"policy": None, "request_router_class": None,
                                   "topology": "openai_ingress",
                                   "deployments": ["LLMServer:m", "OpenAiIngress"]}
    assert manifest["autoscaling"]["verified_deployments"] == ["LLMServer:m"]
    assert "compile_cache_warmup" not in manifest
    assert (root / "input.yaml").read_text() == path.read_text()
    assert _read(root / "benchmark" / "sweep_summary.json") == [ROW]
    assert (root / "telemetry" / "serve_status.jsonl").exists()
    assert _events(root) == ["deploy_started", "deployment_ready", "teardown_started",
                             "teardown_completed"]

    # Each run traces its engines into its own directory, and says so.
    traces = root.resolve() / "telemetry" / "scheduler-traces"
    assert traces.is_dir()
    assert manifest["scheduler_traces"] == {"directory": str(traces)}
    assert load_config(root / "resolved.yaml").analysis.scheduler_trace_dir == traces
    deployed = _read(root / "deployment" / "config.json")
    assert deployed["engine_kwargs"]["scheduler_cls"] == TRACING_SCHEDULER
    assert deployed["runtime_env"]["env_vars"]["AUTOSCALE_HARNESS_TRACE_DIR"] == str(traces)
    assert config.analysis.scheduler_trace_dir is None  # the caller's config is untouched


def test_experiment_with_kv_aware_token_policy(fake_cluster, experiment, monkeypatch):
    config, path = experiment(deployment=TOKEN_POLICY)
    monkeypatch.setattr(runner.AIPerfRunner, "run", lambda self: dict(ROW))
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path)
    manifest = _read(root / "manifest.json")
    assert manifest["routing"]["topology"] == "direct_streaming"
    assert manifest["autoscaling"]["policy"] == "token"
    assert manifest["autoscaling"]["policy_kwargs"] == {"target_prefill_tokens_per_replica": 20000}
    routing = _read(root / "deployment" / "routing.json")
    assert routing["environment"]["RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"] == "1"
    assert routing["verified"]["deployments"] == ["LLMRouter", "LLMServer:m"]


def test_analysis_failure_does_not_fail_run(fake_cluster, experiment, monkeypatch):
    config, path = experiment()
    monkeypatch.setattr(runner.AIPerfRunner, "run", lambda self: dict(ROW))
    monkeypatch.setattr(runner.time, "sleep", lambda s: None)
    # No AIPerf artifacts exist, so the real analysis fails.
    root = runner.run_experiment(config, path)
    manifest = _read(root / "manifest.json")
    assert manifest["status"] == "succeeded"
    assert "analysis_error" in manifest["analysis"]


def test_failed_aiperf_still_analyzes_partial_saved_requests(
    fake_cluster, experiment, monkeypatch
):
    config, path = experiment()
    analyzed = []

    def fail(self):
        records = self.benchmark_root / "agentx-concurrency-ramp" / "profile_export.jsonl"
        records.parent.mkdir(parents=True)
        records.write_text("{}\n")
        raise RuntimeError("generator interrupted")

    monkeypatch.setattr(runner.AIPerfRunner, "run", fail)
    monkeypatch.setattr(runner, "run_analysis",
                        lambda root: analyzed.append(root) or {"warnings": ["partial data"]})
    with pytest.raises(RuntimeError, match="generator interrupted"):
        runner.run_experiment(config, path)
    (root,) = config.runtime.results_dir.iterdir()
    manifest = _read(root / "manifest.json")
    assert analyzed == [root]
    assert manifest["status"] == "failed" and manifest["failure_stage"] == "benchmark"
    assert manifest["analysis"]["warnings"] == ["partial data"]
    assert "generator interrupted" in manifest["error"]
    assert "serve_status.jsonl" in manifest["telemetry"]
    assert fake_cluster.calls[-1] == "teardown"


def test_failed_aiperf_without_requests_is_not_analyzed(fake_cluster, experiment, monkeypatch):
    config, path = experiment()

    def fail(self):
        raise RuntimeError("uvx not on PATH")

    monkeypatch.setattr(runner.AIPerfRunner, "run", fail)
    monkeypatch.setattr(runner, "run_analysis", lambda root: pytest.fail("nothing to analyze"))
    with pytest.raises(RuntimeError, match="uvx not on PATH"):
        runner.run_experiment(config, path)
    (root,) = config.runtime.results_dir.iterdir()
    assert "analysis" not in _read(root / "manifest.json")


def test_preflight_failure_deploys_nothing(fake_cluster, experiment, monkeypatch):
    config, path = experiment()

    def unsupported(config):
        raise RuntimeError("Ray does not provide the policy")

    monkeypatch.setattr(runner, "check_autoscaling_support", unsupported)
    with pytest.raises(RuntimeError, match="does not provide"):
        runner.run_experiment(config, path)
    (root,) = config.runtime.results_dir.iterdir()
    manifest = _read(root / "manifest.json")
    assert (manifest["status"], manifest["failure_stage"]) == ("failed", "autoscaling_preflight")
    assert fake_cluster.calls == []  # nothing deployed, nothing to tear down


@pytest.mark.parametrize("configured, argument, kept", [
    (False, None, False), (True, None, True), (False, True, True), (True, False, False),
])
def test_keep_deployment(fake_cluster, experiment, monkeypatch, configured, argument, kept):
    config, path = experiment(runtime={"keep_deployment": configured})
    monkeypatch.setattr(runner.AIPerfRunner, "run", lambda self: dict(ROW))
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path, keep_deployment=argument)
    assert ("teardown" not in fake_cluster.calls) == kept
    assert _events(root)[-1] == ("teardown_skipped" if kept else "teardown_completed")


def test_warm_runs_before_the_deploy_and_is_recorded(fake_cluster, experiment, monkeypatch):
    config, path = experiment()

    def warm(config, *, replace):
        fake_cluster.calls.append(f"warm replace={replace}")
        return {"replicas": 8, "running_replicas": {"LLMServer:m": 8},
                "cached_devices": ["device_0", "device_1"], "models": {}}

    monkeypatch.setattr(runner, "warm_compile_caches", warm)
    monkeypatch.setattr(runner.AIPerfRunner, "run", lambda self: dict(ROW))
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path, replace=True, warm=True)
    assert fake_cluster.calls == ["warm replace=True", "deploy", "wait_healthy", "teardown"]
    assert _read(root / "manifest.json")["compile_cache_warmup"] == {
        "replicas": 8, "cached_devices": ["device_0", "device_1"],
    }
    assert _read(root / "deployment" / "compile_cache_warmup.json")["running_replicas"] == {
        "LLMServer:m": 8,
    }


def test_run_analysis_never_raises(tmp_path: Path):
    assert runner.run_analysis(tmp_path)["analysis_error"].startswith("FileNotFoundError")
