from __future__ import annotations

import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

from .artifacts import RunArtifacts, format_run_summary, write_json
from .backend import (
    RayServeBackend,
    check_autoscaling_support,
    check_routing_support,
    verify_autoscaling_policy,
    verify_topology,
)
from .benchmark import AIPerfRunner, aiperf_command, workload_row_fields
from .config import AIPERF_SOURCE_DEFAULT, WORKLOAD_DIR, ExperimentConfig, write_config
from .telemetry import TelemetrySession
from .windows import profiling_phase_ns

LIFECYCLE_SCHEMA_VERSION = 1


def _aiperf_version(source: str = AIPERF_SOURCE_DEFAULT) -> str:
    completed = subprocess.run(
        [*aiperf_command(source), "--version"], capture_output=True, text=True, check=True
    )
    return completed.stdout.strip()


def environment_check(
    connect: bool = True, aiperf_source: str = AIPERF_SOURCE_DEFAULT
) -> dict[str, Any]:
    # The driver must share the workers' environment.
    if sys.prefix != sys.base_prefix:
        raise RuntimeError(f"run on the Ray image's Python, not a venv ({sys.prefix})")
    try:
        import ray
    except ImportError as exc:
        raise RuntimeError("Ray is not importable") from exc
    result: dict[str, Any] = {
        "python_version": platform.python_version(),
        "python_executable": sys.executable,
        "aiperf_source": aiperf_source,
        "aiperf_version": _aiperf_version(aiperf_source),
    }
    result.update({"ray_version": ray.__version__, "ray_path": ray.__file__})
    if connect:
        ray.init(address="auto", ignore_reinit_error=True)
        result["cluster_resources"] = ray.cluster_resources()
        if result["cluster_resources"].get("GPU", 0) < 1:
            raise RuntimeError("the connected Ray cluster does not report any GPUs")
    return result


def lifecycle_record(
    config: ExperimentConfig,
    artifact_dir: Path,
    *,
    aiperf_launched_ns: int,
    aiperf_exited_ns: int,
    observation_end_ns: int | None = None,
    observation_skipped: str | None = None,
) -> dict[str, Any]:
    """Epoch-ns boundaries of a run's load, drain and idle periods.

    Load issuance ends ``duration_s`` after profiling starts; AIPerf then
    drains in-flight requests (up to ``grace_period_s``) and exits; the
    harness then observes an idle deployment for ``post_load_observation_s``.
    """
    benchmark = config.benchmark
    start, phase_end = profiling_phase_ns(artifact_dir)

    def offset(seconds: float) -> int | None:
        return None if start is None else start + round(seconds * 1e9)

    return {
        "schema_version": LIFECYCLE_SCHEMA_VERSION,
        "clock": "epoch_ns",
        "aiperf_launched_ns": aiperf_launched_ns,
        "profiling_start_ns": start,
        # Kept for readers of the schema; a session curve has no single ramp.
        "ramp_end_ns": None,
        "load_end_ns": offset(benchmark.duration_s),
        "profiling_phase_end_ns": phase_end,
        "aiperf_exited_ns": aiperf_exited_ns,
        "post_load_observation_s": benchmark.post_load_observation_s,
        "observation_end_ns": observation_end_ns,
        "observation_skipped": observation_skipped,
    }


def run_benchmark(
    config: ExperimentConfig, root: Path, telemetry: TelemetrySession
) -> dict[str, Any]:
    """Run AIPerf, then observe the idle deployment, collecting telemetry throughout.

    Returns AIPerf's summary row and writes it and the lifecycle boundaries
    under ``benchmark/``. An AIPerf failure is recorded and re-raised.
    """
    benchmark_root = root / "benchmark"
    artifact_dir = benchmark_root / WORKLOAD_DIR
    lifecycle_path = benchmark_root / "lifecycle.json"
    runner = AIPerfRunner(config, benchmark_root)
    # Collectors stop and flush on success, benchmark failure, or interrupt.
    with telemetry:
        launched = time.time_ns()
        try:
            row = runner.run()
        except BaseException as exc:  # including Ctrl-C: still record the row
            row = {**workload_row_fields(config.benchmark.workload),
                   "error": f"{type(exc).__name__}: {exc}"}
            raise
        finally:
            # Written even on failure, which skips the observation.
            exited = time.time_ns()
            write_json(benchmark_root / "sweep_summary.json", [row])
            write_json(lifecycle_path, lifecycle_record(
                config, artifact_dir, aiperf_launched_ns=launched, aiperf_exited_ns=exited,
            ))
        if config.benchmark.post_load_observation_s:
            time.sleep(config.benchmark.post_load_observation_s)
            write_json(lifecycle_path, lifecycle_record(
                config, artifact_dir, aiperf_launched_ns=launched, aiperf_exited_ns=exited,
                observation_end_ns=time.time_ns(),
            ))
    return row


def run_analysis(root: Path) -> dict[str, Any]:
    """Derive a run's analysis; failures are recorded, never raised."""
    from .analysis import analyze_run

    try:
        return analyze_run(root)
    except Exception as exc:
        return {"analysis_error": f"{type(exc).__name__}: {exc}"}


def trace_directory(config: ExperimentConfig, root: Path) -> Path:
    """This run's own scheduler-trace directory, created empty."""
    base = config.analysis.scheduler_trace_dir
    directory = (base / root.name if base else root / "telemetry" / "scheduler-traces").resolve()
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def compile_cache_devices() -> list[str]:
    """Per-device vLLM compile caches present on this node (``device_<N>``)."""
    cache = Path.home() / ".cache/vllm/torch_compile_cache/torch_aot_compile"
    return sorted({path.name for path in cache.glob("*/device_*")})


def warm_compile_caches(
    config: ExperimentConfig, *, replicas: int | None = None, replace: bool = False
) -> dict[str, Any]:
    """Start one replica per GPU so every device has a warm vLLM compile cache.

    A replica that lands on a GPU without a compile cache compiles from
    scratch, which roughly doubles its startup time (on B200: ~112s of engine
    init against ~32s). A run with only some devices warm gets fast and slow
    scale-ups mixed together, which skews the result. The cache key is a hash
    of the engine config, so this deploys the experiment's own deployment
    (including the tracing scheduler) at a fixed replica count, requires
    every replica RUNNING, and tears it down.
    """
    import tempfile

    warm = config.model_copy(deep=True)
    warm.deployment.application_name = f"{config.deployment.application_name}-warmup"
    backend = RayServeBackend(warm, replace=replace)
    resources = backend.connect()["cluster_resources"]
    per_replica = warm.deployment.engine.tensor_parallel_size or 1
    count = replicas or int(resources.get("GPU", 0)) // per_replica
    if count < 1:
        raise RuntimeError("the connected Ray cluster does not report any GPUs")
    autoscaling = warm.deployment.autoscaling
    autoscaling.min_replicas = autoscaling.initial_replicas = autoscaling.max_replicas = count
    check_routing_support(warm)
    check_autoscaling_support(warm)
    with tempfile.TemporaryDirectory(prefix="autoscale-harness-warmup-") as traces:
        warm.analysis.scheduler_trace_dir = Path(traces)
        backend.deploy()
        try:
            ready = backend.wait_healthy()
            # /v1/models answers once one replica is up; wait for all of them.
            deadline = time.monotonic() + warm.runtime.readiness_timeout_s
            while True:
                deployments = backend.status()["applications"][
                    warm.deployment.application_name]["deployments"]
                running = {
                    name: (info.get("replica_states") or {}).get("RUNNING", 0)
                    for name, info in deployments.items() if name.startswith("LLMServer")
                }
                if running and all(value == count for value in running.values()):
                    break
                if time.monotonic() > deadline:
                    raise RuntimeError(f"expected {count} RUNNING replicas, got {running}")
                time.sleep(warm.runtime.readiness_poll_s)
        finally:
            backend.teardown()
    return {
        "replicas": count,
        "running_replicas": running,
        "cached_devices": compile_cache_devices(),
        "models": ready["models"],
    }


def run_experiment(
    config: ExperimentConfig,
    config_path: Path,
    *,
    replace: bool = False,
    keep_deployment: bool | None = None,
    warm: bool = False,
) -> Path:
    """Deploy, replay the workload with telemetry, analyze, and tear down.

    With ``warm``, every GPU's compile cache is warmed first (see
    ``warm_compile_caches``). Returns the run directory.
    """
    artifacts = RunArtifacts.create(config, config_path)
    deployed = False
    stage = "environment_check"
    keep = config.runtime.keep_deployment if keep_deployment is None else keep_deployment
    try:
        # Each run traces its engines into its own directory.
        config = config.model_copy(deep=True)
        config.analysis.scheduler_trace_dir = trace_directory(config, artifacts.root)
        write_config(config, artifacts.root / "resolved.yaml")
        artifacts.manifest["scheduler_traces"] = {
            "directory": str(config.analysis.scheduler_trace_dir),
        }
        artifacts.flush_manifest()
        backend = RayServeBackend(config, replace=replace)
        env = environment_check(connect=True, aiperf_source=config.runtime.aiperf_source)
        artifacts.manifest["environment"] = env
        artifacts.flush_manifest()

        stage = "routing_preflight"
        routing = {**backend.routing_summary(), "resolved": check_routing_support(config)}
        routing_path = artifacts.root / "deployment" / "routing.json"
        write_json(routing_path, routing)
        artifacts.manifest["routing"] = routing["resolved"]
        artifacts.flush_manifest()

        stage = "autoscaling_preflight"
        autoscaling = check_autoscaling_support(config)
        autoscaling_path = artifacts.root / "deployment" / "autoscaling.json"
        write_json(autoscaling_path, autoscaling)
        artifacts.manifest["autoscaling"] = autoscaling
        artifacts.flush_manifest()

        if warm:
            stage = "compile_cache_warmup"
            artifacts.record_event("compile_cache_warmup_started")
            warmup = warm_compile_caches(config, replace=replace)
            write_json(artifacts.root / "deployment" / "compile_cache_warmup.json", warmup)
            artifacts.manifest["compile_cache_warmup"] = {
                key: warmup[key] for key in ("replicas", "cached_devices")
            }
            artifacts.flush_manifest()

        stage = "deploy"
        artifacts.record_event("deploy_started")
        write_json(
            artifacts.root / "deployment" / "config.json",
            backend.deployment_spec(),
        )
        status = backend.deploy()
        deployed = True
        write_json(artifacts.root / "deployment" / "initial_status.json", status)

        stage = "readiness"
        ready_started = time.monotonic()
        ready = backend.wait_healthy()
        artifacts.manifest["readiness_s"] = time.monotonic() - ready_started
        artifacts.flush_manifest()
        write_json(artifacts.root / "deployment" / "ready_status.json", ready)
        artifacts.record_event("deployment_ready")

        stage = "topology"
        routing["verified"] = verify_topology(config, ready["serve_status"])
        write_json(routing_path, routing)
        artifacts.manifest["routing"] = {**routing["resolved"], **routing["verified"]}
        artifacts.flush_manifest()

        stage = "autoscaling_policy"
        autoscaling.update(verify_autoscaling_policy(config, backend.serve_details()))
        write_json(autoscaling_path, autoscaling)
        artifacts.manifest["autoscaling"] = autoscaling
        artifacts.flush_manifest()

        stage = "benchmark"
        # Start telemetry immediately before AIPerf.
        telemetry = TelemetrySession(config, artifacts.root)
        try:
            run_benchmark(config, artifacts.root, telemetry)
        finally:
            artifacts.manifest["telemetry"] = telemetry.summary()
            artifacts.flush_manifest()
        stage = "analysis"
        artifacts.manifest["analysis"] = run_analysis(artifacts.root)
        artifacts.flush_manifest()
        artifacts.finish("succeeded")
        return artifacts.root
    except Exception as exc:
        # A failed generator can still leave useful requests and traces. Keep
        # its original failure while producing whatever diagnostics are valid.
        if (stage == "benchmark" and "analysis" not in artifacts.manifest
                and any(artifacts.root.glob("benchmark/*/profile_export.jsonl"))):
            artifacts.manifest["analysis"] = run_analysis(artifacts.root)
        artifacts.finish("failed", failure_stage=stage, error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if deployed and not keep:
            artifacts.record_event("teardown_started")
            try:
                backend.teardown()
                artifacts.record_event("teardown_completed")
            except Exception as exc:
                artifacts.record_event(
                    "teardown_failed", error=f"{type(exc).__name__}: {exc}"
                )
        elif deployed:
            artifacts.record_event("teardown_skipped")
        print(format_run_summary(artifacts.root), flush=True)
