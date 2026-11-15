import copy
import json
import sys
import types
from pathlib import Path

import pytest
import yaml

from serve_profiler.deployment import llm_spec, load_config, resolved_config
from serve_profiler.workload import (
    command,
    COHORT_SIZES,
    PHASE_SECONDS,
    SESSION_POOL_SIZE,
)


def config():
    return {
        "deployment": {
            "model_id": "model",
            "engine": {"tensor_parallel_size": 1},
            "routing": {"policy": "kv_aware"},
        },
        "runtime": {},
    }


def test_config_copy_and_isolated_deployment(tmp_path):
    original = config()
    before = copy.deepcopy(original)
    resolved, name, route = resolved_config(original, tmp_path / "run-abcdef12")
    spec = llm_spec(resolved, tmp_path)
    assert original == before
    assert name == "prefill-profile-abcdef12"
    assert route == "/" + name
    assert spec["deployment_config"]["num_replicas"] == 1
    assert "autoscaling_config" not in spec["deployment_config"]
    assert (
        spec["engine_kwargs"]["scheduler_cls"]
        == "serve_profiler.scheduler.AutoTracingScheduler"
    )
    assert Path(spec["runtime_env"]["py_modules"][0]).name == "serve_profiler"
    assert spec["runtime_env"]["env_vars"]["RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR"] == ""


@pytest.mark.parametrize(
    "setting", ["tensor_parallel_size", "pipeline_parallel_size", "data_parallel_size"]
)
def test_reject_parallelism(tmp_path, setting):
    value = config()
    value["deployment"]["engine"][setting] = 2
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(value))
    with pytest.raises(ValueError, match="one GPU"):
        load_config(path)


def test_phase_commands_use_aiperf_and_remove_delays(tmp_path):
    assert COHORT_SIZES == (8,)
    assert SESSION_POOL_SIZE == 8
    assert PHASE_SECONDS * len(COHORT_SIZES) == 1800
    for phase, cohort_size in enumerate(COHORT_SIZES):
        argv = command(
            config(), "http://localhost:8000/profile", tmp_path, cohort_size, phase
        )
        assert argv[0] == "uvx"
        assert "--ignore-trace-delays" in argv
        assert "--unsafe-override" in argv
        assert "--fixed-schedule" not in argv
        assert argv[argv.index("--concurrency") + 1] == str(SESSION_POOL_SIZE)
        assert argv[argv.index("--agentx-cohort-size") + 1] == str(cohort_size)
        assert argv[argv.index("--random-seed") + 1] == str(42 + phase)
        assert argv[argv.index("--session-header") + 1] == "X-Session-ID"


@pytest.fixture
def scheduler_module(monkeypatch):
    # Test the wrapper protocol and writer without importing CUDA/vLLM.
    class NativeScheduler:
        def __init__(self, *args, **kwargs):
            self.scheduler_config = kwargs["vllm_config"].scheduler_config
            self.requests = {}

        def shutdown(self):
            pass

    class NativeAsync(NativeScheduler):
        pass

    for name, value in [
        ("vllm.v1.core.sched.scheduler", {"Scheduler": NativeScheduler}),
        ("vllm.v1.core.sched.async_scheduler", {"AsyncScheduler": NativeAsync}),
    ]:
        module = types.ModuleType(name)
        module.__dict__.update(value)
        monkeypatch.setitem(sys.modules, name, module)
    import importlib.util

    path = Path(__file__).parents[1] / "serve_profiler/scheduler.py"
    spec = importlib.util.spec_from_file_location("scheduler_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_factory_preserves_resolved_mode_and_flushes(
    scheduler_module, monkeypatch, tmp_path
):
    monkeypatch.setenv("SERVE_PROFILER_TRACE_DIR", str(tmp_path))
    for async_mode in (False, True):
        scheduler_config = types.SimpleNamespace(
            async_scheduling=async_mode, max_num_batched_tokens=16384, max_num_seqs=512
        )
        instance = scheduler_module.AutoTracingScheduler(
            vllm_config=types.SimpleNamespace(scheduler_config=scheduler_config)
        )
        expected = (
            scheduler_module.TracingAsyncScheduler
            if async_mode
            else scheduler_module.TracingScheduler
        )
        assert isinstance(instance, expected)
        instance._emit("test_event")
        instance.shutdown()
    events = [
        json.loads(line)
        for p in tmp_path.glob("*.jsonl")
        for line in p.read_text().splitlines()
    ]
    assert sum(e["event"] == "test_event" for e in events) == 2


def test_writer_rotates_without_loss(scheduler_module, tmp_path):
    writer = scheduler_module._JsonlTraceWriter(str(tmp_path), max_file_bytes=100)
    for index in range(10):
        writer.emit({"event": "item", "index": index})
    writer.close()
    paths = sorted(tmp_path.glob("*.jsonl"))
    assert len(paths) > 1
    events = [json.loads(line) for p in paths for line in p.read_text().splitlines()]
    assert [e["index"] for e in events] == list(range(10))


def test_run_cleans_up_and_preserves_partial_artifacts(monkeypatch, tmp_path):
    import serve_profiler.deployment as deployment
    import serve_profiler.workload as workload
    from serve_profiler.__main__ import run

    source = tmp_path / "config.yaml"
    source.write_text(yaml.safe_dump(config()))
    state = []

    class FakeDeployment:
        def __init__(self, config, root):
            self.url = "http://localhost/profile"

        def connect(self):
            state.append("connect")

        def start(self):
            state.append("deploy")

        def close(self):
            state.append("cleanup")

    def failed_workload(*args):
        raise RuntimeError("workload failure")

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(deployment, "Deployment", FakeDeployment)
    monkeypatch.setattr(workload, "preflight", lambda *args: None)
    monkeypatch.setattr(workload, "run_phases", failed_workload)
    with pytest.raises(RuntimeError, match="workload failure"):
        run(source)
    assert state == ["connect", "deploy", "cleanup"]
    manifest = json.loads(next((tmp_path / "runs").glob("*/run.json")).read_text())
    assert manifest["status"] == "failed"
    assert manifest["stage"] == "workload"


def test_compat_alias_for_renamed_frontend_args(monkeypatch):
    from serve_profiler import compat

    old = "vllm.entrypoints.launchers.cli_args"
    replacement = types.ModuleType("vllm.entrypoints.common.cli_args")
    replacement.FrontendArgs = object()
    monkeypatch.delitem(sys.modules, old, raising=False)
    monkeypatch.setattr(compat.importlib.util, "find_spec", lambda name: None)
    monkeypatch.setattr(compat.importlib, "import_module", lambda name: replacement)
    # Register a placeholder first so monkeypatch restores the global module
    # table after setup replaces it.
    monkeypatch.setitem(sys.modules, old, replacement)
    compat.setup()
    assert sys.modules[old] is replacement


def test_compat_leaves_available_old_path_untouched(monkeypatch):
    from serve_profiler import compat

    monkeypatch.setattr(compat.importlib.util, "find_spec", lambda name: object())
    monkeypatch.setattr(
        compat.importlib, "import_module", lambda name: pytest.fail("unexpected import")
    )
    compat.setup()
