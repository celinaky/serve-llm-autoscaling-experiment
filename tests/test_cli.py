import sys
import types

import pytest
import yaml

from conftest import raw_config
from serve_llm_autoscaling import cli, vllm_compat


@pytest.fixture
def config_path(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_config()))
    return path


@pytest.fixture
def no_setup(monkeypatch):
    """Skip the vLLM import alias the cluster commands install first."""
    calls = []
    monkeypatch.setattr(vllm_compat, "setup", lambda: calls.append("setup"))
    return calls


def test_validate_prints_the_resolved_config(config_path, capsys, no_setup):
    assert cli.main(["validate", str(config_path)]) == 0
    resolved = yaml.safe_load(capsys.readouterr().out)
    assert resolved["benchmark"]["workload"]["seed"] == 42
    assert resolved["deployment"]["autoscaling"]["policy"]["policy_function"]
    assert no_setup == []  # validation needs neither Ray nor vLLM


def test_invalid_config_is_reported_not_raised(config_path, capsys):
    config_path.write_text(yaml.safe_dump(raw_config(mode="concurrency")))
    assert cli.main(["validate", str(config_path)]) == 1
    assert "ERROR: ValidationError" in capsys.readouterr().err


@pytest.mark.parametrize("argv", [["benchmark", "x.yaml", "runs/x"], ["run"], []])
def test_rejects_removed_and_incomplete_commands(argv, capsys):
    with pytest.raises(SystemExit) as exit_info:
        cli.main(argv)
    assert exit_info.value.code == 2


@pytest.mark.parametrize("flags, expected", [
    ([], {"replace": False, "keep_deployment": None, "warm": False}),
    (["--warm", "--replace", "--keep-deployment"],
     {"replace": True, "keep_deployment": True, "warm": True}),
])
def test_run_passes_its_flags(config_path, monkeypatch, capsys, no_setup, flags, expected):
    seen = {}

    def run_experiment(config, path, **kwargs):
        seen.update(kwargs, path=path, name=config.name)
        return "runs/x"

    monkeypatch.setattr(cli, "run_experiment", run_experiment)
    assert cli.main(["run", str(config_path), *flags]) == 0
    assert seen == {**expected, "path": config_path, "name": "agentx-ramp"}
    assert "Run artifacts: runs/x" in capsys.readouterr().out
    assert no_setup == ["setup"]


def test_warm_command(config_path, monkeypatch, capsys, no_setup):
    seen = {}
    monkeypatch.setattr(cli, "warm_compile_caches",
                        lambda config, **kwargs: seen.update(kwargs) or {"replicas": 2})
    assert cli.main(["warm", str(config_path), "--replicas", "2"]) == 0
    assert seen == {"replicas": 2, "replace": False}
    assert '"replicas": 2' in capsys.readouterr().out
    assert no_setup == ["setup"]


def test_run_failure_exits_nonzero(config_path, monkeypatch, capsys, no_setup):
    def fail(*args, **kwargs):
        raise RuntimeError("Serve already has application(s)")

    monkeypatch.setattr(cli, "run_experiment", fail)
    assert cli.main(["run", str(config_path)]) == 1
    assert "ERROR: RuntimeError: Serve already has application(s)" in capsys.readouterr().err


# --- vLLM compatibility hook -------------------------------------------------

OLD = "vllm.entrypoints.launchers.cli_args"
NEW = "vllm.entrypoints.common.cli_args"


def _fake_vllm(monkeypatch, *, old_exists, frontend_args=True):
    new = types.ModuleType(NEW)
    if frontend_args:
        new.FrontendArgs = object
    monkeypatch.setattr(vllm_compat.importlib.util, "find_spec",
                        lambda name: object() if old_exists else None)
    monkeypatch.setattr(vllm_compat.importlib, "import_module", lambda name: {NEW: new}[name])
    monkeypatch.setitem(sys.modules, OLD, None)  # restored after the test
    del sys.modules[OLD]
    return new


def test_setup_aliases_the_moved_frontend_args(monkeypatch):
    monkeypatch.setenv("RAY_RUNTIME_ENV_HOOK", "workspace.hook")
    new = _fake_vllm(monkeypatch, old_exists=False)
    vllm_compat.setup()
    assert sys.modules[OLD] is new
    # Nested Ray drivers must not upload the whole workspace.
    assert "RAY_RUNTIME_ENV_HOOK" not in vllm_compat.os.environ


def test_setup_leaves_an_older_vllm_alone(monkeypatch):
    _fake_vllm(monkeypatch, old_exists=True)
    vllm_compat.setup()
    assert OLD not in sys.modules


def test_setup_rejects_an_unsupported_vllm(monkeypatch):
    _fake_vllm(monkeypatch, old_exists=False, frontend_args=False)
    with pytest.raises(RuntimeError, match="FrontendArgs is unavailable"):
        vllm_compat.setup()
