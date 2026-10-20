import importlib.metadata
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from serve_llm_autoscaling import backend
from serve_llm_autoscaling.artifacts import format_run_summary
from serve_llm_autoscaling.backend import (
    RayServeBackend,
    autoscaling_policy,
    check_autoscaling_support,
    verify_autoscaling_policy,
)
from serve_llm_autoscaling.config import (
    AUTOSCALING_POLICIES,
    AutoscalingPolicyConfig,
    ExperimentConfig,
    load_config,
)

SLA_KWARGS = {"ttft_ms": 300, "itl_ms": 20}
PROMETHEUS = "http://prometheus:9090"


def _config(policy=None) -> ExperimentConfig:
    autoscaling = {"max_replicas": 4}
    if policy is not None:
        autoscaling["policy"] = policy
    return ExperimentConfig.model_validate({
        "name": "autoscaling",
        "deployment": {"model_id": "m", "autoscaling": autoscaling},
    })


def _sla(**kwargs) -> ExperimentConfig:
    return _config({"name": "sla", "kwargs": {**SLA_KWARGS, **kwargs}})


def _details(config, policy, name="LLMServer:m"):
    app = config.deployment.application_name
    deployment = {"deployment_config": {"autoscaling_config": {"policy": policy}}}
    return {"applications": {app: {"deployments": {name: deployment}}}}


@pytest.fixture(autouse=True)
def _no_prometheus_env(monkeypatch):
    monkeypatch.delenv("RAY_PROMETHEUS_HOST", raising=False)


# --- AutoscalingPolicyConfig ------------------------------------------------


def test_default_policy_is_serves_default():
    policy = _config().deployment.autoscaling.policy
    assert (policy.name, policy.kwargs) == ("default", {})
    assert policy.policy_function == "ray.serve.autoscaling_policy:default_autoscaling_policy"


def test_sla_policy_scales_by_default():
    policy = _sla().deployment.autoscaling.policy
    assert policy.policy_function == AUTOSCALING_POLICIES["sla"]
    assert policy.kwargs == {**SLA_KWARGS, "advisory": False}
    assert _sla(advisory=True).deployment.autoscaling.policy.kwargs["advisory"] is True


@pytest.mark.parametrize("policy, match", [
    ({"name": "default", "kwargs": {"ttft_ms": 1}}, "default takes no kwargs"),
    ({"name": "sla", "kwargs": {"ttft_ms": 300}}, r"requires kwargs \['itl_ms'\]"),
    ({"name": "sla"}, r"requires kwargs \['itl_ms', 'ttft_ms'\]"),
    ({"name": "queue"}, "default"),
    ({"name": "sla", "policy_function": "x:y"}, "Extra inputs"),
])
def test_rejects_invalid_policy(policy, match):
    with pytest.raises(ValidationError, match=match):
        _config(policy)


def test_resolved_policy_round_trips():
    config = _sla()
    assert ExperimentConfig.model_validate(config.resolved_dict()) == config


def test_agentx_experiment_uses_default_policy():
    config = load_config("experiments/agentx_kv_aware_autoscaling.yaml")
    assert config.deployment.autoscaling.policy == AutoscalingPolicyConfig()


# --- Deployment spec --------------------------------------------------------


def test_deployment_spec_passes_policy_explicitly():
    autoscaling = RayServeBackend(_config()).deployment_spec()["deployment_config"][
        "autoscaling_config"
    ]
    assert autoscaling["policy"] == {
        "policy_function": AUTOSCALING_POLICIES["default"], "policy_kwargs": {},
    }
    assert autoscaling["max_replicas"] == 4


def test_deployment_spec_is_a_valid_serve_autoscaling_config():
    from ray.serve.config import AutoscalingConfig

    spec = RayServeBackend(_sla()).deployment_spec()
    serve_config = AutoscalingConfig(**spec["deployment_config"]["autoscaling_config"])
    assert serve_config.policy.policy_function == AUTOSCALING_POLICIES["sla"]
    assert serve_config.policy.policy_kwargs == {**SLA_KWARGS, "advisory": False}


def test_sla_policy_uses_driver_prometheus(monkeypatch):
    monkeypatch.setenv("RAY_PROMETHEUS_HOST", PROMETHEUS)
    assert autoscaling_policy(_sla())["policy_kwargs"]["prometheus_address"] == PROMETHEUS
    explicit = _sla(prometheus_address="http://other:9090")
    assert autoscaling_policy(explicit)["policy_kwargs"]["prometheus_address"] == (
        "http://other:9090"
    )
    assert autoscaling_policy(_config())["policy_kwargs"] == {}


# --- Preflight --------------------------------------------------------------


class _Response:
    def __init__(self, error=None):
        self._error = error

    def raise_for_status(self):
        if self._error:
            raise self._error


def _fake_sla_environment(monkeypatch, sklearn="1.6.1", response=None):
    real_version = importlib.metadata.version

    def version(name):
        if name == "scikit-learn":
            if sklearn is None:
                raise importlib.metadata.PackageNotFoundError(name)
            return sklearn
        return real_version(name)

    queries = []

    def get(url, params, timeout):
        queries.append((url, params))
        return response or _Response()

    monkeypatch.setattr(backend.importlib.metadata, "version", version)
    monkeypatch.setattr(backend.requests, "get", get)
    return queries


def test_preflight_default_policy():
    assert check_autoscaling_support(_config()) == {
        "policy": "default",
        "policy_function": AUTOSCALING_POLICIES["default"],
        "policy_kwargs": {},
    }


def test_preflight_sla_policy(monkeypatch):
    monkeypatch.setenv("RAY_PROMETHEUS_HOST", PROMETHEUS)
    queries = _fake_sla_environment(monkeypatch)
    assert check_autoscaling_support(_sla()) == {
        "policy": "sla",
        "policy_function": AUTOSCALING_POLICIES["sla"],
        "policy_kwargs": {**SLA_KWARGS, "advisory": False, "prometheus_address": PROMETHEUS},
        "scikit_learn_version": "1.6.1",
    }
    assert queries == [(f"{PROMETHEUS}/api/v1/query", {"query": "1"})]


@pytest.mark.parametrize("env, sklearn, response, match", [
    (PROMETHEUS, None, None, "requires scikit-learn"),
    (None, "1.6.1", None, "needs Prometheus"),
    (PROMETHEUS, "1.6.1", _Response(RuntimeError("503")), "cannot query Prometheus"),
])
def test_preflight_sla_rejects_environment(monkeypatch, env, sklearn, response, match):
    if env:
        monkeypatch.setenv("RAY_PROMETHEUS_HOST", env)
    _fake_sla_environment(monkeypatch, sklearn, response)
    with pytest.raises(RuntimeError, match=match):
        check_autoscaling_support(_sla())


def test_preflight_reports_missing_policy(monkeypatch):
    monkeypatch.setitem(AUTOSCALING_POLICIES, "default", "ray.serve.nowhere:policy")
    with pytest.raises(RuntimeError, match="does not provide"):
        check_autoscaling_support(_config())


# --- Verification on the deployment -----------------------------------------


def test_verify_applied_policy():
    config = _sla()
    applied = {**autoscaling_policy(config), "_serialized_policy_def": "..."}
    assert verify_autoscaling_policy(config, _details(config, applied)) == {
        "verified_deployments": ["LLMServer:m"],
    }


def test_verify_rejects_other_policy():
    config = _sla()
    with pytest.raises(RuntimeError, match="LLMServer:m runs autoscaling policy"):
        verify_autoscaling_policy(config, _details(config, autoscaling_policy(_config())))
    advisory = autoscaling_policy(_sla(advisory=True))
    with pytest.raises(RuntimeError, match="'advisory': True"):
        verify_autoscaling_policy(config, _details(config, advisory))


def test_verify_requires_llm_server():
    config = _config()
    with pytest.raises(RuntimeError, match="no LLMServer deployment"):
        verify_autoscaling_policy(config, _details(config, {}, name="LLMRouter"))


# --- Run records ------------------------------------------------------------


def _run(monkeypatch, tmp_path, config, applied):
    from serve_llm_autoscaling import runner

    config.runtime.results_dir = tmp_path / "runs"
    monkeypatch.setattr(runner, "environment_check", lambda connect: {})
    monkeypatch.setattr(runner, "check_routing_support", lambda c: {"policy": "p"})
    monkeypatch.setattr(runner, "check_autoscaling_support",
                        lambda c: {"policy": c.deployment.autoscaling.policy.name,
                                   **autoscaling_policy(c)})
    monkeypatch.setattr(runner.RayServeBackend, "deploy", lambda self: {})
    monkeypatch.setattr(runner.RayServeBackend, "teardown", lambda self: None)
    app = config.deployment.application_name
    status = {"applications": {app: {"deployments": {"LLMServer:m": {},
                                                      "OpenAiIngress": {}}}}}
    monkeypatch.setattr(runner.RayServeBackend, "wait_healthy",
                        lambda self: {"serve_status": status})
    monkeypatch.setattr(runner.RayServeBackend, "serve_details",
                        lambda self: _details(config, applied))
    monkeypatch.setattr(runner, "run_benchmarks", lambda *args: [{"error": None}])
    root = None
    try:
        root = runner.run_experiment(config, Path("experiments/baseline.yaml"))
    finally:
        (root,) = (tmp_path / "runs").iterdir()
    return root


def test_run_records_verified_policy(monkeypatch, tmp_path: Path):
    config = _sla()
    root = _run(monkeypatch, tmp_path, config, autoscaling_policy(config))
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["autoscaling"] == {
        "policy": "sla",
        **autoscaling_policy(config),
        "verified_deployments": ["LLMServer:m"],
    }
    recorded = json.loads((root / "deployment" / "autoscaling.json").read_text())
    assert recorded == manifest["autoscaling"]
    assert "Autoscale: sla (advisory=False)\n" in format_run_summary(root)


def test_policy_mismatch_fails_run(monkeypatch, tmp_path: Path):
    config = _sla()
    with pytest.raises(RuntimeError, match="runs autoscaling policy"):
        _run(monkeypatch, tmp_path, config, autoscaling_policy(_config()))
    (root,) = (tmp_path / "runs").iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["failure_stage"] == "autoscaling_policy"
    assert "verified_deployments" not in manifest["autoscaling"]
    assert "Autoscale: sla (advisory=False) (not verified on the deployment)" in (
        format_run_summary(root)
    )
