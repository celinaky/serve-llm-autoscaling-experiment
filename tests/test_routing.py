import os
import subprocess
import sys
import types

import pytest
from pydantic import ValidationError

from conftest import KV_AWARE, REPO, build_config
from serve_llm_autoscaling import backend, runner
from serve_llm_autoscaling.backend import (
    apply_routing_environment,
    check_routing_support,
    verify_topology,
)
from serve_llm_autoscaling.config import KV_AWARE_ROUTER, RoutingConfig

ALL_ENV_VARS = (*backend.ROUTING_ENV_VARS, "RAY_SERVE_ENABLE_HA_PROXY")
BUFFERED = {"routing": {"policy": "kv_aware", "haproxy_request_buffer_bytes": 8_388_608,
                        "haproxy_hard_stop_after_s": 900}}


# --- RoutingConfig ----------------------------------------------------------


def test_default_routing_leaves_the_router_to_serve():
    routing = RoutingConfig()
    assert routing.policy is None and routing.router_class is None
    assert not routing.direct_streaming
    assert routing.environment() == {}
    assert check_routing_support(build_config()) == {
        "policy": None, "request_router_class": None,
    }


def test_kv_aware_enables_direct_streaming_and_body_forwarding():
    routing = RoutingConfig(policy="kv_aware")
    assert routing.direct_streaming
    assert routing.router_class == KV_AWARE_ROUTER == "ray.serve.llm.request_router.KVAwareRouter"
    assert routing.environment() == {
        "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
        "RAY_SERVE_ENABLE_HA_PROXY": "1",
        "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    }
    env = RoutingConfig(**BUFFERED["routing"]).environment()
    assert env["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] == "8388608"
    assert env["RAY_SERVE_HAPROXY_HARD_STOP_AFTER_S"] == "900"


@pytest.mark.parametrize("routing, match", [
    ({"policy": "power_of_two"}, "policy"),
    ({"policy": "consistent_hash"}, "policy"),
    ({"haproxy_request_buffer_bytes": 65536}, "require policy: kv_aware"),
    ({"haproxy_hard_stop_after_s": 600}, "require policy: kv_aware"),
    ({"request_router_kwargs": {"overlap_weight": 2}}, "require policy: kv_aware"),
    ({"policy": "kv_aware", "haproxy_request_buffer_bytes": 512}, "haproxy_request_buffer_bytes"),
    # Both are derived from the policy now.
    ({"policy": "kv_aware", "direct_streaming": True}, "direct_streaming"),
    ({"policy": "kv_aware", "forward_request_body": True}, "forward_request_body"),
])
def test_rejects_invalid_routing(routing, match):
    with pytest.raises(ValidationError, match=match):
        RoutingConfig.model_validate(routing)


# --- Environment ------------------------------------------------------------


def test_apply_routing_environment_sets_and_clears(monkeypatch):
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    for module, _, _ in backend._SNAPSHOTTED_FLAGS:
        monkeypatch.delitem(sys.modules, module, raising=False)
    apply_routing_environment(build_config(deployment=BUFFERED))
    assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
    assert os.environ["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] == "8388608"
    apply_routing_environment(build_config())
    assert not any(name in os.environ for name in backend.ROUTING_ENV_VARS)
    assert os.environ["RAY_SERVE_ENABLE_HA_PROXY"] == "1"  # left to the user


def test_apply_routing_environment_detects_stale_import(monkeypatch):
    for name in ALL_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    stale = types.ModuleType("constants")
    stale.RAY_SERVE_ENABLE_HA_PROXY = False
    stale.RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY = True
    monkeypatch.setitem(sys.modules, "ray.serve._private.constants", stale)
    monkeypatch.delitem(sys.modules, "ray.llm._internal.serve.constants", raising=False)
    with pytest.raises(RuntimeError, match="RAY_SERVE_ENABLE_HA_PROXY"):
        apply_routing_environment(build_config(deployment=KV_AWARE))


# --- Preflight --------------------------------------------------------------


def _fake_dynamo(monkeypatch, version, distribution="ai-dynamo"):
    # Serve is already imported in this process; the ordering is covered by
    # test_preflight_applies_environment_before_first_serve_import.
    monkeypatch.setattr(backend, "apply_routing_environment", lambda config: {})

    real_version = backend.importlib.metadata.version

    def fake_version(name):
        if name not in ("ai-dynamo", "ai-dynamo-runtime"):
            return real_version(name)  # Ray's own imports look packages up too
        if name != distribution or version is None:
            raise backend.importlib.metadata.PackageNotFoundError(name)
        return version

    monkeypatch.setattr(backend.importlib.metadata, "version", fake_version)
    llm = types.ModuleType("dynamo.llm")
    llm.SelectionService = object
    monkeypatch.setitem(sys.modules, "dynamo", types.ModuleType("dynamo"))
    monkeypatch.setitem(sys.modules, "dynamo.llm", llm)
    return llm


@pytest.mark.parametrize("version, match", [
    (None, r"requires ai-dynamo>=1\.4\.0"), ("1.3.2", "found 1.3.2"),
])
def test_preflight_kv_aware_requires_a_recent_ai_dynamo(monkeypatch, version, match):
    _fake_dynamo(monkeypatch, version)
    with pytest.raises(RuntimeError, match=match):
        check_routing_support(build_config(deployment=KV_AWARE))


@pytest.mark.parametrize("distribution", ["ai-dynamo", "ai-dynamo-runtime"])
def test_preflight_accepts_ai_dynamo(monkeypatch, distribution):
    _fake_dynamo(monkeypatch, "1.4.0.post1", distribution)
    assert check_routing_support(build_config(deployment=KV_AWARE)) == {
        "policy": "kv_aware",
        "request_router_class": KV_AWARE_ROUTER,
        "ai_dynamo_version": "1.4.0.post1",
    }


def test_preflight_requires_the_selection_service(monkeypatch):
    del _fake_dynamo(monkeypatch, "1.4.0").SelectionService
    with pytest.raises(RuntimeError, match="SelectionService is not importable"):
        check_routing_support(build_config(deployment=KV_AWARE))


def test_preflight_reports_missing_router(monkeypatch):
    _fake_dynamo(monkeypatch, "1.4.0")
    monkeypatch.setattr(RoutingConfig, "router_class", "ray.serve.nowhere.KVAwareRouter")
    with pytest.raises(RuntimeError, match="does not provide"):
        check_routing_support(build_config(deployment=KV_AWARE))


def test_preflight_applies_environment_before_first_serve_import():
    # A fresh interpreter: in this one, Ray Serve is already imported.
    script = f"""
import os, sys
sys.path.insert(0, {str(REPO / "tests")!r})
for name in {ALL_ENV_VARS!r}:
    os.environ.pop(name, None)
from conftest import KV_AWARE, build_config
from serve_llm_autoscaling import backend
try:
    backend.check_routing_support(build_config(deployment=KV_AWARE))
except RuntimeError:
    pass  # ai-dynamo is optional here; the flags are already snapshotted
from ray.serve._private import build_app, constants
print(build_app.RAY_SERVE_ENABLE_HA_PROXY,
      constants.RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY)
"""
    completed = subprocess.run(
        [sys.executable, "-c", script], capture_output=True, text=True, timeout=120,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")},
    )
    assert completed.returncode == 0, completed.stderr
    assert completed.stdout.split()[-2:] == ["True", "True"]


# --- Topology ---------------------------------------------------------------


def _status(config, *names):
    app = config.deployment.application_name
    return {"applications": {app: {"deployments": {n: {} for n in names}}}}


def test_verify_direct_streaming_topology():
    config = build_config(deployment=KV_AWARE)
    result = verify_topology(config, _status(config, "LLMServer:m", "LLMRouter"))
    assert result == {"topology": "direct_streaming",
                      "deployments": ["LLMRouter", "LLMServer:m"]}
    with pytest.raises(RuntimeError, match="expected LLMRouter"):
        verify_topology(config, _status(config, "LLMServer:m", "OpenAiIngress"))
    with pytest.raises(RuntimeError, match="expected LLMRouter"):
        verify_topology(config, _status(config, "LLMServer:m", "LLMRouter", "OpenAiIngress"))


def test_verify_ingress_topology():
    config = build_config()
    assert verify_topology(config, _status(config, "LLMServer:m", "OpenAiIngress")) == {
        "topology": "openai_ingress", "deployments": ["LLMServer:m", "OpenAiIngress"],
    }
    with pytest.raises(RuntimeError, match="OpenAiIngress topology"):
        verify_topology(config, _status(config, "LLMServer:m", "LLMRouter"))
    with pytest.raises(RuntimeError, match="OpenAiIngress topology"):
        verify_topology(config, {"applications": {}})


def test_topology_failure_fails_run(fake_cluster, experiment, monkeypatch):
    import json

    config, path = experiment(deployment=KV_AWARE)
    # Serve came up with the plain ingress instead of the router.
    monkeypatch.setattr(runner.RayServeBackend, "wait_healthy", lambda self: {
        "serve_status": _status(config, "LLMServer:m", "OpenAiIngress")})
    monkeypatch.setattr(runner.AIPerfRunner, "run", lambda self: pytest.fail("ran AIPerf"))
    with pytest.raises(RuntimeError, match="expected LLMRouter"):
        runner.run_experiment(config, path)
    (root,) = config.runtime.results_dir.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["failure_stage"] == "topology"
    assert fake_cluster.calls == ["deploy", "teardown"]
    routing = json.loads((root / "deployment" / "routing.json").read_text())
    assert routing["policy"] == "kv_aware" and "verified" not in routing
