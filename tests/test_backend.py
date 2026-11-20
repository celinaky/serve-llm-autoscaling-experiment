import os
from pathlib import Path

import pytest

import serve_llm_autoscaling
from conftest import EXPERIMENTS, KV_AWARE, REPO, TOKEN_POLICY, build_config
from serve_llm_autoscaling import backend
from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import AUTOSCALING_POLICIES, KV_AWARE_ROUTER, load_config

PACKAGE = Path(serve_llm_autoscaling.__file__).resolve().parent
DEFAULT_FUNCTION, TOKEN_FUNCTION = (AUTOSCALING_POLICIES[name][0] for name in ("default", "token"))


def _spec(**sections):
    return RayServeBackend(build_config(**sections)).deployment_spec()


# --- Deployment spec --------------------------------------------------------


def test_deployment_spec_forwards_only_set_keys(monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    spec = _spec(deployment={"engine": {}, "autoscaling": {}})
    assert spec == {
        "model_loading_config": {"model_id": "long-context-model",
                                 "model_source": "long-context-model"},
        # Unset keys fall back to vLLM's, Serve's and Ray Serve LLM's defaults.
        "engine_kwargs": {"enable_prompt_tokens_details": True},
        "deployment_config": {"autoscaling_config": {"policy": {
            "policy_function": DEFAULT_FUNCTION, "policy_kwargs": {},
        }}},
        # The harness package and its worker setup hook are always shipped.
        "runtime_env": {
            "py_modules": [str(PACKAGE)],
            "worker_process_setup_hook": "serve_llm_autoscaling.vllm_compat.setup",
            "env_vars": {"PYTHONPATH": str(PACKAGE.parent)},
        },
    }


def test_deployment_spec_forwards_what_the_experiment_sets():
    spec = _spec(deployment={
        "model_source": "s3://bucket/model",
        "accelerator_type": "B200",
        "engine": {"max_num_seqs": 64, "kv_cache_dtype": "fp8", "language_model_only": True},
        "autoscaling": {"max_replicas": 4, "target_ongoing_requests": 8,
                        "look_back_period_s": 60},
        "max_ongoing_requests": 128,
        "experimental_configs": {"stream_batching_interval_ms": 0},
    })
    assert spec["model_loading_config"]["model_source"] == "s3://bucket/model"
    assert spec["accelerator_type"] == "B200"
    assert spec["engine_kwargs"] == {
        "max_num_seqs": 64, "kv_cache_dtype": "fp8", "language_model_only": True,
        "enable_prompt_tokens_details": True,
    }
    autoscaling = spec["deployment_config"]["autoscaling_config"]
    assert {k: v for k, v in autoscaling.items() if k != "policy"} == {
        "max_replicas": 4, "target_ongoing_requests": 8, "look_back_period_s": 60,
    }
    assert spec["deployment_config"]["max_ongoing_requests"] == 128
    assert spec["experimental_configs"] == {"stream_batching_interval_ms": 0}
    # No tracing requested: the scheduler is vLLM's own and nothing is traced.
    assert "scheduler_cls" not in spec["engine_kwargs"]
    assert backend.TRACE_DIR_ENV not in spec["runtime_env"]["env_vars"]


def test_native_offload_capacity_is_forwarded_per_replica():
    spec = _spec(deployment={
        "engine": {"tensor_parallel_size": 2, "enable_prefix_caching": True,
                   "kv_offloading_backend": "native", "kv_offloading_size": 64},
    })
    # Do not divide/multiply by TP or replica count; vLLM owns the split.
    assert spec["engine_kwargs"]["kv_offloading_size"] == 64
    assert spec["engine_kwargs"]["kv_offloading_backend"] == "native"


def test_deployment_spec_names_the_router_and_policy():
    spec = _spec(deployment={
        **TOKEN_POLICY,
        "routing": {"policy": "kv_aware", "request_router_kwargs": {"overlap_weight": 2}},
    })
    config = spec["deployment_config"]
    assert config["request_router_config"] == {
        "request_router_class": KV_AWARE_ROUTER,
        "request_router_kwargs": {"overlap_weight": 2},
    }
    # The token policy is application-scope: application_config() attaches it.
    assert config["autoscaling_config"] == {"max_replicas": 4}


def test_checked_in_experiment_is_a_valid_serve_autoscaling_config():
    from ray.serve.config import AutoscalingConfig, AutoscalingPolicy
    from ray.serve.schema import ServeDeploySchema

    config = load_config(REPO / "experiments/b200_token_64_sessions.yaml")
    serve = RayServeBackend(config)
    spec = serve.deployment_spec()
    autoscaling = spec["deployment_config"]["autoscaling_config"]
    assert "policy" not in autoscaling
    serve_config = AutoscalingConfig(**autoscaling)
    assert (serve_config.min_replicas, serve_config.max_replicas) == (1, 8)
    assert serve_config.look_back_period_s == 60
    # Left unset, the deployment keeps Serve's default policy under the app's.
    assert serve_config.policy.policy_function == DEFAULT_FUNCTION

    app = serve.application_config(spec, config.deployment.routing.environment())
    assert app["autoscaling_policy"]["policy_function"] == TOKEN_FUNCTION
    assert app["autoscaling_policy"]["policy_kwargs"] == {
        "target_prefill_tokens_per_replica": 20000, "post_downscale_cooldown_s": 150,
    }
    AutoscalingPolicy(**app["autoscaling_policy"])
    (parsed,) = ServeDeploySchema(applications=[app]).applications
    assert parsed.name == config.deployment.application_name


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda p: p.name)
def test_checked_in_policy_kwargs_are_the_installed_policys(path):
    """Every experiment uses only kwargs this Ray build's policy takes."""
    config = load_config(path)
    policy = config.deployment.autoscaling.policy
    allowed = {"token": {"target_prefill_tokens_per_replica", "post_downscale_cooldown_s"},
               "default": set()}[policy.name]
    assert set(policy.kwargs) <= allowed
    assert backend.check_autoscaling_support(config)["policy"] == policy.name


# --- Harness package --------------------------------------------------------


@pytest.mark.parametrize("deployment", [{}, KV_AWARE, TOKEN_POLICY], ids=["default", "kv", "token"])
def test_harness_package_is_always_shipped(deployment, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    config = build_config(deployment=deployment)
    assert config.analysis.scheduler_trace_dir is None
    assert RayServeBackend(config).deployment_spec()["runtime_env"] == {
        "py_modules": [str(PACKAGE)],
        "worker_process_setup_hook": backend.COMPAT_SETUP_HOOK,
        "env_vars": {"PYTHONPATH": str(PACKAGE.parent)},
    }


# --- Tracing scheduler ------------------------------------------------------


def test_tracing_scheduler_is_injected_with_the_trace_directory(tmp_path, monkeypatch):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    config = build_config(deployment=TOKEN_POLICY)
    config.analysis.scheduler_trace_dir = tmp_path / "traces"
    spec = RayServeBackend(config).deployment_spec()
    assert spec["engine_kwargs"]["scheduler_cls"] == (
        "serve_llm_autoscaling.scheduler_trace.AutoTracingScheduler"
    )
    assert spec["runtime_env"] == {
        # The harness package itself is shipped to the replicas.
        "py_modules": [str(PACKAGE)],
        "worker_process_setup_hook": "serve_llm_autoscaling.vllm_compat.setup",
        "env_vars": {
            "AUTOSCALE_HARNESS_TRACE_DIR": str((tmp_path / "traces").resolve()),
            "PYTHONPATH": str(PACKAGE.parent),
        },
    }
    # Tracing adds only the scheduler and its output directory.
    untraced = build_config(deployment=TOKEN_POLICY)
    expected = RayServeBackend(untraced).deployment_spec()
    del spec["runtime_env"]["env_vars"]["AUTOSCALE_HARNESS_TRACE_DIR"]
    del spec["engine_kwargs"]["scheduler_cls"]
    assert spec == expected


def test_shipped_package_keeps_the_drivers_pythonpath(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHONPATH", "/opt/site")
    config = build_config()
    expected = f"{PACKAGE.parent}{os.pathsep}/opt/site"
    env = RayServeBackend(config).deployment_spec()["runtime_env"]["env_vars"]
    assert env == {"PYTHONPATH": expected}
    config.analysis.scheduler_trace_dir = tmp_path
    env = RayServeBackend(config).deployment_spec()["runtime_env"]["env_vars"]
    assert env["PYTHONPATH"] == expected


def test_tracing_rejects_replacing_a_custom_scheduler(tmp_path):
    config = build_config(deployment={"engine": {"scheduler_cls": "custom.Scheduler"}})
    assert RayServeBackend(config).deployment_spec()["engine_kwargs"]["scheduler_cls"] == (
        "custom.Scheduler"
    )
    config.analysis.scheduler_trace_dir = tmp_path
    with pytest.raises(ValueError, match="engine.scheduler_cls is already set"):
        RayServeBackend(config).deployment_spec()
    # Naming the harness's own scheduler is not a conflict.
    config = build_config(deployment={"engine": {"scheduler_cls": backend.TRACING_SCHEDULER}})
    config.analysis.scheduler_trace_dir = tmp_path
    assert RayServeBackend(config).deployment_spec()["runtime_env"]["py_modules"] == [str(PACKAGE)]


def test_injected_names_resolve_in_the_package():
    # scheduler_trace itself imports vLLM, so only its presence is checked.
    assert (PACKAGE / "scheduler_trace.py").read_text().count("class AutoTracingScheduler") == 1
    module, _, attribute = backend.COMPAT_SETUP_HOOK.rpartition(".")
    assert callable(getattr(__import__(module, fromlist=[attribute]), attribute))
    assert backend.TRACING_SCHEDULER.rpartition(".")[0] == "serve_llm_autoscaling.scheduler_trace"


# --- Readiness --------------------------------------------------------------


class _Response:
    def __init__(self, ids):
        self.ids = ids

    def raise_for_status(self):
        if self.ids is None:
            raise ConnectionError("refused")

    def json(self):
        return {"data": [{"id": i} for i in self.ids]}


def _readiness(monkeypatch, responses, status=None, **runtime):
    config = build_config(runtime={"readiness_poll_s": 0.001, **runtime})
    urls = []

    def get(url, timeout):
        urls.append(url)
        return _Response(responses.pop(0) if len(responses) > 1 else responses[0])

    monkeypatch.setattr(backend.requests, "get", get)
    monkeypatch.setattr(RayServeBackend, "status",
                        status or (lambda self: {"applications": {}}))
    return RayServeBackend(config), urls


def test_wait_healthy_polls_until_the_model_is_registered(monkeypatch):
    serve, urls = _readiness(monkeypatch, [None, ["other"], ["long-context-model"]],
                             endpoint_url="http://head:8000/")
    ready = serve.wait_healthy()
    assert ready == {"serve_status": {"applications": {}},
                     "models": {"data": [{"id": "long-context-model"}]}}
    assert urls == ["http://head:8000/v1/models"] * 3


def test_wait_healthy_times_out_with_the_last_observation(monkeypatch):
    serve, _ = _readiness(monkeypatch, [["other"]], readiness_timeout_s=0.02)
    with pytest.raises(TimeoutError, match=r"last observation: registered models were \['other'\]"):
        serve.wait_healthy()


@pytest.mark.parametrize("state", ["DEPLOY_FAILED", "ApplicationStatus.DEPLOY_FAILED"])
def test_wait_healthy_fails_fast_when_the_deploy_failed(monkeypatch, state):
    app = build_config().deployment.application_name
    failed = {"applications": {app: {"status": state, "message": "policy import failed"}}}
    # A long timeout: the failure must not wait it out.
    serve, urls = _readiness(monkeypatch, [["long-context-model"]], status=lambda self: failed,
                             readiness_timeout_s=600)
    with pytest.raises(RuntimeError, match="failed to deploy the application: "
                                           "policy import failed"):
        serve.wait_healthy()
    assert urls == []  # the endpoint is not probed once the deploy has failed


def test_wait_healthy_ignores_other_applications_and_transient_states(monkeypatch):
    app = build_config().deployment.application_name
    statuses = [
        {"applications": {"other-app": {"status": "DEPLOY_FAILED", "message": "theirs"}}},
        {"applications": {app: {"status": "DEPLOYING", "message": ""}}},
        {"applications": {app: {"status": "RUNNING", "message": ""}}},
    ]

    def status(self):
        return statuses.pop(0) if len(statuses) > 1 else statuses[0]

    serve, urls = _readiness(monkeypatch, [None, None, ["long-context-model"]], status=status)
    assert serve.wait_healthy()["serve_status"]["applications"][app]["status"] == "RUNNING"
    assert len(urls) == 3


def test_wait_healthy_tolerates_a_failing_status_call(monkeypatch):
    calls = []

    def status(self):
        calls.append(1)
        if len(calls) == 1:
            raise ConnectionError("controller starting")
        return {"applications": {}}

    serve, urls = _readiness(monkeypatch, [["long-context-model"]], status=status)
    assert serve.wait_healthy()["models"] == {"data": [{"id": "long-context-model"}]}
    assert len(calls) == 2 and len(urls) == 1


def test_routing_summary_records_switches():
    summary = RayServeBackend(build_config(deployment=KV_AWARE)).routing_summary()
    assert summary["policy"] == "kv_aware"
    assert summary["request_router_class"] == KV_AWARE_ROUTER
    assert summary["environment"]["RAY_SERVE_ENABLE_HA_PROXY"] == "1"
    assert RayServeBackend(build_config()).routing_summary()["environment"] == {}
