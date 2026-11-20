import json

import pytest
from pydantic import ValidationError

from conftest import KV_AWARE, TOKEN_POLICY, build_config
from serve_llm_autoscaling import runner
from serve_llm_autoscaling.artifacts import format_run_summary
from serve_llm_autoscaling.backend import (
    RayServeBackend,
    autoscaling_policy,
    check_autoscaling_support,
    verify_autoscaling_policy,
)
from serve_llm_autoscaling.config import AUTOSCALING_POLICIES, ExperimentConfig

DEFAULT_FUNCTION, TOKEN_FUNCTION = (AUTOSCALING_POLICIES[name][0] for name in ("default", "token"))
TOKEN_KWARGS = {"target_prefill_tokens_per_replica": 20000}
CUSTOM = "my_policies.scaling:prefill_policy"


def _config(policy=None, **deployment) -> ExperimentConfig:
    autoscaling = {"max_replicas": 4}
    if policy is not None:
        autoscaling["policy"] = policy
    return build_config(deployment={"autoscaling": autoscaling, **deployment})


def _token() -> ExperimentConfig:
    return build_config(deployment=TOKEN_POLICY)


def _details(config, policy=None, name="LLMServer:m", app_policy=None):
    """Serve details with ``policy`` applied on the deployment, ``app_policy`` on the app."""
    app = config.deployment.application_name
    autoscaling = {} if policy is None else {"policy": policy}
    deployment = {"deployment_config": {"autoscaling_config": autoscaling}}
    app_config = {"name": app}
    if app_policy is not None:
        app_config["autoscaling_policy"] = app_policy
    return {"applications": {app: {"deployed_app_config": app_config,
                                   "deployments": {name: deployment}}}}


# --- AutoscalingPolicyConfig ------------------------------------------------


def test_default_policy_is_serves_default():
    policy = _config().deployment.autoscaling.policy
    assert (policy.name, policy.kwargs) == ("default", {})
    assert policy.policy_function == "ray.serve.autoscaling_policy:default_autoscaling_policy"
    assert policy.scope == "deployment"


def test_built_in_policies_name_a_function_and_its_scope():
    assert AUTOSCALING_POLICIES == {
        "default": ("ray.serve.autoscaling_policy:default_autoscaling_policy", "deployment"),
        "token": ("ray.llm._internal.serve.routing_policies.kv_aware.autoscaling_policy"
                  ":LLMTokenAutoscalingPolicy", "application"),
    }


def test_token_policy_passes_kwargs_through():
    kwargs = {**TOKEN_KWARGS, "post_downscale_cooldown_s": 150}
    policy = _config({"name": "token", "kwargs": kwargs}, **KV_AWARE).deployment.autoscaling.policy
    assert policy.policy_function == TOKEN_FUNCTION
    assert policy.scope == "application"
    assert policy.kwargs == kwargs


def test_scope_defaults_to_the_built_ins_and_may_be_restated():
    for name, scope in (("default", "deployment"), ("token", "application")):
        policy = {"name": name, "scope": scope}
        if name == "token":
            policy["kwargs"] = TOKEN_KWARGS
        assert _config(policy, **KV_AWARE).deployment.autoscaling.policy.scope == scope


def test_custom_policy_scope_defaults_to_deployment_and_may_be_application():
    custom = {"name": "mine", "policy_function": CUSTOM}
    assert _config(custom).deployment.autoscaling.policy.scope == "deployment"
    wide = _config({**custom, "scope": "application"}).deployment.autoscaling.policy
    assert (wide.policy_function, wide.scope) == (CUSTOM, "application")
    # A built-in name labelling a replacement does not impose the built-in's scope.
    relabeled = _config({"name": "token", "policy_function": CUSTOM}, **KV_AWARE)
    assert relabeled.deployment.autoscaling.policy.scope == "deployment"
    explicit = _config({"name": "default", "policy_function": CUSTOM, "scope": "application"})
    assert explicit.deployment.autoscaling.policy.scope == "application"


@pytest.mark.parametrize("policy, match", [
    ({"name": "default", "kwargs": {"ttft_ms": 1}}, "default takes no kwargs"),
    ({"name": "token"}, r"token requires kwargs \['target_prefill_tokens_per_replica'\]"),
    ({"name": "token", "kwargs": {"post_downscale_cooldown_s": 150}}, "token requires kwargs"),
    ({"name": "sla", "kwargs": {"ttft_ms": 300, "itl_ms": 20}}, "unknown autoscaling policy 'sla'"),
    ({"name": "queue"}, r"use one of \['default', 'token'\] or set policy_function"),
    ({"name": "mine", "policy_function": "my_policies.scaling.policy"}, "module:attribute"),
    # Naming Serve's default function does not make it accept kwargs.
    ({"name": "default", "policy_function": DEFAULT_FUNCTION, "kwargs": {"a": 1}},
     "takes no kwargs"),
    # A built-in policy runs at its own scope only.
    ({"name": "default", "scope": "application"}, "default is a deployment-scope policy"),
    ({"name": "token", "scope": "deployment", "kwargs": TOKEN_KWARGS},
     "token is a application-scope policy"),
    ({"name": "token", "policy_function": TOKEN_FUNCTION, "scope": "deployment",
      "kwargs": TOKEN_KWARGS}, "application-scope policy"),
    ({"name": "default", "scope": "cluster"}, "'deployment' or 'application'"),
    ({"name": "default", "advisory": True}, "Extra inputs"),
])
def test_rejects_invalid_policy(policy, match):
    with pytest.raises(ValidationError, match=match):
        _config(policy, **KV_AWARE)


def test_token_policy_requires_kv_aware_routing():
    with pytest.raises(ValidationError, match="token requires routing policy kv_aware"):
        _config({"name": "token", "kwargs": TOKEN_KWARGS})


def test_policy_function_overrides_the_built_in_policies():
    """A new policy is evaluated by import path; ``name`` is then only a label."""
    custom = _config({"name": "prefill-v2", "policy_function": CUSTOM,
                      "kwargs": {"target": 3}})
    assert autoscaling_policy(custom) == {
        "policy_function": CUSTOM, "policy_kwargs": {"target": 3},
    }
    # A built-in name can label a replacement, whose kwargs are its own.
    replaced = _config({"name": "token", "policy_function": CUSTOM}, **KV_AWARE)
    assert replaced.deployment.autoscaling.policy.kwargs == {}
    relabeled = _config({"name": "default", "policy_function": CUSTOM, "kwargs": {"target": 3}})
    assert relabeled.deployment.autoscaling.policy.policy_function == CUSTOM


@pytest.mark.parametrize("config", [
    _config(), _token(), _config({"name": "mine", "policy_function": CUSTOM}),
    _config({"name": "mine", "policy_function": CUSTOM, "scope": "application"}),
])
def test_resolved_policy_round_trips(config):
    resolved = config.resolved_dict()
    policy = config.deployment.autoscaling.policy
    # The resolved config names the function and scope that decided the run.
    assert resolved["deployment"]["autoscaling"]["policy"]["scope"] == policy.scope
    assert ExperimentConfig.model_validate(resolved) == config


# --- Deployment spec --------------------------------------------------------


def test_deployment_spec_passes_policy_explicitly():
    autoscaling = RayServeBackend(_config()).deployment_spec()["deployment_config"][
        "autoscaling_config"
    ]
    assert autoscaling == {
        "max_replicas": 4,
        "policy": {"policy_function": DEFAULT_FUNCTION, "policy_kwargs": {}},
    }
    assert autoscaling_policy(_token()) == {
        "policy_function": TOKEN_FUNCTION, "policy_kwargs": TOKEN_KWARGS,
    }
    # A copy: the deployed kwargs cannot alias the config's.
    config = _token()
    assert autoscaling_policy(config)["policy_kwargs"] is not (
        config.deployment.autoscaling.policy.kwargs)


def _app(config, env=None):
    backend = RayServeBackend(config)
    spec = backend.deployment_spec()
    return backend.application_config(spec, env or {}), spec


def test_application_config_builds_the_openai_app_from_the_spec():
    config = _config()
    app, spec = _app(config)
    assert app == {
        "name": config.deployment.application_name,
        "route_prefix": config.deployment.route_prefix,
        "import_path": "ray.serve.llm:build_openai_app",
        "args": {"llm_configs": [spec]},
    }
    # The topology switches reach the task that builds the app.
    env = {"RAY_SERVE_ENABLE_HA_PROXY": "1"}
    assert _app(config, env)[0]["runtime_env"] == {"env_vars": env}


def test_deployment_scope_policy_is_set_on_the_deployment_only():
    for config in (_config(), _config({"name": "mine", "policy_function": CUSTOM})):
        app, spec = _app(config)
        assert "autoscaling_policy" not in app
        assert spec["deployment_config"]["autoscaling_config"]["policy"] == (
            autoscaling_policy(config))


def test_application_scope_policy_is_set_on_the_application_only():
    custom = _config({"name": "mine", "policy_function": CUSTOM, "scope": "application",
                      "kwargs": {"target": 3}})
    for config in (_token(), custom):
        app, spec = _app(config)
        assert app["autoscaling_policy"] == autoscaling_policy(config)
        # Serve's own bounds and delays stay on the deployment.
        assert spec["deployment_config"]["autoscaling_config"] == {"max_replicas": 4}
    assert _app(_token())[0]["autoscaling_policy"] == {
        "policy_function": TOKEN_FUNCTION, "policy_kwargs": TOKEN_KWARGS,
    }


def test_application_config_is_a_valid_serve_deploy_schema():
    from ray.serve.schema import ServeDeploySchema

    for config in (_config(), _token()):
        app, _ = _app(config, config.deployment.routing.environment())
        (parsed,) = ServeDeploySchema(applications=[app]).applications
        assert parsed.name == config.deployment.application_name
        assert parsed.import_path == "ray.serve.llm:build_openai_app"
        applied = parsed.dict(exclude_unset=True).get("autoscaling_policy")
        if config.deployment.autoscaling.policy.scope == "application":
            assert applied["policy_function"] == TOKEN_FUNCTION
            assert applied["policy_kwargs"] == TOKEN_KWARGS
        else:
            assert applied is None


# --- Preflight --------------------------------------------------------------


def test_preflight_default_policy():
    assert check_autoscaling_support(_config()) == {
        "policy": "default",
        "policy_function": DEFAULT_FUNCTION,
        "policy_kwargs": {},
    }


def test_preflight_binds_token_kwargs_to_this_ray_builds_policy():
    kwargs = {**TOKEN_KWARGS, "post_downscale_cooldown_s": 150}
    config = _config({"name": "token", "kwargs": kwargs}, **KV_AWARE)
    assert check_autoscaling_support(config) == {
        "policy": "token", "policy_function": TOKEN_FUNCTION, "policy_kwargs": kwargs,
    }
    # A kwarg from another branch's policy fails here rather than after deploy.
    stale = _config({"name": "token", "kwargs": {**TOKEN_KWARGS, "max_downscale_step": 1}},
                    **KV_AWARE)
    with pytest.raises(RuntimeError, match="does not accept the configured kwargs.*"
                                           "max_downscale_step"):
        check_autoscaling_support(stale)


def test_preflight_imports_a_custom_policy():
    here = _config({"name": "mine", "policy_function": "json:dumps", "kwargs": {"a": 1}})
    assert check_autoscaling_support(here) == {
        "policy": "mine", "policy_function": "json:dumps", "policy_kwargs": {"a": 1},
    }


@pytest.mark.parametrize("function, error", [
    ("ray.serve.nowhere:policy", "ModuleNotFoundError"),
    ("ray.serve.autoscaling_policy:no_such_policy", "AttributeError"),
])
def test_preflight_reports_missing_policy(function, error):
    config = _config({"name": "mine", "policy_function": function})
    with pytest.raises(RuntimeError, match=f"'mine' needs {function}.*does not provide.*{error}"):
        check_autoscaling_support(config)


# --- Verification on the deployment -----------------------------------------


def test_verify_applied_application_policy():
    config = _token()
    applied = {**autoscaling_policy(config), "_serialized_policy_def": "..."}
    assert verify_autoscaling_policy(config, _details(config, app_policy=applied)) == {
        "scope": "application", "verified_deployments": ["LLMServer:m"],
    }


def test_verify_applied_deployment_policy():
    config = _config()
    applied = {**autoscaling_policy(config), "_serialized_policy_def": "..."}
    assert verify_autoscaling_policy(config, _details(config, applied)) == {
        "scope": "deployment", "verified_deployments": ["LLMServer:m"],
    }


def test_verify_rejects_other_application_policy():
    config = _token()
    app = config.deployment.application_name
    # The app runs without a policy: every LLMServer keeps Serve's default.
    with pytest.raises(RuntimeError, match=f"application '{app}' runs autoscaling policy "
                                           "{'policy_function': None"):
        verify_autoscaling_policy(config, _details(config, autoscaling_policy(_config())))
    # The policy on the deployment instead of the application is not the same thing.
    with pytest.raises(RuntimeError, match=f"application '{app}' runs autoscaling policy"):
        verify_autoscaling_policy(config, _details(config, autoscaling_policy(config)))
    other_target = {**autoscaling_policy(config),
                    "policy_kwargs": {"target_prefill_tokens_per_replica": 60000}}
    with pytest.raises(RuntimeError, match="60000"):
        verify_autoscaling_policy(config, _details(config, app_policy=other_target))


def test_verify_rejects_other_deployment_policy():
    config = _config({"name": "mine", "policy_function": CUSTOM, "kwargs": {"target": 3}})
    with pytest.raises(RuntimeError, match="LLMServer:m runs autoscaling policy"):
        verify_autoscaling_policy(config, _details(config, autoscaling_policy(_config())))
    # Applied on the application only, the deployment has no policy of its own.
    with pytest.raises(RuntimeError, match="LLMServer:m runs autoscaling policy "
                                           "{'policy_function': None"):
        verify_autoscaling_policy(
            config, _details(config, app_policy=autoscaling_policy(config)))
    other_kwargs = {**autoscaling_policy(config), "policy_kwargs": {"target": 4}}
    with pytest.raises(RuntimeError, match="'target': 4"):
        verify_autoscaling_policy(config, _details(config, other_kwargs))


def test_verify_checks_every_llm_server_deployment():
    config = _config()
    details = _details(config, autoscaling_policy(config), name="LLMServer:a")
    deployments = details["applications"][config.deployment.application_name]["deployments"]
    deployments["LLMServer:b"] = {"deployment_config": {"autoscaling_config": {}}}
    deployments["LLMRouter"] = {}
    with pytest.raises(RuntimeError, match="LLMServer:b runs autoscaling policy"):
        verify_autoscaling_policy(config, details)
    deployments["LLMServer:b"] = deployments["LLMServer:a"]
    assert verify_autoscaling_policy(config, details) == {
        "scope": "deployment", "verified_deployments": ["LLMServer:a", "LLMServer:b"],
    }


def test_verify_requires_llm_server():
    config = _config()
    with pytest.raises(RuntimeError, match="no LLMServer deployment"):
        verify_autoscaling_policy(config, _details(config, {}, name="LLMRouter"))


# --- Run records ------------------------------------------------------------


def test_run_records_verified_policy(fake_cluster, experiment, monkeypatch):
    config, path = experiment(deployment=TOKEN_POLICY)
    monkeypatch.setattr(runner, "run_benchmark", lambda *args: {"error": None})
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path)
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["autoscaling"] == {
        "policy": "token",
        **autoscaling_policy(config),
        "scope": "application",
        "verified_deployments": ["LLMServer:m"],
    }
    recorded = json.loads((root / "deployment" / "autoscaling.json").read_text())
    assert recorded == manifest["autoscaling"]
    assert "Autoscale: token\n" in format_run_summary(root)


def test_run_records_verified_deployment_policy(fake_cluster, experiment, monkeypatch):
    config, path = experiment()
    monkeypatch.setattr(runner, "run_benchmark", lambda *args: {"error": None})
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path)
    assert json.loads((root / "manifest.json").read_text())["autoscaling"] == {
        "policy": "default",
        "policy_function": DEFAULT_FUNCTION,
        "policy_kwargs": {},
        "scope": "deployment",
        "verified_deployments": ["LLMServer:m"],
    }


def test_policy_mismatch_fails_run(fake_cluster, experiment, monkeypatch):
    config, path = experiment(deployment=TOKEN_POLICY)
    # Serve applied its default deployment policy and no application policy.
    monkeypatch.setattr(runner.RayServeBackend, "serve_details",
                        lambda self: _details(config, autoscaling_policy(_config())))
    monkeypatch.setattr(runner, "run_benchmark", lambda *args: pytest.fail("ran the workload"))
    with pytest.raises(RuntimeError, match="runs autoscaling policy"):
        runner.run_experiment(config, path)
    (root,) = config.runtime.results_dir.iterdir()
    manifest = json.loads((root / "manifest.json").read_text())
    assert manifest["failure_stage"] == "autoscaling_policy"
    assert "verified_deployments" not in manifest["autoscaling"]
    assert "Autoscale: token (not verified on the deployment)" in format_run_summary(root)
    assert fake_cluster.calls[-1] == "teardown"
