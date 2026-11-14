from __future__ import annotations

import importlib
import importlib.metadata
import os
from pathlib import Path
import re
import sys
import time
from dataclasses import dataclass
from typing import Any
from urllib.parse import urljoin

import requests

from .config import ExperimentConfig

# Ray's KVAwareRouter scores replicas with Dynamo's selection service.
MIN_AI_DYNAMO_VERSION = (1, 4, 0)
# Switches that only matter to the direct-streaming topology; unset ones are
# cleared so an earlier deploy in the same process cannot leak into this one.
# RAY_SERVE_ENABLE_HA_PROXY is set when needed but never cleared: HAProxy in
# front of OpenAiIngress is a valid, independent choice.
ROUTING_ENV_VARS = (
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE",
)
# (module, constant, env var) that Ray reads once at import time.
_SNAPSHOTTED_FLAGS = (
    ("ray.serve._private.constants", "RAY_SERVE_ENABLE_HA_PROXY",
     "RAY_SERVE_ENABLE_HA_PROXY"),
    ("ray.serve._private.constants", "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
     "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY"),
    ("ray.llm._internal.serve.constants", "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
     "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING"),
)


def _release(version: str) -> tuple[int, ...]:
    match = re.match(r"\d+(\.\d+)*", version)
    if match is None:
        raise ValueError(f"unparseable version {version!r}")
    return tuple(int(part) for part in match.group().split("."))


def check_routing_support(config: ExperimentConfig) -> dict[str, Any]:
    """Fail before deploying if this environment cannot run the routing policy.

    Checked in the driver, which shares the Anyscale image with the workers.
    This is the first Ray Serve import, so it applies the routing environment.
    """
    routing = config.deployment.routing
    apply_routing_environment(config)
    if routing.policy is None:
        # No router is requested; Serve picks one for the topology.
        return {"policy": None, "request_router_class": None}
    from ray.serve.config import RequestRouterConfig

    import ray

    try:
        router = RequestRouterConfig(
            request_router_class=routing.router_class
        ).get_request_router_class()
    except Exception as exc:
        raise RuntimeError(
            f"routing policy {routing.policy!r} needs {routing.router_class}, which "
            f"Ray {ray.__version__} does not provide ({type(exc).__name__}: {exc})"
        ) from exc
    result: dict[str, Any] = {
        "policy": routing.policy,
        "request_router_class": f"{router.__module__}.{router.__qualname__}",
    }
    if routing.policy == "kv_aware":
        # Without Dynamo, Ray logs a warning and silently load-balances instead.
        wanted = ".".join(map(str, MIN_AI_DYNAMO_VERSION))
        try:
            version = importlib.metadata.version("ai-dynamo")
        except importlib.metadata.PackageNotFoundError:
            try:
                version = importlib.metadata.version("ai-dynamo-runtime")
            except importlib.metadata.PackageNotFoundError:
                raise RuntimeError(
                    f"routing policy kv_aware requires ai-dynamo>={wanted}; install it "
                    f"on every node (pip install 'ai-dynamo>={wanted}')"
                ) from None
        if _release(version) < MIN_AI_DYNAMO_VERSION:
            raise RuntimeError(
                f"routing policy kv_aware requires ai-dynamo>={wanted}, found {version}"
            )
        try:
            importlib.import_module("dynamo.llm").SelectionService
        except (ImportError, AttributeError) as exc:
            raise RuntimeError(
                f"ai-dynamo {version} is installed but dynamo.llm.SelectionService "
                f"is not importable ({type(exc).__name__}: {exc})"
            ) from exc
        result["ai_dynamo_version"] = version
        if (config.analysis.record_router_estimates
                and config.analysis.scheduler_trace_dir is not None):
            from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import KVTokenTracker

            if not hasattr(KVTokenTracker, "_trace_request_event"):
                raise RuntimeError(
                    "record_router_estimates requires the request-audit Ray patch; "
                    "apply patches/router-request-audit.patch to this Ray build "
                    "or disable analysis.record_router_estimates"
                )
            result["router_request_audit_supported"] = True
    return result


def apply_routing_environment(config: ExperimentConfig) -> dict[str, str]:
    """Set the topology switches in this process before Ray Serve is imported.

    Ray snapshots them into module constants on first import, so a module
    imported earlier with different values is an error rather than ignored.
    """
    env = config.deployment.routing.environment()
    for name in ROUTING_ENV_VARS:
        if name not in env:
            os.environ.pop(name, None)
    os.environ.update(env)
    stale = []
    for module_name, constant, name in _SNAPSHOTTED_FLAGS:
        module = sys.modules.get(module_name)
        if module is None or name not in env:
            continue
        if getattr(module, constant, None) is not True:
            stale.append(f"{module_name}.{constant}")
    if stale:
        raise RuntimeError(
            "Ray Serve was imported before the routing environment was applied "
            f"({', '.join(stale)}); export {', '.join(sorted(env))} before starting "
            "the harness"
        )
    return env


def verify_topology(config: ExperimentConfig, status: dict[str, Any]) -> dict[str, Any]:
    """Check that Serve built the ingress topology the routing config asks for."""
    application = config.deployment.application_name
    app = (status.get("applications") or {}).get(application) or {}
    names = sorted(app.get("deployments") or {})
    has_router = "LLMRouter" in names
    has_ingress = "OpenAiIngress" in names
    if config.deployment.routing.direct_streaming:
        if not has_router or has_ingress:
            raise RuntimeError(
                "direct streaming expected LLMRouter without OpenAiIngress, but "
                f"application {application!r} has deployments {names}"
            )
        topology = "direct_streaming"
    else:
        if has_router or not has_ingress:
            raise RuntimeError(
                "expected the OpenAiIngress topology, but application "
                f"{application!r} has deployments {names}"
            )
        topology = "openai_ingress"
    return {"topology": topology, "deployments": names}


def autoscaling_policy(config: ExperimentConfig) -> dict[str, Any]:
    """The Serve ``AutoscalingPolicy`` for LLMServer, as deployed."""
    policy = config.deployment.autoscaling.policy
    kwargs = dict(policy.kwargs)
    host = os.environ.get("RAY_PROMETHEUS_HOST")
    if policy.name == "sla" and "prometheus_address" not in kwargs and host:
        # The policy runs in the controller; pin it to the driver's Prometheus.
        kwargs["prometheus_address"] = host
    return {"policy_function": policy.policy_function, "policy_kwargs": kwargs}


def check_autoscaling_support(config: ExperimentConfig) -> dict[str, Any]:
    """Fail before deploying if the controller cannot run the autoscaling policy.

    Without scikit-learn the SLA policy fails to construct, and without
    Prometheus it silently falls back to the default policy.
    """
    name = config.deployment.autoscaling.policy.name
    policy = autoscaling_policy(config)
    module_name, _, attr = policy["policy_function"].partition(":")
    try:
        getattr(importlib.import_module(module_name), attr)
    except Exception as exc:
        import ray

        raise RuntimeError(
            f"autoscaling policy {name!r} needs {policy['policy_function']}, which "
            f"Ray {ray.__version__} does not provide ({type(exc).__name__}: {exc})"
        ) from exc
    result: dict[str, Any] = {"policy": name, **policy}
    if name == "sla":
        try:
            result["scikit_learn_version"] = importlib.metadata.version("scikit-learn")
        except importlib.metadata.PackageNotFoundError:
            raise RuntimeError(
                "autoscaling policy sla requires scikit-learn in the Serve "
                "controller's environment (pip install scikit-learn)"
            ) from None
        address = policy["policy_kwargs"].get("prometheus_address")
        if not address:
            raise RuntimeError(
                "autoscaling policy sla needs Prometheus; set RAY_PROMETHEUS_HOST "
                "or autoscaling.policy.kwargs.prometheus_address"
            )
        try:
            response = requests.get(
                urljoin(address.rstrip("/") + "/", "api/v1/query"),
                params={"query": "1"},
                timeout=5,
            )
            response.raise_for_status()
        except Exception as exc:
            raise RuntimeError(
                f"autoscaling policy sla cannot query Prometheus at {address} "
                f"({type(exc).__name__}: {exc})"
            ) from exc
    return result


def verify_autoscaling_policy(
    config: ExperimentConfig, details: dict[str, Any]
) -> dict[str, Any]:
    """Check that every LLMServer deployment runs the requested policy.

    ``details`` is the controller's serve instance details, whose deployment
    configs are the ones Serve applied rather than the ones requested.
    """
    expected = autoscaling_policy(config)
    application = config.deployment.application_name
    app = (details.get("applications") or {}).get(application) or {}
    applied = {
        name: (
            (deployment.get("deployment_config") or {}).get("autoscaling_config") or {}
        ).get("policy") or {}
        for name, deployment in (app.get("deployments") or {}).items()
        if name.startswith("LLMServer")
    }
    if not applied:
        raise RuntimeError(f"application {application!r} has no LLMServer deployment")
    for name, policy in applied.items():
        actual = {key: policy.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"{name} runs autoscaling policy {actual}, expected {expected}"
            )
    return {"verified_deployments": sorted(applied)}


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if hasattr(value, "dict"):
        return value.dict()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if hasattr(value, "__dict__"):
        return {k: _jsonable(v) for k, v in vars(value).items() if not k.startswith("_")}
    return value


@dataclass
class RayServeBackend:
    config: ExperimentConfig
    replace: bool = False

    @property
    def base_url(self) -> str:
        return self.config.runtime.endpoint_url.rstrip("/")

    def connect(self) -> dict[str, Any]:
        import ray

        ray.init(address="auto", ignore_reinit_error=True)
        return {
            "ray_version": ray.__version__,
            "ray_path": ray.__file__,
            "cluster_resources": ray.cluster_resources(),
            "available_resources": ray.available_resources(),
        }

    def status(self) -> dict[str, Any]:
        from ray import serve

        return _jsonable(serve.status())

    def serve_details(self) -> dict[str, Any]:
        """The controller's view, including each deployment's applied config."""
        from ray.serve.context import _get_global_client

        return _jsonable(_get_global_client().get_serve_details())

    def deployment_spec(self) -> dict[str, Any]:
        """Return the serializable LLMConfig payload used for deployment."""
        deployment = self.config.deployment
        spec: dict[str, Any] = {
            "model_loading_config": {
                "model_id": deployment.model_id,
                "model_source": deployment.model_source,
            },
            "engine_kwargs": deployment.engine.model_dump(exclude_none=True, exclude={"shared_kv_cache"}),
            "deployment_config": {
                "autoscaling_config": {
                    **deployment.autoscaling.model_dump(
                        exclude_none=True, exclude={"policy"}
                    ),
                    "policy": autoscaling_policy(self.config),
                },
            },
        }
        if deployment.routing.policy is not None:
            spec["deployment_config"]["request_router_config"] = {
                "request_router_class": deployment.routing.router_class,
                "request_router_kwargs": deployment.routing.request_router_kwargs,
            }
        if deployment.experimental_configs:
            spec["experimental_configs"] = deployment.experimental_configs
        if deployment.max_ongoing_requests is not None:
            spec["deployment_config"]["max_ongoing_requests"] = deployment.max_ongoing_requests
        if deployment.accelerator_type:
            spec["accelerator_type"] = deployment.accelerator_type
        env_vars = {}
        shared = deployment.engine.shared_kv_cache
        if shared is not None and shared.enabled:
            env_vars.update({
                "MOONCAKE_CONFIG_PATH": str(shared.config_path.resolve()),
                "PYTHONHASHSEED": "0",
                "VLLM_MOONCAKE_STORE_TIER_LOG": "1",
            })
            spec["engine_kwargs"]["kv_transfer_config"] = {
                "kv_connector": "MooncakeStoreConnector", "kv_role": "kv_both",
                "kv_load_failure_policy": shared.load_failure_policy,
                "kv_connector_extra_config": {
                    "cache_prefix": shared.cache_prefix,
                    "load_async": shared.load_async,
                    "save_decode_cache": shared.save_decode_cache,
                },
            }
            if shared.package:
                spec.setdefault("runtime_env", {})["pip"] = [shared.package]
        if self.config.analysis.scheduler_trace_dir is not None:
            trace_dir = str(self.config.analysis.scheduler_trace_dir.resolve())
            if self.config.analysis.scheduler_trace_backend == "injected":
                scheduler = "serve_profiler.scheduler.AutoTracingScheduler"
                if spec["engine_kwargs"].get("scheduler_cls") not in (None, scheduler):
                    raise ValueError("Cannot replace an existing custom scheduler")
                package = Path(__file__).resolve().parents[3] / "serve-llm-profiler" / "serve_profiler"
                if not (package / "scheduler.py").is_file():
                    raise ValueError(f"Injected scheduler package is missing: {package}")
                spec["engine_kwargs"]["scheduler_cls"] = scheduler
                runtime = spec.setdefault("runtime_env", {})
                runtime.setdefault("py_modules", []).append(str(package))
                runtime["worker_process_setup_hook"] = "serve_profiler.compat.setup"
                env_vars.update({
                    "SERVE_PROFILER_TRACE_DIR": trace_dir,
                    "RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR": "",
                    "PYTHONPATH": str(package.parent) + (
                        os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else ""
                    ),
                })
            else:
                env_vars["RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR"] = trace_dir
            if (self.config.analysis.record_router_estimates
                    and deployment.routing.policy == "kv_aware"):
                env_vars["RAY_SERVE_LLM_ROUTER_REQUEST_TRACE_DIR"] = str(
                    self.config.analysis.scheduler_trace_dir.resolve() / "router-request-audit"
                )
        if env_vars:
            spec.setdefault("runtime_env", {})["env_vars"] = env_vars
        return spec

    def routing_summary(self) -> dict[str, Any]:
        """The requested routing and the Ray switches that implement it."""
        routing = self.config.deployment.routing
        return {
            **routing.model_dump(mode="json"),
            "request_router_class": routing.router_class,
            "environment": routing.environment(),
        }

    def deploy(self) -> dict[str, Any]:
        # Must precede the first Ray Serve import in this process.
        env = apply_routing_environment(self.config)
        from ray import serve

        current = serve.status()
        applications = getattr(current, "applications", {})
        if applications:
            if not self.replace:
                names = ", ".join(sorted(applications))
                raise RuntimeError(
                    f"Serve already has application(s): {names}. Pass --replace to replace them."
                )
            serve.shutdown()
        elif env:
            # Controller options apply only when the controller starts, so an
            # idle controller from an earlier run would keep its environment.
            serve.shutdown()

        # Import after setting the topology switch; Ray snapshots some LLM flags
        # at module import time.
        from ray.serve.llm import LLMConfig, build_openai_app

        deployment = self.config.deployment
        spec = self.deployment_spec()
        # LLMConfig accepts uploaded module URIs, not local directory paths.
        if spec.get("runtime_env", {}).get("py_modules"):
            import tempfile
            import ray
            from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
            from ray._private.runtime_env.setup_hook import upload_worker_process_setup_hook_if_needed

            with tempfile.TemporaryDirectory(prefix="token-metrics-packages-") as scratch:
                for runtime in (spec.get("runtime_env"),):
                    if runtime:
                        upload_py_modules_if_needed(runtime, include_gitignore=True, scratch_dir=scratch)
                        upload_worker_process_setup_hook_if_needed(runtime, ray._private.worker.global_worker)
        app = build_openai_app({"llm_configs": [LLMConfig(**spec)]})
        # HAProxy and its body forwarding are configured in the controller,
        # which does not inherit the driver's environment.
        controller_options = {"runtime_env": {"env_vars": env}} if env else None
        serve.run(
            app,
            name=deployment.application_name,
            route_prefix=deployment.route_prefix,
            controller_options=controller_options,
        )
        return self.status()

    def wait_healthy(self) -> dict[str, Any]:
        runtime = self.config.runtime
        model_id = self.config.deployment.model_id
        models_url = urljoin(self.base_url + "/", "v1/models")
        deadline = time.monotonic() + runtime.readiness_timeout_s
        last_error = "not checked"

        while time.monotonic() < deadline:
            try:
                status = self.status()
                response = requests.get(models_url, timeout=5)
                response.raise_for_status()
                body = response.json()
                ids = [item.get("id") for item in body.get("data", [])]
                if model_id in ids:
                    return {"serve_status": status, "models": body}
                last_error = f"registered models were {ids!r}"
            except Exception as exc:  # readiness must tolerate transient failures
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(runtime.readiness_poll_s)

        raise TimeoutError(
            f"{model_id!r} did not become ready at {models_url} within "
            f"{runtime.readiness_timeout_s}s; last observation: {last_error}"
        )

    def teardown(self) -> None:
        from ray import serve

        serve.shutdown()
