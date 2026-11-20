from __future__ import annotations

import importlib
import importlib.metadata
import inspect
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

# The injected vLLM scheduler that writes per-engine traces, the hook that
# prepares each replica worker for it, and the env var naming its output.
TRACING_SCHEDULER = "serve_llm_autoscaling.scheduler_trace.AutoTracingScheduler"
COMPAT_SETUP_HOOK = "serve_llm_autoscaling.vllm_compat.setup"
TRACE_DIR_ENV = "AUTOSCALE_HARNESS_TRACE_DIR"

# Ray's KVAwareRouter scores replicas with Dynamo's selection service.
MIN_AI_DYNAMO_VERSION = (1, 4, 0)
# Switches that only matter to the KV-aware (direct-streaming) topology; unset
# ones are cleared so an earlier deploy in the same process cannot leak into
# this one. RAY_SERVE_ENABLE_HA_PROXY is set when needed but never cleared:
# HAProxy in front of OpenAiIngress is a valid, independent choice.
ROUTING_ENV_VARS = (
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY",
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE",
    "RAY_SERVE_HAPROXY_HARD_STOP_AFTER_S",
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
    return {"policy_function": policy.policy_function, "policy_kwargs": dict(policy.kwargs)}


def check_autoscaling_support(config: ExperimentConfig) -> dict[str, Any]:
    """Fail before deploying if this Ray build cannot import the policy."""
    name = config.deployment.autoscaling.policy.name
    policy = autoscaling_policy(config)
    module_name, _, attr = policy["policy_function"].partition(":")
    try:
        target = getattr(importlib.import_module(module_name), attr)
    except Exception as exc:
        import ray

        raise RuntimeError(
            f"autoscaling policy {name!r} needs {policy['policy_function']}, which "
            f"Ray {ray.__version__} does not provide ({type(exc).__name__}: {exc})"
        ) from exc
    if inspect.isclass(target):
        # The controller constructs a class policy with the kwargs. A kwarg
        # this Ray build's policy does not take fails there, after deploy,
        # and leaves the deployment without a policy; catch it here instead.
        try:
            inspect.signature(target).bind(**policy["policy_kwargs"])
        except TypeError as exc:
            raise RuntimeError(
                f"autoscaling policy {name!r} ({policy['policy_function']}) in this Ray "
                f"build does not accept the configured kwargs: {exc}. It takes "
                f"{inspect.signature(target)}; is the right Ray branch checked out?"
            ) from exc
    return {"policy": name, **policy}


def verify_autoscaling_policy(
    config: ExperimentConfig, details: dict[str, Any]
) -> dict[str, Any]:
    """Check that Serve applied the requested policy at the requested scope.

    ``details`` is the controller's serve instance details, whose configs
    are the ones Serve applied rather than the ones requested.
    """
    expected = autoscaling_policy(config)
    application = config.deployment.application_name
    app = (details.get("applications") or {}).get(application) or {}
    servers = {
        name: deployment for name, deployment in (app.get("deployments") or {}).items()
        if name.startswith("LLMServer")
    }
    if not servers:
        raise RuntimeError(f"application {application!r} has no LLMServer deployment")
    if config.deployment.autoscaling.policy.scope == "application":
        applied = (app.get("deployed_app_config") or {}).get("autoscaling_policy") or {}
        actual = {key: applied.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"application {application!r} runs autoscaling policy {actual}, "
                f"expected {expected}"
            )
        return {"scope": "application", "verified_deployments": sorted(servers)}
    for name, deployment in servers.items():
        applied = (
            (deployment.get("deployment_config") or {}).get("autoscaling_config") or {}
        ).get("policy") or {}
        actual = {key: applied.get(key) for key in expected}
        if actual != expected:
            raise RuntimeError(
                f"{name} runs autoscaling policy {actual}, expected {expected}"
            )
    return {"scope": "deployment", "verified_deployments": sorted(servers)}


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
            "engine_kwargs": deployment.engine.model_dump(exclude_none=True),
            "deployment_config": {
                "autoscaling_config": deployment.autoscaling.model_dump(
                    exclude_none=True, exclude={"policy"}
                ),
            },
        }
        if deployment.autoscaling.policy.scope == "deployment":
            # An application-scope policy is attached in application_config().
            spec["deployment_config"]["autoscaling_config"]["policy"] = (
                autoscaling_policy(self.config)
            )
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
        self._ship_harness_package(spec)
        trace_dir = self.config.analysis.scheduler_trace_dir
        if trace_dir is not None:
            self._inject_tracing_scheduler(spec, trace_dir.resolve())
        return spec

    @staticmethod
    def _ship_harness_package(spec: dict[str, Any]) -> None:
        """Give every replica worker the harness package and its setup hook.

        The package travels as a ``py_modules`` entry, so replicas need no
        separate install. The hook (``vllm_compat.setup``) runs in each
        worker before vLLM starts its own nested Ray driver; without it that
        driver tries to upload the whole workspace and the engine fails.
        """
        package = Path(__file__).resolve().parent
        runtime = spec.setdefault("runtime_env", {})
        runtime.setdefault("py_modules", []).append(str(package))
        runtime["worker_process_setup_hook"] = COMPAT_SETUP_HOOK
        pythonpath = os.environ.get("PYTHONPATH")
        # The engine core is a spawned subprocess; keep the package
        # importable there too.
        runtime.setdefault("env_vars", {})["PYTHONPATH"] = str(package.parent) + (
            os.pathsep + pythonpath if pythonpath else ""
        )

    @staticmethod
    def _inject_tracing_scheduler(spec: dict[str, Any], trace_dir: Path) -> None:
        """Make every engine write a scheduler trace to ``trace_dir``.

        The engine's ``scheduler_cls`` becomes the harness's tracing subclass
        of vLLM's own scheduler (``scheduler_trace.AutoTracingScheduler``),
        which records request and batch events without changing scheduling.
        """
        if spec["engine_kwargs"].get("scheduler_cls") not in (None, TRACING_SCHEDULER):
            raise ValueError(
                "engine.scheduler_cls is already set; the harness cannot inject "
                "its tracing scheduler over a custom one"
            )
        spec["engine_kwargs"]["scheduler_cls"] = TRACING_SCHEDULER
        spec["runtime_env"]["env_vars"][TRACE_DIR_ENV] = str(trace_dir)

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
            self.teardown()
        elif env:
            # Controller options apply only when the controller starts, so an
            # idle controller from an earlier run would keep its environment.
            self.teardown()

        spec = self.deployment_spec()
        # LLMConfig accepts uploaded module URIs, not local directory paths.
        if spec.get("runtime_env", {}).get("py_modules"):
            import tempfile
            import ray
            from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
            from ray._private.runtime_env.setup_hook import upload_worker_process_setup_hook_if_needed

            runtime = spec["runtime_env"]
            with tempfile.TemporaryDirectory(prefix="autoscale-harness-packages-") as scratch:
                upload_py_modules_if_needed(runtime, include_gitignore=True, scratch_dir=scratch)
                upload_worker_process_setup_hook_if_needed(runtime, ray._private.worker.global_worker)
        # HAProxy and its body forwarding are configured in the controller,
        # which does not inherit the driver's environment.
        controller_options = {"runtime_env": {"env_vars": env}} if env else None
        serve.start(proxy_location="EveryNode", controller_options=controller_options)
        from ray.serve.context import _get_global_client
        from ray.serve.schema import ServeDeploySchema

        # Deployed from a config rather than serve.run(): Serve accepts an
        # application-scope autoscaling policy only through the app config.
        _get_global_client().deploy_apps(
            ServeDeploySchema(applications=[self.application_config(spec, env)])
        )
        return self.status()

    def application_config(self, spec: dict[str, Any], env: dict[str, str]) -> dict[str, Any]:
        """The Serve application config: the OpenAI app built from ``spec``."""
        deployment = self.config.deployment
        app: dict[str, Any] = {
            "name": deployment.application_name,
            "route_prefix": deployment.route_prefix,
            "import_path": "ray.serve.llm:build_openai_app",
            "args": {"llm_configs": [spec]},
        }
        if env:
            # The app is built in a Ray task, where Ray reads the topology
            # switches at import time.
            app["runtime_env"] = {"env_vars": env}
        if deployment.autoscaling.policy.scope == "application":
            app["autoscaling_policy"] = autoscaling_policy(self.config)
        return app

    def wait_healthy(self) -> dict[str, Any]:
        runtime = self.config.runtime
        model_id = self.config.deployment.model_id
        models_url = urljoin(self.base_url + "/", "v1/models")
        started = time.monotonic()
        deadline = started + runtime.readiness_timeout_s
        last_error = "not checked"

        while time.monotonic() < deadline:
            try:
                status = self.status()
            except Exception as exc:  # readiness must tolerate transient failures
                last_error = f"{type(exc).__name__}: {exc}"
                time.sleep(runtime.readiness_poll_s)
                continue
            app = (status.get("applications") or {}).get(
                self.config.deployment.application_name) or {}
            if not app and time.monotonic() - started > 120:
                raise RuntimeError(
                    "Serve has no record of the application two minutes after the "
                    "deploy request; the controller dropped it"
                )
            if "DEPLOY_FAILED" in str(app.get("status")):
                # A config deploy fails asynchronously; do not wait out the timeout.
                raise RuntimeError(
                    f"Serve failed to deploy the application: {app.get('message')}"
                )
            try:
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

    def teardown(self, timeout_s: float = 300) -> None:
        """Shut Serve down and wait until its controller is gone.

        ``serve.shutdown()`` can return while the controller is still
        stopping replicas. A deploy sent to it then is silently ignored
        ("Serve controller is shutting down"), so wait for the actor to exit.
        """
        import ray
        from ray import serve
        from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE
        from ray.serve.context import _set_global_client

        serve.shutdown()
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                ray.get_actor(SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
            except ValueError:
                break  # the controller actor no longer exists
            time.sleep(2)
        else:
            raise TimeoutError(f"the Serve controller did not shut down within {timeout_s}s")
        # Drop this process's client to the dead controller, so a later
        # deploy in the same process starts a new one.
        _set_global_client(None)
