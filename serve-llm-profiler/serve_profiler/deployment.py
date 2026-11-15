"""Small standalone adapter from harness-style YAML to public Serve LLM APIs."""
from __future__ import annotations

import copy
import importlib.metadata
import os
import socket
import time
from pathlib import Path
from urllib.parse import urlparse

import requests
import yaml

from .analysis import write_json

ROUTERS = {
    "power_of_two": "ray.serve._private.request_router:PowerOfTwoChoicesRequestRouter",
    "round_robin": "ray.serve.experimental.round_robin_router.RoundRobinRouter",
    "consistent_hash": "ray.serve.experimental.consistent_hash_router.ConsistentHashRouter",
    "prefix_cache_affinity": "ray.serve.llm.request_router.PrefixCacheAffinityRouter",
    "kv_aware": "ray.serve.llm.request_router.KVAwareRouter",
}


def load_config(path: Path) -> dict:
    config = yaml.safe_load(path.read_text())
    if not isinstance(config, dict) or not config.get("deployment", {}).get("model_id"):
        raise ValueError("Expected application YAML with deployment.model_id")
    engine = config["deployment"].get("engine", {})
    for field in (
        "tensor_parallel_size",
        "pipeline_parallel_size",
        "data_parallel_size",
        "prefill_context_parallel_size",
        "decode_context_parallel_size",
    ):
        if engine.get(field, 1) not in (None, 1):
            raise ValueError(
                f"MVP supports one GPU/engine per replica: {field} must be 1"
            )
    if engine.get("scheduler_cls"):
        raise ValueError("Cannot replace an application's custom scheduler")
    if engine.get("shared_kv_cache"):
        raise ValueError(
            "shared_kv_cache is a harness-managed service; use explicit vLLM kv_transfer_config instead"
        )
    routing = config["deployment"].get("routing", {})
    if routing.get("policy") not in (None, *ROUTERS):
        raise ValueError(f"Unknown routing policy {routing['policy']!r}")
    return config


def routing_environment(config: dict) -> dict:
    routing = config["deployment"].get("routing", {})
    direct = routing.get("direct_streaming")
    if direct is None:
        direct = routing.get("policy") == "kv_aware"
    if routing.get("policy") == "kv_aware" and not direct:
        raise ValueError("kv_aware routing requires direct_streaming")
    env = {}
    if direct:
        env.update(
            RAY_SERVE_ENABLE_HA_PROXY="1", RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING="1"
        )
        forward = routing.get(
            "forward_request_body",
            routing.get("policy") in {"kv_aware", "prefix_cache_affinity"},
        )
        if forward:
            env["RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY"] = "1"
        if routing.get("haproxy_request_buffer_bytes"):
            env["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] = str(
                routing["haproxy_request_buffer_bytes"]
            )
    return env


def resolved_config(config: dict, root: Path) -> tuple[dict, str, str]:
    config = copy.deepcopy(config)
    name = "prefill-profile-" + root.name.split("-")[-1]
    route = "/" + name
    config["deployment"].update(
        application_name=name,
        route_prefix=route,
        autoscaling={"min_replicas": 1, "initial_replicas": 1, "max_replicas": 1},
    )
    config["deployment"].setdefault("engine", {})[
        "scheduler_cls"
    ] = "serve_profiler.scheduler.AutoTracingScheduler"
    config["profiling"] = {
        "num_replicas": 1,
        "trace_dir": str((root / "traces").resolve()),
        "duration_s": 1800,
        "session_tree_pool_size": 8,
        "cohort_sizes": [8],
        "phase_duration_s": 1800,
        "ignore_trace_delays": True,
    }
    return config, name, route


def llm_spec(config: dict, root: Path) -> dict:
    deployment = config["deployment"]
    engine = dict(deployment.get("engine", {}))
    engine.pop("shared_kv_cache", None)
    # Never enable the unrelated Ray built-in trace injection simultaneously.
    env = {
        "SERVE_PROFILER_TRACE_DIR": str((root / "traces").resolve()),
        "RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR": "",
        # The single-node run path is shared. The direct-streaming router
        # builder forwards env_vars but drops py_modules and setup_hook fields.
        "PYTHONPATH": str(Path(__file__).resolve().parent.parent)
        + (
            os.pathsep + os.environ["PYTHONPATH"]
            if os.environ.get("PYTHONPATH")
            else ""
        ),
        **routing_environment(config),
    }
    spec = {
        "model_loading_config": {
            "model_id": deployment["model_id"],
            "model_source": deployment.get("model_source") or deployment["model_id"],
        },
        "engine_kwargs": engine,
        "deployment_config": {"num_replicas": 1},
        "runtime_env": {
            "py_modules": [str(Path(__file__).resolve().parent)],
            "env_vars": env,
            "worker_process_setup_hook": "serve_profiler.compat.setup",
        },
    }
    if deployment.get("accelerator_type"):
        spec["accelerator_type"] = deployment["accelerator_type"]
    for key in ("max_ongoing_requests",):
        if deployment.get(key) is not None:
            spec["deployment_config"][key] = deployment[key]
    if deployment.get("experimental_configs"):
        spec["experimental_configs"] = deployment["experimental_configs"]
    routing = deployment.get("routing", {})
    if routing.get("policy"):
        kwargs = dict(routing.get("request_router_kwargs", {}))
        # The profiler measures offline; no learner can alter routing or budgets.
        kwargs.pop("enable_prefill_queue_learning", None)
        spec["deployment_config"]["request_router_config"] = {
            "request_router_class": ROUTERS[routing["policy"]],
            "request_router_kwargs": kwargs,
        }
    return spec


class Deployment:
    def __init__(self, config: dict, root: Path):
        self.config, self.root = config, root
        self.name = config["deployment"]["application_name"]
        self.attempted = False
        self.ray = self.serve = None
        base = (
            config.get("runtime", {})
            .get("endpoint_url", "http://127.0.0.1:8000")
            .rstrip("/")
        )
        parsed = urlparse(base)
        if parsed.path not in ("", "/") or not parsed.scheme:
            raise ValueError(
                "runtime.endpoint_url must be a Serve HTTP origin, without a route or /v1 suffix"
            )
        self.url = base + config["deployment"]["route_prefix"]

    def connect(self) -> None:
        os.environ.update(routing_environment(self.config))
        import ray

        # Avoid entering Ray's long native retry loop on an unreachable cluster.
        # This check performs no deployment and does not start a local cluster.
        from ray._private.services import canonicalize_bootstrap_address

        address = canonicalize_bootstrap_address("auto")
        if not address:
            raise RuntimeError(
                "No running Ray cluster found; start the GPU cluster first"
            )
        host, port = address.rsplit(":", 1)
        try:
            with socket.create_connection((host.strip("[]"), int(port)), timeout=5):
                pass
        except OSError as exc:
            raise RuntimeError(f"Cannot reach Ray at {address}: {exc}") from exc
        ray.init(
            address="auto",
            namespace="serve-profiler",
            ignore_reinit_error=True,
            runtime_env={
                "py_modules": [str(Path(__file__).resolve().parent)],
                "worker_process_setup_hook": "serve_profiler.compat.setup",
            },
        )
        self.ray = ray
        from ray import serve

        self.serve = serve
        available = ray.available_resources()
        if available.get("GPU", 0) < 1:
            raise RuntimeError(f"Ray has no available GPU: {available}")
        status = serve.status()
        # Direct-streaming switches are controller startup settings. Do not
        # restart a controller belonging to another application to apply them.
        if status.applications:
            raise RuntimeError(
                "Profiling requires an idle Serve instance; existing applications were left untouched: "
                + ", ".join(status.applications)
            )
        versions = {}
        for package in ("ray", "vllm", "torch"):
            try:
                versions[package] = importlib.metadata.version(package)
            except importlib.metadata.PackageNotFoundError:
                versions[package] = "unknown"
        write_json(
            self.root / "environment.json",
            {
                "versions": versions,
                "cluster_resources": ray.cluster_resources(),
                "available_resources": available,
            },
        )

    def start(self) -> None:
        from .compat import setup

        setup()
        from ray.serve.llm import LLMConfig, build_openai_app
        from ray._private.runtime_env.py_modules import upload_py_modules_if_needed
        from ray._private.runtime_env.setup_hook import (
            upload_worker_process_setup_hook_if_needed,
        )

        spec = llm_spec(self.config, self.root)
        # Serve's controller schema accepts package URIs, not driver-local
        # py_modules paths. Upload before serializing the deployment options.
        upload_py_modules_if_needed(
            spec["runtime_env"],
            include_gitignore=True,
            scratch_dir=str(self.root / ".packages"),
        )
        upload_worker_process_setup_hook_if_needed(
            spec["runtime_env"], self.ray._private.worker.global_worker
        )
        write_json(self.root / "deployment.json", spec)
        llm = LLMConfig(**spec)
        app = build_openai_app({"llm_configs": [llm]})
        env = routing_environment(self.config)
        # No serve.shutdown(): this application alone is owned by the profiler.
        self.attempted = True
        self.serve.run(
            app,
            name=self.name,
            route_prefix=self.config["deployment"]["route_prefix"],
            blocking=False,
            controller_options={"runtime_env": {"env_vars": env}} if env else None,
        )
        self.wait_ready()

    def wait_ready(self) -> None:
        """Wait for the owned application and verify scheduler trace output."""
        timeout = self.config.get("runtime", {}).get("readiness_timeout_s", 1800)
        deadline = time.monotonic() + timeout
        last = "not ready"
        while time.monotonic() < deadline:
            try:
                response = requests.get(self.url + "/v1/models", timeout=5)
                response.raise_for_status()
                if self.config["deployment"]["model_id"] in [
                    m["id"] for m in response.json()["data"]
                ]:
                    # A small streaming request warms the runtime and proves the
                    # injected scheduler is writing locally visible trace files.
                    response = requests.post(
                        self.url + "/v1/chat/completions",
                        json={
                            "model": self.config["deployment"]["model_id"],
                            "messages": [{"role": "user", "content": "Hello"}],
                            "max_tokens": 1,
                        },
                        timeout=120,
                    )
                    response.raise_for_status()
                    time.sleep(2)  # background writer flush interval is one second
                    if not list((self.root / "traces").glob("*.jsonl")):
                        raise RuntimeError(
                            "No local scheduler trace files after readiness probe; run on one node with a shared run path"
                        )
                    return
            except requests.RequestException as exc:
                last = str(exc)
            time.sleep(2)
        raise TimeoutError(f"Profiling application not ready: {last}")

    def close(self) -> None:
        if self.attempted and self.serve is not None:
            self.serve.delete(self.name, _blocking=True)
        if self.ray is not None:
            self.ray.shutdown()
