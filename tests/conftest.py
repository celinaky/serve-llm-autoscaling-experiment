import copy
from pathlib import Path

import pytest

from serve_llm_autoscaling.config import ExperimentConfig

REPO = Path(__file__).resolve().parents[1]
EXPERIMENTS = sorted((REPO / "experiments").glob("*.yaml"))

# A long-context deployment replaying a 1 -> 16 session ramp, then a hold.
AGENTX_RAW = {
    "name": "agentx-ramp",
    "deployment": {
        "model_id": "long-context-model",
        "engine": {"max_model_len": 262_144},
        "autoscaling": {"min_replicas": 1, "initial_replicas": 1, "max_replicas": 4},
    },
    "benchmark": {
        "duration_s": 1800,
        "grace_period_s": 300,
        "workload": {
            "concurrency_series": [
                {"time_s": 0, "sessions": 1},
                {"time_s": 1200, "sessions": 16},
            ],
        },
    },
}

# The same deployment with the KV-aware router and the token policy.
KV_AWARE = {"routing": {"policy": "kv_aware"}}
TOKEN_POLICY = {
    **KV_AWARE,
    "autoscaling": {
        "policy": {"name": "token", "kwargs": {"target_prefill_tokens_per_replica": 20000}},
        "max_replicas": 4,
    },
}


def series(*points):
    """A ``concurrency_series`` from ``(time_s, sessions)`` pairs."""
    return [{"time_s": t, "sessions": s} for t, s in points]


def raw_config(workload=None, *, deployment=None, analysis=None, runtime=None, **benchmark):
    """The AgentX experiment as a raw dict, with sections shallowly overridden."""
    raw = copy.deepcopy(AGENTX_RAW)
    raw["benchmark"].update(benchmark)
    raw["benchmark"]["workload"].update(workload or {})
    raw["deployment"].update(copy.deepcopy(deployment or {}))
    if analysis is not None:
        raw["analysis"] = dict(analysis)
    if runtime is not None:
        raw["runtime"] = dict(runtime)
    return raw


def build_config(workload=None, **sections) -> ExperimentConfig:
    return ExperimentConfig.model_validate(raw_config(workload, **sections))


def short_config(workload=None, **sections) -> ExperimentConfig:
    """A 200s development run: ramp 1 -> 4 sessions over 120s, then hold."""
    sections.setdefault("duration_s", 200)
    sections.setdefault("grace_period_s", 30)
    return build_config(
        {"concurrency_series": series((0, 1), (120, 4)), "unsafe_override": True,
         **(workload or {})},
        **sections,
    )


@pytest.fixture
def agentx_config():
    """Build an AgentX config, overriding workload, benchmark or other sections."""
    return build_config


@pytest.fixture
def uvx(monkeypatch):
    """Make AIPerf command construction independent of an installed uvx."""
    monkeypatch.setattr("serve_llm_autoscaling.benchmark.shutil.which",
                        lambda name: f"/usr/bin/{name}")


class FakeCluster:
    """What ``run_experiment`` saw of the faked Ray cluster."""

    def __init__(self):
        self.calls = []      # backend lifecycle calls, in order
        self.sessions = []   # TelemetrySession objects the runner created

    def collectors_alive(self):
        return [c._thread.is_alive() for c in self.sessions[-1].collectors]


@pytest.fixture
def fake_cluster(monkeypatch):
    """Stand in for Ray so ``run_experiment`` runs end to end without a cluster.

    The preflights pass, the backend deploys the topology and autoscaling
    policy the config asks for, and telemetry samples an empty cluster.
    AIPerf is left to each test (``runner.AIPerfRunner.run``).
    """
    from serve_llm_autoscaling import runner
    from serve_llm_autoscaling.backend import autoscaling_policy
    from serve_llm_autoscaling.telemetry import TelemetrySession

    cluster = FakeCluster()
    monkeypatch.setattr(runner, "environment_check", lambda connect, **kwargs: {})
    monkeypatch.setattr(runner, "check_routing_support", lambda config: {
        "policy": config.deployment.routing.policy,
        "request_router_class": config.deployment.routing.router_class,
    })
    monkeypatch.setattr(runner, "check_autoscaling_support", lambda config: {
        "policy": config.deployment.autoscaling.policy.name, **autoscaling_policy(config),
    })

    def ready(self):
        cluster.calls.append("wait_healthy")
        routing = self.config.deployment.routing
        names = ["LLMRouter"] if routing.direct_streaming else ["OpenAiIngress"]
        deployments = {name: {} for name in [*names, "LLMServer:m"]}
        app = self.config.deployment.application_name
        return {"serve_status": {"applications": {app: {"deployments": deployments}}}}

    def details(self):
        # Serve applies the requested application and deployment configs unchanged.
        spec = self.deployment_spec()
        app = self.config.deployment.application_name
        deployment = {"deployment_config": spec["deployment_config"]}
        return {"applications": {app: {
            "deployed_app_config": self.application_config(spec, {}),
            "deployments": {"LLMServer:m": deployment},
        }}}

    def record(name, value):
        return lambda self: cluster.calls.append(name) or value

    monkeypatch.setattr(runner.RayServeBackend, "deploy", record("deploy", {}))
    monkeypatch.setattr(runner.RayServeBackend, "teardown", record("teardown", None))
    monkeypatch.setattr(runner.RayServeBackend, "wait_healthy", ready)
    monkeypatch.setattr(runner.RayServeBackend, "serve_details", details)

    def session(config, root):
        cluster.sessions.append(TelemetrySession(
            config, root, status_fetch=lambda: {"applications": {}},
            discover=lambda: [], fetch_metrics=lambda url: "",
        ))
        return cluster.sessions[-1]

    monkeypatch.setattr(runner, "TelemetrySession", session)
    return cluster


@pytest.fixture
def experiment(tmp_path):
    """A short experiment whose results land in ``tmp_path``; returns (config, path)."""

    def build(workload=None, **sections):
        config = short_config(workload, **sections)
        config.runtime.results_dir = tmp_path / "runs"
        path = tmp_path / "experiment.yaml"
        path.write_text(f"name: {config.name}\n")
        return config, path

    return build
