import pytest
import yaml
from pydantic import ValidationError

from conftest import EXPERIMENTS, KV_AWARE, REPO, TOKEN_POLICY, raw_config, series
from serve_llm_autoscaling.config import (
    AIPERF_SOURCE_DEFAULT,
    EngineConfig,
    ExperimentConfig,
    load_config,
    load_saved_config,
    write_config,
)

# --- Engine -----------------------------------------------------------------


@pytest.mark.parametrize("backend", [None, "native"])
def test_native_offloading_accepts_fractional_gib_and_requires_prefix_caching(backend):
    settings = {"kv_offloading_size": 0.5, "kv_offloading_backend": backend}
    for prefix in (None, False):
        with pytest.raises(ValidationError, match="requires enable_prefix_caching"):
            EngineConfig(**settings, enable_prefix_caching=prefix)
    assert EngineConfig(**settings, enable_prefix_caching=True).kv_offloading_size == 0.5


@pytest.mark.parametrize("size", [0, -1, float("inf"), float("nan")])
def test_rejects_invalid_offload_capacity(size):
    with pytest.raises(ValidationError):
        EngineConfig(kv_offloading_size=size, enable_prefix_caching=True)


def test_rejects_offload_shortcut_overwriting_connector():
    with pytest.raises(ValidationError, match="overwrite the selected connector"):
        EngineConfig(kv_offloading_size=64, enable_prefix_caching=True,
                     kv_transfer_config={"kv_connector": "MultiConnector"})


def test_disabled_offloading_allows_prefix_cache_disabled_and_explicit_connectors():
    config = EngineConfig(enable_prefix_caching=False, kv_offloading_backend="native")
    assert config.kv_offloading_size is None
    EngineConfig(kv_transfer_config={"kv_connector": "MultiConnector"})


# --- Deployment -------------------------------------------------------------


def test_deployment_defaults(agentx_config):
    deployment = agentx_config().deployment
    assert deployment.model_source == deployment.tokenizer == "long-context-model"
    assert deployment.routing.policy is None
    assert deployment.autoscaling.policy.name == "default"
    assert deployment.max_ongoing_requests is None


@pytest.mark.parametrize("autoscaling, match", [
    ({"min_replicas": 3, "max_replicas": 2}, "min_replicas must be <= max_replicas"),
    ({"min_replicas": 1, "initial_replicas": 5, "max_replicas": 4}, "initial_replicas"),
    ({"min_replicas": 2, "initial_replicas": 1}, "initial_replicas"),
])
def test_rejects_bad_replica_bounds(agentx_config, autoscaling, match):
    with pytest.raises(ValidationError, match=match):
        agentx_config(deployment={"autoscaling": autoscaling})


def test_unlisted_engine_and_autoscaling_keys_are_kept(agentx_config):
    config = agentx_config(deployment={
        "engine": {"language_model_only": True},
        "autoscaling": {"look_back_period_s": 60},
    })
    assert config.deployment.engine.model_extra == {"language_model_only": True}
    assert config.deployment.autoscaling.model_extra == {"look_back_period_s": 60}


# --- Session curve ----------------------------------------------------------


def test_concurrency_series_parses(agentx_config):
    workload = agentx_config(
        {"concurrency_series": series((0, 1), (600, 16), (1200, 4))}
    ).benchmark.workload
    assert [(p.time_s, p.sessions) for p in workload.concurrency_series] == [
        (0, 1), (600, 16), (1200, 4)
    ]
    # AIPerf's --concurrency is the peak of the curve, not a separate setting.
    assert workload.target_concurrency == 16
    assert (workload.seed, workload.unsafe_override) == (42, False)


@pytest.mark.parametrize("points, match", [
    (series((0, 1)), "at least 2 items"),
    (series((10, 1), (600, 16)), "start at time_s 0"),
    (series((0, 1), (600, 16), (600, 4)), "strictly increasing"),
    (series((0, 1), (1801, 16)), "after duration_s"),
    (series((0, 1), (600, 0)), "sessions"),
    (series((0, 1), (600, 16.0)), "sessions"),
    (series((0, 1), (-5, 2)), "time_s"),
    ([{"time_s": 0, "sessions": 1}, {"time_s": 60, "qps": 8}], "qps"),
])
def test_rejects_bad_concurrency_series(agentx_config, points, match):
    with pytest.raises(ValidationError, match=match):
        agentx_config({"concurrency_series": points})


def test_series_may_end_exactly_at_duration(agentx_config):
    agentx_config({"concurrency_series": series((0, 1), (1800, 16))})


def test_trajectory_start_window(agentx_config):
    workload = agentx_config({
        "trajectory_start_min_ratio": 0.25, "trajectory_start_max_ratio": 1,
    }).benchmark.workload
    assert (workload.trajectory_start_min_ratio, workload.trajectory_start_max_ratio) == (0.25, 1)
    with pytest.raises(ValidationError, match="cannot exceed"):
        agentx_config({"trajectory_start_min_ratio": 0.5, "trajectory_start_max_ratio": 0.25})


# --- Benchmark --------------------------------------------------------------


def test_short_run_requires_unsafe_override(agentx_config):
    curve = {"concurrency_series": series((0, 1), (60, 4))}
    with pytest.raises(ValidationError, match="duration_s >= 900"):
        agentx_config(curve, duration_s=899)
    agentx_config(curve, duration_s=900)
    assert agentx_config({**curve, "unsafe_override": True}, duration_s=120)


@pytest.mark.parametrize("flag", [
    "--concurrency", "--concurrency-series", "--scenario", "--public-dataset",
    "--session-header", "--benchmark-duration", "--artifact-dir", "--random-seed",
])
def test_rejects_owned_flags_in_extra_args(agentx_config, flag):
    with pytest.raises(ValidationError, match=f"harness-owned options: {flag}"):
        agentx_config(extra_args=[flag, "x"])
    assert agentx_config(extra_args=["--request-timeout-seconds", "600"])


@pytest.mark.parametrize("section, key", [
    ("benchmark", "duration_s"), ("benchmark", "workload"), ("deployment", "model_id"),
])
def test_required_settings(section, key):
    raw = raw_config()
    del raw[section][key]
    with pytest.raises(ValidationError, match=key):
        ExperimentConfig.model_validate(raw)
    del raw[section]
    with pytest.raises(ValidationError, match=section):
        ExperimentConfig.model_validate(raw)


# --- Runtime and analysis ---------------------------------------------------


def test_runtime_and_analysis_defaults(agentx_config):
    config = agentx_config()
    assert config.runtime.aiperf_source == AIPERF_SOURCE_DEFAULT
    assert "@" in AIPERF_SOURCE_DEFAULT  # pinned to a revision
    analysis = config.analysis
    assert (analysis.window_s, analysis.tail_window_s) == (5, 30)
    assert (analysis.status_interval_s, analysis.metrics_interval_s) == (1, 2)
    assert analysis.scheduler_trace_dir is None
    assert (analysis.ttft_slo_ms, analysis.itl_slo_ms, analysis.warmup_s) == (None, None, None)
    assert analysis.slo_percentile == 0.95


def test_rejects_bad_runtime_and_analysis(agentx_config):
    with pytest.raises(ValidationError, match="tail_window_s must be >= window_s"):
        agentx_config(analysis={"window_s": 10, "tail_window_s": 5})
    with pytest.raises(ValidationError, match="aiperf_source"):
        agentx_config(runtime={"aiperf_source": ""})
    assert agentx_config(
        runtime={"aiperf_source": "git+https://example.com/aiperf@abc123"}
    ).runtime.aiperf_source.endswith("@abc123")


# --- Strictness: options the harness no longer has --------------------------


@pytest.mark.parametrize("path, key, value", [
    ((), "notes", "x"),
    (("benchmark",), "mode", "agentx_concurrency_ramp"),
    (("benchmark",), "levels", [1, 2]),
    (("benchmark",), "rate_series", [{"time_s": 0, "qps": 8}]),
    (("benchmark",), "warmup", {"enabled": True}),
    (("benchmark",), "fail_fast", True),
    (("benchmark", "workload"), "type", "agentx"),
    (("benchmark", "workload"), "target_concurrency", 16),
    (("benchmark", "workload"), "concurrency_ramp_duration_s", 1200),
    (("benchmark", "workload"), "input_tokens", 8000),
    (("deployment",), "direct_streaming", True),
    (("deployment", "routing"), "direct_streaming", False),
    (("deployment", "routing"), "forward_request_body", True),
    (("deployment", "autoscaling", "policy"), "advisory", True),
    (("runtime",), "generator", "aiperf"),
    (("analysis",), "prometheus_enabled", True),
    (("analysis",), "generate_plots", True),
    (("analysis",), "scheduler_trace_backend", "injected"),
])
def test_unknown_keys_are_errors(path, key, value):
    raw = raw_config(deployment={"routing": {}}, analysis={}, runtime={})
    raw["deployment"]["autoscaling"]["policy"] = {}
    section = raw
    for name in path:
        section = section[name]
    section[key] = value
    with pytest.raises(ValidationError, match=key):
        ExperimentConfig.model_validate(raw)


# --- Files ------------------------------------------------------------------


def test_load_config_is_strict_and_wants_a_mapping(tmp_path):
    path = tmp_path / "experiment.yaml"
    path.write_text(yaml.safe_dump(raw_config()))
    assert load_config(path) == ExperimentConfig.model_validate(raw_config())
    path.write_text(yaml.safe_dump(raw_config(fail_fast=True)))
    with pytest.raises(ValidationError, match="fail_fast"):
        load_config(path)
    path.write_text("- not\n- a mapping\n")
    with pytest.raises(ValueError, match="must contain a YAML mapping"):
        load_config(path)


def test_resolved_config_round_trips(agentx_config, tmp_path):
    config = agentx_config(deployment=TOKEN_POLICY, extra_args=["--request-timeout-seconds", "600"])
    assert ExperimentConfig.model_validate(config.resolved_dict()) == config
    write_config(config, tmp_path / "resolved.yaml")
    assert load_config(tmp_path / "resolved.yaml") == config
    assert load_saved_config(tmp_path / "resolved.yaml") == config


def _saved(tmp_path, raw):
    path = tmp_path / "resolved.yaml"
    path.write_text(yaml.safe_dump(raw))
    return path


def test_saved_config_drops_retired_keys(tmp_path, capsys):
    """A resolved.yaml written before the harness was reduced still loads."""
    raw = raw_config(
        deployment={**KV_AWARE, "engine": {"max_model_len": 262_144, "language_model_only": True}},
        analysis={"window_s": 10, "prometheus_enabled": True, "generate_plots": True,
                  "diagnostic_plots": True, "record_router_estimates": True},
        runtime={"generator": "aiperf"},
        mode="agentx_concurrency_ramp", fail_fast=True, streaming=True,
        use_server_token_count=True, warmup={"enabled": False},
    )
    raw["benchmark"]["workload"].update(
        {"type": "agentx", "target_concurrency": 16, "concurrency_ramp_duration_s": None})
    raw["deployment"]["routing"].update({"direct_streaming": True, "forward_request_body": True})
    raw["deployment"]["autoscaling"]["look_back_period_s"] = 60
    config = load_saved_config(_saved(tmp_path, raw))
    assert "ignored 15 retired keys" in capsys.readouterr().err
    assert config.benchmark.workload.target_concurrency == 16
    assert config.deployment.routing.direct_streaming
    assert config.analysis.window_s == 10
    # Sections that forward unlisted keys keep them.
    assert config.deployment.engine.model_extra == {"language_model_only": True}
    assert config.deployment.autoscaling.model_extra == {"look_back_period_s": 60}

    capsys.readouterr()
    load_saved_config(_saved(tmp_path, raw_config()))
    assert capsys.readouterr().err == ""


@pytest.mark.parametrize("benchmark, workload", [
    ({"mode": "request_rate_series", "rate_series": [{"time_s": 0, "qps": 8}]},
     {"type": "synthetic", "input_tokens": 8000}),
    ({"mode": "concurrency", "levels": [1, 2]}, {"type": "synthetic"}),
    ({"mode": "agentx_fixed_concurrency"}, {"target_concurrency": 8}),
    # A linear ramp recorded without a curve.
    ({"mode": "agentx_concurrency_ramp"},
     {"target_concurrency": 16, "concurrency_ramp_duration_s": 1200}),
])
def test_saved_config_rejects_other_workloads(tmp_path, benchmark, workload):
    raw = raw_config(**benchmark)
    raw["benchmark"]["workload"] = workload
    with pytest.raises(ValueError, match="only AgentX session-curve runs"):
        load_saved_config(_saved(tmp_path, raw))


def test_saved_config_still_validates_what_it_keeps(tmp_path):
    raw = raw_config({"concurrency_series": series((0, 1), (5000, 16))})
    with pytest.raises(ValidationError, match="after duration_s"):
        load_saved_config(_saved(tmp_path, raw))


# --- Checked-in experiments -------------------------------------------------


def test_expected_experiments_are_checked_in():
    assert [path.stem for path in EXPERIMENTS] == [
        "b200_default_target12_64_sessions", "b200_default_target32_64_sessions",
        "b200_token_100_sessions", "b200_token_32_sessions", "b200_token_64_sessions",
        "smoke",
    ]


@pytest.mark.parametrize("path", EXPERIMENTS, ids=lambda path: path.stem)
def test_checked_in_experiments_are_valid(path):
    config = load_config(path)
    assert config.deployment.routing.policy == "kv_aware"
    assert config.deployment.autoscaling.policy.name in ("default", "token")
    curve = config.benchmark.workload.concurrency_series
    # Each curve spans the whole load period and ramps back down.
    assert curve[-1].time_s == config.benchmark.duration_s
    assert curve[-1].sessions < config.benchmark.workload.target_concurrency
    assert config.analysis.ttft_slo_ms and config.analysis.itl_slo_ms


def _without(config: ExperimentConfig, *paths: str) -> dict:
    resolved = config.resolved_dict()
    for path in paths:
        *parents, leaf = path.split(".")
        section = resolved
        for name in parents:
            section = section[name]
        del section[leaf]
    return resolved


@pytest.mark.parametrize("other", ["b200_default_target12_64_sessions",
                                   "b200_default_target32_64_sessions"])
def test_policy_comparison_differs_only_in_autoscaling(other):
    """The default-policy runs replay the token run's workload on its engine."""
    token = load_config(REPO / "experiments/b200_token_64_sessions.yaml")
    default = load_config(REPO / f"experiments/{other}.yaml")
    ignored = ("name", "deployment.autoscaling.policy",
               "deployment.autoscaling.target_ongoing_requests")
    assert _without(default, *ignored) == _without(token, *ignored)
    assert default.deployment.autoscaling.policy.name == "default"


@pytest.mark.parametrize("sessions", [32, 100])
def test_token_runs_differ_only_in_the_curve(sessions):
    base = load_config(REPO / "experiments/b200_token_64_sessions.yaml")
    other = load_config(REPO / f"experiments/b200_token_{sessions}_sessions.yaml")
    ignored = ("name", "benchmark.duration_s", "benchmark.workload.concurrency_series")
    assert _without(other, *ignored) == _without(base, *ignored)
    assert other.benchmark.workload.target_concurrency == sessions
