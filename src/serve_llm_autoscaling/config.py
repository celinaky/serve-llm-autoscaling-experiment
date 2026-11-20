"""Experiment configuration: a Ray Serve LLM application plus an AgentX workload.

An experiment YAML has four sections:

``deployment``  the Serve LLM application: model, engine, routing, autoscaling
``benchmark``   the AgentX session-concurrency curve AIPerf replays against it
``runtime``     endpoint, AIPerf build, timeouts, results directory
``analysis``    sampling intervals and the SLO targets the run is scored against
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

# AIPerf's artifact directory below ``<run>/benchmark/``.
WORKLOAD_DIR = "agentx-concurrency-ramp"

# The inferencex-agentx-mvp scenario rejects shorter runs unless overridden.
AGENTX_MIN_DURATION_S = 900

# Options the harness always sets on the AIPerf command line.
OWNED_ARGS = {
    "--model", "--model-names", "--tokenizer", "--url", "--streaming",
    "--random-seed", "--benchmark-duration", "--benchmark-grace-period",
    "--artifact-dir", "--output-artifact-dir", "--concurrency",
    "--scenario", "--public-dataset", "--concurrency-series",
    "--trajectory-start-min-ratio", "--trajectory-start-max-ratio",
    "--unsafe-override", "--session-header",
}

# HTTP header that carries session affinity; Serve's default session-id header.
SESSION_HEADER = "X-Session-ID"


class EngineConfig(BaseModel):
    """vLLM engine settings forwarded to Ray Serve LLM.

    Only keys the experiment sets are forwarded; anything left unset takes
    vLLM's own default (which for ``max_num_seqs`` and
    ``max_num_batched_tokens`` depends on the GPU). Keys not listed here are
    forwarded unchanged.
    """

    model_config = ConfigDict(extra="allow")

    tensor_parallel_size: int | None = Field(default=None, ge=1)
    max_model_len: int | None = Field(default=None, ge=1)
    gpu_memory_utilization: float | None = Field(default=None, gt=0, le=1)
    max_num_seqs: int | None = Field(default=None, ge=1)
    max_num_batched_tokens: int | None = Field(default=None, ge=1)
    enable_prefix_caching: bool | None = None
    # Host RAM per replica in GiB, summed across TP ranks. None disables
    # offloading; this is independent of GPU type and HBM capacity.
    kv_offloading_size: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    kv_offloading_backend: Literal["native", "lmcache"] | None = None
    # Reports usage.prompt_tokens_details.cached_tokens so AIPerf can record
    # prefix-cache hits. vLLM only computes the count with prefix caching on.
    enable_prompt_tokens_details: bool = True
    kv_cache_dtype: str | None = None

    @model_validator(mode="after")
    def validate_kv_offloading(self) -> "EngineConfig":
        if self.kv_offloading_size is None:
            return self
        if (self.model_extra or {}).get("kv_transfer_config") is not None:
            raise ValueError(
                "kv_offloading_size cannot be combined with kv_transfer_config: "
                "vLLM's offloading settings overwrite the selected connector"
            )
        if self.kv_offloading_backend in (None, "native"):
            if self.enable_prefix_caching is not True:
                raise ValueError(
                    "native KV offloading requires enable_prefix_caching: true"
                )
        return self


# Built-in autoscaling policies -> (Ray Serve policy_function, scope). The
# policy is always passed explicitly so the deployed config names what
# decided the run.
AUTOSCALING_POLICIES = {
    "default": ("ray.serve.autoscaling_policy:default_autoscaling_policy", "deployment"),
    "token": (
        "ray.llm._internal.serve.routing_policies.kv_aware.autoscaling_policy"
        ":LLMTokenAutoscalingPolicy",
        "application",
    ),
}
# kwargs each built-in policy's constructor requires.
REQUIRED_POLICY_KWARGS = {
    "default": set(),
    "token": {"target_prefill_tokens_per_replica"},
}


class AutoscalingPolicyConfig(BaseModel):
    """The Serve autoscaling policy that decides the replica count.

    ``default`` is Serve's ongoing-requests policy. ``token`` scales the LLM
    server on the look-back mean of outstanding uncached prefill tokens that
    the KV-aware router replicas report, so it requires
    ``routing.policy: kv_aware``.

    To evaluate a new policy, set ``policy_function`` to its
    ``module:attribute`` import path (it must be importable by the Serve
    controller); ``name`` is then only a label. ``scope`` says how Serve
    calls it: a ``deployment`` policy receives the LLM server's
    ``AutoscalingContext``; an ``application`` policy receives every
    deployment's context (including the router's reported metrics) and
    returns a replica count per deployment. ``kwargs`` go to the policy
    unchanged. Serve applies the delays and replica bounds under
    ``autoscaling`` to either kind.
    """

    model_config = ConfigDict(extra="forbid")

    name: str = "default"
    policy_function: str | None = None
    scope: Literal["deployment", "application"] | None = None
    kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_policy(self) -> "AutoscalingPolicyConfig":
        builtin = AUTOSCALING_POLICIES.get(self.name)
        if self.policy_function is None:
            if builtin is None:
                raise ValueError(
                    f"unknown autoscaling policy {self.name!r}; use one of "
                    f"{sorted(AUTOSCALING_POLICIES)} or set policy_function"
                )
            self.policy_function = builtin[0]
        elif ":" not in self.policy_function:
            raise ValueError("policy_function must be a 'module:attribute' import path")
        is_builtin = builtin is not None and self.policy_function == builtin[0]
        if self.scope is None:
            self.scope = builtin[1] if is_builtin else "deployment"
        elif is_builtin and self.scope != builtin[1]:
            raise ValueError(f"autoscaling policy {self.name} is a {builtin[1]}-scope policy")
        if is_builtin and self.name == "default" and self.kwargs:
            raise ValueError("autoscaling policy default takes no kwargs")
        missing = REQUIRED_POLICY_KWARGS.get(self.name, set()) - set(self.kwargs)
        if missing and is_builtin:
            raise ValueError(
                f"autoscaling policy {self.name} requires kwargs {sorted(missing)}"
            )
        return self


class AutoscalingConfig(BaseModel):
    """Serve autoscaling settings; only ``policy`` is always sent.

    Every other key is forwarded only when the experiment sets it, so unset
    keys take Ray Serve's defaults (one replica, ``target_ongoing_requests``
    from Ray Serve LLM, 30s up / 600s down delays). Keys not listed here
    (``look_back_period_s``, ``upscaling_factor``, ...) are forwarded unchanged.
    """

    model_config = ConfigDict(extra="allow")

    policy: AutoscalingPolicyConfig = Field(default_factory=AutoscalingPolicyConfig)
    min_replicas: int | None = Field(default=None, ge=0)
    initial_replicas: int | None = Field(default=None, ge=0)
    max_replicas: int | None = Field(default=None, ge=1)
    target_ongoing_requests: float | None = Field(default=None, gt=0)
    upscale_delay_s: float | None = Field(default=None, ge=0)
    downscale_delay_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_bounds(self) -> "AutoscalingConfig":
        low = 0 if self.min_replicas is None else self.min_replicas
        high = self.max_replicas
        if high is not None and low > high:
            raise ValueError("min_replicas must be <= max_replicas")
        if self.initial_replicas is not None and not (
            low <= self.initial_replicas <= (self.initial_replicas if high is None else high)
        ):
            raise ValueError(
                "initial_replicas must be between min_replicas and max_replicas"
            )
        return self


KV_AWARE_ROUTER = "ray.serve.llm.request_router.KVAwareRouter"


class RoutingConfig(BaseModel):
    """Replica selection and the ingress topology in front of LLMServer.

    ``policy: kv_aware`` deploys Ray's KVAwareRouter behind HAProxy with
    direct streaming and request-body forwarding, which it requires.
    ``policy: null`` sends no router config: Serve picks its own router
    behind the standard OpenAiIngress.
    """

    model_config = ConfigDict(extra="forbid")

    policy: Literal["kv_aware"] | None = None
    # HAProxy's per-buffer cap on the body forwarded to the router; longer
    # bodies are truncated and the router falls back to load balancing.
    haproxy_request_buffer_bytes: int | None = Field(default=None, ge=1024)
    # After each HAProxy reload (every replica change), the old HAProxy process
    # is killed this many seconds later, cutting any stream still open on it.
    # Serve's default is 120s; raise it when requests can wait minutes.
    haproxy_hard_stop_after_s: int | None = Field(default=None, ge=1)
    request_router_kwargs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def validate_haproxy(self) -> "RoutingConfig":
        if self.policy is None and (
            self.haproxy_request_buffer_bytes is not None
            or self.haproxy_hard_stop_after_s is not None
            or self.request_router_kwargs
        ):
            raise ValueError(
                "haproxy_* settings and request_router_kwargs require policy: kv_aware"
            )
        return self

    @property
    def direct_streaming(self) -> bool:
        return self.policy == "kv_aware"

    @property
    def router_class(self) -> str | None:
        return KV_AWARE_ROUTER if self.policy == "kv_aware" else None

    def environment(self) -> dict[str, str]:
        """Ray switches for this topology, needed by the driver and controller."""
        if not self.direct_streaming:
            return {}
        env = {
            "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
            # LLMRouter is an ingress request router, which only HAProxy consults.
            "RAY_SERVE_ENABLE_HA_PROXY": "1",
            "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
        }
        if self.haproxy_request_buffer_bytes is not None:
            env["RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE"] = str(
                self.haproxy_request_buffer_bytes
            )
        if self.haproxy_hard_stop_after_s is not None:
            env["RAY_SERVE_HAPROXY_HARD_STOP_AFTER_S"] = str(self.haproxy_hard_stop_after_s)
        return env


class DeploymentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    application_name: str = "ttft-benchmark"
    model_id: str
    model_source: str | None = None
    tokenizer: str | None = None
    accelerator_type: str | None = None
    route_prefix: str = "/"
    routing: RoutingConfig = Field(default_factory=RoutingConfig)
    engine: EngineConfig = Field(default_factory=EngineConfig)
    autoscaling: AutoscalingConfig = Field(default_factory=AutoscalingConfig)
    # None leaves Ray Serve LLM's default (1e9, effectively unlimited) in
    # place; set it only to cap in-flight requests per replica.
    max_ongoing_requests: int | None = Field(default=None, ge=1)
    # Forwarded only when set; Ray Serve LLM's defaults apply otherwise
    # (for example 50ms of stream batching).
    experimental_configs: dict[str, Any] = Field(default_factory=dict)

    @model_validator(mode="after")
    def fill_defaults(self) -> "DeploymentConfig":
        if self.model_source is None:
            self.model_source = self.model_id
        if self.tokenizer is None:
            self.tokenizer = self.model_source
        return self


class ConcurrencyPoint(BaseModel):
    """One control point of the session-concurrency curve."""

    model_config = ConfigDict(extra="forbid")

    time_s: float = Field(ge=0)
    sessions: int = Field(ge=1, strict=True)


class WorkloadConfig(BaseModel):
    """AgentX trace replay; concurrency counts live session trees, not requests.

    ``concurrency_series`` is a piecewise-linear curve of concurrent sessions
    over time, so one list describes ramp up, hold and ramp down. When the
    curve falls, AIPerf lowers the session limit and retires live sessions at
    their next turn boundary.
    """

    model_config = ConfigDict(extra="forbid")

    public_dataset: str = "semianalysis_cc_traces_weka_062126_256k"
    concurrency_series: list[ConcurrencyPoint] = Field(min_length=2)
    # Window of each trace's recorded duration that AIPerf samples a session's
    # start instant from. Turns before it are replayed as warmup to prime the
    # KV cache. 0/0 starts every trajectory cold at turn zero.
    trajectory_start_min_ratio: float = Field(default=0.0, ge=0, le=1)
    trajectory_start_max_ratio: float = Field(default=0.0, ge=0, le=1)
    seed: int = 42
    # Allows runs shorter than AGENTX_MIN_DURATION_S (development only).
    unsafe_override: bool = False

    @model_validator(mode="after")
    def validate_workload(self) -> "WorkloadConfig":
        if self.trajectory_start_min_ratio > self.trajectory_start_max_ratio:
            raise ValueError(
                "trajectory_start_min_ratio cannot exceed trajectory_start_max_ratio"
            )
        times = [point.time_s for point in self.concurrency_series]
        if times[0] != 0:
            raise ValueError("concurrency_series must start at time_s 0")
        if any(later <= earlier for earlier, later in zip(times, times[1:])):
            raise ValueError("concurrency_series time_s values must be strictly increasing")
        return self

    @property
    def target_concurrency(self) -> int:
        """Peak of the curve; AIPerf's ``--concurrency``."""
        return max(point.sessions for point in self.concurrency_series)


class BenchmarkConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    # How long AIPerf issues load. It then drains in-flight requests for up to
    # ``grace_period_s`` and exits.
    duration_s: float = Field(gt=0)
    grace_period_s: float = Field(default=30, ge=0)
    # Idle telemetry after AIPerf exits, to watch recovery and downscaling
    # with no requests left in flight.
    post_load_observation_s: float = Field(default=0, ge=0)
    workload: WorkloadConfig
    # Extra AIPerf command-line arguments, appended verbatim.
    extra_args: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_benchmark(self) -> "BenchmarkConfig":
        workload = self.workload
        end = workload.concurrency_series[-1].time_s
        if end > self.duration_s:
            raise ValueError(
                f"concurrency_series ends at {end}s, after duration_s {self.duration_s}"
            )
        if self.duration_s < AGENTX_MIN_DURATION_S and not workload.unsafe_override:
            raise ValueError(
                f"AgentX runs require duration_s >= {AGENTX_MIN_DURATION_S}; set "
                "workload.unsafe_override: true for a shorter development run"
            )
        conflicts = sorted(set(self.extra_args) & OWNED_ARGS)
        if conflicts:
            raise ValueError(
                "extra_args cannot override harness-owned options: " + ", ".join(conflicts)
            )
        return self


# The AIPerf build the harness runs through ``uvx --from``: the workday-curve
# fork, whose ``--concurrency-series`` can lower session concurrency.
AIPERF_SOURCE_DEFAULT = (
    "git+https://github.com/celinaky/aiperf@1c61dfbb7644b26c70ce21163f0b4cda893438b0"
)


class RuntimeConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    endpoint_url: str = "http://127.0.0.1:8000"
    # Anything ``uvx --from`` accepts: a pinned release, a local checkout, or
    # ``git+https://...@<sha>``. It must support ``--concurrency-series``.
    aiperf_source: str = Field(default=AIPERF_SOURCE_DEFAULT, min_length=1)
    readiness_timeout_s: float = Field(default=1800, gt=0)
    readiness_poll_s: float = Field(default=2, gt=0)
    keep_deployment: bool = False
    results_dir: Path = Path("runs")


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_s: float = Field(default=5, gt=0)
    # Rolling window for tail latency and window-mean SLO attainment; smooths
    # sparse request windows.
    tail_window_s: float = Field(default=30, gt=0)
    status_interval_s: float = Field(default=1, gt=0)
    metrics_interval_s: float = Field(default=2, gt=0)
    # Debug log: save each replica's GPU KV-cache block store/removal events
    # to telemetry/gpu_kv_events.jsonl (kv_aware routing only). Nothing in
    # the analysis reads it. Uses the endpoints the KV-aware router already
    # consumes; no GPU workload or connector change.
    record_gpu_kv_events: bool = True
    # Base directory for per-run vLLM scheduler traces. Use shared storage
    # when replicas can run on other nodes; by default traces are written to
    # ``<run>/telemetry/scheduler-traces``.
    scheduler_trace_dir: Path | None = None
    # Latency targets the run is scored against.
    ttft_slo_ms: float | None = Field(default=None, gt=0)
    itl_slo_ms: float | None = Field(default=None, gt=0)
    # The SLO holds when this percentile of requests is within the target:
    # 0.95 means "p95 TTFT <= ttft_slo_ms", i.e. 95% of requests within it.
    slo_percentile: float = Field(default=0.95, gt=0, lt=1)
    # Seconds of load excluded from the steady-state score. Use 0 to score
    # the whole curve; by default the first 120s are excluded.
    warmup_s: float | None = Field(default=None, ge=0)

    @model_validator(mode="after")
    def validate_tail_window(self) -> "AnalysisConfig":
        if self.tail_window_s < self.window_s:
            raise ValueError("tail_window_s must be >= window_s")
        return self


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    deployment: DeploymentConfig
    benchmark: BenchmarkConfig
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    analysis: AnalysisConfig = Field(default_factory=AnalysisConfig)

    @model_validator(mode="after")
    def validate_token_policy_routing(self) -> "ExperimentConfig":
        # Engine replicas relay the KV-aware router's work snapshot; any
        # other router reports nothing and the policy holds the replica count.
        if (
            self.deployment.autoscaling.policy.name == "token"
            and self.deployment.routing.policy != "kv_aware"
        ):
            raise ValueError("autoscaling policy token requires routing policy kv_aware")
        return self

    def resolved_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json")


def _read_yaml(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        raw = yaml.safe_load(fh)
    if not isinstance(raw, dict):
        raise ValueError(f"{path} must contain a YAML mapping")
    return raw


def load_config(path: str | Path) -> ExperimentConfig:
    """Load an experiment YAML strictly: unknown keys are errors."""
    return ExperimentConfig.model_validate(_read_yaml(Path(path)))


def _drop_unknown(model: type[BaseModel], raw: Any, where: str, dropped: list[str]) -> Any:
    """Remove keys ``model`` does not declare, recursing into nested models."""
    if not isinstance(raw, dict):
        return raw
    allow_extra = model.model_config.get("extra") == "allow"
    result = {}
    for key, value in raw.items():
        field = model.model_fields.get(key)
        if field is None:
            if allow_extra:
                result[key] = value
            else:
                dropped.append(f"{where}{key}")
            continue
        nested = field.annotation
        if isinstance(nested, type) and issubclass(nested, BaseModel):
            value = _drop_unknown(nested, value, f"{where}{key}.", dropped)
        result[key] = value
    return result


def load_saved_config(path: str | Path) -> ExperimentConfig:
    """Load a run directory's ``resolved.yaml``, tolerating retired options.

    Runs recorded before the harness was reduced to AgentX session curves
    list options that no longer exist. They are dropped with a note so those
    runs can still be re-analyzed; a run of a retired workload is an error.
    """
    path = Path(path)
    raw = _read_yaml(path)
    benchmark = raw.get("benchmark") or {}
    mode = benchmark.get("mode", "agentx_concurrency_ramp")
    workload = benchmark.get("workload") or {}
    if mode != "agentx_concurrency_ramp" or not workload.get("concurrency_series"):
        raise ValueError(
            f"{path} records a {mode!r} run without a concurrency_series; only "
            "AgentX session-curve runs can be analyzed by this harness version"
        )
    dropped: list[str] = []
    config = ExperimentConfig.model_validate(
        _drop_unknown(ExperimentConfig, raw, "", dropped)
    )
    if dropped:
        print(f"note: {path} predates the current config schema; ignored "
              f"{len(dropped)} retired keys", file=sys.stderr)
    return config


def write_config(config: ExperimentConfig, path: Path) -> None:
    with path.open("w") as fh:
        yaml.safe_dump(config.resolved_dict(), fh, sort_keys=False)
