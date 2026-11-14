# Serve LLM Autoscaling Harness

An experiment harness for benchmarking Ray Serve LLM deployments and
autoscaling policies in a dedicated Anyscale workspace.

## Workflow

The harness deploys a controlled Ray Serve LLM application, waits for its
OpenAI-compatible endpoint, generates traffic with AIPerf, saves the results,
and tears Serve down. It supports two kinds of load tests:

- **Sweeps** run a separate benchmark at each configured concurrency or request
  rate. Use these to characterize steady-state performance and locate the
  system's capacity limit.
- **Request-rate series** run one uninterrupted benchmark while the offered
  request rate changes over time. Use these to study autoscaling, overload, and
  recovery behavior.
- **AgentX concurrency ramps** replay recorded agentic coding sessions in one
  uninterrupted benchmark while the number of concurrent sessions ramps up.
  Use these to study autoscaling under a realistic multi-turn workload.
- **AgentX fixed concurrency** replays the same sessions at a constant number
  of concurrent sessions. Use it on a fixed replica count to calibrate a TTFT
  SLO before running autoscaling experiments.

The checked-in baseline uses one real `Qwen/Qwen3-0.6B-FP8` replica and a
synthetic prefill-heavy workload (8,000 input tokens and 50 output tokens).

## Setup in an Anyscale workspace

Run the harness on the Anyscale image's Python so the Ray driver matches the
workers. AIPerf 0.12 runs through `uvx` with an isolated Python 3.11 environment
because its dependencies and Python requirement may differ from the image's.

```bash
pip install -e ".[dev]"
pip install "ai-dynamo>=1.4.0" scikit-learn
autoscale-harness check
```

The second `pip install` is for the cluster rather than the harness:
`ai-dynamo` backs `kv_aware` routing (see [Routing](#routing)) and
`scikit-learn` backs the `sla` autoscaling policy (see
[Autoscaling policy](#autoscaling-policy)). Both run in Ray Serve processes,
so they must be installed on every node; in a workspace, a `pip install` is
added to the workspace dependencies, which Ray installs for every job.

## Run

Validate the experiment without touching the cluster:

```bash
autoscale-harness validate experiments/baseline.yaml
```

Run the complete lifecycle:

```bash
autoscale-harness run experiments/baseline.yaml
```

If the dedicated workspace already has a Serve application, explicitly allow
replacement:

```bash
autoscale-harness run experiments/baseline.yaml --replace
```

Keep the deployment alive for debugging:

```bash
autoscale-harness run experiments/baseline.yaml --keep-deployment
```

## Run changing traffic over time

Set `benchmark.mode` to `request_rate_series` when the request rate needs to
change during one experiment. Unlike a request-rate sweep, this starts only one
AIPerf process, so requests and server state continue across every rate change.

Describe the load curve with `rate_series`. Each point contains:

- `time_s`: seconds from the start of AIPerf's measured profiling period
- `qps`: the offered request rate at that time

AIPerf linearly interpolates between consecutive points. A pair of points with
the same QPS creates a plateau; two points close together create an approximate
step. For example, this configuration sends 8 requests/second for 60 seconds,
steps up to 18 requests/second, stays there until 180 seconds, and then returns
to 8 requests/second for the remainder of the 300-second run:

```yaml
benchmark:
  mode: request_rate_series
  duration_s: 300
  arrival_pattern: poisson   # or constant, or gamma with arrival_smoothness
  rate_series:
    - {time_s: 0, qps: 8}
    - {time_s: 60, qps: 8}
    - {time_s: 60.1, qps: 18}
    - {time_s: 180, qps: 18}
    - {time_s: 180.1, qps: 8}
    - {time_s: 300, qps: 8}
```

The series must contain at least two points. The first must have `time_s: 0`,
times must be strictly increasing, all QPS values must be positive, and the
last point cannot be later than `duration_s`. The `levels` setting used by
static sweeps is ignored in this mode.

`arrival_pattern` controls how arrivals are distributed around the
instantaneous QPS target:

- `constant` spaces requests evenly and is useful for deterministic tests.
- `poisson` introduces random variation and is the default.
- `gamma` supports configurable burstiness through `arrival_smoothness`.

Start with the short 1 -> 2 -> 1 requests/second smoke test:

```bash
autoscale-harness run experiments/smoke_rate_series.yaml
```

Then run the full 8 -> 18 -> 8 requests/second experiment, first against one
fixed replica (the control) and then with autoscaling enabled (1-4 replicas,
`target_ongoing_requests: 4`, `downscale_delay_s: 120` so recovery shows during
the drain). Both use the same model, workload and load curve:

```bash
autoscale-harness run experiments/step_rate.yaml
autoscale-harness run experiments/step_rate_autoscaling.yaml
```

The harness translates the YAML curve into the JSON format expected by AIPerf.
Caution (AIPerf 0.13.0): a linearly interpolated *ramp* segment was delivered
at several times its configured rate in every run that used one, while flat
segments were delivered correctly. Until that is understood, express a rising
load as steps (two points 0.1s apart, as in `experiments/step_rate.yaml`) and
check `analysis/request_timeseries.json`'s `offered_rps` against the curve.

## Ramp AgentX session concurrency

Set `benchmark.mode` to `agentx_concurrency_ramp` to replay the public
AgentX/Weka agentic-coding traces with AIPerf's `inferencex-agentx-mvp`
scenario (see [InferenceX AgentX](https://inferencex.semianalysis.com/agentx)).
One AIPerf process runs the whole experiment, and session concurrency rises
linearly from 1 to `target_concurrency` over `concurrency_ramp_duration_s`,
then holds at the target until `duration_s`.

**Prerequisite:** the default dataset,
`semianalysis_cc_traces_weka_062126_256k`, contains traces up to 256K tokens.
Use a model and deployment whose context window (`engine.max_model_len`) fits
them. The 10K-context Qwen3-0.6B experiments are not suitable.

`experiments/agentx_kv_aware_autoscaling.yaml` is a checked-in short
development run. It serves `Qwen/Qwen3.6-27B-FP8`, a dense hybrid-attention
model with a 262,144-token context, on one >=80GB GPU per replica,
autoscaling from 1 to 4 replicas behind KV-aware routing. It ramps to 32
sessions over 150s, issues load for 210s, drains for up to 30s, then observes
the idle deployment for 90s. It needs `ai-dynamo>=1.4.0` (see
[Routing](#routing)).

```bash
autoscale-harness run experiments/agentx_kv_aware_autoscaling.yaml
```

For a full-length run, adapt this example:

```yaml
deployment:
  model_id: <long-context model>
  engine:
    max_model_len: 262144      # must fit the dataset's traces
  # ...

benchmark:
  mode: agentx_concurrency_ramp
  duration_s: 1800
  grace_period_s: 300
  fail_fast: true
  use_server_token_count: true
  workload:
    type: agentx
    public_dataset: semianalysis_cc_traces_weka_062126_256k
    target_concurrency: 16
    concurrency_ramp_duration_s: 1200
    trajectory_start_min_ratio: 0
    trajectory_start_max_ratio: 0
    seed: 42
    unsafe_override: false
```

The harness runs the equivalent of:

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --concurrency 16 --concurrency-ramp-duration 1200 \
  --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0 \
  --benchmark-duration 1800 --benchmark-grace-period 300 --random-seed 42 \
  --streaming --use-server-token-count ...
```

How to read the load:

- **Concurrency counts live session trees, not requests.** A session tree is
  a root agent conversation plus any subagents it spawns. Subagent fan-out can
  put more requests in flight than the configured concurrency.
- **The ramp raises session concurrency from 1 to the target.** When a session
  tree drains, AgentX replaces it with a new trace. Faster configurations
  therefore complete more sessions and may see a somewhat different trace mix,
  so report throughput together with TTFT and ITL rather than on its own.
- **`trajectory_start_min_ratio` / `trajectory_start_max_ratio`** set the
  window, as a fraction of each trace's recorded duration, from which AIPerf
  samples a session's starting instant. Turns before that instant are
  replayed once during warmup to prime the KV cache, and measurement starts
  at the first turn after it. Both default to `0`, which starts every session
  cold at turn zero with no warmup. The AgentX scenario's own default is
  `0` / `1`: sessions start mid-conversation with their prefix cached.
- **Ramp-up only.** Ramping down, multiple cohorts and arbitrary concurrency
  curves are not supported yet.

The scenario owns ignore-EOS, cache busting and warmup, so the synthetic
settings (`levels`, `rate_series`, `arrival_pattern`, `arrival_smoothness`,
`warmup`) are rejected in this mode, as are harness-owned flags such as
`--scenario` or `--concurrency-ramp-duration` in `extra_args`. The ramp cannot
be longer than `duration_s`, and the scenario requires `duration_s >= 900`.
Set `unsafe_override: true` for a shorter development run; the harness then
passes `--unsafe-override` and AIPerf marks the result as not a valid
submission.

### Observed request rate

Because the configured load is concurrency, the request rate is an outcome,
derived from each request's timestamps rather than estimated from the
configuration (session lengths, think time, latency and subagent fan-out all
vary). Only profiling-phase records count; AgentX warmup is excluded.
Subagent requests are real requests and are included in every total.

| Quantity | Measured from |
| --- | --- |
| Offered QPS | credits issued (`credit_issued_ns`) per second |
| Started QPS | requests started (`request_start_ns`) per second |
| Successful / failed completed QPS | responses finished (`request_end_ns`) per second |
| Root-agent / subagent QPS | offered QPS split by `agent_depth` |

The run's summary row reports whole-run means over the profiling interval
(`duration_s` from the profiling start, so dataset preparation, deployment
readiness, warmup and the grace-period drain do not dilute them), beside
AIPerf's own `request_throughput`, which is kept as a separate field:

```json
{
  "mode": "agentx_concurrency_ramp",
  "level": 16,
  "target_concurrency": 16,
  "ramp_duration_s": 1200,
  "request_count": 1234,
  "failed_requests": 2,
  "mean_offered_qps": 0.82,
  "mean_started_qps": 0.81,
  "mean_successful_qps": 0.80,
  "aiperf_request_throughput": 0.80,
  "error": null
}
```

A difference between `mean_successful_qps` and `aiperf_request_throughput`
usually reflects how each treats the drain period and timestamp boundaries.

`analysis/concurrency_qps.json` puts the configured concurrency next to the
observed rates in each analysis window. The configured value is the ramp at
the window midpoint; the live session count can briefly differ because of
scheduling and tree turnover.

```json
{
  "window_start_s": 300,
  "window_end_s": 305,
  "configured_session_concurrency": 5.0,
  "offered_qps": 1.8,
  "started_qps": 1.8,
  "successful_completed_qps": 1.6,
  "failed_completed_qps": 0.0,
  "in_flight_at_end": 7,
  "offered_root_qps": 1.2,
  "offered_subagent_qps": 0.6,
  "requests_per_second_per_configured_session": 0.36
}
```

`requests_per_second_per_configured_session` is a diagnostic, not a
performance score: it varies with the trace mix, think time, latency and
subagent activity.

In the timeline's traffic panel the left axis shows observed QPS (offered,
started, completed, failed when nonzero, and the root/subagent split), and
the right axis shows the configured session concurrency in session trees. A
vertical line marks the end of the ramp.

For a first real run, use a small target (2-4 sessions) but keep at least 900
seconds. Check `stdout.log`/`stderr.log` for
`Starting session concurrency ramp: 1 → N`, check that the AgentX warmup
issues no cache-priming requests for the initial sessions, and check in
`profile_export.jsonl` that the first root conversations begin at turn zero.

## Calibrate a TTFT SLO with fixed AgentX concurrency

Set `benchmark.mode` to `agentx_fixed_concurrency` to hold AgentX session
concurrency at `target_concurrency` for the whole profiling duration. Unlike
the ramp, the harness does not pass `--concurrency-ramp-duration`, and
`concurrency_ramp_duration_s` is rejected rather than silently ignored. All
other AgentX rules apply: streaming is required, synthetic settings are
rejected, the scenario owns warmup, cache busting and ignore-EOS, and
`duration_s >= 900` unless `unsafe_override: true`.

`experiments/agentx_slo_baseline.yaml` runs the
`agentx_kv_aware_autoscaling.yaml` deployment (same model, GPU,
engine, KV-aware routing, direct streaming, body forwarding, HAProxy buffer,
dataset and seed) on exactly one replica (`min_replicas`, `initial_replicas`
and `max_replicas` all 1), with 8 AgentX sessions for 900s and up to 300s of
drain:

```bash
autoscale-harness validate experiments/agentx_slo_baseline.yaml
autoscale-harness run experiments/agentx_slo_baseline.yaml
```

The harness runs the equivalent of:

```bash
aiperf profile --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis_cc_traces_weka_062126_256k \
  --concurrency 8 \
  --trajectory-start-min-ratio 0 --trajectory-start-max-ratio 0 \
  --benchmark-duration 900 --benchmark-grace-period 300 --random-seed 42 \
  --streaming --use-server-token-count ...
```

Raw artifacts go to `benchmark/agentx-fixed-concurrency/`. The run gets the
same telemetry, lifecycle, request windows and timeline as the other
continuous modes. In the timeline and `analysis/concurrency_qps.json` the
configured session concurrency is 8 from profiling start until `duration_s`
and 0 afterward; there is no ramp-end marker. The summary row has
`mode: agentx_fixed_concurrency`, `level` and `target_concurrency` 8, the
harness-derived mean QPS and AIPerf's TTFT percentiles, and no
`ramp_duration_s`.

The run also writes `analysis/slo_calibration.json`:

```json
{
  "metric": "time_to_first_token",
  "unit": "ms",
  "successful_request_count": 123,
  "failed_request_count": 2,
  "ttft_sample_count": 123,
  "p50_ttft_ms": 1000,
  "p90_ttft_ms": 2400,
  "p99_ttft_ms": 5000,
  "recommended_ttft_slo_ms": 2400,
  "method": "p90 TTFT across successful profiling-phase requests"
}
```

Percentiles cover every successful profiling-phase request. Warmup records
are excluded. Failed requests are counted in `failed_request_count` but
contribute no TTFT samples. The percentiles use the same linear
interpolation as the request windows. The terminal run summary (and
`autoscale-harness analyze`) prints the recommendation:

```text
TTFT SLO:  recommended 2400.0ms (p90 of 123 successful requests; 2 failed). Copy into analysis.ttft_slo_ms: 2400
```

The harness does not edit other configs. To use the SLO, copy
`recommended_ttft_slo_ms` into the autoscaling experiment yourself:

```yaml
# experiments/agentx_kv_aware_autoscaling.yaml
analysis:
  ttft_slo_ms: 2400
```

If `failed_request_count` is not small, the replica was overloaded at this
concurrency, and the p90 describes a saturated system. Lower the concurrency
before you trust the number.

### Session concurrency, ongoing requests and batch size

These three numbers count different things and are not interchangeable:

- **AgentX session concurrency** (`target_concurrency`) is the number of live
  session trees AIPerf keeps active. Each tree alternates between in-flight
  requests and client-side think time, and subagents can add parallel
  requests.
- **Ongoing HTTP requests** (Serve's `ongoing_requests`, the quantity
  `target_ongoing_requests` controls) are the requests a replica is serving
  right now. With 8 sessions this can be below 8, when sessions are between
  turns, or above 8, when subagents fan out.
- **vLLM batch size** (bounded by `max_num_seqs`) is the number of sequences
  the engine schedules in one step, limited further by KV-cache capacity and
  `max_num_batched_tokens`.

> **Warning:** `target_ongoing_requests: 8` does **not** mean eight AgentX
> sessions per replica. The autoscaler counts in-flight HTTP requests. To
> relate the two, compare the baseline's Serve ongoing requests (pressure
> panel) and client in-flight counts with its configured session concurrency.

## Routing

`deployment.routing` selects how Serve picks an LLM replica and the ingress
topology in front of it:

```yaml
deployment:
  routing:
    policy: kv_aware                  # see the table below
    direct_streaming: true            # LLMRouter + HAProxy instead of OpenAiIngress
    forward_request_body: true        # let HAProxy pass the prompt to the router
    haproxy_request_buffer_bytes: 8388608
    request_router_kwargs: {}         # passed to the router class
```

| `policy` | Ray router class | Notes |
| --- | --- | --- |
| `power_of_two` | `PowerOfTwoChoicesRequestRouter` | Serve's default for the ingress topology |
| `round_robin` | `ray.serve.experimental.round_robin_router.RoundRobinRouter` | |
| `consistent_hash` | `ray.serve.experimental.consistent_hash_router.ConsistentHashRouter` | AIPerf sends `--session-header X-Session-ID`, so each session's turns pin to one replica |
| `prefix_cache_affinity` | `ray.serve.llm.request_router.PrefixCacheAffinityRouter` | Routes on the prompt text |
| `kv_aware` | `ray.serve.llm.request_router.KVAwareRouter` | Scores prompt/KV-cache overlap; needs direct streaming, body forwarding and `ai-dynamo>=1.4.0` |

With no `policy` set (the default), the harness sends no request router
config and Serve picks the router itself: power of two for the ingress
topology, round robin under direct streaming. Set a policy to pin one.

- **Direct streaming** makes `LLMServer` the ingress deployment and attaches
  `LLMRouter` as its ingress request router. HAProxy asks `LLMRouter` for a
  replica and streams the response straight from it. The harness sets
  `RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1` and `RAY_SERVE_ENABLE_HA_PROXY=1`
  in the driver and passes them to the Serve controller through
  `controller_options`. Because controller options apply only when the
  controller starts, the harness shuts down any idle Serve instance first.
- **Body forwarding** (`RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY=1`)
  gives body-aware routers (`prefix_cache_affinity`, `kv_aware`) the prompt;
  it is on by default for them under direct streaming. HAProxy truncates
  bodies longer than `haproxy_request_buffer_bytes` (Ray's default is 256KiB),
  and the router then falls back to load balancing. Long AgentX prompts need a
  larger buffer.
- `kv_aware` turns on direct streaming and body forwarding when they are left
  unset, and rejects an explicit `false`.

Before deploying, `run` and `deploy` check that the installed Ray provides the
router class and, for `kv_aware`, that `ai-dynamo>=1.4.0` and
`dynamo.llm.SelectionService` are importable. Without Dynamo, Ray would only
log a warning and silently load-balance. Install it on every node:

```bash
pip install "ai-dynamo>=1.4.0"
```

After the deployment is ready, the harness verifies the Serve topology:
`LLMRouter` without `OpenAiIngress` under direct streaming, and
`OpenAiIngress` without `LLMRouter` otherwise. A mismatch fails the run at
stage `topology`. The requested routing, the environment switches, the
resolved router class and the verified deployments are written to
`deployment/routing.json` and summarized under `routing` in `manifest.json`.

## Native CPU KV cache offloading

Enable vLLM's native CPU cache in any experiment with:

```yaml
deployment:
  routing:
    policy: kv_aware
  engine:
    enable_prefix_caching: true
    kv_offloading_backend: native
    kv_offloading_size: 64  # GiB of host RAM per replica, across all TP ranks
```

Omit `kv_offloading_size` (or set it to `null`) to disable offloading. Existing
experiments remain unchanged. Native offloading requires explicit
`enable_prefix_caching: true`; a positive finite size is required when set.
Do not combine the size shortcut with `kv_transfer_config` (including Mooncake
or MultiConnector): vLLM overwrites the connector selection. The harness rejects
that combination. Offloading works with other routing policies too; `kv_aware`
additionally tracks native CPU-resident blocks and discounts their routing
credit to account for reload cost. Ray configures the native connector/event
path for that router; no new package or standalone service is required.

`experiments/h100_native_cpu_offload.yaml` is a fixed two-replica AgentX trial
using the current H100 model/settings and 64 GiB per replica. Validate without
deploying or touching GPUs:

```bash
autoscale-harness validate experiments/h100_native_cpu_offload.yaml
```

Run it only once the experiment's GPUs are available. The harness replaces
existing Serve applications only with `--replace`; that option shuts Serve
down. The sample deliberately uses the existing benchmark app/route and must
not be launched alongside the active experiment. Keep the no-offload baseline's
model, routing, concurrency, seed and engine settings identical; remove just
the two `kv_offloading_*` keys for that comparison.

The same engine settings apply to H100 and B200 with a compatible Ray/vLLM
stack; change `accelerator_type` to the target GPU and tune the CPU budget per
node. 64 GiB is a starting budget, not a measured optimum: two replicas consume
up to 128 GiB of CPU cache, eight up to 512 GiB, plus other process/object-store
memory and transfer buffers. Budget for the maximum simultaneously live
replicas, including replacement overlap. No automatic sizing from GPU token
slots is performed, especially for hybrid attention/Mamba models. B200 runtime
compatibility and the model's actual offload/reload behavior require testing.

Telemetry retains external prefix hits/queries, native store/load bytes,
transfer size/time, lookup delays, allocation failures, CPU-cache usage, and
computed-prefill histograms when exported by the installed vLLM/Ray build.
Inspect `telemetry/serve_metrics.jsonl`; counter deltas and histogram sums can
be compared per replica. For exact computed-token counts, prefer
`prompt_tokens_by_source_total{source="local_compute"}` deltas: the installed
Ray exporter may omit the computed-prefill histogram for a replica, and its
histogram sums may be approximate. `request_success_total` counts completed
requests. A useful offload test needs GPU cache eviction followed
by reuse on the same engine: successful stores without subsequent loads do not
demonstrate a benefit. Compare TTFT, actual prefill computed, external hits,
and miss overhead. CPU cache is private to the engine and is lost on engine
restart/scale-down; cross-replica/restart persistence remains the Mooncake test.

See [Ray's offloading guide](https://docs.ray.io/en/master/serve/llm/user-guides/kv-cache-offloading.html)
for native backend and KV-aware routing behavior.

## Mooncake shared KV cache

Use `deployment.engine.shared_kv_cache` to replace the private native CPU
cache with an external Mooncake DRAM pool:

```yaml
deployment:
  engine:
    enable_prefix_caching: true
    shared_kv_cache:
      enabled: true
      config_path: /mnt/cluster_storage/mooncake-kv/requester.json
      cache_prefix: MODEL_AND_EXACT_REVISION/tp1/kv-fp8/experiment-name
      package: mooncake-transfer-engine-cuda13==0.3.14
      load_async: true
      save_decode_cache: true
      load_failure_policy: recompute
      health_url: http://HEAD_INTERNAL_IP:9300/health
      master_metrics_url: http://HEAD_INTERNAL_IP:9003/metrics
      client_metrics_url: http://HEAD_INTERNAL_IP:9300/metrics
```

Set `enabled: false` or omit the stanza to disable the shared tier. The
stanza is a harness option, not a vLLM engine argument: the harness translates
it to `MooncakeStoreConnector`, `kv_role: kv_both`, the namespace/options, and
the replica runtime environment. Remove `kv_offloading_backend`,
`kv_offloading_size`, and any explicit `kv_transfer_config` when enabling it;
those combinations are rejected instead of silently overwriting a connector.
Disabling shared caching alone leaves GPU prefix caching enabled. Choose the
native experiment YAML to switch back to private CPU offloading.

The requester JSON must specify `mode: standalone-store` and
`global_segment_size: 0`. Its path must be readable at the same absolute path
by every replica and vLLM worker. On Anyscale use shared storage for this small
configuration file; the live cache remains Mooncake-owned DRAM. Configure a
1 GiB requester `local_buffer_size` initially. Set one shared namespace for
compatible replicas, and pin the model/tokenizer revision and KV layout/dtype.
Use a fresh `cache_prefix` for a cold experiment; intentionally reusing it can
reuse data across separate deployments/runs.

The package is installed in each Ray replica/worker runtime environment, so
a driver-only virtualenv is insufficient. The pinned CUDA 13 wheel matches
this workspace. For a CUDA 12 image select the matching
`mooncake-transfer-engine` wheel instead; set `package: null` when the image
already supplies the compatible dependency. Ray and vLLM are not reinstalled.

Start the master and one resource-owning client independently of Serve. In
this workspace the isolated setup, service manager and cross-replica smoke
test are in `/home/ray/default/mooncake-kv`:

```bash
/home/ray/default/mooncake-kv/.venv/bin/python /home/ray/default/mooncake-kv/services.py start --pool-gib 256
autoscale-harness run experiments/h100_workday_curve_mooncake_scale_down.yaml
```

The 256 GiB allocation is one pool shared by all replicas, not 256 GiB per
replica. Harness preflight checks the requester topology, master connectivity
and optional storage-client health, then saves the effective requester JSON
and its hash in `deployment/shared_kv_cache.json`. Serve teardown does not stop
the standalone services, so the pool survives replica removal. Stopping the
storage process or Workspace loses the pool.

For long contexts over TCP, the service manager uses
`mooncake_master --default_kv_lease_ttl=120s`. The upstream 10-second default
expired during a measured 158K-token load. The connector defaults to
`load_failure_policy: recompute`, allowing vLLM to recompute after a failed
cache load instead of its default error finish. Neither setting makes a
transfer fast; failed operations and fallback work must still be measured.

The toggle does not change the selected router or its implementation. The
current KV-aware router still models replica-local residency; this integration
does not add shared-pool awareness. Correctness comes from the engine's own
Mooncake lookup, and a miss falls back to prefill. Inspect routing and actual
scheduler work separately when interpreting the token autoscaler.

`telemetry/serve_metrics.jsonl` includes per-replica Mooncake operation counts,
bytes, failure counts and latency. Optional master/client endpoints are saved
in `telemetry/mooncake_metrics.jsonl`; master occupancy/capacity and eviction
counters describe the external pool, rather than in-flight transfer workspace.
The ordinary automatic diagnostic plots and HTML gallery remain enabled.

The longer H100 experiment is
[`experiments/h100_workday_curve_mooncake_decay_long.yaml`](experiments/h100_workday_curve_mooncake_decay_long.yaml).
It adds one concurrency slot every 30 seconds to reach 32 sessions at 930s,
holds for 2790s, then lowers the limit at the same rate over 930s. The paired
control points compensate for AIPerf's nearest-integer interpolation. Limit
updates have polling tolerance; reducing the limit retires session trees at
turn boundaries rather than cancelling every in-flight request immediately.
Its upscale/downscale delays are 40s/250s, and it sets:

```yaml
deployment:
  routing:
    policy: kv_aware
    request_router_kwargs:
      prefill_decay_tokens_per_second: 10000
```

This option needs the local Ray branch's decay implementation. It estimates
remaining prefill using one FIFO service budget per engine replica, shared
among that replica's requests. It changes the token autoscaling signal; it
does not change Dynamo's selection load or add awareness of shared residency.
Zero disables decay. Keep the same rate and token target when interpreting
this experiment; the rate is an assumption, not measured GPU throughput.

The 16-session comparison rerun is
[`experiments/h100_workday_curve_mooncake_decay_400s.yaml`](experiments/h100_workday_curve_mooncake_decay_400s.yaml).
It keeps the original 1050-second up/hold workload and 40-second upscale delay,
adds 10,000 tokens/sec per-replica prefill estimate decay, and sets a 400-second
downscale delay. Its 600-second idle observation allows the longer downscale
delay to elapse. Each run uses a fresh cache namespace.

## Autoscaling policy

`deployment.autoscaling.policy` is the only autoscaling key the harness
always sends. Every other `deployment.autoscaling` key (`min_replicas`,
`max_replicas`, `target_ongoing_requests`, the delays, ...) and every
`deployment.engine` key except `enable_prompt_tokens_details` and the translated
`shared_kv_cache` setting is forwarded
only when the experiment sets it; unset keys take Ray Serve's, Ray Serve
LLM's or vLLM's own defaults. The same holds for `max_ongoing_requests`,
`experimental_configs` and `routing.policy`.

`deployment.autoscaling.policy` selects the Serve autoscaling policy that
decides `LLMServer`'s replica count:

```yaml
deployment:
  autoscaling:
    policy:
      name: sla            # default | sla
      kwargs:              # passed to the policy class unchanged
        ttft_ms: 300       # mean TTFT target
        itl_ms: 20         # mean inter-token latency target
        advisory: false    # the harness default for sla
    upscaling_factor: 1.0  # recommended for sla: keep its exact recommendation
    min_replicas: 1
    max_replicas: 4
```

| `name` | Ray `policy_function` | Notes |
| --- | --- | --- |
| `default` | `ray.serve.autoscaling_policy:default_autoscaling_policy` | Scales on ongoing requests per replica (`target_ongoing_requests`); takes no kwargs |
| `sla` | `ray.serve.experimental.llm_sla_autoscaling_policy:SLAOnlineAutoscalingPolicy` | Learns per-replica capacity from Prometheus and scales to meet `ttft_ms` and `itl_ms`; needs `scikit-learn` |
| `token` | `ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy` | Scales on the time-weighted mean over `upscale_lookback_s` of the KV-aware router's outstanding uncached prefill tokens divided by `target_prefill_tokens_per_replica`; requires `routing.policy: kv_aware` (`experiments/b200_workday_curve_prefill.yaml`) |

The policy is always passed explicitly, including `default`. For `sla`:

- `advisory` defaults to `false`. With `advisory: true` the policy only
  reports its recommendation (`serve_sla_autoscaling_recommended_replicas`)
  and Serve scales exactly as with `default`; set it explicitly to compare.
- `prometheus_address` defaults to the driver's `RAY_PROMETHEUS_HOST`, because
  the policy runs in the Serve controller.
- Until the policy has enough metrics and training coverage, it falls back to
  the default policy's decision. Its `sla_reason` state is logged by the Serve
  controller, which the harness tails into `telemetry/sla_policy_events.jsonl`.
  With the policy's defaults the model needs `min_training_rows` (30)
  observations taken `observation_window_s` (30s) apart while the topology has
  been stable for `topology_stabilization_s` (60s): about 15 minutes of
  cumulative stable time before it leaves `model_not_ready`. Size the load
  so the decisions you want to grade come after that, or lower those kwargs
  together.
- `analysis.ttft_slo_ms` and `analysis.itl_slo_ms` default to the policy's
  `ttft_ms` and `itl_ms`, and must match them if set, so the run is scored
  against the targets the policy was given. `experiments/b200_sla_smoke.yaml`
  checks the machinery at a load one replica handles; the
  `b200_workday_ramp_{default,sla,sla_advisory}.yaml` trio is the same
  40-minute AgentX ramp under each policy, scored on mean TTFT 3s and mean
  ITL 50ms.

Before deploying, `run` and `deploy` check that the installed Ray provides the
policy class and, for `sla`, that `scikit-learn` is installed and Prometheus
answers a query. After the deployment is ready, the harness reads the
autoscaling config Serve applied to every `LLMServer` deployment and fails at
stage `autoscaling_policy` unless its `policy_function` and `policy_kwargs`
match the request. The policy, its kwargs and the verified deployments are
written to `deployment/autoscaling.json`, summarized under `autoscaling` in
`manifest.json` and in the run summary, and `autoscaling_policy` is recorded in
the analysis summary. The SLA policy's diagnostic gauges (recommended
replicas, model readiness, training rows, predicted TTFT and ITL) are collected
with the other Prometheus metrics.

## Autoscaling telemetry and timeline

While a continuous run (request-rate series, AgentX ramp or fixed AgentX
concurrency) is in progress,
two background collectors sample the cluster from immediately before AIPerf
starts, through its grace/drain period, to the end of any post-load
observation:

- **Serve status** (public `serve.status()`, every `analysis.status_interval_s`,
  default 1s) records realized capacity: replica counts by state (`STARTING`,
  `RUNNING`, `STOPPING`, ...). A missing application is recorded as unavailable,
  never as zero replicas.
- **Prometheus metrics** (every `analysis.metrics_interval_s`, default 2s)
  record what the autoscaler and routers see: desired/target replicas, ongoing
  requests at replicas, router and handle queues, replica health and startup
  latency; the SLA policy's gauges (recommendation, model readiness, training
  rows, predicted TTFT/ITL); and per-replica vLLM engine metrics (TTFT and
  time-per-output-token histograms, running and waiting requests, prefix-cache
  hits), which are what the SLA policy learns from. The harness scrapes each
  Ray node's metrics export port directly; no Prometheus server is required.
- **SLA policy log** (`sla` policy only, every `analysis.status_interval_s`)
  tails the Serve controller log for the policy's state transitions
  (`model_not_ready`, `unsupported_workload`, `capacity`, `safe_scale_down`,
  `observed_slo_violation`, ...), which Serve exposes nowhere else, into
  `telemetry/sla_policy_events.jsonl`.

Both are stamped with the same epoch clock as AIPerf and aligned to AIPerf's
profiling start. Collection failures are recorded as error samples and never
stop the benchmark. Static sweeps collect no telemetry.

Set `benchmark.post_load_observation_s` to keep collecting after AIPerf exits,
with no requests in flight, to watch recovery and downscaling:

```text
0 ─── load issuance ─── duration_s ── drain (≤ grace_period_s) ── AIPerf exits ── idle observation ──
```

The harness records each boundary in epoch nanoseconds in
`benchmark/lifecycle.json`: AIPerf launch, profiling start, ramp end (AgentX ramp only),
load-issuance end (`profiling start + duration_s`), profiling-phase end (drain
complete), AIPerf exit, and observation end. Observation is skipped when a
`fail_fast` failure aborts the run, or when `benchmark` runs without
telemetry. Only continuous modes accept a non-zero value.

After the run, the harness derives the analysis from the raw artifacts:

- **Request windows** (default 5s, `analysis.window_s`): offered load by credit
  issue time, actual starts, successful and failed completions, in-flight
  requests at each boundary, client queue delay, and TTFT and ITL mean and
  percentiles grouped by request start. Windows continue until the last
  request finishes, or to the end of post-load observation if that is later.
- **Rolling latency**: TTFT and ITL mean and p99 over the preceding
  `analysis.tail_window_s` (default 30s), which is stable even when a 5s window
  has few samples and matches the SLA policy's observation window. Set
  `analysis.ttft_slo_ms` and `analysis.itl_slo_ms` to also report SLO
  attainment and the [autoscaling evaluation](#autoscaling-evaluation).
- **Serve-side latency**: vLLM's mean TTFT and ITL per scrape, as the ratio of
  the histogram counters' increases over the same rolling window summed across
  the deployment's replicas. This is the policy's view of latency; it excludes
  router and proxy queueing that the client sees.
- **Metrics inventory**: every Serve metric the workspace exported, with labels,
  source nodes and sample counts, plus recommended metrics that were absent.
- **Timeline** (`autoscaling_timeline.png`), five panels on one time axis,
  six with the SLA policy:
  1. Traffic: offered, started and completed QPS with the configured load:
     QPS for a request-rate series, session concurrency (on a second axis)
     for an AgentX ramp or fixed-concurrency run.
  2. TTFT and 3. ITL: p50/p90/p99 per window, the rolling mean and p99, the
     SLO, and shading on every window whose rolling mean exceeds it.
  4. Pressure: client in-flight requests and queue delay, Serve ongoing
     requests and router queue length, vLLM running and waiting requests.
  5. Autoscaling: desired, target and SLA-recommended replicas (metrics) with
     running, starting and stopping replicas (status) as step functions.
  6. SLA policy: predicted TTFT/ITL against the server (vLLM) and client
     means, when the model became ready, and each logged state transition.

  Vertical lines mark the load curve's control points, or the end of an
  AgentX ramp. Dotted lines mark the end of load issuance, draining, AIPerf's
  exit and observation. The drain period (requests still in flight) is shaded
  orange, and the idle period (no requests) is shaded green. `plot_data.json`
  carries these as `lifecycle.markers` and `lifecycle.periods`. Signals that
  were not collected are omitted with a warning rather than drawn as zero.

### Automatic AgentX diagnostic plots

Every AgentX run with `analysis.generate_plots: true` now generates the same
four diagnostic plots used in the results site, in addition to the scaling
timeline:

- Arrival statistics: p95, p90 and mean load at arrival, queue time and client
  TTFT in 180-second windows, with request counts.
- Replica percentiles: 180-second windows, blank below 10 requests per replica.
- Policy versus engine mean: the policy's lookback (60 seconds by default),
  comparing its router signal with remaining prefill from scheduler traces.
- Session lanes: recorded initial lane IDs when available, otherwise explicitly
  labeled inferred request-span rows. Bar ends are last observed requests,
  not verified session completion. The CSV lists all observed root replicas.

Open `runs/<run>/analysis/index.html` for the local gallery. Generation status,
warnings and the exact trace inputs are saved in `diagnostic_plots.json`;
each command has its own generation log. Missing traces and failed charts are
reported explicitly and do not turn a successful benchmark into a failed run.

The harness enables scheduler tracing automatically for new AgentX runs and
passes the same absolute path through `LLMConfig.runtime_env` to every replica
and vLLM worker. It uses a unique directory for each run, so other experiments'
load cannot pollute the charts. On a single node it defaults to that run's
`telemetry/scheduler-traces` directory. For **multiple nodes**, set a shared
base directory visible at the same path on every node:

```yaml
analysis:
  generate_plots: true
  diagnostic_plots: true
  scheduler_trace_dir: /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces
```

`RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR` is also accepted as the base if the config
does not set one. The resolved per-run path is recorded in `resolved.yaml` and
the run manifest. This needs a Ray build with the scheduler tracing hook; a
build without it will produce a clear missing-trace notice. Set
`analysis.diagnostic_plots: false` to opt out of the extra plots and automatic
tracing, or `generate_plots: false` to disable all plots.

To regenerate the gallery without using a GPU:

```bash
python -m serve_llm_autoscaling.cli analyze runs/<run>
# For older runs whose traces are elsewhere:
python -m serve_llm_autoscaling.cli analyze runs/<run> --traces-dir /path/to/traces
```

New continuous runs using `routing.policy: kv_aware` also save the existing
vLLM KV-event streams to `telemetry/gpu_kv_events.jsonl` by default
(`analysis.record_gpu_kv_events: false` disables this). The recorder discovers
initial and autoscaled replicas from Serve's advertised routing metadata and
uses the replay endpoint to recover missed batches. It records sequence gaps
and collection errors explicitly. This needs `pyzmq` and `msgspec` in the driver
environment (both are present with vLLM); it does not start a model or change
the router or connector.

Each batch retains its publisher timestamp, receive timestamp, replica/actor
ID, DP rank, sequence number and block hashes by cache group and medium.
Prompt token IDs are omitted. For a GPU prefix-removal timeline, select
`BlockRemoved` events with `medium: GPU`; keep hybrid-model cache groups
separate. `AllBlocksCleared` is a separate cache reset, and Mooncake DRAM-pool
evictions remain separate metrics. Timestamps identify the publishing batch,
not the exact allocator instruction. Do not infer zero evictions from an empty
file or a period containing sequence gaps or collection errors. Earlier runs
without this file cannot recover the individual event history.

For cache-credit attribution, new traced KV-aware runs also enable
`analysis.record_router_estimates` (default `true`). The router saves its actual
selection-time prompt length, GPU/CPU overlap and effective-prefill booking,
then each request's remaining decayed contribution at autoscaling snapshot time,
under `<scheduler_trace_dir>/router-request-audit/`. Engine admission traces
separately retain local cached tokens, external cached tokens and initial GPU
prefill tokens. These share request IDs and epoch timestamps. No prompt token
IDs are written. This requires the local Ray changes preserved in
`patches/router-request-audit.patch`; the routing preflight rejects an unsupported
Ray build instead of silently claiming collection succeeded.

After a run, produce a per-request comparison without using a GPU:

```bash
python -m serve_llm_autoscaling.diagnostics.router_cache_credit runs/<run>
# Optionally account for a verified 75% host-cache credit in the selector:
python -m serve_llm_autoscaling.diagnostics.router_cache_credit runs/<run> --host-credit .75
```

Results are `analysis/router_cache_credit/requests.csv` and `audit.json`.
A large router booking alongside external hits and little GPU computation is
evidence of an estimate/computation disagreement. To attribute it specifically
to missed external credit, compare the router's GPU overlap with the engine's
local hits and allow for intentional host-transfer weighting. Snapshot records
then show whether decay and delayed completion carried that request's excess
into the plotted mean. Missing historical selection logs remain unavailable.

For a controlled proof, use one fixed replica and an unchanged decay setting:
compute prefix A, repeat A while GPU-resident, fill with distinct prefixes until
A's GPU removals are recorded, then repeat A and confirm an external hit. Join
the three A requests' router bookings to their engine cache splits/computation.
This removes replica placement and autoscaling as confounders. Test any event
format fix separately afterward; changing it during evidence collection would
mix the baseline and the proposed correction.

Legacy shared trace directories are filtered by this run's request IDs before
plotting. The five original `tools/*.py` entry points remain available and now
use the packaged implementations.

## Autoscaling evaluation

Every analyzed continuous run also gets `analysis/autoscaling_evaluation.json`,
a scorecard for the autoscaling policy, summarized at the end of the run
summary and by `analyze`. It is computed from `plot_data.json` and the
per-request records only. Its sections:

- **Periods.** `steady` is the load period after a warmup: the concurrency
  ramp, or `analysis.warmup_s`, or 120s of a run without a ramp (half the load
  if shorter). Scores are reported for `steady` and for the `whole_run`; the
  steady figures are the ones to compare between policies, because no policy
  can absorb the cold-start burst of a run that opens every session at once.
- **SLO attainment**, two ways, because they answer different questions.
  *Per request*: the share of successful requests whose TTFT (and ITL, and
  both) were within the SLO, with mean/p50/p90/p95/p99 and failures. The
  verdict (`slo_met`, "SLO PASS/FAIL" in the summary) is whether the
  `analysis.slo_percentile` (default 0.95) of requests was within the target,
  i.e. measured p95 <= `ttft_slo_ms`; the measured percentile is reported
  beside the target. TTFT is also split into cold prompts (under half of the
  prompt read from the prefix cache, per the server's usage report) and warm
  ones: a cold 100K-token prompt costs seconds of prefill whatever the
  replica count, so the warm figure is the one autoscaling can move. It is what a
  user feels and is dominated by the latency tail, which a policy has little
  control over. *Per window*: the share of rolling windows (ending in the
  period, with at least one request) whose rolling mean TTFT/ITL met the
  target, and the seconds spent over it. It is what a mean-latency policy like
  `sla` actually controls. A healthy single replica on the AgentX workload
  shows high window attainment with middling request attainment; both low
  means under-provisioned.
- **Cost**: running and starting replica-seconds (the integral of the status
  step function), mean and maximum running replicas, and GPU-hours (running
  plus starting replicas times `tensor_parallel_size`), per scoring period and
  per lifecycle period (load, drain, idle). Attainment against load GPU-hours
  is the trade-off to iterate on: any policy can meet the SLO at
  `max_replicas`.
- **Scale events**: every change of the desired replica count with the time
  the running count first matched it (reaction plus startup lag), and, for the
  `sla` policy, the state it was in and its raw recommendation at that moment;
  summarized as counts, first upscale and mean/max upscale lag.
- **Latency by replica count**: steady-state mean and p90 TTFT/ITL grouped by
  the replicas running when each request started.
- **Raw prefill workload**: telemetry also captures
  `ray_serve_router_uncached_prefill_tokens` when routers export it. Each
  application/deployment/handle/actor series is the router's latest raw snapshot,
  independent of the selected policy. Routers report replicated global views,
  so do not sum these series to estimate total workload.
- **Token policy** (when the controller's `serve_autoscaling_policy_metric`
  gauge was collected): which term, prefill or decode, asked for each
  scale-up and, if the policy also publishes its raw inputs, how strongly
  each input predicts latency: Spearman rank correlation and a quartile table of the policy's
  input per running replica at each request's start against the latency
  that request saw (prefill backlog against warm-request TTFT and against
  TTFT in excess of the prompt's own prefill floor, active decodes against
  ITL, plus the cross pairs). Because Ray exports gauges only every ~10s,
  the in-flight prefill is also rebuilt from the request records (uncached
  tokens of requests started but not yet at their first token) and scored
  the same way, as `in_flight_prefill_vs_*`.
- **SLA policy** (when its gauges or log were collected): when the model
  became ready and first made a `capacity` decision, the state timeline and
  time spent in each state, and prediction error. Predictions are for the
  recommended replica count, so only scrapes where that equals the running
  count (with the model ready) are compared: predicted minus server mean
  (vLLM, the policy's own input) and minus client mean, as bias, MAE and MAPE;
  plus client minus server, the queueing the policy cannot see.

Put several analyzed runs side by side; same workload, seed and SLO make the
rows comparable:

```bash
autoscale-harness compare runs/<default run> runs/<sla advisory run> runs/<sla run>
```

Regenerate the analysis with different windows without rerunning the
experiment; this needs no Ray cluster or GPU:

```bash
autoscale-harness analyze runs/<run-directory> --window-s 10 --tail-window-s 60
autoscale-harness analyze runs/<run-directory> --no-plot
```

A request-rate-series run directory contains the following. An AgentX run
has `benchmark/agentx-concurrency-ramp/` or
`benchmark/agentx-fixed-concurrency/` instead, without `rate_series.json`.
It also writes `analysis/concurrency_qps.json`, and a fixed-concurrency run
writes `analysis/slo_calibration.json`.

```text
benchmark/
├── request-rate-series/          # raw AIPerf artifacts
│   ├── rate_series.json          # curve passed to AIPerf
│   ├── command.json              # exact command used
│   ├── stdout.log
│   ├── stderr.log
│   ├── phase_manifest.json       # profiling start time (the time origin)
│   ├── profile_export.jsonl      # per-request AIPerf records
│   └── profile_export_aiperf.json
├── lifecycle.json                # load / drain / idle boundaries (epoch ns)
└── sweep_summary.json            # aggregate metrics for the complete series
deployment/
├── config.json                   # LLMConfig payload
├── routing.json                  # policy, env switches, resolved router, topology
└── ...
telemetry/                        # raw samples, never modified by analysis
├── serve_status.jsonl
├── serve_metrics.jsonl
└── sla_policy_events.jsonl       # sla policy only: controller log transitions
analysis/                         # derived; regenerated by `analyze`
├── request_timeseries.json
├── metrics_inventory.json
├── plot_data.json                # aligned data the timeline is drawn from
├── autoscaling_evaluation.json   # the scorecard (see Autoscaling evaluation)
└── autoscaling_timeline.png
```

The summary contains one row with `mode: "request_rate_series"` and
`level: "series"`. It summarizes the complete run; use the analysis artifacts
for behavior over time.

## Results

Results are written below `runs/<UTC timestamp>-<experiment name>/`. Every run
contains the original and resolved configuration, environment metadata,
deployment status snapshots, AIPerf logs and native exports, and a normalized
`sweep_summary.json`.

Findings from past investigations, with the runs and configs behind them:

- `findings-token-policy/README.md`: the token-based autoscaling policy
  versus Serve's default, best-of-each on five workload shapes, which term
  binds, signal strength, tuning, and recommended next steps. Evidence in
  `sweep.md` and `pairs.md` beside it.
- `findings-claude-code-overnight/b200_investigation.md`: replica limits,
  prefix-cache hit rates, and routing on 4x B200.

## Commands

```text
autoscale-harness check
autoscale-harness validate <config>
autoscale-harness deploy <config> [--replace]
autoscale-harness benchmark <config>
autoscale-harness status <config>
autoscale-harness teardown <config>
autoscale-harness run <config> [--replace] [--keep-deployment]
autoscale-harness analyze <run-directory> [--window-s S] [--tail-window-s S] [--no-plot]
autoscale-harness compare <run-directory>...
```

`benchmark` runs against an existing deployment. For a continuous run it
collects telemetry when it can connect to Ray, and otherwise runs without it and
warns that the analysis has no Serve telemetry.

The workspace is assumed to be dedicated to the experiment. Teardown calls
`serve.shutdown()` and therefore removes the complete Serve instance, while
leaving the Ray cluster running.
