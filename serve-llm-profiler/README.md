# Serve LLM prefill queue profiler

Standalone MVP for collecting a replica's remaining uncached prefill work at
engine arrival versus estimated initial execution queue time. It has no imports
from the experiment harness and does not enable the online queue learner.

## Install and run

Use the Ray image's Python on a single-node GPU Ray cluster. **Only the Ray
cluster must already be running**, with at least one free GPU. You do not
predeploy the model or start a Serve application.

The `run` command performs the entire application lifecycle:

1. Read your YAML and deploy a new Serve LLM application with one model replica
   and the injected tracing scheduler. Serve starts automatically if needed.
2. Wait for that application to become healthy, then send the AIPerf workload.
3. Delete the profiling application, reconstruct its traces, and write results.

For this MVP, the Serve instance must have no existing applications. The
profiler refuses to replace an existing application; it does not send workload
to one.

```bash
cd serve-llm-profiler
python -m pip install -e '.[dev]'
python -m serve_profiler run /path/to/application.yaml
```

Run on the cluster node: the driver and engine need to see the same absolute run
directory. The package is uploaded to the replica using Ray `py_modules`; an
editable install on the driver alone is not assumed to exist on workers.

The input is the existing experiment YAML shape. Only `deployment` and relevant
`runtime` settings are used. Other benchmark and analysis settings are ignored:
the MVP workload is intentionally hardcoded. The original file is never edited.

`runtime.endpoint_url` is the base HTTP address of Serve on this Ray cluster,
not the URL of an already deployed model. The profiler creates a unique route
below that address and directs AIPerf to the application it just deployed.

```yaml
deployment:
  model_id: Qwen/Qwen3.6-27B-FP8
  accelerator_type: H100
  engine:
    tensor_parallel_size: 1
    max_model_len: 262144
    gpu_memory_utilization: 0.92
    max_num_seqs: 512
    max_num_batched_tokens: 16384
    enable_prefix_caching: true
    enable_prompt_tokens_details: true
    kv_cache_dtype: fp8
    language_model_only: true
runtime:
  endpoint_url: http://127.0.0.1:8000
  readiness_timeout_s: 1800
```

An optional `deployment.routing` uses the same `policy`, `direct_streaming`,
`forward_request_body`, `haproxy_request_buffer_bytes`, and
`request_router_kwargs` fields as the experiment configs. These require a Ray
build supporting that router/topology. The original accelerator and engine
options, including native vLLM offloading, are preserved. Harness-only
`shared_kv_cache` service configuration is rejected: this profiler does not
start an external KV store. Tensor, pipeline, data, and context parallelism
above one and an existing custom scheduler are rejected in this MVP.

The profiler creates a unique application name and HTTP route, forces exactly
one LLM replica (`num_replicas: 1`), and retains the ingress built by Serve.
It deletes only its own application on completion or failure. It never calls
`serve.shutdown()`. Partial artifacts remain under the printed run directory.

## Workload: AIPerf, not a custom replay driver

One deployment runs one continuous 30-minute AIPerf phase, keeping eight session trees:

| Phase | Session-tree pool | Maximum requests per cohort | Profiling duration | Seed |
|---|---:|---:|---:|---:|
| 1 | 8 | 8 | 1800 s | 42 |

The 30 minutes count profiling issuance, not model startup, dataset preparation,
warmup, or up to 300 seconds of request drain. There is no phase boundary or
mid-run session-tree restart; the phase seed is 42. This replaces the previous
4/8-request, 900-second-per-phase schedule for the cohort-8 experiment.

The cohort gate holds dependency-ready requests and releases at most one per
session tree in each wave. Every response in the wave must finish before the
next wave starts. Partial waves proceed when fewer trees have ready work; there
is no minimum-size wait. Subagents and joins use the same gate, so they cannot
silently exceed the wave size. This changes branch overlap as well as arrival
timing. It is an issuance barrier, not a guarantee of simultaneous engine arrival.

The first requests start cold sessions. Later waves contain eligible follow-ups
and new cold replacements only when an entire tree finishes. There is no explicit
cold/warm quota, early session replacement, or cache reset. Follow-ups can still
miss the cache. Eight trees and drained waves do not guarantee low KV occupancy;
check CPU cache hits, preemptions, and KV pressure in the resulting traces.

Settings live in `workload.py`. Until the cohort change has a published revision,
the default AIPerf source is the sibling `../aiperf` checkout, on branch
`celina/agentx-cohort-gate` based on `1c61dfbb`. Keep the modified fork beside this
folder when handing it off. `runtime.aiperf_source` can select a different local
checkout or a published Git revision supporting `--agentx-cohort-size`.
AIPerf runs through `uvx --python 3.11`; Ray remains in the image's Python.

Dataset: `semianalysis_cc_traces_weka_062126_256k`, with streaming,
`X-Session-ID`, `first_turn_prefix` cache busting, recorded output lengths, and
trajectories starting at turn zero. Prompts/prefixes are reconstructed by AIPerf
from the dataset; this is not byte-for-byte replay of the original Claude text.

`--ignore-trace-delays` removes recorded think times. The
`inferencex-agentx-mvp` scenario retains AIPerf's agentic replay engine and
spawn/join dependencies; `--unsafe-override` permits the intentional timing
change and the 1800-second phase. These are modified AgentX profiling runs, not
valid AgentX benchmark submissions. Do not add `--fixed-schedule`.

The profiler checks the pinned CLI supports the required flags before deploying.
Runtime validation of the selected AIPerf revision and its dependency behavior
is still necessary: CLI flag availability alone does not establish correctness.
If AIPerf rejects the combination, the phase fails with its console log saved;
there is no silent fallback to synthetic workload or trace-timed replay.

Network access is required for first-time AIPerf installation, the public
dataset, and uncached model/tokenizer downloads. `uvx` must be on PATH.

## Injected scheduler

`scheduler.py` adapts the previous Ray tracing wrapper into a standalone module.
There are no GPU-worker changes. The `AutoTracingScheduler` factory chooses the
tracing sync/async subclass after vLLM resolves `async_scheduling`; it does not
force the application into a different scheduling mode.

vLLM receives `scheduler_cls=serve_profiler.scheduler.AutoTracingScheduler`.
`SERVE_PROFILER_TRACE_DIR` selects output. The unrelated Ray built-in
`RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR` is disabled for this deployment, so a second
injection path cannot overwrite or reject the custom class.

The scheduler calls native methods and records `request_queued`,
`request_admitted`, `batch_scheduled`, `batch_completed`, `request_first_token`,
`request_preempted`, and `request_finished`. Decode-only batches are retained.
Events enter a bounded nonblocking queue; a background thread writes/flushes
rotating JSONL files. This has logging overhead, even though disk I/O is off
the scheduler thread. Files are flushed during scheduler shutdown before
analysis. The readiness probe verifies that trace files are visible locally.

The wrapper depends on private vLLM V1 interfaces. It was adapted against the
workspace's tracing implementation and installed vLLM interfaces; compatibility
across arbitrary Ray/vLLM versions is not promised. No installed packages are
patched. Live validation requires a compatible GPU runtime.

The workspace's Ray build still imports `FrontendArgs` from a path renamed in
vLLM 0.33. `compat.py` provides a process-local module alias when that old path
is absent. The same startup hook is applied to the model and router actors;
installed Ray/vLLM files are not edited. Serve deployment options contain
uploaded GCS module URIs, rather than driver-local module paths. The
single-node shared package path also lets the router's environment-only
runtime configuration locate the hook.

## Reconstruction

All requests on each engine contribute to its workload, including preparation,
warmup, and requests excluded from the sample set. Normal analysis selects only
AIPerf profiling records as observations and keeps phase identities.

For each request episode, sum observed scheduled prefill tokens, add that work
at arrival (or at preemption for recomputation), then subtract each chunk when
its batch's output callback arrives. This follows the original trace method.
Using observed chunks handles the engine's alignment/tail behavior; admission
prompt-minus-cache counts are exported separately for comparison. Decode tokens
are never counted. Rotated files are combined before replay.

Each point samples the reconstructed work strictly before its engine arrival,
excluding its own arrival increment. This includes dispatched prefill chunks
whose callbacks have not yet returned. Cache information and observed chunks
are learned later: this is an offline diagnostic reference, not a live router
feature or an exact instantaneous GPU-work measurement.

```text
estimated_start = max(first_prefill_batch_scheduled,
                      previous_batch_output_callback)
queue_ms = 1000 * (estimated_start - engine_arrival)
```

Use captured monotonic timestamps, not JSONL file-write time. Skip samples with
no previous batch, invalid timing, or preemption before first execution. This
label ends at the first prefill batch; it does not include later inter-chunk
waiting, full prefill duration, tokenization, or client-side queueing.

A cancelled/unresolved request's unknown workload cannot silently become zero.
The analyzer conservatively excludes observations from that request's arrival
onward for that engine. Dropped or malformed events prevent publishing fitted
points from that analysis. Reasons and excluded counts are in `quality.json`.
This intentionally favors trustworthy partial results over optimistic coverage.

## Results and reanalysis

```bash
python -m serve_profiler analyze runs/<run>
```

Outputs under `analysis/`:

- `points.csv`: load, queue estimate, native scheduling wait, own size/cache
  counts, observed prefill work, phase, request, engine, and batch IDs.
- `coverage.csv`: zero-load bin, then `(0,20000]`, `(20000,40000]`, etc.,
  including empty intervening bins. No minimum count or upper-load cutoff.
- `fit.json`: pooled and per-phase ordinary least squares, slope in ms/token,
  intercept in ms, R², RMSE, MAE, and observed load range. Degenerate or
  nonpositive-slope fits are explicitly marked.
- `queue_vs_load.png`: points by phase, fitted line, and residuals.
- `quality.json`: input provenance, exclusions, incomplete requests, and loss.

The intended policy conversion is
`(target_queue_ms - intercept_ms) / slope_ms_per_token`. This is a mean model,
not a tail guarantee. Targets below the intercept are infeasible under the
model; extrapolation outside the observed range is unvalidated. The profiler
does not modify an autoscaling policy or require minimum coverage/R².

Analyze old traces without AIPerf records explicitly:

```bash
python -m serve_profiler analyze runs/historical-validation \
  --traces /path/to/old/scheduler-traces --trace-only
```

Trace-only analysis includes every reconstructible traced request, including
warmup, and must not be presented as a fresh 30-minute experiment.

## Limitations and follow-ups

The fit is specific to the model, GPU/engine config, and AgentX coding workload
with modified pacing. Removing think times changes cache residency and decode
mix. CPU offloading, KV transfers/pressure, sequence limits, priorities,
preemption, and varying decode load can introduce other waits. The previous
output callback is not the exact GPU end time; async execution may overlap
callbacks and input preparation. Record these conditions when interpreting
results. Automated alerts, GPU timing, explicit cold/warm mixing, adaptive coverage,
configurable workload settings, and policy integration are deferred.

Run checks without Ray or a GPU:

```bash
python -m pytest
```

The scheduler protocol/writer tests use fake native schedulers. They do not
replace a GPU compatibility test.
