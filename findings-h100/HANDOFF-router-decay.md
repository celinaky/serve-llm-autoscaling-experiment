# Brief: plan decay for the KV router's outstanding-prefill gauge

You are planning, not implementing. Produce a design and an implementation
plan for making the KV-aware router's outstanding uncached prefill count decay
as the engines actually prefill, so the token autoscaling policy scales on
the work still left to do. Evidence and background are in `README.md` next to
this file; everything you need to start is below.

## Where things live

- Ray checkout: `/mnt/local_storage/ray`, branch
  `celina/prefill-token-autoscaling-policy`, HEAD `e643369b6e`. It is
  symlinked into site-packages, so edits take effect on the next deploy.
- Harness (runs, analysis tools, experiment configs):
  `/home/ray/default/serve-llm-autoscaling-experiment`.
- Versions: vLLM 0.31.0, `ai-dynamo-runtime` 1.7.0 (closed native library;
  Python stubs at `site-packages/dynamo/_core.pyi`).
- Hardware: 8xH100 80GB. Model `Qwen/Qwen3.6-27B-FP8`, hybrid attention and
  Mamba-style layers, 4 KV cache groups, 1568-token blocks,
  `max_num_batched_tokens` 16384, so a prefill chunk is up to ~16K tokens and
  one step takes about 1 to 2s on H100.

## How the signal works today

All paths relative to the Ray checkout.

1. **Routing adds a whole prompt.**
   `python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py`,
   `select_worker` calls Dynamo `select_and_reserve`, which returns
   `effective_prefill_tokens` = prompt length minus the overlap the router's
   Dynamo index believes is cached on the chosen worker.
   `_track_request_state` (line 638) adds it to
   `_outstanding_effective_prefill_tokens` (line 663).
2. **The first output token removes it in full.** On the engine replica,
   `routing_policies/kv_aware/vllm/token_tracking.py`,
   `RequestTokenTracker.on_output` reports `on_prefill_complete` on the first
   output token (line 125). Events travel through `LifecycleEventForwarder`:
   queued locally, batched, ordered, broadcast to every LLMRouter replica,
   best-effort (dropped on failure, never blocks generation). They arrive at
   `core/ingress/router.py:298` `on_lifecycle_events`, then
   `KVTokenTracker.on_lifecycle_events` (line 666), which only accepts hooks in
   `LIFECYCLE_HOOKS` (line 49). `on_prefill_complete` (line 693) subtracts the
   whole amount (line 700); `on_request_completed` and
   `_remove_autoscaling_load` (line 736) cover requests that end before a first
   token.
3. **Nothing in between.** No event reports prefill progress, so a 100K prompt
   counts as 100K for its whole 10 to 15s prefill.
4. **The policy averages it.** `_get_autoscaling_snapshot` (line 347) and
   `_publish_autoscaling_snapshot` (line 355) publish the total on every change.
   `python/ray/serve/experimental/llm_token_autoscaling_policy.py` takes a
   time-weighted mean over `upscale_lookback_s` (60s) per router replica,
   takes the max across router replicas, divides by
   `target_prefill_tokens_per_replica` (20K), and asks for the ceiling. It
   publishes the un-rounded value as `prefill_required_replicas`.

Tests: `python/ray/llm/tests/serve/cpu/deployments/routers/kv/test_kv_token_tracker.py`
and `python/ray/serve/tests/unit/test_llm_token_autoscaling_policy.py`.

## Why change it

Measured with `tools/policy_vs_engine_mean.py`, which rebuilds the engines'
remaining uncached prefill from vLLM scheduler traces, each chunk leaving when
its batch completes:

- **B200, 32 sessions:** the router's 60s mean averaged 15.4K against 8.2K
  actually queued. About 5.2K of the 7.2K gap is the missing decay; ~0.8K is the
  router holding each request ~130 ms longer than the engine; ~1.2K is warm
  turns over-counted because the index lacked blocks the engine had.
- **Ratio router / engine, median:** 1.97 on B200, 1.46 and 1.74 on the two H100
  runs. The policy asks for roughly twice the replicas the queued work implies,
  and spikes to 2 to 2.6 replicas on a single cold prompt.

## The complication you must design around

On H100 the router sometimes reads *below* the engine, which a no-decay gauge
should never do: 19 to 25% of loaded time, with stretches up to 97K under.
Every such stretch coincides with a returning session whose context the engine
had evicted (H100's cache is 1.26M tokens against B200's 4.1M). In those cases
the router added a few thousand tokens, or none, while the engine re-prefilled
80K to 170K from scratch: the router's index still believed the context was
cached. Read README section 5 for the evidence.

The cause is not settled. The leading hypothesis is per-group eviction in the
hybrid model: the Mamba state evicted while attention blocks survive, so the
engine misses but the index sees a hit. Removal events do reach the index in
the simple case (tested).

**Consequence for the design:** decay that only burns down the router's own
estimate makes this worse. It lowers the signal everywhere, including where it
is already too low. A design that takes the engine's measured uncached tokens
at admission fixes both problems at once.

## Data the engine side already has

The branch has a scheduler wrapper,
`python/ray/llm/_internal/serve/engines/vllm/scheduler_trace.py`
(`SchedulerTraceMixin`, enabled by `RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR`), which
wraps vLLM's `schedule()` and already sees, per request and per step:

- at admission: `prompt_tokens` and `cached_prefix_tokens`, the engine's true
  uncached count;
- per step: `scheduled_tokens`, `computed_before`, `remaining_prefill_after`,
  phase;
- preemptions, which discard KV and add recompute work back.

Today it only writes these to JSONL. The `RequestOutput` stream that drives the
lifecycle hooks sees nothing until the first token.

## Options to evaluate

Not exhaustive; compare them and recommend one.

1. **Router-side time decay.** Burn each request down at an assumed per-replica
   rate. No new events. It is wrong whenever the replica is shared or the rate
   differs (H100 ~7K tok/s, B200 ~16K), and it cannot fix the under-count.
2. **Engine-reported progress through the lifecycle channel.** New hooks, for
   example one at admission carrying the engine's uncached count, which
   replaces the router's estimate, and one per prefill chunk. This gives exact
   decay and fixes the under-count. Costs: event volume (a 100K prompt is
   about 7 chunk events, broadcast to every router replica), ordering against
   `on_prefill_complete`, and getting events out of the scheduler process into
   the replica's asyncio loop where the forwarder lives.
3. **Per-replica gauge from the engines.** Each replica publishes its own queued
   prefill, and the policy sums replicas instead of reading the router. This is
   simplest and exact for what is queued, but it loses requests in flight
   between routing and queueing, and it adds metric latency. Check how the
   policy would receive per-replica metrics.

## Questions the plan must answer

1. Which option, and why, given the under-count.
2. How the scheduler process reports to the replica process, and at what
   batching, so generation never waits.
3. Preemption and abort: when tokens are added back, and how the count is
   guaranteed to return to exactly 0 with no leaks. Today's invariant: the
   gauge never goes negative and returns to 0.
4. Multiple router replicas: every replica receives every event today, and the
   policy takes the max across them. Does that still hold?
5. Publication rate: a snapshot per chunk makes the metric churn. Look at
   `_defer_autoscaling_snapshot` and decide on coalescing.
6. Target recalibration: with decay the signal roughly halves, so 20K no longer
   means what it did. Use the H100 queue profile, where queue time is about 1s
   per 10K uncached tokens queued ahead on a replica, linear to 220K. Say what
   target you would start from on H100 and on B200, and how to confirm it.
7. Whether Dynamo's own per-worker prefill load, used for routing decisions,
   should get the same correction. That is optional and may be out of reach in
   the closed library.

## How to verify

- **Unit tests** in the two test files above: decay, the admission correction,
  preemption, abort before first token, and return to 0.
- **End to end:** rerun `experiments/h100_workday_curve_prefill_mean.yaml`
  (16 sessions; set `max_replicas` back to 8 if no other engine holds a GPU).
  Use `RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR=/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces`
  and `autoscale-harness run <config> --replace`.
- **Check with the same tools as before:**
  `python tools/load_breakdown.py <run>`, then
  `python tools/policy_vs_engine_mean.py <run> --mark-evictions`.
  Success: router mean / engine mean median near 1.0, and no stretch of 20s or
  more with the router over 10K below the engine. The previous 16-session run
  (`runs/20261103T182949Z-h100-workday-curve-prefill-mean`) is the baseline.
- **Don't trust raw gauge scrapes** in `telemetry/serve_metrics.jsonl` for
  averages: they are 10s sample-and-hold. Use the published
  `prefill_required_replicas` x target.

## Out of scope

Pinning and session placement (new replicas get no existing sessions), a
per-replica scaling signal, HAProxy stream cuts on replica add, and the open
question of why the index misses some evictions. Note in the plan where
those interact with your design, but don't solve them.
