# Qwen3.6-27B-FP8 on 4x B200: replica limits, cache hits, autoscaling

Working notes from an overnight investigation (2026-10-08). Nothing here is
committed. Scripts live in `findings-claude-code-overnight/scripts/`.

## Setup
- Node: one head node `4xB200:92CPU-960GB`, Ray 2.58.0, vLLM 0.26.0,
  ai-dynamo 1.5.0, AIPerf (uvx).
- Workload: AgentX `semianalysis_cc_traces_weka_062126_256k`, replayed by
  AIPerf's `inferencex-agentx-mvp` scenario.
- Prior runs:
  - `runs/20261007T230910Z-*`: target_ongoing 24, upscale 30s.
  - `runs/20261008T001704Z-*`: target_ongoing 8, upscale 10s.

## 1. The dataset is synthetic text over real cache structure
- Each request is only `in`/`out` token counts plus `hash_ids` (one per
  64-token block). AIPerf synthesizes text per hash id, so prefix reuse is
  fully determined by hash-id overlap.
- AIPerf replays it as `DELTAS_WITH_RESPONSES`: each turn appends the
  recorded assistant reply plus the new delta, and live replies are
  discarded.
- The cache-bust marker is per trajectory, built from the trace id
  (`aiperf/timing/strategies/cache_bust.py:106`), and prefixed to the system
  message. It prevents cross-session sharing only, not reuse within a
  session.

### Ideal hit rates (scripts/ideal_hits.py, whole dataset)
| scope | ideal hit rate |
|---|---|
| any earlier request in the trace | 96.2% |
| earlier requests of the same stream (what replay can hit) | 95.3% |
| same stream, floored to vLLM's 1568-token align block | **94.6%** |

- At that ceiling, uncached tokens per request: p50 2.4K, p90 9.3K,
  p99 62K, mean 5.4K.
- 17% of requests keep <50% of the previous same-stream prompt
  (compaction or side calls). Those are real cold-ish requests.

**Observed:** cumulative vLLM prefix-cache hit rates of 25-66% per replica,
and ~3% on a single replica for the first 5 minutes of the 24-target run.
With one replica there is no routing involved, so the miss is inside the
engine or the request formatting.

## 2. vLLM prefix caching for this hybrid model
- Qwen3.6 mixes 48 Gated DeltaNet (linear attention) layers with 16 full
  attention layers. vLLM 0.26 falls back to `mamba_cache_mode="align"`
  (model lacks "all" support). In align mode the linear-attention state is
  saved only at the end of a scheduler step that lands on a block boundary
  (`vllm/config/cache.py:139`).
- Prefill chunks are clipped to block boundaries so the state of the
  prompt's last full block is saved (`v1/core/sched/scheduler.py:355`).
- Attention block size is forced to **1568 tokens** so attention and
  mamba pages are equal size (replica log).

## 3. Single replica, engine only (standalone `vllm serve`, same engine args, GPU 0)
Probe: `scripts/replica_probe.py` (random token-id prompts via /v1/completions).

| probe | result |
|---|---|
| cold prefill 8K / 32K / 64K / 128K | 21.3K / 20.0K / 18.8K / 16.8K tok/s |
| warm: 64K prompt + 512 new tokens | **0.18s**, 64,288 cached (=41 x 1568) |
| warm: + 2K / + 8K new tokens | 0.20s / 0.59s |
| 2K request issued 1s after a 128K cold prompt starts | **6.9s** (0.11s alone) |
| 4 / 8 concurrent cold 32K prompts | aggregate ~20K tok/s; p50 TTFT 6.2s / 9.4s |

- Prefill is compute-bound at ~20K tok/s per B200 replica; concurrency does not raise it.
- Prefix caching works: a warm agent turn costs ~0.2s. The 300ms goal is
  reachable for warm turns with no queueing.
- **Head-of-line blocking:** vLLM's scheduler gives running prefills the
  token budget first, so a short request waits for a long prefill's
  remaining chunks.

### Real AgentX requests through a logging proxy (`scripts/logging_proxy.py`)
- 2 sessions, 29 requests. Every turn after a session's first big request
  hit 95%+ (e.g. 114,053 prompt / 111,328 cached).
- Each session starts with ~1 tiny request, then a ~64K-token
  single-message request that is completely cold (per-session cache bust).
  This is the cold start that dominates ramps.

### vLLM's "Avg prompt throughput" log line is lumpy
- It counts uncached prompt tokens, booked when a prefill *completes*
  (`vllm/v1/metrics/loggers.py:146`, `stats.py:328`).
- A 10s window can show 0 while the GPU is busy on a long prefill. Average
  it over minutes.

## 4. Why the earlier runs had low hit rates
`scripts/run_cache_ceiling.py`: the ideal hit rate for the requests a run
actually sent. A conversation's turn i can reuse at most turn i-1's prompt.

| run | ideal hit (whole run) | ideal hit, first 4 min | uncached arrival during ramp |
|---|---|---|---|
| target 24 | 79.5% | ~1-17% | 4-10K tok/s |
| target 8 | 82.5% | ~1-44% | 5-10K tok/s |

- The low early hit rates were inherent to the workload, not an engine or
  routing bug. A ramp from zero (`trajectory_start_ratio: 0`) means every
  session starts with a ~64K cold request, so the first minutes are almost
  pure cold prefill.
- Later in the runs the ideal is 86-96% per minute. Observed cumulative
  hit rates trail it because they include the cold start, and because of
  cache placement after scale-up (see below).

## 5. Session affinity is broken: KV-aware routing moves sessions off their cache
`scripts/routing_affinity.py` maps each AIPerf `x_request_id` to the
LLMServer replica that served it (Serve access logs), then checks whether
each follow-up turn stayed on the replica that served the previous turn.

| run | follow-up turns that moved replica | TTFT p50/p90, stayed | TTFT p50/p90, moved | reusable tokens re-prefilled |
|---|---|---|---|---|
| target 24 (2 replicas) | 50/277 = 18% | 0.9s / 7.4s | 9.4s / 62s | 2.5M |
| target 8 (4 replicas) | 134/344 = **39%** (44% once all 4 up) | **0.8s / 7.1s** | **6.1s / 19.9s** | **7.5M** (~6 min of GPU prefill) |

- `scripts/move_vs_load.py` rebuilds per-replica in-flight counts from the
  access logs. Of the moved turns in the target-8 run, the cached replica was
  busier in 93/156, equal in 16, and **less busy in 47**. Mean in-flight was
  9.6 (cached) vs 6.3 (chosen).
- Moves happen throughout the steady state (16-27 per minute), not just
  after scale-up.
- The router does get KV events: one event worker registered per replica,
  router block size 1568, no indexer errors.
- A move to an equal-or-busier replica should not happen if a 70K-token
  overlap were scored. Either the overlap signal is missing (router tokens
  differ from engine tokens, or hybrid-model events don't index) or the
  load term dominates. Diagnostic: `experiments/diag_kv_affinity_2rep.yaml`
  + `scripts/affinity_probe.py`.

## 6. Single-replica capacity on the real workload (standalone vLLM, AgentX fixed concurrency, 600s)
`scripts/run_levels.sh` + `scripts/level_summary.py`. "Cold start" is the
first 180s, when every session opens with a ~64K cold request at once;
"steady" is the rest.

| sessions | cold start TTFT p50/p90 | steady TTFT p50/p90/p99 | steady ITL p50/p90 | req/s | ideal hit |
|---|---|---|---|---|---|
| 4 | 2.2 / 17.3s | 1.4 / 7.0 / 16.0s | 11 / 22ms | 0.12 | 82% |
| 8 | 7.4 / 37.9s | **0.6 / 3.9 / 8.3s** | 13 / 24ms | 0.29 | 85% |
| 16 | 8.5 / 38.2s | **0.8 / 3.9 / 13.0s** | 28 / 49ms | 0.55 | 86% |
| 32 | 72 / 132s | **14.3 / 30.3 / 48.4s** | 106 / 174ms | 0.38 | 71% |

- One B200 replica handles ~16 concurrent AgentX sessions (~0.55 req/s)
  in steady state, and is overloaded by 32. Throughput *drops* at 32: the
  closed loop slows as TTFT explodes.
- At 32 sessions the engine hit 50.2% (from the /metrics counters) vs
  71.3% achievable. 32 sessions x ~100K tokens of retained context
  approaches the 4.18M-token KV cache, so prefixes are evicted during
  think time.
- The cold-start storm (all sessions open with ~64K cold prompts at once)
  costs 30-130s TTFT p90 at any concurrency >= 8. No steady-state policy
  can fix that; it needs capacity before the burst.
- One standalone replica at 16 sessions (0.55 req/s) beat the 4-replica
  autoscaled kv_aware run at a similar request rate (0.51-0.64 req/s):
  steady p50 0.8s vs ~2.2s.

## 7. torch.compile cache is keyed by physical GPU index
- vLLM stores compiled graphs under
  `~/.cache/vllm/torch_compile_cache/<hash>/rank_0_0_dev{N}`.
- In the target-8 run, replicas on GPUs 0-1 reused caches from the earlier
  run (compile 7s, init ~37s); GPUs 2-3 compiled cold (68s, init ~108s).
- All four devN caches now exist, so runs after 2026-10-08 ~01:00 start
  faster. **Do not compare scale-up times across that boundary.**
- In production the first replica per GPU index pays ~60s extra unless
  the cache is pre-seeded.

## 8. Ray Serve overhead is negligible (1 replica, 16 sessions, 600s)
Run `runs/20261008T143225Z-b200-single-replica-c16`
(`experiments/b200_single_replica_c16.yaml`, kv_aware + direct streaming,
HAProxy hard-stop raised to 900s in the Serve controller env).

| steady state (t>=180s) | standalone vLLM | Ray Serve 1 replica |
|---|---|---|
| TTFT p50/p90/p99 | 0.78 / 3.94 / 13.0s | 0.89 / 4.04 / 8.3s |
| ITL p50/p90 | 28 / 49ms | 30 / 56ms |
| req/s | 0.55 | 0.53 |
| engine prompt tok/s avg (windows with waiting) | 2.7K (4/47) | 2.9K (1/46) |

- The cold phase (first 180s) has TTFT p50 8.5s standalone vs 19.0s
  Serve, but p90 (38 vs 41s) and the engine stats (6.5K vs 5.9K tok/s,
  waiting in 11/18 vs 13/18 windows) match. Likely ordering noise, not a
  confirmed cost.
- **Conclusion:** the bad autoscaling results come from routing and cold
  starts, not from the Serve path.

### Open question: effective cold prefill is ~6K tok/s under the real mix
- During the cold-start storm, with requests waiting, both setups computed
  only ~6K uncached tok/s on average (6.6K at 32 sessions, waiting in every
  window). The isolated probe does ~20K tok/s.
- Effective capacity for cold work on this workload is therefore ~1/3 of
  the synthetic number.

## 9. Autoscaling: routing comparison with everything else equal
Both runs: target_ongoing 8, upscale 10s, downscale 60s, 1-4 replicas,
32 sessions ramped over 150s, HAProxy hard-stop 900s, warm compile cache on
all GPUs. Only `routing.policy` differs. Report: `scripts/autoscale_report.py`.

| | kv_aware (`runs/20261008T163724Z-b200-autoscale-kv-aware`) | consistent_hash (`runs/20261008T152545Z-b200-autoscale-consistent-hash`) |
|---|---|---|
| scale-up: desired 2/3/4 at | 72 / 112 / 142s | 67 / 97 / 127s |
| scale-up: running 2/3/4 at | 171 / 205 / 236s | 171 / 192 / 220s |
| follow-up turns moved replica | **43.7%** | **7.2%** (only at scale events) |
| reusable context re-prefilled | 10.4M tokens | 1.7M tokens |
| TTFT, 1 replica (ramp) p50 | 35.6s | 37.3s |
| steady t=360-600 TTFT p50/p90/p99 | 2.01 / 10.1 / 20.8s | **0.65 / 3.9 / 12.2s** |
| steady ITL p90 | 61ms | 29ms |
| completed requests | 500 | 566 |
| errors | 0 | 3 (scale-down at t~621s killed in-flight requests) |

**The improvement is routing, not the HAProxy fix.** KV-aware routing moves
~40% of follow-up turns even with the fix and no mid-load scale-down;
moved turns had TTFT p50 5.0s vs 1.26s for turns that stayed.

## 10. Why requests exceed 120s
- 5-9% of requests ran over 120s end to end. 84-86% of their time was
  decode, at a median ITL of 134-219ms (a quiet replica runs ~12ms).
- Their outputs are modest (p50 574-808 tokens). Decode is slowed because
  each engine step carrying a prefill chunk (up to 16,384 tokens, ~0.8s)
  stalls every decoding request on that replica.
- At 12ms ITL only 1-4 of them would exceed 120s: genuine 10K-32K-token
  outputs, forced by `ignore_eos`.
- End-to-end p99 > 1 min even on an unloaded replica (8 sessions steady:
  87s), because of forced output lengths. Use TTFT, not end-to-end, to
  judge capacity.

## 11. Where the autoscaling policy falls short (with sticky routing)
1. **Reaction plus startup lag ~2.5-3 min.** The first scale-up decision
   comes at t~67-72s (30s look-back + 10s delay), and replicas take
   ~95-105s to serve even with a warm compile cache. One replica absorbs
   the cold-start burst: TTFT p50 ~35-40s.
2. **Scale-up moves sessions.** Ring changes remap a share of sessions,
   which then re-prefill (23 moves during scale-up).
3. **Scale-down mid-load is destructive.** It moved 10 sessions (p90 TTFT
   26s in that window), and in-flight requests on the removed replica were
   killed: STOPPING t=626-660s, Serve `graceful_shutdown_timeout_s`
   default 20s, versus agent requests that run for minutes.
4. **Request count is a poor signal.** 10 cold 64K requests overload a
   replica while 20 warm ones are easy.
5. **Per-replica floor:** steady TTFT p90 ~4s and p99 ~12s at ~8
   sessions/replica, the same as 1 replica at 8 sessions. Cold starts of
   new sessions and head-of-line blocking dominate; more replicas don't
   fix it.
