# Prefill-token autoscaling on a pinned-session workday curve: session notes

Working notes from the Claude Code session of 2026-10-30 to 2026-11-02
(Celina Yang). This is the summary a reader on another workspace needs to
pick the work up. The raw transcript is archived at
`/mnt/cluster_storage/serve-llm-autoscaling-experiment/sessions/2026-10-30_to_11-02_prefill-arrival-p95-session_c79a47ff.jsonl.gz`
(shared cluster storage; ~48 MB gzipped). The earlier investigations this
builds on are `findings-token-policy/` and `findings-pd-token-policy/`.

## The question

Can a Ray Serve autoscaling policy hold a target amount of uncached prefill
work per replica, so that requests rarely wait behind other requests'
prefill? "Uncached prefill tokens ahead of a request on its engine at
arrival" was established earlier (`runs/burst-sweep-pooled`, the
load-vs-TTFT breakdowns) as the quantity that predicts queue time: about 1s
of engine queue per 16K tokens ahead on a B200.

## Setup

- Node: 8x B200. Model `Qwen/Qwen3.6-27B-FP8`, one GPU per replica, replicas
  1 to 8, replica boot ~90s from decision to serving.
- Ray checkout `/mnt/local_storage/ray`, symlinked into site-packages.
  Branches on the fork `celinaky/ray`:
  - `celina/prefill-token-autoscaling-policy` at `ed547dd6dc`: the
    **mean policy**. Signal is the KV-aware router's gauge of total
    outstanding uncached prefill tokens across all replicas; replicas =
    `ceil(60s time-weighted mean / target)`. Prefill-only (the decode term
    was removed this session, commit `ed547dd6dc`). Also carries the vLLM
    scheduler tracing commit `02b42b7932`.
  - `celina/prefill-token-autoscaling-p95` at `debcf256b6`: the
    **arrival-p95 policy**. The router records, per request, the uncached
    prefill tokens already outstanding on the replica it chose (own tokens
    excluded); the policy takes the p95 of those samples over
    `upscale_lookback_s` and asks for `ceil(current_replicas x p95 /
    target)`. `d2648c05bc` added a `min_arrival_samples` gate (hold the
    count until the window has N arrivals) and a one-replica-per-decision
    step cap; `debcf256b6` removed the step cap and kept the gate.
  - Uncommitted on both: a try/except import shim in
    `python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/tokenizer.py`
    for vLLM 0.31's moved `chat_utils`. Needed to deploy.
  - Ray unit tests for the p95 policy were not updated (user's choice).
  - As of 2026-11-02 the checkout is on `celina/prefill-token-autoscaling-policy`.
- Dynamo runtime 1.7.0 wheel from `/mnt/cluster_storage/kvfinal4/`
  (`pip install --no-deps --force-reinstall`), needed for
  `session_affinity_mode`. Both branches use it.
- Routing: `kv_aware` with session pinning (ray-project/ray#67224), soft
  affinity via the `x-session-id` header, TTL 1800s. AIPerf sends
  `--session-header X-Session-ID` (harness `benchmark.py`).
- Engine-side ground truth: `RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR=/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces`
  must be set when launching. `tools/load_breakdown.py <run>` turns the
  traces into `analysis/load_requests.csv` with, per request,
  `engine_load_tokens` (uncached tokens outstanding on its engine at
  arrival, own excluded), `queue_ms`, `prefill_ms`, `client_ttft_ms`,
  `own_uncached`, `steps`. Every chart below reads that file.
- Workload: AgentX replay `semianalysis_cc_traces_weka_062126_256k`, seed
  42, one cold session every 30s from 1 to 32 over 930s, hold at 32 for
  930s, no ramp down. A session's first turn is tiny, its second turn is a
  47K to 64K cold prompt, later turns are prefix-cached; subagent fan-outs
  add 40K to 170K cold prompts at random. About 270 uncached tok/s per
  pinned session. 4 sessions per replica keeps p95 load at arrival under
  6K, 5 gives ~15K, 6 or more misses 20K. 32 sessions need 3 to 4 replicas.
- Launch:
  `RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR=... autoscale-harness run experiments/<file>.yaml --replace`.
  Before launching: no live `aiperf` processes (orphaned workers from a
  killed launcher contaminate the next run; `pkill` of the launcher does not
  kill them), no `core` dump files in the repo (the runtime env uploads the
  working dir with a 512 MB cap), and Serve fully torn down.

## Scoring: "meeting the target"

Per window of 100 requests, the window meets the target when the p95 of
load at arrival (engine side, own tokens excluded) is at or under 20K.
Equivalently, at most 5% of the window's requests found more than 20K
ahead. `tools/arrival_load_windows.py <run> --per-requests 100` prints and
plots it.

## Runs on this curve

All with `experiments/b200_workday_curve_prefill*.yaml` or
`b200_workday_curve_default.yaml`, target 20K, 60s or 120s lookback,
max 8 replicas, scheduler tracing on.

| Run dir | Policy | Serve delays up/down | Windows meet/miss | Replicas mean, max | GPU-h load | TTFT p95 | cold TTFT p95 |
|---|---|---|---|---|---|---|---|
| `runs/20261031T000711Z-b200-workday-curve-prefill` | mean, 60s | 0 / 180 | (earlier curve variant) | up to 8 | | | |
| `runs/20261031T182704Z-b200-workday-curve-prefill` | mean, 60s | 0 / 180 | 9 / 5 | 2.9, 8 | 2.10 | 5.0s | 16.5s |
| `runs/20261031T230121Z-b200-workday-curve-prefill` | arrival p95, 120s, gate 50 | 10 / 600 | 13 / 3 | 5.1, 8 | 3.10 | 2.9s | 14.2s |
| `runs/20261102T085146Z-b200-workday-curve-prefill-single-replica` | none (1 replica) | | 0 / 12 | 1, 1 | | 7.9s | |
| `runs/20261102T203807Z-b200-workday-curve-prefill-mean` | mean, 60s | 40 / 1800 | 11 / 5 | 2.1, 3 | 1.15 | 2.7s | 6.3s |
| `runs/20261102T223604Z-b200-workday-curve-default` | Serve default, target 8 ongoing | 40 / 1800 | 14 / 2 | 2.3, 3 | 1.26 | 2.5s | 6.0s |

Failed or aborted attempts (deploy failures, orphaned-AIPerf contamination,
stopped early), safe to delete: `20261030T223130Z`, `20261030T234941Z`,
`20261030T235750Z`, `20261030T235955Z`, `20261031T222130Z`,
`20261031T224047Z`, `20261031T224136Z`, `20261031T224248Z`,
`20261031T224502Z` (all `-b200-workday-curve-prefill`).

## Findings, in the order they were learned

1. **The mean-of-total-backlog signal under-counts.** At 32 pinned sessions
   the 60s mean reads about one replica's worth (10 to 25K) while the p95 of
   load at arrival says 2 to 3 are needed. The mean measures work, the p95
   measures how often requests collide, and the target is about collisions.
   In the mean-policy runs every window where one replica held 6 or more
   sessions missed the target while the policy's own number sat under 20K.
2. **Scale-ups without pinning cause re-prefill storms**: new replicas pull
   sessions whose context is cached elsewhere. **Scale-downs with pinning
   evict sessions**, and the evicted sessions' re-prefill drives a wave to
   the replica cap. Every scale-down in `20261031T000711Z` and
   `20261031T182704Z` did this. Hence the long downscale delays later.
3. **The ratio rule `current x p95 / target` compounds under pinning.** With
   p95 roughly constant at 27 to 38K (one replica's hot sessions), the p95
   run went 1, 2, 4, 6, 7, 8 in five minutes because the ratio is applied to
   the growing count. The first p95 attempt without the sample gate took a
   single collision among 15 samples as a p95 of 65K and went 1 to 4 to 7 in
   two minutes. A step rule (+1 when p95 > target, -1 when p95 < target/2)
   was discussed, not implemented.
4. **Nearly every over-target wait is behind one unsplittable cold prompt.**
   `tools/arrival_collision_causes.py --simple`: in the mean-policy run 43
   over-target requests were behind one prompt that alone had more than 20K
   left, 4 were stacked behind several smaller ones. Spreading requests
   cannot fix the first kind; only a replica per cold start would.
5. **Per replica, the distribution of load at arrival is a spike at zero
   with a thin tail.** 68% to 100% of requests in every 120s window found
   an empty engine on their replica (`tools/replica_arrival_bars.py`). The
   p95 moves on 2 or 3 collisions per window; the mean never moves.
6. **The router gauge reads about 2x the engine.** `tools/policy_vs_engine_mean.py`:
   the policy's 60s mean averaged 17.9K over the mean-policy run while the
   engines' remaining uncached prefill, rebuilt from traces, averaged 8.2K;
   ratio median 2.0, p10 0.7, p90 4.2, widest during cold bursts (55K vs
   24K at 840s). The shapes match. Main suspected cause: the gauge holds a
   request's whole uncached prompt until first token while the engine burns
   it down ~16K per step. Other candidates: the router's uncached estimate
   vs the engine's real cache hit, batched lifecycle-event lag, missed
   decrements. The counter is `_outstanding_effective_prefill_tokens` in
   `python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py`
   (incremented at admission ~line 644, decremented in
   `on_prefill_complete` ~line 693 and `on_request_completed` ~line 723).
   Options discussed for the gauge: calibrate the target to the gauge;
   FCFS linear drain model in the router at ~16K tok/s; online drain-rate
   calibration from first-token timings; engine progress reports.
7. **Serve's 40s upscale delay removed the single-prompt false starts.** In
   the mean-policy run the timer started five times and reset three times
   (4s, 16s, 4s) during single cold-prompt humps; the two that passed were
   40s+ sustained humps. With 0s delay each reset would have added a
   replica.
8. **With no scale-down and a 40s delay, the mean policy reached 3 replicas
   and the hold was clean** (every hold window under 7K p95, mostly under
   3K) at a third of the p95 run's GPU-hours. The cost was the ramp: 5
   missed windows, all before the third replica, while replica 1 held 6 to
   11 sessions at 30 to 57K p95.
9. **Serve's default policy did slightly better on this curve.** Ongoing
   requests track active sessions (one request in flight per session), so
   at target 8 it reached the same 3 replicas but 3 minutes earlier (first
   scale-up 422s vs 602s), giving 14/16 windows. It knows nothing about
   prompt size, and the result depends on the target choice as much as the
   token target does. Both policies miss the same first two windows, which
   no policy can act on.
10. **Single replica baseline**: p95 load at arrival rises to 220K, p95
    engine queue to 13s, p95 TTFT to 15.5s by the end of the hold; 0 of 12
    windows meet. Queue time is the entire difference between runs: client
    TTFT minus engine queue is the same 2 to 3s floor in every run (own
    prefill plus proxy and router time).

## What changed in the harness this session

- `config.py`: token policy requires only `target_prefill_tokens_per_replica`;
  `SESSION_PINNED_ROUTING = ("consistent_hash", "kv_aware")`; token policy
  requires kv_aware routing.
- `benchmark.py`: sends `--session-header X-Session-ID` for pinned routing.
- `analysis.py`, `evaluation.py`, `plotting.py`, `telemetry.py`: series for
  `prefill_arrival_p95_tokens`, `prefill_arrival_samples`, prefill-only
  evaluation (decode-tolerant), token panel gated on the policy's metrics.
  Tests updated; 5 pre-existing SLA test failures remain.
- Experiments: `experiments/b200_workday_curve_prefill.yaml` (p95),
  `..._prefill_mean.yaml` (mean, 40/1800), `..._prefill_single_replica.yaml`,
  `experiments/b200_workday_curve_default.yaml`.

## Chart tools added (all take a run dir, write into `runs/<run>/analysis/`, need `load_breakdown.py` first)

| Tool | What it draws |
|---|---|
| `arrival_load_windows.py --per-requests 100` | p95 of load at arrival per 100-request window, green/red vs target, replica count under each bar; `--style lines --mean` |
| `arrival_stats_lines.py` | all requests per 120s window: p95, p90, mean of load at arrival; `--match <run>` keeps another run's K-per-inch |
| `arrival_stats_stack.py` | same, three panels: load at arrival, engine queue time, client TTFT |
| `arrival_stats_compare.py <run>... --labels "a;b"` | 3x3 grid, rows load/queue/TTFT, columns mean/p90/p95, one line per run |
| `replica_percentile_lines.py --window 120` | per replica, faint line per replica: p95, p90, mean panels |
| `replica_arrival_bars.py` | per replica per window, requests stacked by load-ahead bucket (grey 0, blues under target, reds over) |
| `replica_arrival_strips.py` | per replica per window, one dot per request at its load ahead, p95/p50 bars |
| `replica_spread_timeline.py`, `replica_sessions_timeline.py` | per-replica p95 bars, request rows, root sessions per replica |
| `replica_load_timeline.py --window 60` | per-engine outstanding uncached prefill from traces, session-start markers |
| `arrival_collision_causes.py --simple --per-requests 100` | share over target split into behind-one-cold-prompt vs stacked |
| `policy_decision_timeline.py --backlog --per-requests 100` | 3 panels: engine percentiles plus the policy's signal; share over target per 100 requests; replicas with the ratio and mean rules; works for mean, p95, default and fixed-replica runs |
| `policy_vs_engine_mean.py [--raw]` | the policy's 60s mean of the router gauge vs the engines' remaining prefill from traces; upscale-delay timer episodes marked on the x axis; settings line under the legend |
| `backlog_vs_replicas.py` | raw gauge and 60s mean with replica add/remove markers |
| `arrival_load_timeline.py`, `arrival_load_bands.py`, `sessions_vs_arrival_load.py`, `replica_arrival_windows.py` | earlier variants, kept |

Scale decisions on every chart: solid line at the decision labelled
`before->after`, shaded band until the new replica is running
(`running_matched_at_s` from the harness evaluation), dashed line at ready.
Runs without a token policy are scored against 20K.

Comparison figure across runs: `runs/compare_arrival_stats120_20261102T203807Z_vs_20261102T085146Z.png`
(this one marks the moment the new replica was serving, not the decision).

## Open threads

- Why the router gauge is ~2x the engine (finding 6). Handed to a separate
  investigation; the data pointers are in finding 6.
- Policy design not yet implemented: step rule instead of the ratio;
  session-count-aware, one-at-a-time scale-down so pinned sessions are not
  evicted in waves; a drain model in the router; a collision-rate (share
  over target per N requests) signal, which is what the scoring uses.
- Ray unit tests on the p95 branch; the tokenizer shim is uncommitted.
- The default policy's result depends on `target_ongoing_requests`; a small
  sweep (6, 8, 10, 12) would show whether its 14/16 was luck.
