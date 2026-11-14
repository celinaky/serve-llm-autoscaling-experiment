# H100: mean token policy, KV-aware routing, and the router's prefill gauge

Working notes from 2026-11-02 to 2026-11-03, two Claude Code sessions on the
8xH100 workspace:

- `fd1b87e8`: migration from B200, first H100 run (32 sessions), single-GPU
  queue-profiling setup.
- `c5ad060d`: the 16-session run, per-replica and per-session analysis, and the
  investigation of the router gauge against the engines.

It builds on `findings-token-policy/` and `findings-prefill-arrival-p95/`
(`RESTORE.md` section 7 covers the B200 to H100 move). The brief for the next
piece of work, adding decay to the router's gauge, is `HANDOFF-router-decay.md`.

## Setup

- 8xH100 80GB. Model `Qwen/Qwen3.6-27B-FP8`, a hybrid of attention and
  Mamba-style layers: vLLM runs it with 4 KV cache groups and a 1568-token
  block. TP1, one GPU per replica.
- Ray checkout `/mnt/local_storage/ray`, branch
  `celina/prefill-token-autoscaling-policy` at `e643369b6e`, on top of
  `upstream/kv-router-session-pin` (soft session affinity, 1800s TTL).
  vLLM 0.31.0, `ai-dynamo-runtime` 1.7.0.
- Policy `token`, `upscale_aggregation: mean`: replicas =
  ceil(60s time-weighted mean of the router gauge / 20K). Upscale delay 40s,
  downscale delay 1800s, so nothing scales down during a run.
- Workload: AgentX trace replay through the workday-curve AIPerf fork, seed 42,
  every session cold at turn 0.

## H100 against B200, same engine settings

| | B200 | H100 |
|---|---|---|
| KV cache per replica | 4.12M tokens | 1.26M tokens |
| Cold prefill rate per replica | ~16K tok/s | 6.6 to 7.8K tok/s |
| Replica boot | ~94s | 120 to 190s |
| `max_num_seqs` | 1024 | 512 (vLLM needs it at or under the 824 Mamba cache blocks) |

## Runs

| | B200 · 32 sessions | H100 · 32 sessions | H100 · 16 sessions |
|---|---|---|---|
| Run | `20261102T203807Z-b200-workday-curve-prefill-mean` | `20261103T064300Z-h100-workday-curve-prefill-mean` | `20261103T182949Z-h100-workday-curve-prefill-mean` |
| Curve | 1 to 32 over 930s, hold to 1860s | same | 1 to 16 over 450s, hold to 1050s |
| 100-request windows meeting 20K | 11 / 16 | 6 / 11 | 3 / 5 |
| Requests completed / failed | 1593 / 8 | 1048 / 39 | 418 / 15 |
| Replicas mean / max | 2.1 / 3 | 4.6 / 8 (cap) | 2.3 / 4 |
| Scale-up decisions | 602, 842s | 197, 487, 767 (3 to 7), 1147s | 195, 475, 715s |
| GPU-hours during load | 1.15 | 2.78 | 0.79 |
| TTFT p95 | 2.7s | 10.4s | 7.7s |
| ITL p95 | 33 ms | 97 ms | 67 ms |

The 16-session config is `experiments/h100_workday_curve_prefill_mean.yaml`. It
now has `max_replicas: 7` while the profiling engine holds a GPU; the run above
used 8 and never went past 4.

## 1. 32 sessions on H100 is a capacity and cache test, not a policy test

The first H100 run copied the B200 config. It hit the 8-replica cap, and the
3-to-7 scale-up at 767s landed after the spike it answered, since a replica
takes about 190s to boot. In the hold, the 200 to 250K gauge spikes were not
aggregate capacity: replica 1, holding the oldest pinned sessions, served 62
requests of which 33 were cold (2.9M tokens, about 89K each) at 87% KV, while
replicas 4 to 8 served 13 to 90 requests each with none over target. That led
to the 16-session curve.

## 2. 16 sessions: what the policy did

- Each cold prompt lifts the 60s mean to 2 to 2.6 replicas for about a minute,
  and it decays under 0.5 within about 90s. Each of the three scale-ups was a
  spike that outlasted the 40s delay.
- With 418 requests over 1050s, a 120s window holds 19 to 81 requests, so its
  p95 is the top one or two. The charts for this run use 180s or 50-request
  windows.
- New replicas got almost no sessions. Pinning happens on a session's first
  request, and after replica 3 came up at 602s there were no new root sessions:
  the lanes were full of long-running ones. Replica 3 got 0 root sessions and
  19 subagent requests, replica 4 one 42-second session. The policy's total
  signal asked for replicas that pinning kept from taking any of replica 1's
  load.
- AIPerf's concurrency counts open session trees, and a tree stays open while
  it waits out recorded think time (root inter-turn gaps: median 1s, p90 76s,
  p95 182s, max 433s). 13 sessions were live through the hold of the
  16-session run; 23 to 26 of 32, peak 28, in the 32-session runs. 17 sessions
  ran in 14 lanes; 11 lanes kept one session for the whole run.
- The 15 failed requests are HAProxy stream cuts: every one ended within a
  second of 120s after a replica became healthy (435s, 721s, 957s). The B200
  run had 8 and the 32-session H100 run 39, by the same mechanism.

## 3. Why replica 1 re-prefilled: cache size plus pinning

Replica 1 held 13 to 22 conversations whose latest prompts add up to 1.15M to
1.42M tokens, more than the 1.26M-token cache, while requests in flight took
another 25 to 75% of it. vLLM evicts least-recently-used blocks, so the
conversation idle longest loses its context.

- On replica 1 every conversation idle longer than 181s came back fully
  uncached, seven cases. The only gap over 150s that survived was 268s for a
  44K subagent.
- At 957 to 993s three such returns re-prefilled 305K tokens in 36s (70K,
  65K, 170K). The next request waited 14.5s.
- On B200 the same turns, matched by prompt length, arrived with 0.5K to 3.2K
  uncached.
- Replica 1's prefix-cache hit rate fell from 0.93 to 0.66 late in the run;
  replica 2, with fewer and younger conversations, stayed at 0.86 to 0.98.

So the per-replica limit on H100 is resident context: about 12 to 14 long
agentic conversations fill the cache.

## 4. Three different quantities have been called "outstanding prefill tokens"

The policy charts and the arrival charts used the same words for different
things. Use these names:

| Name | Definition | Where it appears |
|---|---|---|
| Prompts in prefill | Each request's whole uncached prompt, counted from routing until its first token, summed over replicas. No decay. | The router gauge, the policy's signal |
| Queued work | Uncached prompt tokens still waiting to be prefilled; each chunk leaves when its batch completes. Decays. | Green line of `tools/policy_vs_engine_mean.py` |
| Queued ahead at arrival | What one request found waiting on its own replica when it arrived, its own prompt excluded | Arrival-stats and replica charts |

In the minute 383 to 443s of the 16-session run these were 85K, 30 to 50K,
and 5.6K. The first is about twice the second because it holds whole prompts.
The third is lower again because requests don't arrive at random moments: a
session's next turn arrives after its previous one finished, so the sessions
whose prompts were prefilling were not arriving.

## 5. The router gauge against the engines

### What the router counts

`kv_token_tracker.py` in the Ray branch adds a request's
`effective_prefill_tokens` (prompt minus the overlap its Dynamo index reports
as cached) when the request is routed, and subtracts it in full when the
replica reports the first output token. Nothing is subtracted while prefill
runs. The policy publishes its 60s mean as `prefill_required_replicas`
(mean / target), so mean in tokens = `prefill_required_replicas` x 20K.

### B200: the router reads high, as expected

From session `0f3c0e18` (see memory note "token-policy gauge gap
decomposition"): on the B200 32-session run the router's mean averaged 15.4K
against 8.2K for queued work.

| Part of the 7.2K gap | Size |
|---|---|
| Whole prompt held until first token, no decay | ~5.2K |
| Router holds each request ~130 ms longer than the engine's span | ~0.8K |
| Some warm turns counted at full length (index missing blocks the engine had) | ~1.2K |

### H100: the router sometimes reads low

Same tool and definition (`tools/policy_vs_engine_mean.py`, queued work), all
three runs:

| | B200 · 32 | H100 · 32 | H100 · 16 |
|---|---|---|---|
| Router mean / engine mean, median | 1.97 | 1.46 | 1.74 |
| p10 of that ratio | 1.12 | 0.72 | 0.60 |
| Loaded time with router below engine | 3% | 25% | 19% |
| Stretches of 20s+ with router over 10K below | none | 355-415s (-31K), 797-845s (-47K), 1079-1151s (-97K), 1313-1335s (-25K) | 353-413s (-29K) |

A gauge with no decay can only fall under the decayed engine line if it added
too little for some requests. Every such stretch coincides with re-prefills of
evicted contexts (the `--mark-evictions` charts mark them).

### Direct evidence for the 353 to 413s dip (16-session run)

Raw gauge scrapes are 10s sample-and-hold, but each value is an exact integer
sum of what the router held, so individual states can be read:

| Scrape | Router gauge | Engine at that moment |
|---|---|---|
| 345.2s | 3,421 | dcea9a turn 2, 158,653 uncached, prefilling 344 to 364s |
| 365.2s | 6,028 | dcea9a still prefilling until 364.5s |
| 385.2s | 15,422 | d13287 turn 6 (79,710) and 4701d9 turn 2 (46,153) prefilling |
| 415.2s | 88,278 | exactly 2efb65 turn 0 (a new session, 88,278); ea262f turn 5 (99,973) and d85c0e turn 4 (49,694) also outstanding |

So for those returning sessions the router added little or nothing while the
engine re-prefilled them from scratch: its index believed their contexts were
cached and the engine had evicted them. Not every eviction is missed: dcea9a's
160K re-prefill at 650s was counted (the router mean rose above the engine).

### What is and isn't established about the cause

- Ruled out: the `ParentBlockNotFound` warnings (27 in the 16-session run, 86 in
  the 32-session run). Those are skipped *stores*, which leave the index
  missing blocks and push the router *up*. They also occur outside the dips.
- Tested and working: on a live engine with the router's own index (Dynamo
  `SelectionService`, same settings), a 100K prompt evicted by 1.3M tokens of
  other prompts dropped to 0 blocks in the index and the router's estimate
  became 100,000, matching the engine (`tools/kv_index_eviction_probe.py`). So
  removal events do reach the index in the simple case.
- Not tested: the leading hypothesis is the hybrid model. A prefix only hits
  in vLLM if every KV cache group has it, including the Mamba state. If a
  context's Mamba state is evicted while its attention blocks survive, the
  engine finds 0 cached but an index tracking attention blocks still sees the
  prefix. `tools/kv_event_group_probe.py` decodes the engine's events per
  group next to the index's view to check this; it did not run because the
  profiling engine was shut down.
- Per-request proof needs the router's `effective_prefill_tokens` logged next
  to the engine's `cached_prefix_tokens`; the branch doesn't log it.

## 6. Queue profile on one H100

`runs/20261103T180742Z-h100-queue-profile` (321 requests, burst schedule in
`experiments/h100_queue_profile.json`, engine outside Serve via
`tools/standalone_engine.py`). Engine queue time is linear in the uncached
tokens queued ahead and the same for every request size: about 1s per 10K up
to 220K, steepening to about 80s at 600K. Requests under 704 uncached tokens
(the chunk remainder) sit slightly lower at low load. Chart:
`analysis/load_queue_by_size.png`.

## 7. Operational notes

- The Dynamo 1.7.0 wheel is not on PyPI and is lost on workspace restart;
  reinstall from `/home/ray/default/migration/` with `--no-deps
  --force-reinstall`.
- Every replica add reloads HAProxy, and any stream still open 120s later is
  cut with `ClientPayloadError`. This affects every run with scale-ups.
- `tools/standalone_engine.py start [--kv-events]` runs one engine on a GPU that
  a detached Ray actor reserves, so harness runs see 7 GPUs. As of 23:13 UTC on
  2026-11-03 it is stopped and all 8 GPUs are free.

## Charts

- Comparison page (B200 and both H100 runs): https://claude.ai/artifact/AFPtH9nAqa7hdn4gTjFCoG
- 16-session run page: https://claude.ai/artifact/U4HcDyL4J559fPgVB5fDmb
- In each run's `analysis/`: `policy_vs_engine_mean60_evictions_*.png`,
  `session_lanes_*.png`, `mean_signal_breakdown60_*.png`,
  `router_vs_engine_signal60_*.png`, `router_undercount_330_460.png`,
  `kv_fill_1754314_0_1050.png`.

## Tools added (uncommitted)

| Tool | What it does |
|---|---|
| `tools/policy_vs_engine_mean.py --mark-evictions` | Existing chart plus a red tick for each returning session whose context the engine had evicted |
| `tools/router_vs_engine_signal.py` | Router's published mean against the engine rebuilt with the router's own definition (whole prompt to first token) |
| `tools/router_undercount_zoom.py` | Zoom on a window, each big prompt marked new or evicted |
| `tools/mean_signal_breakdown.py` | Policy mean total, per replica, and queued ahead at arrival |
| `tools/session_lanes.py` | Which session ran in each AIPerf lane, coloured by pinned replica |
| `tools/replica_kv_fill.py`, `tools/replica_kv_residency.py` | One replica's KV cache: running requests vs cached prefixes, LRU reconstruction |
| `tools/standalone_engine.py`, `tools/standalone_profile_run.py` | One engine outside Serve, and the queue-profile driver |
| `tools/kv_index_eviction_probe.py`, `kv_index_lag_probe.py`, `kv_index_partial_eviction_probe.py`, `kv_event_group_probe.py` | Probes of the router's index against a live engine |

## Open questions

1. Why the router under-counts some re-prefills (section 5). Run
   `tools/kv_event_group_probe.py` against `standalone_engine.py start
   --kv-events`.
2. What target to use on H100. 20K was calibrated on B200 against a gauge with
   no decay; on H100 20K of queued work is about 2s of queue.
3. Whether the policy needs a per-replica view. Pinning leaves new replicas
   idle while one replica holds the oldest, largest contexts.
