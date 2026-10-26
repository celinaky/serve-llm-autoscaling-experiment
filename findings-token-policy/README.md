# Token-based autoscaling policy vs Serve's default: findings

Working notes from the investigation of 2026-10-22 to 2026-10-26. This file
is the summary; the run-by-run evidence is in `sweep.md` (target sweep and
signal isolation on the workday ramp) and `pairs.md` (best-of-each
comparisons on five workload shapes). Start here, go to those for numbers
behind any claim.

## Setup

- Node: one head node `4xB200:92CPU-960GB`. Model `Qwen/Qwen3.6-27B-FP8`,
  one GPU per replica, `max_replicas: 4`, replica boot ~65s.
- Ray: checkout at `/mnt/local_storage/ray`, symlinked into site-packages
  (`import ray.serve.experimental.llm_token_autoscaling_policy` resolves
  there). Policy work was on branch `celina/prefill-token-autoscaling-policy`;
  the five diagnostic gauges the harness scrapes
  (`serve_token_autoscaling_*`) are committed on
  `celina/autoscaling-pd-disagg` (commit `f49d565d16`), which is the branch
  checked out as of 2026-10-26.
- Policy: `LLMTokenAutoscalingPolicy`, replicas =
  `max(ceil(outstanding_uncached_prefill_tokens / P),
  ceil(active_decode_requests / D), 1)`. Both inputs come from the kv-aware
  router's work snapshot, so `routing.policy: kv_aware` is required.
- Default policy: Serve's `target_ongoing_requests` per replica.
- Workload: AgentX trace replay (`semianalysis_cc_traces_weka_062126_256k`)
  through AIPerf 0.13.0, every session starting cold at turn 0 (its first
  request is a 64K-170K-token prompt; later turns hit the prefix cache),
  plus synthetic 32K-prompt shapes with prefix caching off.
- SLO scored at p95 over the whole load period: TTFT <= 5000ms, ITL <= 50ms.
  3000ms was the first target and is infeasible (section 1).
- Harness: `autoscale-harness run <config>`; `autoscale-harness compare
  <run dirs>`. Each run directory has
  `analysis/autoscaling_evaluation.json` (SLO attainment per request and
  per window, GPU-hours, every scale event with the binding term, TTFT
  decomposition, Spearman signal strength) and
  `analysis/autoscaling_timeline.png`.
- Measured noise: identical-config repeats differ by about +-0.3s on p95
  TTFT. Differences under that are not treated as evidence.

## Configs

| what | where |
| --- | --- |
| committed best token config, workday ramp | `experiments/b200_workday_ramp_token.yaml` (P=60K, D=8, `upscale_delay_s: 0`, `downscale_delay_s: 300`, `metrics_interval_s: 2`, `look_back_period_s: 10`) |
| default policy, same ramp and delays | `experiments/b200_workday_ramp_default.yaml` (target 8), `experiments/sweep/default_target{4,6,12}.yaml` |
| sweep variants (P/D isolation, delays, chunk size, storm, warm) | `experiments/sweep/` |
| paired shapes, both policies | `experiments/pairs/{storm,bigramp,mixed,synth32k,synthcold32k,synthstep32k}_*.yaml` |
| queue-free TTFT floor | `scripts/ttft_floor.py` |

## 1. The SLO: 3s TTFT is infeasible, 5s is the right target

22-25% of requests are cold (<50% of the prompt cached): session first
turns, subagent first turns, and root follow-up turns whose session was
moved by kv-aware routing. A B200 prefills at ~16-20K tok/s, so a 64K
prompt is 4s before any queueing. The queue-free floor (5th percentile
TTFT per power-of-two uncached-token bucket, `scripts/ttft_floor.py`) puts
p95 TTFT at 3.8s with zero waiting. Sweep run 1 held four replicas from
t=310s and still scored 5017ms. No replica count shortens a prompt's own
prefill. The 5s target is the floor plus the ~1s of queueing and noise
that remains, and the committed config met it in three of three runs.

ITL p95 <= 50ms was met by every configuration that did not saturate a
single replica.

## 2. Head-to-head, best of each policy, five shapes

Every pair ran with identical engine, routing, delays (`upscale_delay_s:
0`, `downscale_delay_s: 300`) and 2s metric push, with a small sweep of
each policy's own knob. Numbers are p95 TTFT ms / p95 ITL ms / GPU-h.

| shape | token (best) | default (best) | result | why |
| --- | --- | --- | --- | --- |
| workday ramp: 32 cold sessions over 10 min, 5 min hold | P=60K D=8: 4511 / 28 / 0.87 | target 8: 4829 / 38 / 0.70 | token wins both latencies, +24% GPU; at matched cost (target 4, 0.85 GPU-h) default is 0.4s worse | second replica serving at ~70s vs 169s; sessions arriving in between queued on one replica |
| storm: 32 cold sessions in 150s | P=30K D=8: 5042 / 37 / 0.99 | target 4: 5462 / 50 / 0.96 | token wins both, +3% GPU; neither meets 5s (instant-startup bound 4.75s) | asks for all four replicas at 13-31s so they boot in parallel; default steps 1->2->3->4 at 42/60/73s |
| big ramp: 64 cold sessions over 20 min, 5 min hold | P=60K D=8: 5536 / 42 / 1.62 | target 6: 5778 / 47 / 1.45 | token ahead on ITL; TTFT inside noise; waiting ordering 1659 < 1949 < 2124ms is not; +12% GPU | at the cap by ~100s vs 401s; after that both saturated at 16 sessions/replica |
| stepped synthetic: 32K cold prompts at 0.3 / 0.8 / 1.3 req/s, 5 min each | P=60K D=8: 3015 / 48 / 0.80 | target 3: 2861 / 41 / 0.82 | tie (inside noise, same cost) | same-size prompts: ongoing requests = rate x lifetime measures the prefill work as well as tokens do |
| synthetic overload from t=0 (AIPerf ramp overshoot) | 42986 / 202 / 0.98 | target 3: 9538 / 201 / 0.99 | no policy difference | both asked for the cap within 40s, within seconds of each other; the token run's replicas took ~140s to boot instead of ~70s (436 vs 208 starting replica-s), which is infrastructure, not policy |
| mixed warm/cold ramp | dropped | dropped | -- | AIPerf primes mid-trace sessions with a burst that pins both policies at the cap before measurement |

Run directories: workday ramp chain 5 `runs/20261023T230816Z-*`; storm
`runs/20261024T050550Z-*`; big ramp `runs/20261024T055946Z-*`; stepped
synthetic `runs/20261024T084125Z-*`; overload `runs/20261024T073727Z-*`.

## 3. Where the token policy's advantage comes from

**It scales earlier on agentic ramps, because a request count cannot see
a lone cold session.** A new session is one request carrying a 64K-128K
cold prefill. To `target_ongoing_requests: 8` that is one request, so the
default waits for eight or more sessions in flight. To the token policy it
is 64K+ tokens against P=60K, so it asks on the first snapshot. On the
workday ramp the token policy's three decisions came at 11, 99 and 371s
versus 149, 381 and 561s for the default, and its waiting at p95 was 813ms
versus 1028ms. It landed within 60ms of the instant-startup ceiling for
the shape (4452ms), i.e. it extracted essentially all the gain available
to any autoscaler.

**In a burst it asks for several replicas at once.** A few concurrent 64K
prefills put 200K+ tokens on the backlog; divided by P=30K that is four
replicas in one step, so three boot in parallel instead of in sequence.
That is the storm result (waiting 1295 vs 1802ms).

**It adds nothing when the count already measures the work.** With
same-sized prompts at a steady rate, ongoing requests tracks prefill work
exactly and the prefill term has no extra information. Under overload from
the first second both signals saturate at once. In those cases the two
policies make the same decisions within seconds of each other.

**Its cost is earlier replicas**: +3% to +24% GPU time over the default's
cheapest passing setting. At matched cost it was never worse.

## 4. Which term binds

The policy takes the max of its two terms. From the evaluation files, the
binding term for each change in the running replica count:

| shape | 1->2 | 2->3 | 3->4 |
| --- | --- | --- | --- |
| workday ramp | prefill, 11s | decode, 99s | prefill, 371s |
| storm P=30K | prefill, 13s (to 3) | | prefill, 31s |
| big ramp | prefill, 13s | both, 103s (to 4) | |
| stepped synthetic | prefill, 18s | prefill, 122s | decode, 618s |
| overload synthetic | both, 23s | prefill, 35s | prefill, 41s |

The prefill term makes the first decision in every run. The decode term
takes over as sessions reach the generating phase and is what holds the
count up (it rises smoothly; the prefill term is a spike train that falls
back to zero as each prompt clears). After the cap the prefill term pulses
the desired count to 5-22 and the decode term pulls it back to 4; Serve
clamps all of it and the running count never moved. On the stepped
synthetic the prefill term's two early scale-ups were over-provisioning
(step one fit on one replica); the one that mattered came from the decode
term. Removing the prefill term reproduces the default policy within noise
(sweep run 9 vs 10); removing the decode term leaves a deployment that
either never scales under a 10s upscale delay (run 7, ITL p95 75ms at one
replica) or flaps on spikes.

## 5. The signals themselves

**Active decode requests** is a moderate predictor of ITL (Spearman +0.26
to +0.73 across runs; p90 ITL 24 -> 40ms across its quartiles). D=8 held
ITL p95 at 29-45ms in every run; D=10 is the cost floor at 46ms. It is
nearly the same quantity as Serve's ongoing-request count (decode is most
of a request's life), which is why the decode-only configuration and the
default at target 8 land within noise of each other.

**Uncached prefill backlog** is a real predictor of waiting, but it has to
be measured correctly. Against the scraped gauge the Spearman with
per-request waiting is +0.01..+0.17, because Ray exports gauges every 10s
(`metrics_report_interval_ms`) and the scraped value lags reality by
12-14s (cross-correlation peak, rho +0.46..+0.49). Rebuilt from the
request records (uncached tokens of requests started but not yet at first
token, which is what the KV tracker counts between routing and
`on_prefill_complete`) it predicts waiting at **+0.20..+0.36**, the same
order as decodes -> ITL. It is moderate rather than strong because kv-aware
routing pins a session to one replica, so the waiting a request sees is
behind other prefills on its own replica, which a global sum cannot see.

**The prefill term is a spike train and the upscale delay silences it.**
One replica clears a 64K prompt in ~5s, so the backlog is zero most of the
time and spikes to 100-340K tokens for a few scrapes. With
`upscale_delay_s: 10` the P=60K term exceeded the running count 14 times in
a run and never for 10s together; the deployment stayed at one replica.
With `upscale_delay_s: 0` the same term scaled at 46s versus 237s for the
decode term and 175s for the default. This is why the committed config has
no upscale delay, and why `downscale_delay_s: 300` does the anti-flap work
instead (the desired count oscillated at the cap; running replicas never
did).

**Faster delivery changes when it scales, not the result.** Recording and
pushing the router snapshot every 2s instead of 6s/10s moved the first
scale-up from 50s to 10s and the second from 259s to 99s; TTFT p95 stayed
in the same band (4956 vs 4924/4985ms). The 2s push is kept for reaction
time; it is not where the remaining latency is.

## 6. Tuning: both policies need it, for different reasons

The default's best target was 8 (ramp), 4 (storm), 6 (big ramp), 3
(steps), and a wrong one fails outright: target 12 scored 5653ms on the
ramp; target 5 on the steps scaled to four on the step-2 transient, saw
ongoing requests below target for 300s, dropped to two at 638s exactly as
the 1.3 req/s step arrived, and scored 15.8s. A count has no fixed meaning
in GPU work, so the right target moves with both request size and arrival
pattern.

The token policy's P moved too (60K on the ramp, 30K in the storm), but
for one reason only: P converts a backlog *snapshot* into replicas, and
the right conversion depends on how much more work arrives during a 65s
boot. In a storm the same 100K snapshot should mean "four replicas"; in a
trickle it should mean "this spike will clear itself". Request size does
not move P through total work (eight 8K prompts and one 64K prompt are the
same backlog, which is the policy's genuine advantage over a count), but
it leaks in through peak height: a lone 128K prompt against P=30K asks
for five replicas for one session. Aligning the default's delays and
delivery with the token config also mattered: target 8 went from 5234ms
(10s delay, 10s push) to 4829ms and a pass. Part of the token policy's
margin in the first sweep was the delivery path, not the signal.

Not a lever: prefill chunk size. 8K and 32K `max_num_batched_tokens` were
both worse than 16K on every column.

## 7. What remains is not an autoscaling problem

On every shape the TTFT decomposition puts most of p95 TTFT in the
prompt's own prefill (1.6-4.1s) and most of the rest in waiting behind
other prefills on the request's own replica. Startup-lag waiting (what an
instant-booting replica would remove) is 0-0.3s. Both policies sit within
~0.1s of the instant-startup ceiling on the agent ramp. Getting below that
means fewer session moves (74 root follow-up turns per run arrived with
zero cached tokens because kv-aware routing had moved the session: see
`findings-claude-code-overnight/b200_investigation.md` section 5) or
engine scheduling (head-of-line blocking behind long prefills), not a
different autoscaler.

## 8. Recommended next steps for the policy

1. **A rate-based prefill term**: uncached tokens *arriving* per second
   over the last 30-60s, divided by a replica's prefill throughput (~16K
   tok/s here). It is in capacity units (invariant to request size, like
   P) and averages over a window longer than the spikes (invariant to
   burstiness, unlike P), so it survives `upscale_delay_s` and would need
   retuning only when the hardware changes.
2. **A per-replica backlog** from the KV tracker, which already knows
   which replica each request went to, so the policy can scale on the max
   replica queue rather than the global sum. That is where the remaining
   waiting lives.
3. **A size-shift experiment** that none of the five shapes covers: hold
   0.6 req/s, send 8K cold prompts for five minutes then 64K at the same
   rate (and the reverse). Prediction: a default tuned on the first phase
   keeps one or two replicas through the switch because the count barely
   moves; the token policy scales within one scrape. This isolates the one
   thing the token signal knows that a count cannot.

## 9. Harness caveats worth knowing before the next run

- AIPerf 0.13 `request_rate_series` ramp segments overshoot the configured
  rate by 3-15x; plateaus are delivered correctly. Use steps, not ramps
  (`synthstep32k_*`). Noted in the top-level README.
- AIPerf primes every session that starts mid-trace by replaying its
  earlier turns at full speed, and that burst is load the autoscaler sees.
  `trajectory_start_min/max_ratio: 0/0` (every session cold) avoids it; a
  realistic warm/cold mix is not possible with AIPerf today.
- Replica boot time varies (70-140s per replica on this workspace). Check
  `starting replica-seconds` before attributing a difference to a policy.
- 1-10 requests per run fail with `ClientPayloadError` in batches that
  mostly coincide with a replica entering STARTING (HAProxy reload on
  topology change). A Serve-side issue, not the policy.
- Five `sla` preflight tests fail because the installed Ray has no SLA
  module (that policy was abandoned; its work is on branch
  `sla-autoscaling-evaluation`). Everything else passes.
- Each paired row above is one run per configuration. The +-0.3s noise
  figure comes from two identical-config repeats.
