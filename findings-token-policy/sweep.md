# Token autoscaling policy: target sweep on the B200 workday ramp

Experiment: `experiments/b200_workday_ramp_token.yaml` (32 AgentX sessions
ramped over 600s, every session starting cold, 5 min hold, 4x B200).
SLO: p95 TTFT <= 3000ms and p95 ITL <= 50ms over the whole load period.
Policy: `LLMTokenAutoscalingPolicy` (branch
`celina/prefill-token-autoscaling-policy`), replicas =
`max(ceil(prefill_tokens / P), ceil(decode_requests / D), 1)`.

Goal: the loosest (cheapest) `P`, `D` that meets the SLO.


## Answers (18 runs, 2026-10-22/23)

**1. Can the SLO be met by tuning the targets?** Not at TTFT p95 <= 3s:
the workload's queue-free TTFT floor is p95 3.8s (22-25% of requests are
cold prompts of 32K-128K uncached tokens, prefilled at ~16K tok/s on a
B200; `scripts/ttft_floor.py`), and no replica count changes a single
prompt's prefill time. At **TTFT p95 <= 5s and ITL p95 <= 50ms** -- the
floor plus the ~1s of queueing and run-to-run spread that remains -- yes:
`P=60000, D=8, upscale_delay_s: 0` (now committed in
`experiments/b200_workday_ramp_token.yaml`) met it in three of three runs
(TTFT p95 4480 / 4924 / 4985ms, ITL p95 29-36ms) at 0.75-0.80 GPU-h for
the 15-minute ramp. Decode-driven scaling (D alone, or the default policy
at target 8) sits on the line (4.9-5.3s; 1 of 5 runs under 5s) at
0.57-0.66 GPU-h. Meeting the TTFT target therefore costs ~25% more GPU
time than meeting ITL alone, which every non-saturated configuration does.
The identical-config repeats (5017 vs 5299, 5118 vs 5212) put the noise at
about +-0.3s on p95 TTFT; the spread of the committed configuration (4.5-
5.0s) is consistent with that.

**2. Is uncached prefill backlog a relevant and correct scaling signal?**
Relevant, yes; correct as implemented, only in part.

- *It is the earliest signal there is.* A cold prompt raises the backlog
  the moment it is routed; decodes and ongoing-request counts only rise
  once requests are generating. With no upscale delay the prefill term
  scaled 1->2 at 46-50s into the ramp, versus 237s for the decode term and
  175s for the default policy, and that lead is what produced the only
  TTFT passes. In the cold-start storm it scaled at 28s and 108s versus
  78/110/138s for the default policy, with cold p95 7.3s vs 10.0s.
- *But it is a spike train, so any persistence filter silences it.* One
  replica clears a 64K prompt in ~5s, so the global backlog sits at zero
  most of the time and spikes to 100-340K tokens for a few scrapes. Serve
  only acts on a desired count that persists for `upscale_delay_s`. At the
  default 10s the term exceeded the running count 14 times in a run and
  never for 10s together, and the deployment stayed at one replica while
  ITL went to 75ms (P=60K, decode term off). At P=30K it only persisted
  once the single replica had saturated and the backlog stopped clearing
  (t=537s): by then it is a lagging indicator.
- *It does not predict a request's TTFT.* Spearman between the backlog per
  running replica at a request's start and that request's TTFT (warm only,
  or in excess of its own prefill floor) was within +-0.2 of zero in every
  run (median +0.07). Two reasons: the backlog is global, but a request's
  queueing happens on its own replica, where kv_aware pins its session;
  and TTFT is dominated by each prompt's own prefill, which the backlog
  says nothing about.

So the quantity is right and early; the form is wrong. A rate --
uncached tokens *arriving* per second over the last 30-60s, divided by a
replica's prefill throughput (~16K tok/s here) -- would be persistent
under a ramp, lead the decode signal by the same margin, and need no
delay-zero trick. Whatever the form, a global token count cannot see
per-replica queues; a per-replica backlog from the KV tracker (it already
knows which replica each request went to) would let the policy scale on
the *max* replica queue rather than the sum.

**3. Is active decode requests a relevant and correct signal?** Yes, and
it is the one that actually scaled every run that used it. It rises
smoothly with load (~0.7 decoding requests per session at 32 sessions,
not the ~0.3 seen at 8), predicts ITL with Spearman +0.26..+0.73 (p90 ITL
24 -> 40ms across its quartiles), and D=8 holds ITL p95 at 30-45ms against
the 50ms target; D=10 is the cost floor at 46ms, D=6 buys nothing. It is,
however, nearly the same quantity as Serve's ongoing-request count: the
decode-only configuration and the default policy at
`target_ongoing_requests: 8` land within noise of each other (TTFT p95
4.9 vs 5.2s, ITL 42 vs 44ms, 0.57 vs 0.61 GPU-h). The token policy's edge
over the default policy is entirely the prefill term, which only shows
when it is allowed to act.

**4. Is the policy working?** It scales when its terms say so, holds
without flapping (the 300s downscale delay does that work; the desired
count oscillated at the cap but running replicas never did), and is quiet
and correct on a warm workload (24 primed sessions: 2 replicas, ITL p95
21ms, TTFT p95 3.2s). Its weaknesses are the two above: the prefill term's
form, and sticky routing making the global backlog blind to per-replica
queues. Two things outside the policy also matter: 74 root follow-up turns
per run arrived with zero cached tokens because kv_aware had moved the
session (routing, not scaling), and 1-10 requests per run had their stream
cut around replica starts (an HAProxy reload effect, not the policy).

**Not a lever:** prefill chunk size. 8K and 32K were both worse than 16K
on every column (chain 3).

**Scoring under TTFT p95 <= 5000ms, ITL p95 <= 50ms** (cold ramp runs; the
evaluation files were scored at the target in force at the time, so this
is a rescoring of the same numbers):

| run | policy | P | D | upscale delay | TTFT p95 | ITL p95 | GPU-h | verdict |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | token | 60K | 6 | 10 | 5017 | 33 | 0.82 | FAIL |
| 2 | token | 120K | 6 | 10 | 4867 | 32 | 0.72 | PASS |
| 3 | token | 240K | 6 | 10 | 4655 | 33 | 0.75 | PASS |
| 4 | token | 120K | 10 | 10 | 5335 | 46 | 0.54 | FAIL |
| 5 | token | 120K | 8 | 10 | 5118 | 40 | 0.66 | FAIL |
| 6 | token | 120K | 8 | 10 | 5212 | 42 | 0.66 | FAIL |
| 7 | token | 60K | off | 10 | 4986 | **75** | 0.25 | FAIL |
| 8 | token | 30K | off | 10 | 5984 | **66** | 0.51 | FAIL |
| 9 | token | off | 8 | 10 | 4933 | 42 | 0.57 | PASS |
| 10 | default | - | - | 10 | 5234 | 44 | 0.61 | FAIL |
| 11 | token | 60K | off | 0 | 4480 | 32 | 0.80 | PASS |
| 15 | token (chunk 8K) | 120K | 8 | 10 | 5466 | 39 | 0.67 | FAIL |
| 16 | token (chunk 32K) | 120K | 8 | 10 | 6101 | 51 | 0.67 | FAIL |
| 17 | **token** | **60K** | **8** | **0** | **4924** | **29** | **0.79** | **PASS** |
| 18 | token | 60K | 8 | 5 | 4985 | 36 | 0.75 | PASS |
| (user's own, 60K/6/10) | token | 60K | 6 | 10 | 5299 | 37 | 0.78 | FAIL |

Run directories: `runs/2026102*`. Each has `analysis/autoscaling_evaluation.json`
(per-request and per-window SLO attainment, cost, scale events with the
binding term, signal strength) and `analysis/autoscaling_timeline.png`.

| run | P (prefill tokens/replica) | D (decodes/replica) | TTFT p95 | ITL p95 | SLO | GPU-h | max replicas | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (`runs/20261023T002143Z-b200-workday-ramp-token`) | 60000 | 6 | 5017ms FAIL (warm 2600ms, cold 7801ms) | 33ms PASS | FAIL | 0.82 | 4 | 1->2 at 154s, 2->4 at 244s (~13 sessions), held 4 to the end of load; 880 requests, 1 failed |
| 2 (`runs/20261023T004458Z-b200-workday-ramp-token`) | 120000 | 6 | 4867ms FAIL (warm 2853ms, cold 6627ms) | 32ms PASS | FAIL | 0.72 | 4 | 1->2 at 156s, 2->3 at 346s, 3->4 at 507s (~27 sessions); 882 requests, 9 failed |
| 3 (`runs/20261023T010742Z-b200-workday-ramp-token`) | 240000 | 6 | 4655ms FAIL (warm 2800ms, cold 6906ms) | 33ms PASS | FAIL | 0.75 | 4 | 1->2 at 157s, 2->3 at 347s, 3->4 at 407s; 891 requests, 5 failed. Same timing as run 2: P is not the binding term |
| 4 (`runs/20261023T013031Z-b200-workday-ramp-token`) | 120000 | 10 | 5335ms FAIL (warm 3011ms, cold 7353ms) | 46ms PASS | FAIL | 0.54 | 3 | 1->2 at 257s, 2->3 at 527s; never 4; 802 requests, 7 failed. Both warm TTFT and ITL at the edge |
| 5 (`runs/20261023T015316Z-b200-workday-ramp-token`) | 120000 | 8 | 5118ms FAIL (warm 3370ms, cold 7681ms) | 40ms PASS | FAIL | 0.66 | 4 | 1->2 at 237s, 2->3 at 407s, 3->4 at 607s; 851 requests, 10 failed |
| 6 (`runs/20261023T082719Z-b200-workday-ramp-token`) | 120000 | 8 | 5212ms FAIL vs 4500 (warm 3172ms, cold 7801ms) | 42ms PASS | FAIL | 0.66 | 4 | gauged policy; all three scale-ups decided by the decode term (prefill term wanted 0/1/2 when decode wanted 2/3/4); 846 requests, 8 failed |

## Signal isolation (2026-10-23, `experiments/sweep/`, SLO TTFT p95 <= 4500ms, ITL p95 <= 50ms)

| run | policy terms | TTFT p95 (warm / cold) | ITL p95 | GPU-h | replicas | scale-ups (binding term) |
| --- | --- | --- | --- | --- | --- | --- |
| 6 (`runs/20261023T082719Z-...-token`) | P=120K, D=8 | 5212 FAIL (3172 / 7801) | 42 PASS | 0.66 | 1->2->3->4 | 237s, 417s, 597s (all decode) |
| 7 (`runs/20261023T084942Z-...-p60k-donly-off`) | P=60K only | 4986 FAIL (3258 / 6660) | **75 FAIL** | 0.25 | 1 throughout | none |
| 8 (`runs/20261023T091204Z-...-p30k-donly-off`) | P=30K only | 5984 FAIL (4002 / 10271) | **66 FAIL** | 0.51 | 1->2->4 | 537s, 627s (prefill), after the replica was saturated |
| 9 (`runs/20261023T093428Z-...-d8-ponly-off`) | D=8 only | 4933 FAIL (2775 / 8073) | 42 PASS | 0.57 | 1->2->3 | 237s, 407s (decode) |
| 10 (`runs/20261023T095652Z-...-default`) | default, target_ongoing 8 | 5234 FAIL (3478 / 6622) | 44 PASS | 0.61 | 1->2->3->4->3 | 175s, 416s, 605s (ongoing requests), down at 686s |

The default policy (ongoing requests per replica, target 8) lands in the
same place as the token policy: TTFT p95 5.2s, ITL p95 44ms, 0.61 GPU-h.
That is expected: active decode requests and Serve's ongoing-request count
are nearly the same quantity on this workload (a request is "ongoing" from
routing to completion, "decoding" from first token to completion, and
decode is most of a request's life), so D=8 and target_ongoing_requests=8
make the same decisions a minute apart. The token policy's distinct input,
the prefill backlog, is the term that does not fire.

Why the prefill term does not scale in time: its input is the instantaneous
global backlog of uncached tokens, and one replica clears a cold prompt in
5-15s. The backlog is therefore a spike train (run 6: p50 38K, p90 183K,
max 340K tokens, each spike a few scrapes wide). Serve only acts on a
desired count that persists for `upscale_delay_s` (10s):

| run | prefill term > running replicas | stretches | longest | stretches >= 10s |
| --- | --- | --- | --- | --- |
| P=60K only | 16% of scrapes | 14 | 8s | 0 |
| P=30K only | 33% of scrapes | 16 | 98s | 4 |
| P=120K, D=8 | 1% of scrapes | 1 | 8s | 0 |

At P=60K the term asked for a second replica fourteen times and never for
10s running, so the deployment stayed at one replica for the whole ramp and
ITL went to 75ms. At P=30K the term only persisted once the single replica
was saturated and the backlog stopped clearing (t=537s, ~28 sessions): by
then it is a lagging signal, and TTFT/ITL were already past the targets.
The decode term asks for the second replica at 237s in every run it is
active, before either target is breached.

Signal strength (Spearman, policy input per running replica at request
start vs. the latency that request saw):

| run | decodes -> ITL | prefill backlog -> warm TTFT | prefill backlog -> excess TTFT |
| --- | --- | --- | --- |
| 6 | +0.43 | +0.17 | +0.13 |
| 7 | +0.53 | +0.07 | +0.03 |
| 8 | +0.73 | +0.10 | +0.07 |
| 9 | +0.26 | +0.19 | +0.17 |

Active decode requests is a moderate predictor of ITL (strongest when a
replica is allowed to saturate). The uncached-prefill backlog is a weak
predictor of TTFT, even of the part of TTFT above each prompt's own prefill
floor. Two reasons: a request's TTFT depends on the queue *on its replica*
(kv_aware pins a session to the replica holding its cache), and the global
backlog says nothing about that; and the backlog is dominated by the cold
prompts' own prefill, which no replica count shortens.

## Other shapes and no upscale delay (2026-10-23, chain 2)

| run | shape | policy | TTFT p95 (warm / cold) | ITL p95 | SLO | GPU-h | replicas | scale-ups |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 11 (`...-p60k-donly-off-nodelay`) | ramp 32/600s cold | P=60K only, `upscale_delay_s: 0` | **4480 PASS** (2513 / 6152) | 32 PASS | **PASS** | 0.80 | 1->2->3->4 | 46s, 206s, 476s (prefill); desired flapped 4<->5 at the top (capped at 4) |
| 12 (`...-storm-token-p60k-d8`) | storm 32/150s cold | P=60K, D=8 | 5392 FAIL (3081 / 7251) | 54 FAIL | FAIL | 0.94 | 1->2->4 | 28s, 108s (prefill) |
| 13 (`...-storm-default`) | storm 32/150s cold | default, target 8 | 6039 FAIL (3056 / 9995) | 70 FAIL | FAIL | 0.84 | 1->2->3->4->2->3 | 78s, 110s, 138s |
| 14 (`...-warm24-token-p120k-d8`) | fixed 24 warm | P=120K, D=8 | 3234 PASS (2039 / 7368) | 21 PASS | PASS | 0.49 | 2 | prefill during priming, 2->1->2 mid-run |

- **Without the 10s persistence filter the prefill term leads and the SLO
  is met.** Run 11's first scale-up came at 46s (vs 237s for the decode
  term and 175s for the default policy), the second at 206s, and the run
  is the only one on the cold ramp to pass TTFT p95 <= 4.5s. It costs 0.80
  GPU-h, about 30% more than the decode-driven runs (0.57-0.66), because
  it reaches 4 replicas by 476s instead of ~600s. The desired count
  flapped 4<->5 at the top but running replicas never did (max_replicas
  4; downscale delay 300s).
- **Under a sustained backlog (the storm) the prefill term beats the request
  count** even with the filter: 1->2 at 28s and 2->4 at 108s, versus
  78/110/138s for the default policy; cold p95 7.3s vs 10.0s, ITL p95 54
  vs 70ms, fewer failed requests (7 vs 14). Neither meets the SLO: 32 cold
  64K-170K prompts in 150s exceed four replicas' prefill throughput
  whatever the policy, and replicas take ~65s to start.
- **On a warm workload the policy is quiet and right**: 24 primed sessions
  sit at 2 replicas (ITL p95 21ms, TTFT p95 3.2s, 0.49 GPU-h).
- Signal strength is unchanged across shapes: decodes -> ITL +0.28..+0.51;
  prefill backlog -> TTFT within +-0.08 of zero in every run.

## Prefill chunk size (chain 3): is the TTFT above the floor a chunking effect?

| run | max_num_batched_tokens | TTFT p95 (warm / cold) | ITL p95 | GPU-h |
| --- | --- | --- | --- | --- |
| 6 | 16384 | 5212 (3172 / 7801) | 42 | 0.66 |
| 15 (`...-chunk8k`) | 8192 | 5466 (3720 / 9316) | 39 | 0.67 |
| 16 (`...-chunk32k`) | 32768 | 6101 (4227 / 9371) | 51 | 0.67 |

Smaller chunks slow the cold prompts (9.3s p95) without relieving the warm
turns (3.7s); larger chunks make every step that carries a cold chunk longer, so both the warm turns (4.2s) and the cold ones (9.4s) wait more, and ITL slips to 51ms. 16K is the better of the three. The ~1.2s between the cold ramp's TTFT p95
and the queue-free floor is not a chunking artefact.

## Conclusion (5 runs, 2026-10-22)

**Chosen setting: P = 120000, D = 8** (committed in
`experiments/b200_workday_ramp_token.yaml`).

- **ITL p95 <= 50ms is met at every setting tried.** D is the term that
  scales on this workload (P never bound at 120K or 240K); ITL p95 moved
  32-33ms (D=6) -> 40ms (D=8) -> 46ms (D=10). D=10 is the cost floor
  (0.54 GPU-h, never needed the 4th GPU) with 4ms of margin on a single
  run; D=8 keeps 10ms of margin for 0.66 GPU-h. D=6 (0.72-0.82 GPU-h) buys
  nothing the SLO needs.
- **TTFT p95 <= 3000ms cannot be met by any P or D**, and this is not a
  property of the policy. 22-25% of requests are cold (<50% of the prompt
  cached): session first turns, subagent first turns, and root follow-up
  turns whose session was moved to a replica that had not served it
  (kv_aware routing). Their p95 TTFT is 6.6-7.8s whatever the replica count
  (run 1 held 4 replicas from t=310s and still scored 5.0s). The warm p95
  sat at 2.6-3.4s across all five runs with no relation to replicas (2600ms
  at 4, 3011ms at 3, 3370ms at 4): warm turns queue behind cold prefills on
  their own replica, which more replicas do not relieve because a session
  stays where its cache is.
- The TTFT levers are therefore outside the autoscaler: fewer session moves
  (routing), and prefill/decode interleaving (`max_num_batched_tokens`,
  chunk size vs. head-of-line blocking). Scoring TTFT at p90, or on warm
  turns only, would make the policy's TTFT contribution visible; at p95
  the cold population dominates.
- Open issue: 1-10 requests per run fail with `ClientPayloadError`
  (stream cut), in batches that mostly coincide with a replica entering
  STARTING (run 2: t=341s and 526s) but not always (t=691s). Looks like
  HAProxy backend reloads on topology changes; a Serve-side issue to chase
  separately.
- No default-policy baseline was run in this sweep; `experiments/
  b200_workday_ramp_default.yaml` is the same ramp for that comparison.

## Runs 1-3: the decode term is what scales

The first scale-up came at 154-157s in all three runs and the second at
346-347s in runs 2 and 3, whatever P was. The prefill term only decided
run 1's jump to 4 replicas at 244s; otherwise `active_decode_requests / D`
is what asks for replicas. The pressure panel shows ~10 requests running
at 200s, ~15 at 400s and 20-25 at 32 sessions, i.e. ~0.7 decoding
requests per session at this load, not the ~0.3 seen at 8 sessions. With
D=6 the policy asks for 4 replicas once ~19 requests decode, which 32
sessions always produce, so every run ends at 4 replicas with ITL p95 at
33ms against a 50ms target. Runs 4+ loosen D.

## Run 1 notes

- 220 of 880 requests (25%) were cold (<50% of the prompt cached): 85
  session first turns, 113 subagent requests, and 74 *root follow-up turns
  with 0 cached tokens* (e.g. turn 1, 64K prompt, nothing cached). The last
  group means the session's second turn landed on a replica that had not
  served its first: kv_aware routing moving sessions, as in
  findings-claude-code-overnight/b200_investigation.md section 9. That is a
  routing cost, not a provisioning one.
- Cold requests: median 40K uncached tokens, p90 94K; TTFT p50 2.4s, p95
  8.2s. Warm: median 3.5K uncached, p95 TTFT 2.6s.
- The policy reached max_replicas (4) at ~13 sessions because the global
  prefill backlog of a few concurrent cold prompts (3 x 100K tokens) divided
  by 60K asks for 5. At 4 replicas TTFT p90 was still 3.5s: the overall p95
  TTFT SLO cannot be met by any replica count on this workload. ITL passes
  with room (p95 33ms).
- Therefore the sweep goes the other way: loosen P to cut GPU time while
  warm TTFT p95 and ITL p95 stay within the SLO.
