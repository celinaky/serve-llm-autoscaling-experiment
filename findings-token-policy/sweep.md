# Token autoscaling policy: target sweep on the B200 workday ramp

Experiment: `experiments/b200_workday_ramp_token.yaml` (32 AgentX sessions
ramped over 600s, every session starting cold, 5 min hold, 4x B200).
SLO: p95 TTFT <= 3000ms and p95 ITL <= 50ms over the whole load period.
Policy: `LLMTokenAutoscalingPolicy` (branch
`celina/prefill-token-autoscaling-policy`), replicas =
`max(ceil(prefill_tokens / P), ceil(decode_requests / D), 1)`.

Goal: the loosest (cheapest) `P`, `D` that meets the SLO.

| run | P (prefill tokens/replica) | D (decodes/replica) | TTFT p95 | ITL p95 | SLO | GPU-h | max replicas | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 (`runs/20261023T002143Z-b200-workday-ramp-token`) | 60000 | 6 | 5017ms FAIL (warm 2600ms, cold 7801ms) | 33ms PASS | FAIL | 0.82 | 4 | 1->2 at 154s, 2->4 at 244s (~13 sessions), held 4 to the end of load; 880 requests, 1 failed |
| 2 (`runs/20261023T004458Z-b200-workday-ramp-token`) | 120000 | 6 | 4867ms FAIL (warm 2853ms, cold 6627ms) | 32ms PASS | FAIL | 0.72 | 4 | 1->2 at 156s, 2->3 at 346s, 3->4 at 507s (~27 sessions); 882 requests, 9 failed |
| 3 (`runs/20261023T010742Z-b200-workday-ramp-token`) | 240000 | 6 | 4655ms FAIL (warm 2800ms, cold 6906ms) | 33ms PASS | FAIL | 0.75 | 4 | 1->2 at 157s, 2->3 at 347s, 3->4 at 407s; 891 requests, 5 failed. Same timing as run 2: P is not the binding term |
| 4 (`runs/20261023T013031Z-b200-workday-ramp-token`) | 120000 | 10 | 5335ms FAIL (warm 3011ms, cold 7353ms) | 46ms PASS | FAIL | 0.54 | 3 | 1->2 at 257s, 2->3 at 527s; never 4; 802 requests, 7 failed. Both warm TTFT and ITL at the edge |
| 5 (`runs/20261023T015316Z-b200-workday-ramp-token`) | 120000 | 8 | 5118ms FAIL (warm 3370ms, cold 7681ms) | 40ms PASS | FAIL | 0.66 | 4 | 1->2 at 237s, 2->3 at 407s, 3->4 at 607s; 851 requests, 10 failed |

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
