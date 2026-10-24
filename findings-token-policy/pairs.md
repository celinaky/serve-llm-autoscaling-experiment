# Token vs default policy, best-of-each, on several cold-session workloads

Follow-up to `sweep.md`. Each shape is run under both policies with every
other setting equal (engine, routing, SLO p95 TTFT <= 5s / ITL <= 50ms,
`upscale_delay_s: 0`, `downscale_delay_s: 300`, 2s metric push), with a
small sweep of each policy's own knob so the comparison is best against
best. Configs: `experiments/pairs/`.

| shape | config stem | what it probes |
| --- | --- | --- |
| workday ramp (reference) | `b200_workday_ramp_*`, `sweep/default_target*` | 32 cold sessions over 10 min; see sweep.md chain 5 |
| storm | `storm_*` | 32 cold sessions in 150s: a sustained prefill backlog |
| mixed | `mixed_*` | 32 sessions over 10 min, each starting in the first half of its trace: cold starts mixed with resumed sessions |
| bigramp | `bigramp_*` | 64 cold sessions over 20 min, 5 min hold: needs all four GPUs |
| synthcold32k | `synthcold32k_*` | 32K-token synthetic prompts, prefix caching off, Poisson arrivals: prefill-heavy and decode-light |

## Reference: workday ramp (chain 5)

Token P=60K D=8: TTFT p95 4511ms, ITL p95 28ms, 0.87 GPU-h. Default's best
(target 8): 4829ms, 38ms, 0.70 GPU-h; at matched cost (target 4, 0.85
GPU-h) 4932ms. Token wins on both latencies; its 0.3s gain is entirely
waiting removed (813 vs 1028ms) and it is within 60ms of the
instant-startup ceiling (4452ms). Details in sweep.md.

## synth32k, first attempt: mis-specified, discarded

`synth32k_token` and `synth32k_default_t3` both saturated: desired counts of
28-36 replicas against a cap of 4 within 100s, cold p95 TTFT 62-68s, SLO
failed by miles for both. Two configuration mistakes, both the harness's
fault rather than the policies':

- AIPerf delivered ~2.4x the configured request-rate series (offered 1.15
  -> 3.95 req/s against a configured 0.1 -> 1.6), so the load exceeded
  four replicas' prefill capacity.
- The synthetic prompts shared a prefix: 80% of prompt tokens were read
  from the prefix cache, so the stream was not the all-cold one intended.

The shape is re-specified as `synthcold32k_*`: prefix caching off (every
request a full 32K prefill) and the series scaled down to 0.05 -> 0.65 so
the effective peak stays inside four replicas. Even in the broken form
the two policies ranked the same way as elsewhere (token p95 7.3s vs
default-t3 8.1s, ITL 71 vs 84ms), but nothing from those runs is used.

## Storm: 32 cold sessions in 150s (chain 7, aligned)

| run | policy | TTFT p95 | ITL p95 | GPU-h | replicas 2/3/4 asked for at | p95 = own prefill + waiting | instant-startup p95 |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `runs/20261024T050550Z-...-storm-token-p30k` | **token P=30K D=8** | **5042** | **37** | 0.99 | 13 / 13 / 31s (prefill term) | 3747 + 1295 | 4753 |
| `...-storm-token` | token P=60K D=8 | 5645 | 41 | 0.96 | 12 / 90 / 90s (prefill term) | 3820 + 1826 | 5645 |
| `...-storm-default-t4` | default target 4 | 5462 | 50 | 0.96 | 42 / 60 / 73s | 3660 + 1802 | 5462 |
| `...-storm-default` | default target 8 | 5729 | 60 FAIL | 0.87 | 60 / 90 / 120s | 4125 + 1604 | 5729 |

**Best of each: token P=30K vs default target 4. Token wins both latencies
(0.4s on TTFT p95, 13ms on ITL p95) for 3% more GPU time. Neither meets
the 5s target; nothing can on this shape** -- even an instant-startup
autoscaler would land at ~4.75s, because 32 cold 64K-170K prompts in 150s
outrun four replicas' prefill throughput for the first few minutes.

Why the token policy wins here, and why by less than on the ramp:

- In a storm the request count explodes almost as fast as the prefill
  backlog, so the prefill term's lead shrinks from ~140s (ramp) to
  30-60s: default-t4 asked for its second replica at 42s, the token policy
  at 12-13s. Both reach four replicas within the first 2.5 minutes.
- The win comes from asking for *all four at once*. At P=30K the first
  cold prompts (a few concurrent 64K prefills = 200K+ tokens) make the
  prefill term demand 3-4 replicas in one step at 13-31s, so all three new
  replicas load in parallel and are serving by ~95s. The default policy
  steps 1->2->3->4 as ongoing requests climb (42/60/73s), and P=60K stepped
  2->4 only at 90s. Waiting at p95: 1295ms vs 1802ms (default-t4) vs
  1826ms (P=60K).
- Tighter P costs almost nothing on this shape (0.99 vs 0.96 GPU-h)
  because every configuration ends at four replicas anyway; the only
  difference is how soon. On the gradual ramp P=30K would over-provision
  (see sweep.md run 8, where P=30K alone scaled late *and* the ramp runs
  where P=60K already reached the cap), so the right P is shape-dependent:
  the backlog a single replica can absorb before the policy should act.
- The default policy's own knob trades the same way: target 8 is cheaper
  (0.87) but fails ITL (60ms) and waits 1604ms; target 4 is its best.

Verdict on this shape: token > default on latency at equal cost; the SLO
is unreachable for both; the token policy's advantage is parallel scale-out
from a sustained backlog, which a request count cannot express as early.

## Mixed warm/cold ramp: not a scaling test as configured, dropped

`mixed_token` (sessions start anywhere in the first half of their trace)
ran at four replicas for the entire measured period (mean 4.0, 1.00 GPU-h,
zero starting time): the policy scaled 1->3->4 at t=-460s and -400s,
*before profiling began*. AIPerf primes every session that starts
mid-trace by replaying its earlier turns at full speed first, and that
priming burst (dozens of cold 64K prefills at once) is a storm any policy
scales to the cap on. The default policy would do the same, so the pair
would compare two deployments pinned at four replicas. The shape is
dropped; its remaining two runs were cancelled to give the time to the
big ramp and the synthetic cold-prefill shape. A realistic warm/cold mix
needs AIPerf to prime sessions *without* counting the priming as load the
autoscaler sees, which it cannot do today.

For the record, token on this shape: TTFT p95 5146ms (own prefill 3669 +
waiting 1476), ITL p95 16ms, 465 requests, cold p95 11.3s (resumed
sessions carry 100K+ contexts, so their first measured turns are large).

## Big ramp: 64 cold sessions over 20 min, 5 min hold (chain 8, aligned)

| run | policy | TTFT p95 | ITL p95 | GPU-h | replicas 2/3/4 asked for at | p95 = own prefill + waiting |
| --- | --- | --- | --- | --- | --- | --- |
| `runs/20261024T055946Z-...-bigramp-token` | **token P=60K D=8** | **5536** | **42** | 1.62 | 13 / 103 / 103s (prefill, then both terms) | 3877 + 1659 |
| `...-bigramp-default-t6` | default target 6 | 5778 | 47 | 1.45 | 149 / 263 / 401s | 3829 + 1949 |
| `...-bigramp-default` | default target 8 | 5974 | 46 | 1.36 | 151 / 385 / 595s | 3849 + 2124 |

**Best of each: token vs default target 6. Token ahead on both latencies
(0.24s TTFT p95, 5ms ITL p95) for 12% more GPU time; the TTFT margin is
at the edge of the +-0.3s run-to-run noise, the waiting ordering (1659 <
1949 < 2124ms) is not.** Neither meets 5s: 64 sessions is 16 per replica
at the cap, the per-replica load the overnight notes identified as the
ITL/TTFT knee, and the decomposition shows no startup-lag waiting left
to remove in any of the three runs (instant-startup p95 = actual).

Why: the token policy reached the cap at ~100s with 6 sessions running,
six to eight minutes before either default setting (401s / 595s), so
sessions 7-30 arrived to four replicas instead of two or three. That is
the whole latency difference and the whole cost difference. The prefill
term made the first two steps (13s, 103s); from 480s on, the decode term
asked for 5-7 replicas it could not have (D=8 at 40+ concurrent decodes),
which is the ITL margin shrinking toward the knee, not a policy fault.
On this shape "scale to the cap immediately" is simply right, and a
request-count policy cannot know that until the requests have arrived.

## Synthetic cold prefill, ramped (synthcold32k): an overload-recovery test by accident

AIPerf 0.13 delivered the configured 0.05 -> 0.65 req/s ramp at 0.75 ->
2.5 req/s while holding the 0.65 plateau correctly (the ramp segment of
`--request-rate-series` overshoots by 3-15x; the first synthetic attempt
showed the same thing and was misread as a constant factor). One replica
prefills 32K-token prompts at ~0.5 req/s, so every run was saturated from
its first second. What remains is a comparison of recovery from an
initial overload, which is a real scenario but not the one designed.

| run | policy | TTFT p95 | ITL p95 | GPU-h | replicas 2/3/4 asked for at | starting replica-s | failed |
| --- | --- | --- | --- | --- | --- | --- | --- |
| `runs/20261024T073727Z-...-synthcold32k-token` | token P=60K D=8 | 42986 | 202 | 0.98 | 23 / 35 / 41s | 436 | 54 |
| `...-synthcold32k-default-t3` | default target 3 | 9538 | 201 | 0.99 | 21 / 33 / 39s | 209 | 0 |
| `...-synthcold32k-default-t5` | default target 5 | 10981 | 201 | 0.98 | 29 / 42 / 42s | 207 | 1 |

Both policies ask for the cap within 40s of the first request, within a
few seconds of each other: when the load is already past one replica's
capacity, the prefill backlog and the ongoing-request count both scream
at once and there is nothing left for a signal to add. The token run's
far worse p95 is not the policy: its three new replicas took twice as
long to come up (436 vs ~208 starting replica-seconds, i.e. ~140s instead
of ~70s each), so 130 requests queued on one replica instead of 69. The
rest of each run, at four replicas, is identical (mean TTFT 2.5-2.6s,
ITL p95 201ms in all three: 32K prefills interleaving with decodes at
~1.6 req/s per replica is simply past the ITL target).

Verdict: no difference attributable to the policy; replica boot time
decides overload recovery. The intended shape (load crossing one, then
two, then three replicas' capacity) is `synthstep32k_*`, which uses rate
plateaus AIPerf delivers correctly.
