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

## Synthetic cold prefill, stepped (synthstep32k, chain 9): a tie

Load delivered as configured this time: 0.3, 0.8 and 1.3 req/s of 32K
cold prompts for five minutes each (one, two and three replicas' worth).

| run | policy | TTFT p95 | ITL p95 | GPU-h | replicas 2/3/4 asked for at | p95 = own prefill + waiting |
| --- | --- | --- | --- | --- | --- | --- |
| `runs/20261024T084125Z-...-synthstep32k-token` | token P=60K D=8 | 3015 PASS | 48 | 0.80 | 18 / 122 / 618s (prefill, prefill, decode) | 1609 + 1406 |
| `...-synthstep32k-default-t3` | **default target 3** | **2861 PASS** | **41** | 0.82 | 32 / 332 / 332s | 1607 + 1254 |
| `...-synthstep32k-default-t5` | default target 5 | 15789 FAIL | 199 | 0.81 | 34 / 330 / 338s, then **4->2 at 638s** | 1611 + 14179 |

**Best of each: a tie.** Default target 3 is marginally ahead on both
latencies (0.15s, 7ms) at the same cost; both inside the +-0.3s noise.
Why no token advantage here: on a stream of same-sized cold prompts the
request count is as good a measure of prefill work as the token backlog.
Each request lives ~4.5s (2s prefill + 2.5s decode), so ongoing requests
= rate x 4.5 tracks the arrival rate faithfully, and a target of 3 per
replica maps onto one replica's ~0.6 req/s of prefill capacity. The
prefill term's edge, seeing work before it starts, carries no extra
information when every arrival is the same known-size prefill. The token
policy also over-provisioned early (three replicas by 122s under a load
one replica could serve) because every lone 32K prompt is a 32K spike
against P=60K; it cost nothing here only because the default ran at
three replicas too.

The default's other setting shows its failure mode: target 5 scaled to
four on the step-2 transient, then saw ongoing requests (~3.6 at 0.8
req/s) below target for 300s and dropped to two replicas at 638s, just as
the 1.3 req/s step arrived. Thirty seconds later it was asking for six.
The 300s downscale delay that protects a ramp does not protect a step
that lands at the end of a lull.

## Across all shapes: when the token policy wins, and why

| shape | token (best) | default (best) | token vs default | mechanism |
| --- | --- | --- | --- | --- |
| workday ramp, 32 cold sessions / 10 min | 4511 / 28 / 0.87 | t8: 4829 / 38 / 0.70 | token wins both latencies, +24% GPU; at equal cost default is 0.4s worse | first replica at ~70s vs 169s: sessions arriving in between queued on one replica |
| storm, 32 cold sessions / 150s | P=30K: 5042 / 37 / 0.99 | t4: 5462 / 50 / 0.96 | token wins both, +3% GPU | asks for all four at 13-31s vs 42/60/73s; parallel start-up |
| big ramp, 64 cold sessions / 20 min | 5536 / 42 / 1.62 | t6: 5778 / 47 / 1.45 | token ahead (TTFT within noise, ITL not), +12% GPU | at the cap by ~100s vs 401s |
| stepped synthetic 32K prefill | 3015 / 48 / 0.80 | t3: 2861 / 41 / 0.82 | tie | request count tracks same-size prefill work as well as tokens do |
| synthetic overload from t=0 | 42986 / 202 / 0.98 | t3: 9538 / 201 / 0.99 | no policy difference | both at the cap in 40s; replica boot time decides |
| mixed warm/cold | dropped | dropped | -- | AIPerf priming pins both at the cap before measurement |

(TTFT p95 ms / ITL p95 ms / GPU-h; SLO p95 TTFT <= 5s, ITL <= 50ms.)

1. **The token policy's advantage is the prefill term, and it shows
   exactly when a request count is slow to express new work**: agent
   sessions arriving one at a time, each a large cold prefill that takes
   seconds before it registers as a "request in progress" anywhere a
   count would see it, and bursts where the backlog asks for several
   replicas in one step. That is the agent ramp and the storm.
2. **It brings nothing when the request count already measures the
   work**: same-sized prefills at a steady rate (stepped synthetic), or
   an overload that saturates everything at once. There the two policies
   make the same decisions within seconds of each other.
3. **Its cost is earlier replicas** (+3% to +24% GPU time over the
   default's cheapest passing setting). At matched cost it is never
   worse and usually clearly better, because a default policy tuned to
   spend the same amount spends it later.
4. **The remaining latency is not scalable.** In every shape the
   decomposition puts most of p95 TTFT in the prompt's own prefill
   (1.6-4.1s) and most of the rest in waiting on the request's own
   replica behind other prefills; startup-lag waiting is 0-0.3s. Both
   policies sit within ~0.1s of the instant-startup ceiling on the agent
   ramp. Below that is routing or engine scheduling, not autoscaling.
5. **Tuning matters as much as the policy.** The default policy with the
   same delays and 2s metric delivery passed the ramp SLO it had failed
   with its defaults; its best target differs per shape (8, 4, 6, 3) and
   a wrong one fails outright (target 12 on the ramp, target 5 on the
   steps). The token policy's P also wants to be shape-aware (60K on a
   ramp, 30K in a storm): the right P is the backlog one replica can
   absorb before a second one is worth starting.
