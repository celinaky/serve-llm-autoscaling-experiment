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
