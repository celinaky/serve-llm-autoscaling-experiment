# Engine queue time: what is measured and why

This explains how engine queue time is defined when it is reconstructed from the
vLLM scheduler traces: the "estimated first-execution queue time" of a request.
It is written for an agent or engineer who needs to change, port or defend the
definition.

The same rule is used in two places:

- **This harness.** `diagnostics/load_breakdown.py` uses it to time every engine
  step (`started_at[i] = max(completed_at[i - 1], scheduled_at[i])`), which is
  what `queue_ms` in `analysis/load_requests.csv` and the arrival statistics
  plot are built on.
- **The NFER prefill queue profile** ([anyscale/nfer#373](https://github.com/anyscale/nfer/pull/373)),
  where it is the Y value of the fitted line
  `queue_ms = intercept + slope × load_tokens`. The numbers in section 3 come
  from a run of that profile.

The definition, as written in NFER's
[`analysis.py`](https://github.com/anyscale/nfer/blob/feat/prefill-queue-profile/nfer/suites/prefill_queue_profile/analysis.py):

```python
start = max(scheduled[batch], completed[batch - 1])
queue_ms = (start - arrival) * 1000
```

where `batch` is the first engine step that carries prefill tokens for the
request. In words: **queue time runs from the request's arrival at the engine to
the moment its first prefill step starts executing, and that moment is taken to
be the later of (a) the step being scheduled and (b) the previous step's output
callback.**

## 1. What we want to measure

The fit predicts how long a request waits behind work already on the engine. The
wait we care about ends when the GPU starts working on the request's own tokens.
Everything after that is the request's own prefill, not queueing.

There is no direct timestamp for "the GPU started this step". The tracing
scheduler runs in the engine's CPU process and sees three events per step:

| Event | Timestamp in the trace | Taken at |
| --- | --- | --- |
| Request arrives | `request_queued.queued_at_monotonic_s` | `add_request()` is called |
| Step is scheduled | `batch_scheduled.scheduled_at_monotonic_s` | `schedule()` returns |
| Step's output is processed | `batch_completed.completed_at_monotonic_s` | `update_from_output()` is called |

The start of execution has to be inferred from these.

## 2. Batch starting logic

### The naive answer and why it is wrong

The obvious choice is "the step starts when it is scheduled". vLLM's own
`scheduled_ts - queued_ts` uses this, and it is what the trace reports as
`request_admitted.queue_wait_ms`.

It is wrong whenever async scheduling is on. vLLM 0.33 leaves the setting on
"auto" unless a recipe sets it, and it resolved to on in the run cited in
section 3 (`trace_started.async_scheduling` in the trace records the resolved
value). With async scheduling the engine loop does this
(`step_with_batch_queue` in `vllm/v1/engine/core.py`):

1. If the batch queue has room, schedule the next step and submit it to the
   worker **without waiting**.
2. Only when the queue is full, block on the oldest step's result.
3. Process that result (`update_from_output`), then go back to 1.

The queue holds two steps. So step N+1 is scheduled and handed to the worker
while step N is still running on the GPU. The worker runs them one after the
other. Step N+1's scheduling timestamp is therefore early: the step exists, but
it is waiting behind step N.

For a request whose first prefill chunk lands in step N+1, "time until
scheduled" stops the clock up to one whole step too soon. During prefill-heavy
load one step is roughly 0.9 s on a B200, so the error is not small.

### The rule

A step cannot start executing before both of these are true:

- **It has been scheduled.** Nothing can run a step that does not exist yet.
- **The step before it has finished.** The worker executes steps in order, one
  at a time.

The step starts at whichever of the two happens last:

```text
start(N) = max(scheduled(N), finished(N − 1))
```

We do not have `finished(N − 1)` from the GPU. The closest CPU-side observation
is the previous step's output callback, `completed[N − 1]`. Hence the rule in
the code.

### Why `max`, case by case

| Situation | Which term wins | Why that is right |
| --- | --- | --- |
| Engine is busy, async scheduling | `completed[N − 1]` | Step N was scheduled while N − 1 was running. It starts when N − 1 ends. |
| Engine was idle before the request arrived | `scheduled[N]` | The previous step ended long ago. The step starts as soon as it is scheduled. |
| Async scheduling is off | `scheduled[N]` | The loop processes N − 1's output before scheduling N, so `scheduled[N]` is already the later of the two. |

One rule covers all three, so the analysis does not need to know whether async
scheduling is on. `AutoTracingScheduler` picks the sync or async scheduler to
match the engine, and the same analysis applies to both.

### Why the previous step and not an earlier one

The batch queue holds at most two steps when pipeline parallelism is 1
(`max_concurrent_batches` in `vllm/config/vllm.py`). When step N is scheduled,
at most one other step, N − 1, can be in flight. So N − 1 is the only step N can
be waiting behind. Batch indices in the trace count only steps that scheduled
tokens, so `batch − 1` is always the previous executed step.

### Why not use `dispatch_to_output_ms`

The trace also records `dispatch_to_output_ms` per step, which looks like "how
long the step took". With async scheduling it is not: it runs from dispatch to
output, so it includes the time spent waiting behind the previous step. On the
30-minute B200 run, full-chunk steps average 1491 ms by this measure and 900 ms
measured from the previous step's output. Do not use it for step duration or to
back out a start time.

## 3. Evidence that the choice matters

Same 336 samples from one 30-minute, cohort-8 run on a local B200 replica
(an NFER profiling run, local to the original workspace and not in any
repository), fitted against each definition. NFER writes both values to
`analysis/points.csv` as `queue_ms` and `scheduling_queue_ms`.

| Queue time ends at | Slope (ms/token) | Intercept (ms) | RMSE (ms) | R² |
| --- | ---: | ---: | ---: | ---: |
| Estimated execution start (this definition) | 0.05047 | −89.1 | 232.5 | 0.984 |
| First scheduling only | 0.04476 | −269.3 | 394.7 | 0.945 |

- The two definitions differ by 263 ms per request on average and by up to
  1174 ms.
- 294 of 336 requests were scheduled before the previous step finished.
- The scheduling-only label gives a flatter slope and a much more negative
  intercept, because it removes up to a step of waiting from the loaded
  requests.

An independent check, detailed in NFER's
[step-time line hand-off](https://github.com/anyscale/nfer/blob/feat/prefill-queue-profile/docs/prefill-queue-step-time-line.md): the measured
duration of a full-chunk step, divided by the chunk size, is 0.054 ms/token. That is prefill throughput measured directly, and
it agrees with the execution-start slope (0.050) far better than with the
scheduling-only slope (0.045).

The same relabeling was done earlier on an H100 online-learning run. Changing
only the training label from native scheduling to estimated execution reduced
prediction RMSE from 1224.9 ms to 984.1 ms on 561 evaluation requests.

## 4. What the estimate does not capture

- **It is a CPU callback, not a GPU timestamp.** `completed[N − 1]` is taken when
  the engine process handles the output, slightly after the GPU finished. The
  estimated start is late by that handling time, and queue time is overstated by
  the same amount. It has not been compared against CUDA timings.
- **Only the wait before the first prefill step counts.** A long prompt is
  prefilled over several steps, and waits between those steps are not included.
  This is initial queue time, not total time-to-first-token.
- **The very first step has no predecessor.** A request whose first step is the
  first step in the trace is excluded (`no_previous_batch`), because the rule
  cannot be evaluated.
- **Requests preempted before their first step are excluded**
  (`preempted_before_first_execution`).
- **Pipeline parallelism breaks the assumption.** With pipeline parallel size
  above 1, vLLM runs several steps at once on different stages, so "the previous
  step must finish first" no longer holds and this rule would overstate the
  wait. Neither this harness nor NFER's `validate_service` currently rejects
  such configurations. Treat results from them as unvalidated.
- **Small requests can start earlier than the load suggests.** A request that
  fits in the spare token budget of a step that is about to run starts with that
  step. Its queue time is measured correctly, but it sits below the fitted line.
  This is a property of the linear model, not of the queue-time definition.

## 5. How this harness's `queue_ms` differs slightly

`load_breakdown.py` times steps with the rule above, then reports

```text
queue_ms = engine_ttft_ms − (duration of the steps the request had tokens in)
```

So the harness's queue time is everything before first token that is not the
request's own prefill steps. Besides the wait for its first step to start, that
includes any step between admission and first token in which the request got no
tokens (`idle_steps` in `load_requests.csv`). NFER's `queue_ms` stops at the
start of the first prefill step. The two agree for a request that prefills in
consecutive steps.

The README and plot labels describe the middle panel as the wait "before the
request's first prefill chunk was scheduled". That wording is shorthand: the
value plotted is the one described here, not time-until-scheduled.

## 6. If you change this

- The X value and the Y value depend on the same event. Outstanding load is
  reduced at `completed[N]`, and execution start is estimated from
  `completed[N − 1]`. Changing one without the other makes the fit inconsistent.
- In NFER, keep `scheduling_queue_ms` in `points.csv`. It costs nothing and lets
  anyone re-run the comparison in section 3.
- A different start estimate, for example from GPU-side timing, should be
  validated by repeating the section 3 table on the same run. A better
  definition should move the slope toward the directly measured step throughput
  and reduce RMSE.
- Online consumers must use the same definition as the offline fit. The
  recovered Ray learner exposes it as
  `calibration_queue_time_metric: estimated_execution`; the native scheduling
  label is still its default.
