# Online prefill queue learning and trace comparison

Recorded 2026-11-10. This note lives in `/home/ray/default/agent-context`, beneath the
workspace directory preserved over restarts (`~/default`). It provides context for future
agent sessions investigating uncached prefill tokens, queue-time learning,
relabeling, and prediction error.

The experiment fits a linear relationship between a replica's outstanding
uncached prefill token workload and a request's queue time. Changing the target
from first scheduling to estimated first execution improves prediction error
against the new target. Reconstructing the token workload from scheduler traces
improves it further. These are offline prediction results on one recorded run;
they do not establish lower queue times or better live autoscaling.

## Checkouts and recovery history

Paths and branches observed when this note was written:

| Component | Local checkout | Branch |
|---|---|---|
| Recovered Ray learner | `/home/ray/default/ray-online-prefill` | `celina/online-prefill-execution-queue` |
| Experimental harness | `/home/ray/default/online-prefill-harness` | `online-prefill-execution-queue-verification` |

The original Ray experiment was on `celina/online-prefill-token-calibration`
at `/mnt/local_storage/ray-online-prefill`. The original harness branch was
`online-prefill-calibration-verification`. Source snapshots and patches were
archived on November 6, then recovered and extended with execution queue timing
on November 9. Consult the manifests instead of assuming the historical paths
are still active:

- [Original archive manifest](../online-prefill-archive/20261106T190109Z/manifest.json)
- [Execution queue recovery manifest](../online-prefill-archive/20261109-execution-queue-recovery/manifest.json)
- [Harness documentation](../online-prefill-harness/README.md), sections
  “Online token-budget calibration” and “Estimated first-batch execution queue”.

The recovered Ray checkout has local changes and is a partial historical
checkout. The README documents compatibility problems with the installed Ray
wheel's Serve controller and metrics interfaces. A compatible runtime is needed
before deploying this learner; importing selected source modules alone is
insufficient. The analysis below requires no deployment.

## Model and queue definitions

The router stores the selected replica's outstanding prefill workload at routing
time, before tracking the new request. It pairs that snapshot with the request's
queue label and fits:

```text
queue_time_ms = intercept_ms + slope_ms_per_token * workload_tokens
token_budget = (target_queue_time_ms - intercept_ms) / slope_ms_per_token
```

Each router learns independently. Defaults are a 1,000-sample rolling window,
100 samples before the first fit, and subsequent refreshes every 120 seconds.
Fits use ordinary least squares with an intercept and require a positive finite
slope. The token budget targets mean queue time, not a tail-latency guarantee.

| Metric | Definition |
|---|---|
| `native_scheduling` | Engine arrival to first scheduling, using native `scheduled_ts - queued_ts`; still the default label. |
| `estimated_execution` | Engine arrival to estimated first-batch execution start, including async pipeline waiting, excluding later prefill gaps. |
| Original trace total wait | Waiting before and between prefill steps; a different quantity, not the evaluation target below. |

The execution estimate is:

```text
estimated_start = max(first_batch_scheduled, previous_batch_output_callback)
execution_queue_ms = 1000 * (estimated_start - engine_arrival)
```

For the first batch, use its scheduling time without a preceding callback.
This is a CPU-callback estimate, not a GPU execution-start measurement. “Arrival”
here means engine arrival, not client arrival or router admission.

The recovered online implementation supports:

```yaml
enable_prefill_queue_learning: true
calibration_queue_time_metric: estimated_execution
```

These are router kwargs. See
[the example configuration](../online-prefill-harness/experiments/h100_online_execution_queue_profile.yaml).
Online learning can use the execution timing without JSONL trace files. The
offline comparison below specifically uses saved scheduler traces.

## How offline relabeling worked

The recorded run is
`online-prefill-harness/runs/20261106T052548Z-h100-online-queue-profile-traced-short`.
Its original learner consumed native scheduling labels.

1. Read engine arrival, first admitted batch, batch scheduling, and batch output
   callbacks from the saved scheduler traces.
2. Compute the execution queue estimate above and join it to each original
   learner sample by request ID.
3. Keep the original router workload, sample order, and saved training windows;
   replace only the queue-time training label.
4. Replay the fits. Predict each later observation using only earlier available
   fit snapshots, then clamp predictions to zero.
5. Score both models against the same trace-estimated execution queue labels.

Original audits and fits were preserved. This is retraining with a new target
definition, not evidence that measured request delays decreased.

The initial replay is implemented in
[replay_execution_queue_learning.py](../online-prefill-harness/tools/replay_execution_queue_learning.py).
Trace parsing and workload reconstruction are in
[compare_router_scheduler_trace.py](../online-prefill-harness/tools/compare_router_scheduler_trace.py).

## Side by side prediction results

All rows below use the same **561 evaluation requests**, following bootstrap,
and the same **15 saved fit windows**. The original replay contains 661 accepted
samples. Every model is evaluated against **trace-estimated first-execution
queue time**, regardless of its training label.

| Token workload input | Queue-time training label | RMSE ms | MAE ms |
|---|---|---:|---:|
| Router estimate at routing | Native first scheduling | 1224.888 | 436.799 |
| Router estimate at routing | Trace-estimated first execution | 984.070 | 397.060 |
| Trace remaining workload at engine arrival | Trace-estimated first execution | 626.437 | 232.400 |
| Trace remaining workload at routing | Trace-estimated first execution | 726.834 | 317.059 |

RMSE is the square root of mean squared prediction error; it penalizes large
misses more than MAE, the mean absolute error. Lower values are better.
Changing the label reduces RMSE about 20%. Replacing the router workload with
trace workload at engine arrival reduces it a further 36%.

Trace workload is remaining uncached prefill work on the selected engine,
excluding the arriving request's own increment. Reconstruction adds each
request's observed prefill work at arrival, subtracts chunks when their batches
complete, and accounts for recomputation at preemption. It includes all traced
engine requests, not just benchmark requests.

Engine arrival is later than routing. The fourth row samples reconstructed work
at routing time to check that timing difference; its RMSE is still below the
router estimate's 984 ms. Trace workload uses observed future prefill chunks in
its offline reconstruction, so treat it as a diagnostic reference rather than
an immediately deployable feature or a prospective prediction result. These
comparisons also retain the original workload and reservations rather than
simulating new autoscaling decisions.

Validation reproduced the original two models' saved MAE and RMSE to relative
tolerance `1e-10`. One learner request absent from the completed-request
comparison CSV was reconstructed directly from raw traces, preserving the full
evaluation set. Trace quality reported zero dropped events and zero incomplete
trace requests.

## Saved evidence and reproduction

Relative to the recorded run directory above:

- `telemetry/prefill-calibration/*.jsonl`: original samples and fit windows.
- `analysis/router-trace-comparison/points.csv`: joined router and trace data.
- `analysis/execution-queue-replay/summary.json` and `index.html`: initial
  relabeling comparison.
- `analysis/queue-model-comparison/summary.json`: four-model results and caveats.
- `analysis/queue-model-comparison/predictions.csv`: per-request predictions and
  the shared target.
- `analysis/queue-model-comparison/fits.json`: coefficients for saved windows.
- `deployment/profile_supervisor.json`: raw scheduler trace directory provenance.

The raw traces used were stored at:

```text
/mnt/cluster_storage/serve-llm-autoscaling-experiment/paired-scheduler-traces/h100_online_queue_profile_traced_short-1793942745650028954
```

Reproduce the four-model comparison from `~/default`:

```bash
MPLCONFIGDIR=/tmp/queue-model-matplotlib \
  .venvs/online-prefill/bin/python \
  online-prefill-harness/tools/compare_queue_model_rmse.py \
  online-prefill-harness/runs/20261106T052548Z-h100-online-queue-profile-traced-short
```

The script reads the joined CSV and original audit windows, consults the raw
trace directory recorded in `profile_supervisor.json` for missing CSV requests,
and verifies the first two models against the saved original replay. It uses a
100-sample minimum to match this run. If storage paths or environments change,
locate the archived evidence before reproducing; the table above is preserved
here even if external trace storage is unavailable.
