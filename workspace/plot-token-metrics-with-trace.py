"""Overlay exported router/policy gauges with the existing trace reconstruction."""
import bisect
import collections
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

RUN = Path('/home/ray/default/serve-llm-autoscaling-experiment/runs/20261111T002035Z-h100-token-metrics-injected-scheduler-local')
OUT = RUN / 'analysis/token-metrics-overlay'
traces = json.loads((RUN / 'analysis/scheduler_trace_inputs.json').read_text())
changes = collections.defaultdict(float)
counts = collections.Counter()
for filename in traces:
    queued, chunks, completed, preempted = {}, collections.defaultdict(list), {}, collections.defaultdict(list)
    offset = None
    for line in Path(filename).open():
        event = json.loads(line)
        kind = event['event']
        counts[kind] += 1
        if offset is None and 'wall_time_s' in event and 'monotonic_s' in event:
            offset = event['wall_time_s'] - event['monotonic_s']
        if kind == 'request_queued':
            queued[event['request_id']] = event['monotonic_s']
        elif kind == 'batch_scheduled':
            for request in event['requests']:
                if request['phase'] == 'prefill':
                    chunks[request['request_id']].append((event['batch_index'], request['scheduled_tokens']))
        elif kind == 'batch_completed':
            completed[event['batch_index']] = event['monotonic_s']
        elif kind == 'request_preempted':
            preempted[event['request_id']].append((event['batch_index'], event['monotonic_s']))
    if offset is None:
        raise RuntimeError(f'No clock alignment in {filename}')
    for request_id, queued_at in queued.items():
        request_chunks = chunks.get(request_id, [])
        if not request_chunks or any(batch not in completed for batch, _ in request_chunks):
            counts['excluded_incomplete_requests'] += 1
            continue
        counts['reconstructed_requests'] += 1
        # Same accounting as diagnostics/policy_vs_engine_mean.py: add observed
        # uncached work at queue arrival (and recomputation at preemption), then
        # remove each prefill chunk at its batch's output callback.
        starts = [(0, queued_at)] + sorted(preempted.get(request_id, []))
        for index, (batch, started_at) in enumerate(starts):
            until = starts[index + 1][0] if index + 1 < len(starts) else math.inf
            changes[started_at + offset] += sum(n for b, n in request_chunks if batch <= b < until)
        for batch, tokens in request_chunks:
            changes[completed[batch] + offset] -= tokens

times = sorted(changes)
levels, areas = [], []
level = area = 0.0
previous = times[0]
for timestamp in times:
    area += level * (timestamp - previous)
    level += changes[timestamp]
    levels.append(level)
    areas.append(area)
    previous = timestamp
assert min(levels) >= 0 and levels[-1] == 0, 'Reconstructed workload did not balance'
assert counts['excluded_incomplete_requests'] == 0, counts

def integral(at):
    index = bisect.bisect_right(times, at) - 1
    return 0 if index < 0 else areas[index] + levels[index] * (at - times[index])

start = datetime(2026, 11, 11, 0, 24, tzinfo=timezone.utc).timestamp()
end = datetime(2026, 11, 11, 0, 47, tzinfo=timezone.utc).timestamp()
grid = list(range(int(start), int(end) + 1))
mean = [(integral(t) - integral(t - 60)) / 60 for t in grid]
stamp = lambda t: datetime.fromtimestamp(float(t), timezone.utc)

plt.rcParams.update({'font.size': 11, 'axes.spines.top': False, 'axes.spines.right': False})
fig, ax = plt.subplots(figsize=(15, 6.5))
for key, label, color, width in [
    ('router', 'KV tracker: current uncached prefill tokens', '#3298bd', 1.35),
    ('policy_mean', 'Policy: exported 60-second lookback mean', '#8b3dbb', 2.4),
]:
    result = json.loads((OUT / f'{key}-prometheus.json').read_text())['data']['result'][0]
    points = result['values']
    ax.plot([stamp(t) for t, _ in points], [float(v) for _, v in points],
            label=label, color=color, linewidth=width, zorder=4 if key == 'policy_mean' else 2)
low = bisect.bisect_right(times, start) - 1
high = bisect.bisect_right(times, end)
raw_times = [start] + times[max(0, low + 1):high] + [end]
raw_levels = [levels[low] if low >= 0 else 0] + levels[max(0, low + 1):high] + [levels[high - 1] if high else 0]
ax.step([stamp(t) for t in raw_times], raw_levels, where='post',
        color='#e69a43', alpha=0.55, linewidth=0.8, label='Trace: remaining uncached prefill tokens', zorder=1)
ax.plot([stamp(t) for t in grid], mean, color='#179567', linewidth=2.3,
        label='Trace: reconstructed 60-second mean', zorder=3)
ax.set(title='Router and policy metrics vs. trace-reconstructed prefill workload',
       xlabel='November 11, 2026 · UTC', ylabel='Uncached prefill tokens', xlim=(stamp(start), stamp(end)), ylim=(0, None))
ax.xaxis.set_major_formatter(mdates.DateFormatter('%H:%M', tz=timezone.utc))
ax.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))
ax.grid(alpha=0.18)
ax.legend(loc='lower left', bbox_to_anchor=(0, 1.07), ncol=2, frameon=False, fontsize=10)
fig.text(0.08, 0.04, 'Blue/purple: exported Prometheus gauges at 10-second resolution. Orange/green: scheduler-trace reconstruction; green is the time-weighted trailing 60-second mean.', fontsize=9, color='#555555')
fig.text(0.08, 0.01, 'Trace workload enters at engine queue arrival and decreases as prefill batches complete. All 322 traced requests reconstructed; no incomplete requests excluded.', fontsize=9, color='#555555')
fig.tight_layout(rect=(0, 0.09, 1, 0.98))
for extension in ('png', 'svg'):
    fig.savefig(OUT / f'router-policy-and-trace.{extension}', dpi=180)
plt.close(fig)
with (OUT / 'trace-reconstruction.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['epoch_s', 'remaining_prefill_tokens'])
    writer.writerows(zip(times, levels))
with (OUT / 'trace-mean-60s.csv').open('w') as stream:
    writer = csv.writer(stream)
    writer.writerow(['epoch_s', 'mean_prefill_tokens'])
    writer.writerows(zip(grid, mean))
(OUT / 'trace-reconstruction-summary.json').write_text(json.dumps(dict(counts), indent=2) + '\n')
print(OUT / 'router-policy-and-trace.png')
print(dict(counts))
print('Trace maximum', max(levels), 'trace mean maximum', max(mean))
