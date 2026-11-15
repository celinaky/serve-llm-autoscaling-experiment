"""Plot the measured policy mean for the H100 attempt that stayed at one replica."""
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path('/home/ray/default/serve-llm-autoscaling-experiment/runs/20261112T180926Z-h100-token-freeze-150s-injected-scheduler')
data = json.loads((ROOT / 'analysis/plot_data.json').read_text())
t0 = data['time_origin']['profiling_start_ns'] / 1e9
status = [r for r in data['serve_status_samples'] if not r.get('error') and 0 <= r['relative_time_s'] <= 1050]
assert status and all(r['running_replicas'] == 1 for r in status)
means = {}
for line in (ROOT / 'telemetry/serve_metrics.jsonl').open():
    record = json.loads(line)
    if record.get('error'):
        continue
    t = record['scrape_started_at_ns'] / 1e9 - t0
    if not 0 <= t <= 1050:
        continue
    for sample in record.get('samples', []):
        labels = sample.get('labels', {})
        if (sample['name'] == 'ray_serve_autoscaling_policy_metric'
                and labels.get('metric') == 'prefill_tokens_lookback_mean'
                and 'LLMServer' in labels.get('deployment', '')):
            means[t] = sample['value']
assert means
points = sorted(means.items())
fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True,
                         gridspec_kw={'height_ratios': [4, 1]})
for ax in axes:
    ax.axvspan(0, 450, color='#e8f0fb')
    ax.axvspan(450, 1050, color='#f1f1ee')
    ax.grid(axis='y', alpha=0.25)
    ax.spines[['top', 'right']].set_visible(False)
axes[0].plot([t for t, v in points], [v / 1000 for t, v in points],
             color='#6250d6', linewidth=2, label='Exported policy mean (60-second lookback)')
axes[0].axhline(20, color='#555', linestyle=':', label='20K-token target per replica')
axes[0].set(ylabel='Mean uncached prefill tokens (K)', ylim=(0, None))
axes[0].legend(frameon=False)
axes[0].text(225, 1.02, 'Ramp: 1 → 16 sessions', transform=axes[0].get_xaxis_transform(), ha='center')
axes[0].text(750, 1.02, 'Hold: 16 sessions', transform=axes[0].get_xaxis_transform(), ha='center')
axes[1].step([r['relative_time_s'] for r in status], [r['running_replicas'] for r in status],
             where='post', color='#17834c', linewidth=2)
axes[1].set(xlabel='Seconds since profiling started', ylabel='Running\nreplicas',
            yticks=[0, 1, 2], ylim=(0, 2), xlim=(0, 1050))
fig.suptitle('H100: one running replica throughout the workload', fontsize=15)
fig.text(0.08, 0.025, 'Initial freeze attempt: scale-out replicas failed startup; this is an observed single-replica run, not a controlled fixed-replica baseline.', fontsize=9)
fig.tight_layout(rect=(0, 0.06, 1, 0.94))
out = ROOT / 'analysis/single_running_replica_policy_mean.png'
fig.savefig(out, dpi=170)
print(out)
print('Peak exported mean:', max(means.values()))
