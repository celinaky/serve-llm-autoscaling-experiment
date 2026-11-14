"""Align the saved router/GPU curves with observed Mooncake data movement."""
from pathlib import Path
from datetime import datetime, timezone
from collections import defaultdict
import argparse
import bisect
import csv
import importlib.util
import json
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
REPO = RUN.parent.parent
EVIDENCE = json.loads((OUT/'evidence.json').read_text())
REPORT = json.loads((RUN/'analysis/mooncake_report.json').read_text())
T0 = EVIDENCE['time_origin_epoch_s']
END = 1050
BIN = 10
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--gpu-kv', action='store_true',
    help='Add saved per-replica GPU KV block usage and write a separate figure.')
args = parser.parse_args()

# Same chunk-burn-down definition and time samples as the existing green curve.
events = sorted(x for r in EVIDENCE['requests'] for x in r['engine_deltas'])
times, levels, areas = [], [], []
level, area, last = 0., 0., events[0][0]
for t, delta in events:
    area += level*(t-last)
    level += delta
    times.append(t);levels.append(level);areas.append(area);last = t
assert abs(level) < 1e-6 and min(levels) > -1e-6

def integral(at):
    i = bisect.bisect_right(times, at)-1
    return areas[i]+levels[i]*(at-times[i]) if i >= 0 else 0.

def gpu_mean(at):
    return (integral(at)-integral(at-60))/60

purple = [(t,v) for t,v in EVIDENCE['published_router_mean'] if 0 <= t <= END]
green_times = [t for t,_ in EVIDENCE['raw_router_gauge'] if 0 <= t <= END]
green_values = [gpu_mean(t) for t in green_times]

# Obtain write-volume bins from the same filtered, per-series counter deltas
# as the run report. Read-only: this does not build or edit the report/Site.
spec = importlib.util.spec_from_file_location('cache_report',REPO/'tools/report_mooncake.py')
cache_report = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache_report)
totals, buckets, _ = cache_report.operation_data(RUN,set(REPORT['replica_ids']),round(T0*1e9))

# GET logs carry request IDs, tier bytes and completion timestamps. Infer the
# worker's display-time offset by requiring completion within that request's
# observed engine lifecycle; do not assume its wall-clock display timezone.
gets = []
year = datetime.fromtimestamp(T0,timezone.utc).year
for r in EVIDENCE['requests']:
    for line in r['get_log_lines']:
        stamp = re.search(r'INFO (\d\d-\d\d \d\d:\d\d:\d\d)',line)[1]
        local_epoch = datetime.strptime(f'{year}-{stamp}','%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
        end = r['first_token_s'] if r['first_token_s'] is not None else r['finish_s']
        assert end is not None, r['request_id']
        offsets = [h for h in range(-14,15) if r['start_s']-1 <= local_epoch+3600*h-T0 <= end+2]
        assert len(offsets) == 1, (r['request_id'],offsets)
        completed = local_epoch+3600*offsets[0]-T0
        size = int(re.search(r"'memory':\s*(\d+)",line)[1])
        gets.append(dict(request_id=r['request_id'],arrival_s=r['start_s'],
            completion_s=completed,bytes=size,worker_display_to_utc_hours=offsets[0],
            prompt_tokens=r['prompt_tokens'],cached_tokens=r['cached_tokens'],
            gpu_computed_tokens=r['computed_before_first_token'],
            router_selection_time_miss_tokens=None))
assert len(gets) == 51
assert sum(r['bytes'] for r in gets) == totals['load_get_bytes_ok']
assert len({r['request_id'] for r in gets}) == 51

get_bins = defaultdict(int)
for r in gets:
    get_bins[int(r['completion_s']//BIN)*BIN] += r['bytes']
bins = np.arange(0,END,BIN)
get_rates = [get_bins[int(t)]/(BIN*1024**2) for t in bins]
put_rates = [buckets.get(int(t),{}).get('save_put_bytes_ok',0)/(BIN*1024**2) for t in bins]
arrivals = [r['arrival_s'] for r in gets if 0 <= r['arrival_s'] <= END]

gpu_rows, gpu_summary = [], []
if args.gpu_kv:
    mapping = json.loads((RUN/'analysis/monitoring/root_request_routing.json').read_text())['engine_to_replica_id']
    trace_dir = Path('/mnt/cluster_storage/mooncake-kv/traces')/RUN.name
    # Extract capacity from the saved initialization logs, including replicas
    # with no requests; do not infer pool capacity from observed utilization.
    log_path = REPO/'findings-h100/mooncake-decay-16-400s/run.log'
    clean_log = re.sub(r'\x1b\[[0-9;]*m', '', log_path.read_text())
    capacities = {float(v) for v in re.findall(r'Available KV cache memory: ([\d.]+) GiB', clean_log)}
    assert len(capacities) == 1, capacities
    capacity_gib = capacities.pop()
    token_capacities = {}
    for pid, slots in re.findall(r'\(EngineCore pid=(\d+)\)[^\n]*GPU KV cache size: ([\d,]+) tokens', clean_log):
        token_capacities[pid] = int(slots.replace(',', ''))
    paths = sorted(trace_dir.glob('*.jsonl'))
    assert len(paths) == len(mapping) == 5
    for path in paths:
        pid = re.search(r'_pid-(\d+)', path.name)[1]
        replica_id = mapping[pid]
        offset = None
        samples = []
        for line in path.open():
            row = json.loads(line)
            if row.get('event') == 'trace_started':
                offset = row['wall_time_s']-row['monotonic_s']
            if row.get('event') != 'batch_scheduled' or 'kv_cache_usage' not in row:
                continue
            assert offset is not None
            at = row['monotonic_s']+offset-T0
            if not 0 <= at <= END:
                continue
            fraction = float(row['kv_cache_usage'])
            assert 0 <= fraction <= 1
            sample = dict(replica_id=replica_id, engine_pid=pid, time_s=at,
                active_block_fraction=fraction, capacity_gib_rounded=capacity_gib,
                approximate_active_gib=fraction*capacity_gib)
            samples.append(sample)
        gpu_rows.extend(samples)
        peak = max(samples,key=lambda r:r['active_block_fraction']) if samples else None
        gpu_summary.append(dict(replica_id=replica_id, engine_pid=pid,
            capacity_gib_rounded=capacity_gib,
            reported_token_slots=token_capacities[pid], batch_samples=len(samples),
            peak_usage_pct=100*peak['active_block_fraction'] if peak else None,
            peak_time_s=peak['time_s'] if peak else None))

purple_c, green_c, get_c, put_c = '#6250d6','#1baf7a','#df852e','#8493a1'
if args.gpu_kv:
    fig, (top,bottom,kv) = plt.subplots(3,1,figsize=(15,12.4),sharex=True,
        gridspec_kw={'height_ratios':[1.1,1,1.05],'hspace':.36})
else:
    fig, (top,bottom) = plt.subplots(2,1,figsize=(15,9),sharex=True,
        gridspec_kw={'height_ratios':[1.15,1],'hspace':.28})
fig.patch.set_facecolor('#fcfcfb')
top.plot([t for t,v in purple],[v/1000 for t,v in purple],color=purple_c,lw=2.1,
    label='Router-reported mean')
top.plot(green_times,np.array(green_values)/1000,color=green_c,lw=2.1,
    label='Engine-reconstructed remaining GPU prefill')
top.axhline(20,color='#777',lw=.9,ls=':',label='20K-token autoscaling target per replica')
top.set_ylabel('Prefill tokens · trailing 60s mean (K)')
top.set_title('Router estimate and actual remaining GPU-prefill work',loc='left',fontsize=12,pad=10)
top.set_ylim(bottom=0)
top.legend(loc='upper left',frameon=False,fontsize=9)

bottom.bar(bins+BIN/2,get_rates,width=8.5,color=get_c,alpha=.85,
    label='GET: DRAM cache reloads (bytes at logged completion)')
bottom.plot(bins+BIN/2,put_rates,color=put_c,lw=1.6,
    label='PUT: cache writes (exported byte-counter deltas)')
ceiling = max(max(get_rates),max(put_rates))*1.16
rug_y = -ceiling*.065
bottom.scatter(arrivals,[rug_y]*len(arrivals),marker='v',s=21,color=get_c,
    edgecolor='white',linewidth=.35,zorder=5,
    label=f'Arrival of a request that later required GET ({len(arrivals)} requests)')
bottom.axhline(0,color='#aaa',lw=.8)
bottom.set_ylim(-ceiling*.13,ceiling)
bottom.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}' if v >= 0 else ''))
bottom.set_ylabel('Observed data movement (MiB/s · 10s bins)')
bottom.set_title('Mooncake movement, on the same time axis',loc='left',fontsize=12,pad=10)
bottom.legend(loc='upper left',frameon=False,fontsize=8.5)
if args.gpu_kv:
    colors = ['#3077b5','#c15b87','#009b87','#8771b2','#b49424']
    for summary,color in zip(gpu_summary,colors):
        samples = [r for r in gpu_rows if r['replica_id'] == summary['replica_id']]
        if not samples:
            continue
        xs, ys = [], []
        for sample in samples:
            # No extrapolation through idle periods without measured samples.
            if xs and sample['time_s']-xs[-1] > 1:
                xs.append(float('nan'));ys.append(float('nan'))
            xs.append(sample['time_s']);ys.append(sample['active_block_fraction']*100)
        label = summary['replica_id']
        if summary is gpu_summary[0]:
            label += ' (original)'
        kv.plot(xs,ys,color=color,lw=1.2,marker='.',markersize=1.1,
            rasterized=True,label=label)
    kv.axhline(100,color='#525252',lw=1.1,ls='--',
        label=f'Pool capacity per replica: ≈{capacity_gib:g} GiB (100%)')
    peak = max(gpu_rows,key=lambda r:r['active_block_fraction'])
    kv.annotate(f"Original replica: {100*peak['active_block_fraction']:.1f}% at {peak['time_s']:.0f}s",
        xy=(peak['time_s'],100*peak['active_block_fraction']),xytext=(peak['time_s']+90,40),
        fontsize=9,color=colors[0],arrowprops={'arrowstyle':'->','color':colors[0],'lw':.9})
    kv.set_ylim(0,108)
    kv.set_ylabel('Active / non-free KV blocks (% of pool)')
    kv.set_title('GPU KV-cache usage per replica · observed scheduler batches; gaps have no samples',
        loc='left',fontsize=12,pad=10)
    kv.legend(loc='upper left',bbox_to_anchor=(0,-.23),ncol=3,frameon=False,fontsize=8.5)
    secondary = kv.secondary_yaxis('right',functions=(lambda pct:pct*capacity_gib/100,
        lambda gib:gib*100/capacity_gib))
    secondary.set_ylabel('Approximate active KV (GiB)')
    kv.set_xlabel('Seconds since profiling start')
else:
    bottom.set_xlabel('Seconds since profiling start')

for ax in ((top,bottom,kv) if args.gpu_kv else (top,bottom)):
    ax.set_facecolor('#fcfcfb');ax.grid(axis='y',alpha=.2);ax.set_axisbelow(True)
    ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0,END);ax.xaxis.set_major_locator(MultipleLocator(100))
    for t in (500,700,850):
        ax.axvline(t,color='#b8b5b0',lw=.8,ls=':',zorder=0)
    for lo,hi in ((430,520),(670,750),(800,890)):
        ax.axvspan(lo,hi,color='#e8c792',alpha=.08,lw=0,zorder=0)

fig.subplots_adjust(left=.085,right=.925 if args.gpu_kv else .985,
    bottom=.21 if args.gpu_kv else .15,top=.91 if args.gpu_kv else .86,
    hspace=.36 if args.gpu_kv else .28)
fig.suptitle('H100 · 16 sessions · 10K tokens/sec decay · 400s downscale',fontsize=15,y=.97)
if args.gpu_kv:
    fig.text(.085,.105,'GPU usage = non-free blocks, not all resident prefixes: reclaimable blocks may still contain cached KV. The GPU pool is preallocated.',fontsize=8.5,color='#555')
    no_samples = ', '.join(s['replica_id'] for s in gpu_summary if not s['batch_samples'])
    fig.text(.085,.085,f'Capacity comes from initialization logs (rounded). Replica {no_samples} had no scheduled batches; no usage curve is inferred.',fontsize=8.5,color='#555')
fig.text(.085,.06,'GET-request ticks identify confirmed external reloads, not measured router miss-token estimates (those were not logged).',fontsize=8.5,color='#555')
fig.text(.085,.04 if args.gpu_kv else .032,'GET timing: worker completion logs, 1s precision. PUT timing: metric export bins. Rates are observed bytes per bin, not transfer-duration measurements.',fontsize=8.5,color='#555')

stem = 'router_engine_mooncake_gpu_kv_aligned' if args.gpu_kv else 'router_engine_mooncake_aligned'
png = OUT/f'{stem}.png'
svg = OUT/f'{stem}.svg'
fig.savefig(png,dpi=180,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.15)
fig.savefig(svg,facecolor=fig.get_facecolor(),bbox_inches='tight',pad_inches=.15)
with (OUT/'router_engine_mooncake_movements.csv').open('w',newline='') as f:
    writer=csv.DictWriter(f,fieldnames=list(gets[0]));writer.writeheader();writer.writerows(gets)
with (OUT/'router_engine_mooncake_bins.csv').open('w',newline='') as f:
    writer=csv.writer(f);writer.writerow(['bin_start_s','get_completed_bytes','put_exported_bytes'])
    writer.writerows((int(t),get_bins[int(t)],buckets.get(int(t),{}).get('save_put_bytes_ok',0)) for t in bins)
audit=dict(run=RUN.name,plot_window_s=[0,END],bin_s=BIN,get_count=len(gets),
    get_bytes_from_deduplicated_logs=sum(r['bytes'] for r in gets),
    get_bytes_exported_counter=totals['load_get_bytes_ok'],
    get_log_bytes_match_exported_counter=True,
    log_display_to_utc_offsets=sorted({r['worker_display_to_utc_hours'] for r in gets}),
    router_per_request_miss_tokens_recorded=False,site_modified=False,
    note='Read-only saved-artifact analysis. No request was sent and no Site was edited.')
if args.gpu_kv:
    audit['gpu_kv'] = dict(source='batch_scheduled.kv_cache_usage',
        capacity_source=str(log_path.relative_to(REPO)),
        metric_definition='1 - free_blocks / (total_blocks - null_block); reclaimable cached blocks count as free',
        idle_gaps_over_s=1, extrapolate_missing_samples=False, replicas=gpu_summary)
    with (OUT/'router_engine_mooncake_gpu_kv_samples.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(gpu_rows[0]));writer.writeheader();writer.writerows(gpu_rows)
audit_name = 'router_engine_mooncake_gpu_kv_plot_audit.json' if args.gpu_kv else 'router_engine_mooncake_plot_audit.json'
(OUT/audit_name).write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2))
print(png)
