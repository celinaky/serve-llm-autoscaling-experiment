"""Extend the original plot with measured Mooncake traffic and request credit."""
from pathlib import Path
from collections import defaultdict, Counter
from datetime import datetime, timezone
import csv
import importlib.util
import json
import re
import runpy
import sys

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import MultipleLocator, FuncFormatter
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parents[1]
REPO = RUN.parents[1]
END, BIN, WINDOW = 1050, 10, 60
data = json.loads((RUN/'analysis/plot_data.json').read_text())
T0_NS = data['time_origin']['profiling_start_ns']
T0 = T0_NS/1e9
TRACE = Path('/mnt/cluster_storage/mooncake-kv/traces')/RUN.name
LOG = REPO/'findings-h100/mooncake-no-decay-16-audit-400s/run.log'

records = []
for raw in csv.DictReader((RUN/'analysis/router_cache_credit/requests.csv').open()):
    r = dict(raw)
    for k,v in r.items():
        if k not in ('request_id','engine','router_replica_id','accounting_consistent'):
            r[k] = float(v) if v else None
    records.append(r)
by_id = {r['request_id']:r for r in records}
external = [r for r in records if r['engine_external_cached_tokens'] > 0]
uncredited = [r for r in external if r['router_initial_excess_vs_gpu_tokens'] > 0]
credited = [r for r in external if r['router_initial_excess_vs_gpu_tokens'] == 0]
local = [r for r in records if r['engine_external_cached_tokens'] == 0
         and r['engine_local_cached_tokens'] > 0 and r['router_initial_excess_vs_gpu_tokens'] > 0]
assert (len(external),len(uncredited),len(credited),len(local)) == (17,10,7,32)
assert all(r['router_effective_prefill_tokens'] == r['router_prompt_tokens']-r['router_gpu_overlap_tokens'] for r in records)
assert all(r['router_gpu_overlap_tokens'] == r['router_cpu_overlap_tokens'] == 0
           and r['router_initial_excess_vs_gpu_tokens'] == r['engine_external_cached_tokens'] for r in uncredited)
assert all(r['router_initial_excess_vs_gpu_tokens'] == r['engine_local_cached_tokens']-r['router_gpu_overlap_tokens'] for r in local)
uncredited_ids = {r['request_id'] for r in uncredited}
local_ids = {r['request_id'] for r in local}

# Reuse the original plotting implementation and its actual axis/line artists.
# It writes a separate reference copy: the original PNG is never overwritten.
saved_argv = sys.argv
sys.argv = ['policy_vs_engine_mean',str(RUN),'--lookback','60','--end',str(END),
            '--traces',str(TRACE),'--out',str(OUT/'original_reference.png')]
try:
    original = runpy.run_module('serve_llm_autoscaling.diagnostics.policy_vs_engine_mean',run_name='__main__')
finally:
    sys.argv = saved_argv
fig,top = original['fig'],original['ax']


class StepSeries:
    def __init__(self, points):
        self.t,self.v = np.array(sorted(points,key=lambda p:p[0])).T
        self.area = np.r_[0,np.cumsum(self.v[:-1]*np.diff(self.t))]

    def integral(self, at):
        at = np.asarray(at)
        idx = np.searchsorted(self.t,at,side='right')-1
        i = np.maximum(idx,0)
        return np.where(idx<0,0,self.area[i]+self.v[i]*(at-self.t[i]))

    def mean(self, at):
        at = np.asarray(at)
        return (self.integral(at)-self.integral(at-WINDOW))/WINDOW


# Original green series uses K tokens; this reconstruction keeps identical events.
engine = StepSeries(list(zip(original['eng_times'],np.array(original['eng_levels'])*1000)))
assert np.allclose(engine.mean(original['ts'])/1000,original['engine_mean'])
snapshots = []
for path in sorted((TRACE/'router-request-audit').glob('*.jsonl')):
    for line in path.open():
        r = json.loads(line)
        if r['event'] != 'router_autoscaling_snapshot':continue
        assert abs(sum(x['remaining_estimate_tokens'] for x in r['requests'])-r['outstanding_effective_prefill_tokens']) <= 1
        extra_external = extra_local = 0
        for request in r['requests']:
            rid = request['request_id']
            # Decay is disabled: logged contributions retain their initial values.
            assert request['remaining_estimate_tokens'] == request['initial_effective_prefill_tokens']
            if rid in uncredited_ids:extra_external += by_id[rid]['router_initial_excess_vs_gpu_tokens']
            if rid in local_ids:extra_local += by_id[rid]['router_initial_excess_vs_gpu_tokens']
        snapshots.append((r['wall_time_s']-T0,r['outstanding_effective_prefill_tokens'],extra_external,extra_local))
assert snapshots
router = StepSeries([(t,s) for t,s,_,_ in snapshots])
external_component = StepSeries([(t,e) for t,_,e,_ in snapshots])
local_component = StepSeries([(t,l) for t,_,_,l in snapshots])

# GET completion logs give real bytes and request IDs. De-duplicate Ray forwarding.
gets,seen = [],set()
for raw in LOG.open():
    if 'Mooncake load tier summary:' not in raw:continue
    clean = re.sub(r'\x1b\[[0-9;]*m','',raw)
    match = re.search(r'\(Worker pid=(\d+)\).*?INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?(Mooncake load tier summary:.*)',clean)
    assert match,clean
    payload = re.sub(r'\s+\[repeated \d+x across cluster\]$', '',match[3].strip())
    key = (match[1],match[2],payload)
    if key in seen:continue
    seen.add(key)
    rid = re.search(r'req_id=(\S+)',payload)[1]
    rid = re.sub(r'^(?:chatcmpl-|cmpl-)','',rid).rsplit('-',1)[0]
    assert rid in by_id,rid
    request = by_id[rid]
    local_epoch = datetime.strptime('2026-'+match[2],'%Y-%m-%d %H:%M:%S').replace(tzinfo=timezone.utc).timestamp()
    upper = request['engine_first_token_wall_time_s'] or request['engine_admitted_wall_time_s']
    offsets = [h for h in range(-14,15) if request['router_selected_wall_time_s']-1 <= local_epoch+3600*h <= upper+2]
    assert len(offsets)==1,(rid,offsets)
    completed = local_epoch+3600*offsets[0]-T0
    size = int(re.search(r"'memory':\s*(\d+)",payload)[1])
    gets.append(dict(request_id=rid,time_s=completed,bytes=size,
        initial_credit='uncredited_external' if rid in uncredited_ids else 'already_credited',
        arrival_s=request['router_selected_wall_time_s']-T0))
assert len(gets) == len(external)
assert {r['request_id'] for r in gets} == {r['request_id'] for r in external}

spec = importlib.util.spec_from_file_location('mooncake_report',REPO/'tools/report_mooncake.py')
report = importlib.util.module_from_spec(spec);spec.loader.exec_module(report)
replicas = report.replica_ids(LOG,'ttft-benchmark')
totals,buckets,_ = report.operation_data(RUN,replicas,T0_NS)
assert sum(r['bytes'] for r in gets) == totals['load_get_bytes_ok']
bins = np.arange(0,END,BIN)
missing_bytes,credited_bytes = defaultdict(int),defaultdict(int)
for r in gets:
    target = missing_bytes if r['initial_credit']=='uncredited_external' else credited_bytes
    target[int(r['time_s']//BIN)*BIN] += r['bytes']
get_missing = np.array([missing_bytes[int(t)]/(BIN*1024**2) for t in bins])
get_credited = np.array([credited_bytes[int(t)]/(BIN*1024**2) for t in bins])
put = [buckets.get(int(t),{}).get('save_put_bytes_ok',0)/(BIN*1024**2) for t in bins]

# GPU events: count removed prefix hashes, preserving hybrid-group semantics.
original_replica = min(records,key=lambda r:r['router_selected_wall_time_s'])['router_replica_id'].rsplit('#',1)[-1]
removed = defaultdict(int)
groups = {}
event_errors = []
for line in (RUN/'telemetry/gpu_kv_events.jsonl').open():
    r = json.loads(line)
    if r.get('error'):event_errors.append(r)
    for event in r.get('events',[]):
        if event.get('medium') != 'GPU' or r['replica_id'] != original_replica:continue
        group = event.get('group_idx')
        if event.get('kv_cache_spec_kind'):groups[group] = event['kv_cache_spec_kind']
        if event['type'] == 'BlockRemoved':
            t = r['timestamp_ns']/1e9-T0
            if 0 <= t <= END:removed[(int(t//BIN)*BIN,group)] += event['block_count']
assert not event_errors
full_groups = [g for g,kind in groups.items() if kind=='full_attention']
assert full_groups == [3],groups
removal_counts = [removed[(int(t),3)] for t in bins]

# Resize/reposition the original top axis, then append new shared-time panels.
for legend in list(fig.legends):legend.remove()
for text in list(fig.texts):text.remove()
fig.set_size_inches(15,18.6)
grid = fig.add_gridspec(5,1,left=.085,right=.985,bottom=.115,top=.94,
    hspace=.55,height_ratios=[1.3,.92,.65,.95,1.18])
top.set_position(grid[0].get_position(fig))
movement,evictions,credit,gap = [fig.add_subplot(grid[i],sharex=top) for i in range(1,5)]
purple,green,red,blue,teal,gray = '#6250d6','#1baf7a','#d86437','#3f82bd','#21877c','#8b939b'
top.set_title('1. Original router-versus-engine curves · decay disabled',loc='left',fontsize=12,pad=10)
top.set_ylabel('Outstanding prefill (K tokens)\n60s trailing mean')
top.set_xlabel('')
top.legend(handles=[Line2D([],[],color=purple,lw=2,label='Published router/policy mean'),
    Line2D([],[],color=green,lw=2,label='Reconstructed remaining GPU prefill'),
    Line2D([],[],color='#555',ls=':',lw=1,label='20K target per replica')],
    loc='upper center',fontsize=8.5,frameon=False)

movement.bar(bins+5,get_missing,8.5,color=red,label='GET · no prefix credit at routing (10 requests)')
movement.bar(bins+5,get_credited,8.5,bottom=get_missing,color=teal,label='GET · prefix already credited at routing (7 requests)')
movement.plot(bins+5,put,color=gray,lw=1.4,label='PUT · exported byte-counter deltas')
movement.set_ylabel('Mooncake traffic\nMiB/s · 10s bins')
movement.set_title('2. Actual Mooncake transfers · GET bytes grouped by the request’s initial router credit',loc='left',fontsize=11,pad=10)
movement.legend(loc='upper left',fontsize=8.5,frameon=False)
movement.set_ylim(bottom=0)

evictions.bar(bins+5,removal_counts,8.5,color=blue,alpha=.8)
evictions.set_ylabel('Removed prefix hashes\nper 10s')
evictions.set_title(f'3. GPU prefix removals on original replica {original_replica} · full-attention group 3 only',loc='left',fontsize=11,pad=10)
evictions.set_ylim(bottom=0)

for rows,color,label in [(uncredited,red,'External/Mooncake reuse absent from initial credit (10)'),
                         (local,blue,'Local GPU reuse beyond initial router credit (32)')]:
    x = [r['router_selected_wall_time_s']-T0 for r in rows]
    y = [r['router_initial_excess_vs_gpu_tokens']/1000 for r in rows]
    credit.vlines(x,0,y,color=color,alpha=.7,lw=1)
    credit.scatter(x,y,s=22,color=color,zorder=4,label=label)
credit.scatter([r['router_selected_wall_time_s']-T0 for r in credited],[-9]*len(credited),
    marker='v',color=teal,s=34,label='Mooncake hit, but initial credit already matched GPU work (7)')
credit.axhline(0,color='#888',lw=.8)
credit.set_ylim(-17,200)
credit.yaxis.set_major_formatter(FuncFormatter(lambda v,_:f'{v:g}' if v>=0 else ''))
credit.set_ylabel('Initial router estimate minus\nGPU computation (K tokens)')
credit.set_title('4. Matched request evidence · each stem is one request, positioned at router selection',loc='left',fontsize=11,pad=10)
credit.legend(loc='upper left',fontsize=8.3,frameon=False)
example = next(r for r in uncredited if r['request_id'].startswith('e8174d'))
at = example['router_selected_wall_time_s']-T0
credit.annotate('Router 160.9K · GPU 2.5K\n158.4K externally reused',
    xy=(at,158.368),xytext=(at-145,110),fontsize=8.5,color=red,
    arrowprops={'arrowstyle':'->','color':red,'lw':.8})

x = np.arange(0,END+.25,.25)
eng = engine.mean(x)/1000
audit_gap = (router.mean(x)-engine.mean(x))/1000
ext = external_component.mean(x)/1000
loc = local_component.mean(x)/1000
other = audit_gap-ext-loc
assert np.allclose(audit_gap,ext+loc+other)
gap.fill_between(x,0,ext,color=red,alpha=.65,label='Uncredited external-prefix component')
gap.fill_between(x,ext,ext+loc,color=blue,alpha=.55,label='Additional local-prefix credit gap')
gap.plot(x,other,color=gray,lw=1.3,label='Other signed accounting difference¹')
gap.plot(x,audit_gap,color='#303030',lw=1.5,label='Total gap from request-audit snapshots')
px = np.array(original['pub_t']);pv = np.array(original['pub_v'])
gap.plot(px,pv-engine.mean(px)/1000,color=purple,lw=1.7,ls='--',label='Published purple minus green (top panel)')
gap.axhline(0,color='#777',lw=.8)
gap.set_ylabel('Router minus engine\nK tokens · 60s trailing mean')
gap.set_xlabel('Seconds since profiling start')
gap.set_title('5. How much of the gap do those requests contribute? · actual logged request lifetimes',loc='left',fontsize=11,pad=10)
gap.legend(loc='upper left',fontsize=8.2,frameon=False,ncol=2)
gap.set_ylim(min(-10,float(np.min(other))*1.15),max(float(np.max(audit_gap)),float(np.max(pv-engine.mean(px)/1000)))*1.38)

for i,ax in enumerate((top,movement,evictions,credit,gap)):
    ax.set_facecolor('#fcfcfb');ax.spines[['top','right']].set_visible(False)
    ax.set_xlim(0,END);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
    ax.xaxis.set_major_locator(MultipleLocator(100))
    if i < 4:ax.tick_params(labelbottom=False)
    if i:
        for t in (379.3,659.7,1001.7):ax.axvline(t,color='#a99a8e',lw=.7,ls=':',alpha=.7,zorder=0)

fig.suptitle('H100 · 16 sessions · Mooncake · no decay · measured cache-credit evidence',fontsize=14,y=.985)
fig.text(.085,.075,'¹ Other = total audited gap − the two colored cache-credit components; it includes prefill-progress accounting, negative credit gaps, timing and unmatched requests.',fontsize=8.5,color='#555')
fig.text(.085,.059,'The dashed/black difference is between the controller’s published mean and the exact router audit. Cache components are not fitted to the published curve.',fontsize=8.5,color='#555')
fig.text(.085,.043,'GET bars use deduplicated worker completion logs (1s precision); PUT bars use metric exports. Removal events do not record a capacity-eviction reason.',fontsize=8.5,color='#555')
fig.text(.085,.027,'Local-credit gaps compare selection with later engine reuse; stale indexing and newly available in-flight prefixes are not distinguished. All recorded requests are included.',fontsize=8.5,color='#555')

png = OUT/'router_mooncake_cache_credit_context.png'
svg = OUT/'router_mooncake_cache_credit_context.svg'
fig.savefig(png,dpi=170,bbox_inches='tight',pad_inches=.15,facecolor=fig.get_facecolor())
fig.savefig(svg,bbox_inches='tight',pad_inches=.15,facecolor=fig.get_facecolor())
with (OUT/'request_credit_points.csv').open('w',newline='') as f:
    fields = ['request_id','selection_s','category','router_estimate','gpu_compute','external_reuse','local_reuse','initial_excess']
    writer = csv.DictWriter(f,fieldnames=fields);writer.writeheader()
    for category,rows in [('external_uncredited',uncredited),('local_extra_reuse',local),('external_already_credited',credited)]:
        for r in rows:
            writer.writerow(dict(request_id=r['request_id'],selection_s=r['router_selected_wall_time_s']-T0,
                category=category,router_estimate=r['router_effective_prefill_tokens'],gpu_compute=r['engine_initial_gpu_prefill_tokens'],
                external_reuse=r['engine_external_cached_tokens'],local_reuse=r['engine_local_cached_tokens'],initial_excess=r['router_initial_excess_vs_gpu_tokens']))
with (OUT/'gap_components.csv').open('w',newline='') as f:
    writer = csv.writer(f);writer.writerow(['time_s','engine_mean_K','audited_gap_K','external_component_K','local_component_K','other_signed_K'])
    writer.writerows(zip(x,eng,audit_gap,ext,loc,other))
audit = dict(run=RUN.name,original_top_curve_implementation_reused=True,source_trace_directory=str(TRACE),
    external_requests=len(external),external_uncredited_requests=len(uncredited),external_already_credited_requests=len(credited),
    local_extra_reuse_requests=len(local),get_bytes_from_logs=sum(r['bytes'] for r in gets),
    get_bytes_from_exported_counters=totals['load_get_bytes_ok'],get_bytes_match=True,
    original_replica=original_replica,gpu_full_attention_group=3,gpu_prefix_hash_removals=sum(removal_counts),
    request_audit_snapshots=len(snapshots),no_decay_verified=True,site_modified=False,
    load_window_mean_K=dict(audited_router_minus_engine=float(np.trapezoid(audit_gap,x)/END),
        uncredited_external_component=float(np.trapezoid(ext,x)/END),
        additional_local_reuse_component=float(np.trapezoid(loc,x)/END),
        other_signed_accounting_difference=float(np.trapezoid(other,x)/END)),
    note='Colored gap components are initial excess tokens for requests still present in actual router snapshots. They do not assume every Mooncake GET lacked credit.')
(OUT/'plot_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
print(json.dumps(audit,indent=2));print(png)
