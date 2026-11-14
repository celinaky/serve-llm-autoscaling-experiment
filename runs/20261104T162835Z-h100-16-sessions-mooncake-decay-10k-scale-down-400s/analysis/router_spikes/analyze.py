"""Read-only reconstruction of router spikes from this run's saved evidence."""
from pathlib import Path
from collections import defaultdict
import json, re, bisect, math, csv
import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
REPO = RUN.parent.parent
T0 = json.loads((RUN/'analysis/plot_data.json').read_text())['time_origin']['profiling_start_ns']/1e9
TRACES = Path('/mnt/cluster_storage/mooncake-kv/traces')/RUN.name
meta = {}
for p in RUN.glob('benchmark/*/profile_export.jsonl'):
    for line in p.open():
        d = json.loads(line)
        meta[d['metadata']['x_request_id']] = d

def aid(rid):
    return re.sub(r'^(chatcmpl-|cmpl-)', '', rid).rsplit('-', 1)[0]

records = []
for p in sorted(TRACES.glob('*.jsonl')):
    q, ad, ft, finish, done, preempt = {}, {}, {}, {}, {}, defaultdict(list)
    chunks = defaultdict(list)
    off = None
    for line in p.open():
        x = json.loads(line); ev = x['event']
        if off is None and 'wall_time_s' in x:
            off = x['wall_time_s']-x['monotonic_s']
        if ev == 'request_queued': q[x['request_id']] = x
        elif ev == 'request_admitted': ad[x['request_id']] = x
        elif ev == 'request_first_token': ft[x['request_id']] = x
        elif ev == 'request_finished': finish[x['request_id']] = x
        elif ev == 'request_preempted': preempt[x['request_id']].append((x['batch_index'], x['monotonic_s']+off-T0))
        elif ev == 'batch_completed': done[x['batch_index']] = x['monotonic_s']+off-T0
        elif ev == 'batch_scheduled':
            for r in x['requests']:
                if r['phase'] == 'prefill': chunks[r['request_id']].append((x['batch_index'], r['scheduled_tokens']))
    for rid, queued in q.items():
        profile = meta.get(aid(rid), {}); m = profile.get('metadata', {})
        admission = ad.get(rid, {}); first = ft.get(rid)
        cs = chunks.get(rid, [])
        stop_batch = first['batch_index'] if first else math.inf
        ds = []
        if cs and all(b in done for b, _ in cs):
            starts = [(0, queued['monotonic_s']+off-T0)] + sorted(preempt[rid])
            for i, (batch, at) in enumerate(starts):
                until = starts[i+1][0] if i+1 < len(starts) else math.inf
                ds.append((at, sum(n for b, n in cs if batch <= b < until)))
            ds.extend((done[b], -n) for b, n in cs)
        records.append(dict(request_id=aid(rid), engine_request_id=rid, engine=p.stem,
            start_s=queued['monotonic_s']+off-T0,
            client_start_s=(m['request_start_ns']/1e9-T0) if m.get('request_start_ns') else None,
            first_token_s=first['monotonic_s']+off-T0 if first else None,
            finish_s=finish[rid]['monotonic_s']+off-T0 if rid in finish else None,
            trace_id=m.get('source_trace_id','')[-6:], conversation_id=m.get('conversation_id'),
            turn=m.get('turn_index'), depth=m.get('agent_depth'), prompt_tokens=queued['prompt_tokens'],
            cached_tokens=admission.get('cached_prefix_tokens'),
            computed_before_first_token=sum(n for b,n in cs if b <= stop_batch),
            queue_wait_s=first['queue_wait_ms']/1000 if first else None,
            prefill_span_s=first['prefill_span_ms']/1000 if first else None,
            engine_ttft_s=first['engine_ttft_ms']/1000 if first else None,
            profile_error=profile.get('error'), preemptions=preempt[rid],
            engine_deltas=sorted(ds), get_bytes=0, get_log_lines=[]))
by_id = {r['request_id']:r for r in records}
ansi = re.compile(r'\x1b\[[0-9;]*m')
seen_get_lines = set()
for line in (REPO/'findings-h100/mooncake-decay-16-400s/run.log').open():
    if 'Mooncake load tier summary:' not in line: continue
    clean = ansi.sub('',line).strip()
    normalized = re.search(r'\(Worker pid=\d+\).*?INFO (\d\d-\d\d \d\d:\d\d:\d\d).*?(Mooncake load tier summary:.*)', clean)
    if normalized:
        worker = re.search(r'\(Worker pid=(\d+)\)', clean)[1]
        payload = re.sub(r'\s+\[repeated \d+x across cluster\]$', '', normalized[2])
        key = (worker, normalized[1], payload)
        if key in seen_get_lines: continue
        seen_get_lines.add(key)
    m = re.search(r'req_id=(\S+)',clean)
    if not m or aid(m[1]) not in by_id: continue
    r = by_id[aid(m[1])]
    match = re.search(r"'memory':\s*(\d+)",clean)
    r['get_bytes'] += int(match[1]) if match else 0
    r['get_log_lines'].append(clean)

raw, published = [], []
for line in (RUN/'telemetry/serve_metrics.jsonl').open():
    d = json.loads(line)
    if d.get('error'):continue
    t = d['scrape_started_at_ns']/1e9-T0
    values = {}
    for s in d.get('samples',[]):
        if s['name'] == 'ray_serve_autoscaling_policy_metric' and 'LLMServer' in s['labels'].get('deployment',''):
            values[s['labels']['metric']] = s['value']
    if 'outstanding_effective_prefill_tokens' in values:
        raw.append((t, values['outstanding_effective_prefill_tokens']))
    if 'prefill_required_replicas' in values:
        published.append((t, values['prefill_required_replicas']*20000))

def integral_mean(events, at, window=60):
    events = sorted(events)
    lo = at-window
    level = sum(d for t,d in events if t <= lo)
    acc, prev = 0., lo
    for t,d in events:
        if t <= lo:continue
        if t > at:break
        acc += level*(t-prev);level += d;prev = t
    return (acc+level*(at-prev))/window

engine_ds = sorted(v for r in records for v in r['engine_deltas'])
raw_ds=[];last=0
for t,v in raw:raw_ds.append((t,v-last));last=v

# Counterfactual reconstruction: use one FIFO 10K/s budget per engine,
# initialize with the actual scheduler-computed prefix and remove at first token.
# This is not a claim that these were the router's selection-time estimates.
def perfect_credit_replay():
    events=[]
    for r in records:
        if not r['engine_deltas']:continue
        events.append((r['start_s'],1,r['request_id'],r['engine'],r['computed_before_first_token']))
        end=r['first_token_s'] if r['first_token_s'] is not None else r['finish_s']
        if end is not None:events.append((end,0,r['request_id'],r['engine'],0))
    events.sort();pending=defaultdict(dict);last=0.;out=[];cur=0
    # 10ms integration gives a close visual reference, not exact router sampling.
    for t in np.arange(0,1050.01,.01):
        for eng in pending:
            budget=10000*(t-last)
            while pending[eng] and budget>0:
                rid=next(iter(pending[eng]));used=min(pending[eng][rid],budget)
                pending[eng][rid]-=used;budget-=used
                if pending[eng][rid]<=1e-8:del pending[eng][rid]
        while cur<len(events) and events[cur][0]<=t:
            _,kind,rid,eng,n=events[cur]
            if kind:pending[eng][rid]=float(n)
            else:pending[eng].pop(rid,None)
            cur+=1
        out.append(sum(sum(d.values()) for d in pending.values()));last=t
    grid=np.arange(0,1050.01,.01)
    mean=np.cumsum(out)*.01
    return grid,mean

grid,area=perfect_credit_replay()
def perfect_mean(at):
    hi=min(len(grid)-1,int(at/.01));lo=max(0,int((at-60)/.01))
    return float((area[hi]-area[lo])/60)

windows=[(425,530),(650,760),(810,910)]
spikes=[]
for lo,hi in windows:
    at,pub=max((t,v) for t,v in published if lo<=t<=hi)
    # Peak by value, not by timestamp.
    at,pub=max(((t,v) for t,v in published if lo<=t<=hi),key=lambda p:p[1])
    contributing=[]
    for r in records:
        contribution=integral_mean(r['engine_deltas'],at)
        if r['first_token_s'] and r['start_s']<at and r['first_token_s']>at-60:
            contributing.append({**r,'engine_mean_contribution':contribution})
    candidates=[r for r in records if r['get_bytes'] and r['start_s']<=at and r.get('first_token_s') and r['first_token_s']>=at-60]
    spikes.append(dict(window=[lo,hi],peak_time_s=at,published_mean_tokens=pub,
        engine_remaining_mean_tokens=integral_mean(engine_ds,at),
        exported_latest_gauge_reconstructed_mean=integral_mean(raw_ds,at),
        perfect_credit_fifo_counterfactual_mean=perfect_mean(at),
        get_requests=sorted(candidates,key=lambda r:r['start_s']),
        engine_top_contributions=sorted(contributing,key=lambda r:r['engine_mean_contribution'],reverse=True)[:8]))

result=dict(run=RUN.name,time_origin_epoch_s=T0,lookback_s=60,rate_tokens_per_second_per_engine=10000,
    router_selection_time_per_request_estimates_recorded=False,
    note='Router means are exported controller gauges. Engine counts are reconstructed from completed GPU prefill chunks, including preemptions, and exclude cache-transfer bytes/time. Positive GET logs establish DRAM loads; cached prefix tokens can include both local GPU and external hits.',
    requests=records,raw_router_gauge=raw,published_router_mean=published,spikes=spikes)
(OUT/'evidence.json').write_text(json.dumps(result,indent=2)+'\n')
summary=[]
for s in spikes:
    summary.append({k:s[k] for k in ['window','peak_time_s','published_mean_tokens','engine_remaining_mean_tokens','exported_latest_gauge_reconstructed_mean','perfect_credit_fifo_counterfactual_mean']})
    print(json.dumps(summary[-1]))
    print('GET candidates:',json.dumps([{'t':round(r['start_s'],2),'id':r['request_id'],'trace':r['trace_id'],'turn':r['turn'],'prompt':r['prompt_tokens'],'cached':r['cached_tokens'],'computed':r['computed_before_first_token'],'queue_s':r['queue_wait_s'],'prefill_s':r['prefill_span_s'],'get_GiB':round(r['get_bytes']/1024**3,3)} for r in s['get_requests']]))
(OUT/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
