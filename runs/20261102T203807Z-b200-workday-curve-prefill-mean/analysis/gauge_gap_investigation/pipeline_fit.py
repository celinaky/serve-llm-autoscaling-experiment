"""Fit the telemetry pipeline (router 2s record tick, controller evaluation/export tick)
so the trace rebuild can be compared to the exported gauge and published mean exactly."""
import json, bisect, math, statistics as st, itertools, sys
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
rows=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0; d={}
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and 'LLMServer' in s['labels'].get('deployment',''): d[s['labels']['metric']]=s['value']
    if 'outstanding_effective_prefill_tokens' in d: rows.append((t,d['outstanding_effective_prefill_tokens'],d['prefill_required_replicas']*20000))
rows=[r for r in rows if 0<=r[0]<=1860]
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def level_fn(lead,hold):
    return step([(r['q']-lead,r['own']) for r in reqs]+[(r['ft']+hold,-r['own']) for r in reqs])
EARLY=650
best=[]
leads=[0.0,0.05,0.1,0.2,0.3]; holds=[0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8]
fns={(l,h):level_fn(l,h) for l in leads for h in holds}
psis=[round(5.5+0.1*i,2) for i in range(43)]; phis=[round(0.05*i,2) for i in range(40)]
results=[]
for psi in psis:
    for phi in phis:
        # evaluation time per scrape, record sample time
        samp=[]
        for t,v,_ in rows:
            e=psi+10*math.floor((t-psi)/10); s=phi+2*math.floor((e-phi)/2); samp.append((s,v,t))
        for (l,h),(T,L) in fns.items():
            m_early=sum(1 for s,v,t in samp if t<=EARLY and abs(at(T,L,s)-v)<=1)
            m_all=sum(1 for s,v,t in samp if abs(at(T,L,s)-v)<=1)
            results.append((m_early,m_all,psi,phi,l,h))
results.sort(reverse=True)
n_early=sum(1 for r in rows if r[0]<=EARLY)
print(f"scrapes: early (<= {EARLY}s) {n_early}, all {len(rows)}")
print("top fits by exact matches of exported raw gauge vs rebuild sampled at the implied record tick (early matches, all matches, eval phase mod 10, record phase mod 2, lead s, hold s):")
for r in results[:12]: print("  ",r)
# baseline: how many exact matches do you get ignoring the pipeline (sample rebuild at scrape time)?
T,L=fns[(0.0,0.0)]
print("baseline, rebuild sampled at scrape time, no pipeline model: early", sum(1 for t,v,_ in rows if t<=EARLY and abs(at(T,L,t)-v)<=1), "all", sum(1 for t,v,_ in rows if abs(at(T,L,t)-v)<=1))
json.dump(results[:50],open('pipeline_fit_top.json','w'))
