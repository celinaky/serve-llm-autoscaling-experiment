import json, bisect, math, statistics as st, random
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values()); G=[tuple(x) for x in S['G']]
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']*20000))
def step(d):
    d=sorted(d); times=[t for t,_ in d]; lv=0; levels=[]
    for _,x in d: lv+=x; levels.append(lv)
    return times,levels
def at(times,levels,t):
    i=bisect.bisect_right(times,t)-1; return levels[i] if i>=0 else 0
def wmean(times,levels,t,w=60.0):
    a=t-w; i0=bisect.bisect_right(times,a)
    pts=[(a,at(times,levels,a))]+[(times[i],levels[i]) for i in range(i0,bisect.bisect_right(times,t))]
    tot=0.0
    for (t1,v1),(t2,_) in zip(pts,pts[1:]+[(t,None)]):
        if t2>t1: tot+=v1*(t2-t1)
    return tot/w
A=step([(r['q'],r['own']) for r in reqs]+[(r['ft']+0.13,-r['own']) for r in reqs])
print("window: mean published policy mean vs mean 60s-mean of rebuild (engine own_uncached, +130ms hold, shifted 6s), ratio")
for lo,hi in [(60,300),(300,600),(600,700),(700,800),(800,940),(940,1040),(1040,1200),(1200,1400),(1400,1600),(1600,1860)]:
    P=[v for t,v in pub if lo<=t<hi]; X=[wmean(*A,t-6) for t,v in pub if lo<=t<hi]
    print(f"  {lo:4d}-{hi:4d}: pub {st.mean(P):6.0f}  rebuild {st.mean(X):6.0f}  diff {st.mean(P)-st.mean(X):6.0f}  ratio {st.mean(P)/st.mean(X):.2f}")
# states explained only by counting a warm request at full isl
Gs=[(t,v) for t,v in G if 0<=t<=1860]
states=[]
for t,v in Gs:
    if v>0 and (not states or states[-1][1]!=v or t-states[-1][2]>4.5): states.append([t,v,t])
    elif v>0: states[-1][2]=t
def subset(target, vals, tol=1):
    sums={0:()}
    for i,v in enumerate(vals):
        new={}
        for s,sub in sums.items():
            ns=s+v
            if ns<=target+tol and ns not in sums and ns not in new: new[ns]=sub+(i,)
        sums.update(new)
    for s in range(target-tol,target+tol+1):
        if s in sums: return sums[s]
    return None
print("\nstates not explained by engine own_uncached but explained when some request counts at full prompt:")
for t,v,t2 in states:
    c=[r for r in reqs if r['q']<=t+1 and r['ft']>=t-14]
    if not c or len(c)>18: continue
    if subset(int(v),[r['own'] for r in c]) is not None: continue
    vals=[r['own'] for r in c]+[r['isl']-r['own'] for r in c]
    sub=subset(int(v),vals)
    if sub is None: continue
    n=len(c); full=[c[i-n] for i in sub if i>=n]; owns=[c[i] for i in sub if i<n]
    print(f"  t={t:.1f} value={int(v)}: full-prompt counted: {[(r['isl'],r['cached'],r['kind'],round(r['q'],1),r['eng'][-6:]) for r in full]}; plus own of {[r['own'] for r in owns]}")
