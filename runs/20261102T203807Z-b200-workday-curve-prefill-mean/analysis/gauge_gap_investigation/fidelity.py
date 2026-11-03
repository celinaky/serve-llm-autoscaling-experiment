import json, bisect, math, statistics as st, random
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']*20000))
pubL=[(t,v) for t,v in pub if 60<=t<=1860]
def step(d):
    d=sorted(d); times=[t for t,_ in d]; lv=0; levels=[]
    for _,x in d: lv+=x; levels.append(lv)
    return times,levels
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def wmean(T,L,t,w=60.0):
    a=t-w; i0=bisect.bisect_right(T,a)
    pts=[(a,at(T,L,a))]+[(T[i],L[i]) for i in range(i0,bisect.bisect_right(T,t))]
    tot=0.0
    for (t1,v1),(t2,_) in zip(pts,pts[1:]+[(t,None)]):
        if t2>t1: tot+=v1*(t2-t1)
    return tot/w
def sampled_mean(T,L,t,phase,w=60.0,dt=2.0):
    # policy-style: 2s samples of the gauge, time-weighted over the window with carry-in
    k0=math.floor((t-w-phase)/dt); samples=[]
    for k in range(k0,int((t-phase)/dt)+1):
        ts_=phase+k*dt
        if ts_<=t: samples.append((ts_,at(T,L,ts_)))
    samples=[s for s in samples if s[0]>=t-w-dt]
    tot=0.0; a=t-w
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(t,None)]):
        lo,hi=max(t1,a),min(t2,t)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/w
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
A=step([(r['q'],r['own']) for r in reqs]+[(r['ft']+0.13,-r['own']) for r in reqs])
y=[v for _,v in pubL]
print("published policy mean vs exact rebuild (engine counts, +130ms):")
for sh in (0,4,6,8,10):
    x=[wmean(*A,t-sh) for t,_ in pubL]; print(f"  shift {sh:2d}s: corr {corr(x,y):.3f}  rmse {rmse(x,y):.0f}")
# how much noise does the policy's own 2s sampling add? compare exact rebuild mean to 2s-sampled rebuild mean
random.seed(0)
ex=[wmean(*A,t) for t,_ in pubL]
for phase in (0.0,0.7,1.3):
    sm=[sampled_mean(*A,t,phase) for t,_ in pubL]
    print(f"  2s-sampling noise alone (phase {phase}): rmse exact-vs-sampled rebuild {rmse(ex,sm):.0f}, corr {corr(ex,sm):.3f}")
# sampled rebuild vs published at best shift
best=None
for sh in (4,6,8,10):
    for phase in (0.0,0.5,1.0,1.5):
        sm=[sampled_mean(*A,t-sh,phase) for t,_ in pubL]; c=corr(sm,y)
        if best is None or c>best[0]: best=(c,sh,phase,rmse(sm,y))
print(f"  2s-sampled rebuild vs published: best corr {best[0]:.3f} at shift {best[1]}s phase {best[2]} (rmse {best[3]:.0f})")
# time-resolved fit: by 300s block, rmse at shift 6
x=[wmean(*A,t-6) for t,_ in pubL]
for lo in range(60,1860,300):
    idx=[i for i,(t,_) in enumerate(pubL) if lo<=t<lo+300]
    print(f"  {lo:4d}-{lo+300:4d}: corr {corr([x[i] for i in idx],[y[i] for i in idx]):.3f} rmse {rmse([x[i] for i in idx],[y[i] for i in idx]):.0f}  mean pub {st.mean(y[i] for i in idx):.0f} rebuild {st.mean(x[i] for i in idx):.0f}")
