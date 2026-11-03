import json, bisect, math, statistics as st
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
pubL=[(t,v) for t,v in pub if 0<=t<=1860]
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
def rb(sel,lag=0.0): return step([(r['q'],r['own']) for r in sel]+[(r['ft']+lag,-r['own']) for r in sel])
cold=[r for r in reqs if r['kind']=='cold']; warm=[r for r in reqs if r['kind']=='warm']
y=[v for _,v in pubL]
for lag in (0.0,0.13):
    C=rb(cold,lag); W=rb(warm,lag); A=rb(reqs,lag)
    for sh in (6,):
        xc=[wmean(*C,t-sh) for t,_ in pubL]; xw=[wmean(*W,t-sh) for t,_ in pubL]; xa=[wmean(*A,t-sh) for t,_ in pubL]
        s11=sum(a*a for a in xc); s22=sum(b*b for b in xw); s12=sum(a*b for a,b in zip(xc,xw)); s1y=sum(a*v for a,v in zip(xc,y)); s2y=sum(b*v for b,v in zip(xw,y))
        det=s11*s22-s12*s12; ac=(s1y*s22-s2y*s12)/det; aw=(s11*s2y-s12*s1y)/det
        resid=[v-(ac*a+aw*b) for a,b,v in zip(xc,xw,y)]
        print(f"lag {lag*1000:.0f} ms, shift {sh}s: published ~ {ac:.2f} x cold + {aw:.2f} x warm (rmse {math.sqrt(st.mean(r*r for r in resid)):.0f}); single slope {sum(a*v for a,v in zip(xa,y))/sum(a*a for a in xa):.2f}; means: pub {st.mean(y):.0f} cold {st.mean(xc):.0f} warm {st.mean(xw):.0f} all {st.mean(xa):.0f}")
# also with intercept, single slope
A=rb(reqs,0.13); xa=[wmean(*A,t-6) for t,_ in pubL]
mx,my=st.mean(xa),st.mean(y); b=sum((a-mx)*(v-my) for a,v in zip(xa,y))/sum((a-mx)**2 for a in xa); print(f"with intercept: slope {b:.2f} intercept {my-b*mx:.0f}")
