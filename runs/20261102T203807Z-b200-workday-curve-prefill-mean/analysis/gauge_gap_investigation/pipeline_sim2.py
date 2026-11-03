import json, bisect, math, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def sim(T,L,stale,w=60.0):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10)          # evaluation/export time
        last=e-stale                              # latest sample the controller has
        k=math.floor((last-PHI)/2); ticks=[PHI+2*j for j in range(k-32,k+1)]
        samples=[(s,at(T,L,s)) for s in ticks]
        a=e-w; tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/w)
    return out
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("early RMSE / bias of published mean vs simulated, by series staleness at evaluation and by lead/hold:")
for lead,hold in [(0.0,0.1),(0.1,0.1),(0.2,0.1),(0.0,0.0),(0.2,0.0)]:
    T,L=step([(r['q']-lead,r['own']) for r in reqs]+[(r['ft']+hold,-r['own']) for r in reqs])
    line=f"  lead {lead:.1f} hold {hold:.1f}:"
    for stale in (0.0,1.0,2.0,3.0,4.0):
        s=sim(T,L,stale)
        line+=f"  stale {stale:.0f}s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} |"
    print(line)
# residual of late region at best early setting
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
for stale in (0.0,2.0):
    s=sim(T,L,stale)
    print(f"stale {stale:.0f}s, lead .1 hold .1: late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):.0f} bias {st.mean(pub[i]-s[i] for i in late):+.0f}; whole-run mean pub {st.mean(pub):.0f} sim {st.mean(s):.0f}")
json.dump(sim(T,L,2.0),open('sim_best.json','w'))
# 'not counted' cases: where is the record tick relative to the request window?
print("\n'not counted' cases: sample time relative to the request's engine window [q, ft]:")
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in reqs if r['q']<=s<r['ft']+0.1]
    if v==0 and held and all(r['eng'][-6:] in ('180495','173074') for r in held):
        for r in held: print(f"  t={t:.1f} sample s={s:.1f}: request q={r['q']:.2f} ft={r['ft']:.2f} (s-q={s-r['q']:+.2f}s, ft-s={r['ft']-s:+.2f}s) own {r['own']} eng {r['eng'][-6:]}")
