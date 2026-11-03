import json, bisect, math, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
S=json.load(open('gauge_decomp_state.json')); reqs=sorted(S['reqs'].values(), key=lambda r:r['q'])
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI=9.6
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
LEAD,HOLD=0.1,0.1
T,L=step([(r['q']-LEAD,r['own']) for r in reqs]+[(r['ft']+HOLD,-r['own']) for r in reqs])
def tick_schedule(phi, c, end=1900.0):
    """record ticks every 2s after the previous one completes; a tick that falls inside a
    blocked interval [q-lead-c*isl, q-lead] (tokenizing on the loop before admission) is
    delayed to the block's end."""
    blocks=sorted((r['q']-LEAD-c*r['isl'], r['q']-LEAD) for r in reqs)
    bs=[b[0] for b in blocks]
    ticks=[]; t=phi
    while t<end:
        # find a block containing t
        i=bisect.bisect_right(bs,t)-1
        if i>=0 and blocks[i][0]<=t<blocks[i][1]: t=blocks[i][1]
        ticks.append(t); t+=2.0
    return ticks
def sim(ticks, w=60.0):
    out=[]; raw=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10)
        j=bisect.bisect_right(ticks,e)
        samples=[(s,at(T,L,s)) for s in ticks[max(0,j-35):j]]
        a=e-w; tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/w); raw.append(samples[-1][1] if samples else 0)
    return out, raw
pub=[p for _,_,p in rows]; rawv=[v for _,v,_ in rows]
early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("tick slip = c x prompt tokens per admission; early/late RMSE and bias of published mean vs simulation; exact raw matches")
for c in [0, 2e-6, 4e-6, 6e-6, 8e-6, 10e-6, 12e-6, 15e-6, 20e-6]:
    best=None
    for phi in [0.1*i for i in range(20)]:
        s,raw=sim(tick_schedule(phi,c))
        r_e=rmse([s[i] for i in early],[pub[i] for i in early])
        if best is None or r_e<best[0]: best=(r_e,phi,s,raw)
    r_e,phi,s,raw=best
    print(f"  c={c*1e6:4.0f} us/token phi={phi:.1f}: early rmse {r_e:5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} | late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):5.0f} bias {st.mean(pub[i]-s[i] for i in late):+5.0f} | raw exact: early {sum(1 for i in early if abs(raw[i]-rawv[i])<=1)}/{len(early)} late {sum(1 for i in late if abs(raw[i]-rawv[i])<=1)}/{len(late)} | whole-run sim mean {st.mean(s):.0f} vs pub {st.mean(pub):.0f}")
