import json, bisect, math, statistics as st, collections
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
def pm(e, stale, w=60.0):
    last=e-stale; k=math.floor((last-PHI)/2)
    samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
    if not samples: return 0.0
    a=max(e-w,samples[0][0]); tot=0.0
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
        lo,hi=max(t1,a),min(t2,e)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/(e-a)
# one entry per export tick
ticks={}
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(v,p))
out=[]
for e,(v,p) in sorted(ticks.items()):
    if e<0: continue
    cands=[(abs(pm(e,d)-p),d) for d in [0.5*i for i in range(0,25)]]
    r0=abs(pm(e,0)-p); r,d=min(cands)
    out.append((e,p,pm(e,0),r0,d,r))
early=[o for o in out if o[0]<=650]; late=[o for o in out if o[0]>650]
for name,grp in (('early',early),('late',late)):
    print(f"{name}: ticks {len(grp)}; residual with stale=0: rmse {math.sqrt(st.mean(o[3]**2 for o in grp)):.0f}; best per-tick staleness: rmse {math.sqrt(st.mean(o[5]**2 for o in grp)):.0f}; staleness distribution: {sorted(collections.Counter(o[4] for o in grp).items())}")
print("\nearly ticks with residual > 1500 at stale=0: (e, pub, sim0, best stale, residual after)")
for o in early:
    if o[3]>1500: print(f"  e={o[0]:.1f} pub {o[1]:6.0f} sim {o[2]:6.0f} diff {o[1]-o[2]:+6.0f} -> stale {o[4]:4.1f}s residual {o[5]:5.0f}")
