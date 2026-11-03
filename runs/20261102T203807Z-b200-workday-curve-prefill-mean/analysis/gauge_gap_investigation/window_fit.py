import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]
def sim(w, drop_first=0, mode='full'):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        ticks=[PHI+2*j for j in range(k-40,k+1)]
        samples=[(s,at(T,L,s)) for s in ticks if s>=e-w][drop_first:]
        if not samples: out.append(0); continue
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
print("published mean vs simulated with different effective window lengths (early region):")
for w in (20,30,40,50,58,60,70,80,100):
    s=sim(w); print(f"  window {w:3d}s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
# alternative: samples carry forward only 2s each (no carry to 'now'), i.e. plain average of samples in window
def sim_avg(w):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        vals=[at(T,L,PHI+2*j) for j in range(k-40,k+1) if PHI+2*j>=e-w]
        out.append(st.mean(vals))
    return out
s=sim_avg(60); print(f"  plain average of 2s samples, 60s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
# what if the series the controller holds is only the last N samples per push (e.g. a cap)?
for n in (5,10,15,20,25,30):
    def sim_cap(n):
        out=[]
        for t,v,p in rows:
            e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
            samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-n+1,k+1)]
            a=samples[0][0]; tot=0.0
            for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
                if t2>t1: tot+=v1*(t2-t1)
            out.append(tot/(e-a))
        return out
    s=sim_cap(n); print(f"  last {n:2d} samples only: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
