"""Model the router's KV index as an LRU of C blocks per worker fed by the traces; derive each
request's router-side uncached count and push it through the telemetry pipeline."""
import json, bisect, math, statistics as st
from collections import OrderedDict
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
B=784
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid),{}); r=dict(r); r['conv']=a.get('conversation_id',rid); r['osl']=a.get('output_sequence_length') or 0; r['turn']=a.get('turn_index'); R.append(r)
R.sort(key=lambda r:r['q'])
def router_counts(C, refresh_on_hit, scope='worker'):
    idx={}; evicted={}
    out={}
    for r in R:
        key=r['eng'] if scope=='worker' else 'all'
        d=idx.setdefault(key,OrderedDict()); ev=evicted.setdefault(r['eng'],set())
        nb_cached=r['cached']//B; nb_total=(r['isl']+r['osl'])//B
        # router overlap = engine's cached prefix, cut at the first block the router index evicted
        n=nb_cached
        for k in range(nb_cached):
            if (r['eng'],r['conv'],k) in ev: n=k; break
        out[r['q']]=r['own']+(nb_cached-n)*B
        if refresh_on_hit:
            for k in range(0, nb_cached):
                if (r['eng'],r['conv'],k) in d: d.move_to_end((r['eng'],r['conv'],k))
        for k in range(nb_cached, nb_total):
            d[(r['eng'],r['conv'],k)]=None; d.move_to_end((r['eng'],r['conv'],k)); ev.discard((r['eng'],r['conv'],k))
        if C is not None:
            while len(d)>C:
                (e_,c_,k_),_=d.popitem(last=False); evicted.setdefault(e_,set()).add((e_,c_,k_))
    return out
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def sim(T,L,w=60.0):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("router index as LRU of C blocks per worker: published mean vs simulated rebuild")
for scope,refresh in (('worker',False),('worker',True),('all',False),('all',True)):
    for C in (None, 8192, 6144, 5120, 4608, 4096, 3584, 3072, 2560, 2048):
        cnt=router_counts(C, refresh, scope)
        T,L=step([(r['q']-0.1,cnt[r['q']]) for r in R]+[(r['ft']+0.1,-cnt[r['q']]) for r in R])
        s=sim(T,L)
        print(f"  scope={scope:6} refresh_on_hit={refresh!s:5} C={str(C):5}: early rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} | late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):5.0f} bias {st.mean(pub[i]-s[i] for i in late):+5.0f} | whole-run sim mean {st.mean(s):6.0f} vs pub {st.mean(pub):.0f}")
