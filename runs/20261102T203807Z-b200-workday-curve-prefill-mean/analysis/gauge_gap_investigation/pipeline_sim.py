import json, bisect, math, statistics as st, collections
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
PSI,PHI,LEAD,HOLD=9.6,0.3,0.0,0.1
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
T,L=step([(r['q']-LEAD,r['own']) for r in reqs]+[(r['ft']+HOLD,-r['own']) for r in reqs])
def policy_mean(e,w=60.0):
    # samples at record ticks <= e; time-weighted over [e-w, e] with carry-in and carry-forward (as _time_weighted_mean)
    k_last=math.floor((e-PHI)/2); ticks=[PHI+2*k for k in range(k_last-31,k_last+1)]
    samples=[(s,at(T,L,s)) for s in ticks if s<=e]
    a=e-w; tot=0.0
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
        lo,hi=max(t1,a),min(t2,e)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/w
sim=[]; raw_pred=[]
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    sim.append(policy_mean(e)); raw_pred.append(at(T,L,s))
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
pub=[p for _,_,p in rows]
print("published mean vs pipeline-simulated rebuild mean:")
for lo,hi in [(0,650),(650,1860),(0,1860)]:
    idx=[i for i,(t,_,_) in enumerate(rows) if lo<=t<hi]
    x=[sim[i] for i in idx]; y=[pub[i] for i in idx]
    print(f"  {lo:4d}-{hi:4d}: corr {corr(x,y):.3f} rmse {rmse(x,y):.0f} mean pub {st.mean(y):.0f} sim {st.mean(x):.0f} (pub-sim {st.mean(y)-st.mean(x):+.0f})")
for lo in range(0,1860,150):
    idx=[i for i,(t,_,_) in enumerate(rows) if lo<=t<lo+150]
    x=[sim[i] for i in idx]; y=[pub[i] for i in idx]
    print(f"    {lo:4d}-{lo+150:4d}: rmse {rmse(x,y):5.0f}  pub {st.mean(y):6.0f} sim {st.mean(x):6.0f} diff {st.mean(y)-st.mean(x):+6.0f}   raw mismatches {sum(1 for i in idx if abs(raw_pred[i]-rows[i][1])>1)}/{len(idx)}")
json.dump({'rows':rows,'sim':sim,'raw_pred':raw_pred,'params':[PSI,PHI,LEAD,HOLD]},open('pipeline_sim.json','w'))

# classify raw mismatches
print("\nraw gauge mismatches (export != rebuild at implied record tick), classified:")
def subset(target, vals, tol=1):
    sums={0:()}
    for i,v in enumerate(vals):
        new={}
        for s_,sub in sums.items():
            ns=s_+v
            if ns<=target+tol and ns not in sums and ns not in new: new[ns]=sub+(i,)
        sums.update(new)
    for s_ in range(target-tol,target+tol+1):
        if s_ in sums: return sums[s_]
    return None
def aid(r): return None
cls=collections.Counter(); details=[]
for i,(t,v,p) in enumerate(rows):
    pred=raw_pred[i]
    if abs(pred-v)<=1: continue
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in reqs if r['q']-LEAD<=s<r['ft']+HOLD]           # what rebuild counts at s
    diff=int(round(v-pred))
    near=[r for r in reqs if r['q']-3<=s<=r['ft']+3]                  # requests near the sample
    tag=None; info=''
    if diff>0:
        # extra: a held warm request counted at full prompt?
        ex=[r for r in held if abs((r['isl']-r['own'])-diff)<=1]
        if ex: tag='held request counted at FULL prompt'; info=f"isl {ex[0]['isl']} own {ex[0]['own']} cached {ex[0]['cached']} eng {ex[0]['eng'][-6:]}"
        if tag is None:
            sub=subset(diff,[r['isl']-r['own'] for r in held if r['isl']>r['own']])
            if sub: tag='several held requests counted at FULL prompt'
        if tag is None:
            late=[r for r in near if r['ft']+HOLD<=s and abs(r['own']-diff)<=1]
            if late: tag='decrement later than first token'; info=f"own {late[0]['own']} first token {s-late[0]['ft']:.2f}s before sample"
        if tag is None:
            early=[r for r in near if s<r['q']-LEAD and abs(r['own']-diff)<=1]
            if early: tag='admitted before engine queued'; info=f"own {early[0]['own']} queued {early[0]['q']-s:.2f}s after sample"
        if tag is None:
            sub=subset(diff,[r['own'] for r in near if not (r['q']-LEAD<=s<r['ft']+HOLD)])
            if sub: tag='timing: nearby requests held at sample (late decrement / early admit)'
    else:
        miss=[r for r in held if abs(r['own']+diff)<=1]
        if miss: tag='held request NOT counted (or decremented early)'; info=f"own {miss[0]['own']} eng {miss[0]['eng'][-6:]} q {miss[0]['q']:.1f} ft {miss[0]['ft']:.1f}"
        if tag is None:
            sub=subset(-diff,[r['own'] for r in held])
            if sub: tag='several held requests not counted'
    if tag is None:
        # maybe record tick jitter: does any s' within +-2s match exactly?
        for ds in [x/10 for x in range(-20,21)]:
            if abs(at(T,L,s+ds)-v)<=1: tag=f'record-tick jitter (matches at {ds:+.1f}s)'; break
    if tag is None: tag='unexplained'
    cls[tag]+=1; details.append((round(t,1),int(v),int(pred),diff,tag,info))
for k,n in cls.most_common(): print(f"  {n:3d}  {k}")
print("\nmismatch details (t, exported, predicted, diff, class, info):")
for d in details: print("  ",d)
