import json, math, collections
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
JOINS=[698.0,935.0]
conv={}; R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['conv']=a['conversation_id'] if a else None; r['turn']=a.get('turn_index') if a else None; R.append(r)
    if a: conv.setdefault(a['conversation_id'],[]).append(r)
def context_before(c,t): return max((x['isl'] for x in conv.get(c,[]) if x['q']<t), default=0)
for r in R:
    jb=[j for j in JOINS if j<r['q']]
    r['pre']=min(r['cached'],context_before(r['conv'],max(jb))) if (jb and r['kind']=='warm') else 0
    r['pre']=(r['pre']//784)*784
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
# distinct exported states at export ticks
states={}
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); states.setdefault(e,v)
tally=collections.Counter(); per_window=collections.defaultdict(collections.Counter)
for e,v in sorted(states.items()):
    if e<0 or v<=0: continue
    s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in R if r['q']-0.5<=s<r['ft']+0.1]
    if not held or len(held)>16: tally['skipped (0 or >16 held)']+=1; continue
    w=int(e//100*100)
    if subset(int(v),[r['own'] for r in held]) is not None: k='exact with engine counts'
    elif subset(int(v),[r['own'] for r in held]+[r['pre'] for r in held if r['pre']>0]) is not None: k='exact only if pre-join blocks count as uncached'
    elif subset(int(v),[r['own'] for r in held]+[r['isl']-r['own'] for r in held if r['isl']>r['own']]) is not None: k='exact only if some warm turn counts full prompt'
    else: k='unexplained'
    tally[k]+=1; per_window[w][k]+=1
print("distinct nonzero exported gauge states (one per 10s export tick):")
for k,n in tally.most_common(): print(f"  {n:3d}  {k}")
print("\nby 100s window: exact / pre-join-blocks / full-prompt / unexplained")
for w in sorted(per_window):
    c=per_window[w]; print(f"  {w:4d}: {c['exact with engine counts']:2d} / {c['exact only if pre-join blocks count as uncached']:2d} / {c['exact only if some warm turn counts full prompt']:2d} / {c['unexplained']:2d}")
# ages of correctly counted warm turns before the first join (rules out gradual aging up to that age)
ages=[]
for e,v in sorted(states.items()):
    if not (0<e<698) or v<=0: continue
    s=PHI+2*math.floor((e-PHI)/2); held=[r for r in R if r['q']-0.5<=s<r['ft']+0.1]
    if not held or len(held)>16: continue
    sub=subset(int(v),[r['own'] for r in held])
    if sub:
        for i in sub:
            r=held[i]
            if r['kind']=='warm' and r['cached']>0: ages.append((round(r['q']-min(x['q'] for x in conv[r['conv']])), r['cached']//784))
print("\nbefore the first join, warm turns counted exactly at engine counts: (session age s, cached blocks) max age:", max(ages) if ages else None, "; n=",len(ages), "; ages>400s:", sorted(a for a,_ in ages if a>400))
