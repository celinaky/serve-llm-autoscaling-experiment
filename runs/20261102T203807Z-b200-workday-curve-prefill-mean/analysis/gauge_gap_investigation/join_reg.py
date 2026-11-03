import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json')); PSI=9.6
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
JOINS=[698.0, 935.0]   # replica 2 and 3 RUNNING
conv={}
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['conv']=a['conversation_id'] if a else None; r['turn']=a.get('turn_index') if a else None; R.append(r)
    if a: conv.setdefault(a['conversation_id'],[]).append(r)
def context_before(c, t):
    """prompt length of the conversation's last turn queued before t (0 if none)"""
    return max((x['isl'] for x in conv.get(c,[]) if x['q']<t), default=0)
# per warm request: cached blocks that predate the most recent join before it
for r in R:
    joins_before=[j for j in JOINS if j<r['q']]
    if r['kind']=='warm' and joins_before:
        j=max(joins_before); r['pre']=min(r['cached'], context_before(r['conv'], j)); r['post']=r['cached']-r['pre']
    else: r['pre']=0; r['post']=r['cached'] if r['kind']=='warm' else 0
ticks={}
for i,(t,v,p) in enumerate(rows):
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(p,sim[i]))
Xpre=[];Xpost=[];Xcold=[];Y=[];E=[]
for e,(p,s) in sorted(ticks.items()):
    if e<60: continue
    held=[r for r in R if r['ft']>e-60 and r['q']<e]
    w=lambda r: max(0,min(r['ft']+0.1,e)-max(r['q'],e-60))/60.0
    Xpre.append(sum(r['pre']*w(r) for r in held)); Xpost.append(sum(r['post']*w(r) for r in held)); Xcold.append(sum(r['own']*w(r) for r in held if r['kind']=='cold'))
    Y.append(p-s); E.append(e)
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y)); return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy) if sx and sy else 0
print(f"residual (published - simulated) vs exposure-weighted cached tokens of held warm turns:")
print(f"  corr with cached blocks created BEFORE the latest replica join: {corr(Xpre,Y):+.3f}")
print(f"  corr with cached blocks created AFTER the latest replica join:  {corr(Xpost,Y):+.3f}")
# 2-var regression through origin
s11=sum(a*a for a in Xpre); s22=sum(b*b for b in Xpost); s12=sum(a*b for a,b in zip(Xpre,Xpost)); s1y=sum(a*y for a,y in zip(Xpre,Y)); s2y=sum(b*y for b,y in zip(Xpost,Y))
det=s11*s22-s12*s12; a=(s1y*s22-s2y*s12)/det; b=(s11*s2y-s12*s1y)/det
res=[y-(a*x1+b*x2) for x1,x2,y in zip(Xpre,Xpost,Y)]
print(f"  residual ~ {a:+.2f} x pre-join cached + {b:+.2f} x post-join cached ; rmse of fit {math.sqrt(st.mean(r*r for r in res)):.0f} vs raw residual rmse {math.sqrt(st.mean(y*y for y in Y)):.0f}")
late=[i for i,e in enumerate(E) if e>700]
print(f"  late region only: mean residual {st.mean(Y[i] for i in late):+.0f}; mean pre-join cached volume {st.mean(Xpre[i] for i in late):.0f} -> {a:.2f} x that = {a*st.mean(Xpre[i] for i in late):+.0f}")
# how much of the gauge mean would be explained: a * pre-join volume over the run
print(f"  whole-run: mean pre-join cached volume {st.mean(Xpre):.0f}, implied over-count {a*st.mean(Xpre):+.0f} tokens on the mean")
# sanity: residual by whether any pre-join sessions are held
grp=[(y,x>0) for y,x in zip(Y,Xpre)]
print(f"  mean residual when no pre-join cached volume held: {st.mean(y for y,f in grp if not f):+.0f} (n={sum(1 for y,f in grp if not f)}); when some: {st.mean(y for y,f in grp if f):+.0f} (n={sum(1 for y,f in grp if f)})")
