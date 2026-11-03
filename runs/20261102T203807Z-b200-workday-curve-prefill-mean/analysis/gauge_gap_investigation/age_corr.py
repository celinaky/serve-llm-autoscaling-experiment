import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json')); PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
t0=json.load(open('/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
conv_start={}
for rid,r in reqs.items():
    a=recs.get(aid(rid))
    if a: conv_start[a['conversation_id']]=min(conv_start.get(a['conversation_id'],1e9), a['request_start_ns']/1e9-t0)
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['age']=(r['q']-conv_start[a['conversation_id']]) if a else 0; r['turn']=a.get('turn_index') if a else None; R.append(r)
# per export tick: residual (pub - sim) vs. cached volume held in the 60s window, split by session age
ticks={}
for i,(t,v,p) in enumerate(rows):
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(p,sim[i]))
X_old=[];X_young=[];X_cold=[];Y=[];E=[]
for e,(p,s) in sorted(ticks.items()):
    if e<60: continue
    held=[r for r in R if r['ft']>e-60 and r['q']<e]
    w=lambda r: max(0,min(r['ft'],e)-max(r['q'],e-60))/60.0   # exposure fraction in window
    X_old.append(sum(r['cached']*w(r) for r in held if r['age']>600)); X_young.append(sum(r['cached']*w(r) for r in held if r['age']<=600)); X_cold.append(sum(r['own']*w(r) for r in held if r['kind']=='cold'))
    Y.append(p-s); E.append(e)
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y)); return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy) if sx and sy else 0
print("residual (published - simulated rebuild) vs exposure-weighted volume held in the window:")
print(f"  corr with cached tokens of sessions older than 600s: {corr(X_old,Y):+.3f}")
print(f"  corr with cached tokens of sessions younger than 600s: {corr(X_young,Y):+.3f}")
print(f"  corr with cold uncached tokens: {corr(X_cold,Y):+.3f}")
# regression Y ~ a*X_old + b*X_young (through origin)
s11=sum(a*a for a in X_old); s22=sum(b*b for b in X_young); s12=sum(a*b for a,b in zip(X_old,X_young)); s1y=sum(a*y for a,y in zip(X_old,Y)); s2y=sum(b*y for b,y in zip(X_young,Y))
det=s11*s22-s12*s12; a=(s1y*s22-s2y*s12)/det; b=(s11*s2y-s12*s1y)/det
print(f"  residual ~ {a:+.3f} x old-session cached + {b:+.3f} x young-session cached   (fraction of cached tokens the router appears to miss)")
# excursion at 1240-1300: held requests with ages
print("\nheld requests at export tick 1279.6 (pub peak) with session age and cached blocks:")
e=1279.6; held=sorted([r for r in R if r['ft']>e-60 and r['q']<e], key=lambda r:-r['cached'])
for r in held[:14]: print(f"  q {r['q']:.1f} ft {r['ft']:.1f} own {r['own']:6d} cached {r['cached']:6d} ({r['cached']//784:3d} blk) turn {r['turn']} session age {r['age']:5.0f}s eng {r['eng'][-6:]}")
print(f"  pub {ticks[e][0]:.0f} sim {ticks[e][1]:.0f}")
# time trend of residual vs. mean session age of held warm requests, per 300s
print("\nper 300s: mean residual, mean session age of held warm requests, old-session cached volume")
for lo in range(60,1860,300):
    idx=[i for i,e in enumerate(E) if lo<=e<lo+300]
    print(f"  {lo:4d}-{lo+300:4d}: residual {st.mean(Y[i] for i in idx):+6.0f}  old-session cached volume {st.mean(X_old[i] for i in idx):8.0f}  young {st.mean(X_young[i] for i in idx):8.0f}")
