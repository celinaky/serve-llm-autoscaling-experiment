import json, bisect, math, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json'))
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICY, REBUILD, ENGINE = "#6250d6", "#c9480f", "#1baf7a"
ts=[t for t,_,_ in rows]; pub=[p for _,_,p in rows]
# engine 60s mean via the same pipeline (2s samples at phase 0.3, eval at tick)
import glob, os, datetime
eng=json.load(open('eng_deltas.json')) if os.path.exists('eng_deltas.json') else None
if eng is None:
    # rebuild engine-style deltas quickly from traces
    T='/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces'
    plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
    m=json.load(open(f'{R}/manifest.json')); rs=datetime.datetime.fromisoformat(m['started_at']).timestamp(); re_=datetime.datetime.fromisoformat(m['finished_at']).timestamp()
    eng=[]
    for path in sorted(glob.glob(f'{T}/*.jsonl')):
        mt=os.path.getmtime(path)
        if mt<rs or mt>re_+600: continue
        chunks={}; comp={}; queued={}; pre={}; off=None
        for line in open(path):
            try: e=json.loads(line)
            except ValueError: continue
            if off is None and 'wall_time_s' in e and 'monotonic_s' in e: off=e['wall_time_s']-e['monotonic_s']
            ev=e['event']
            if ev=='batch_scheduled':
                for r in e['requests']:
                    if r['phase']=='prefill': chunks.setdefault(r['request_id'],[]).append((e['batch_index'],r['scheduled_tokens']))
            elif ev=='batch_completed': comp[e['batch_index']]=e['monotonic_s']
            elif ev=='request_queued': queued[e['request_id']]=e
            elif ev=='request_preempted': pre.setdefault(e['request_id'],[]).append((e['batch_index'],e['monotonic_s']))
        rel=lambda mono: mono+off-t0
        for rid,q in queued.items():
            cs=chunks.get(rid)
            if not cs or any(b not in comp for b,_ in cs): continue
            starts=[(0,q['monotonic_s'])]+sorted(pre.get(rid,[]))
            for k,(batch,at_) in enumerate(starts):
                until=starts[k+1][0] if k+1<len(starts) else math.inf
                eng.append((rel(at_),sum(t for b,t in cs if batch<=b<until)))
            eng.extend((rel(comp[b]),-t) for b,t in cs)
    json.dump(eng,open('eng_deltas.json','w'))
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
PSI,PHI=9.6,0.3
def pipeline(T,L,w=60.0):
    out=[]
    for t in ts:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
Te,Le=step(eng); eng_s=pipeline(Te,Le)
fig,ax=plt.subplots(figsize=(14,5.8),dpi=150); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2,labelsize=8); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.axvspan(0,930,color="#e8f0fb",zorder=0,lw=0); ax.axvspan(930,1860,color="#f1f1ee",zorder=0,lw=0)
ycap=60000
ax.text(465,ycap*1.07,"ramp up: 1→32 sessions",ha='center',va='top',fontsize=8,color=INK2); ax.text(1395,ycap*1.07,"hold: 32→32 sessions",ha='center',va='top',fontsize=8,color=INK2)
ax.axvline(697.7,color=INK2,lw=0.8,ls='--',alpha=0.6); ax.axvline(934.7,color=INK2,lw=0.8,ls='--',alpha=0.6)
ax.text(700,ycap*1.0,"replica 2 running",fontsize=7.5,color=INK2); ax.text(937,ycap*1.0,"replica 3 running",fontsize=7.5,color=INK2)
ax.step(ts,[min(v,ycap) for v in pub],where='post',color=POLICY,lw=1.8)
ax.step(ts,[min(v,ycap) for v in sim],where='post',color=REBUILD,lw=1.8,ls=(0,(4,2)))
ax.step(ts,[min(v,ycap) for v in eng_s],where='post',color=ENGINE,lw=1.8)
ax.axhline(20000,color=INK2,lw=1,ls=':')
ax.set_ylim(0,ycap*1.1); ax.set_xlim(0,1860); ax.set_xlabel("seconds since profiling start",color=INK2); ax.set_ylabel("total outstanding uncached prefill tokens, all replicas",color=INK2)
early=[i for i,t in enumerate(ts) if t<=650]; late=[i for i,t in enumerate(ts) if t>650]
rm=lambda idx,a,b: math.sqrt(st.mean((a[i]-b[i])**2 for i in idx))
handles=[Line2D([],[],color=POLICY,lw=1.8,label=f"policy: 60s mean as published (load-phase average {st.mean(pub)/1000:.1f}K)"),
 Line2D([],[],color=REBUILD,lw=1.8,ls=(0,(4,2)),label=f"rebuild: engine's own uncached counts held queued→first token, pushed through the same telemetry pipeline: 2s router samples, controller 60s mean, 10s metrics export ({st.mean(sim)/1000:.1f}K)"),
 Line2D([],[],color=ENGINE,lw=1.8,label=f"engine: chunk burn-down, same pipeline ({st.mean(eng_s)/1000:.1f}K)"),
 Line2D([],[],color=INK2,lw=1,ls=':',label="target 20K per replica"),
 Line2D([],[],color=INK2,lw=0.8,ls='--',alpha=0.6,label=f"new replica running  ·  policy vs rebuild: RMSE {rm(early,pub,sim)/1000:.1f}K before 650s, {rm(late,pub,sim)/1000:.1f}K after")]
fig.legend(handles=handles,loc='lower center',fontsize=8,frameon=False,ncol=1,bbox_to_anchor=(0.5,-0.2))
fig.suptitle("b200-workday-curve-prefill-mean · policy vs rebuild with the telemetry pipeline modeled (same sampling, same export)",x=0.01,ha='left',fontsize=10,color=INK)
out=f"{R}/analysis/policy_vs_engine_rebuild60_pipeline_0_1860.png"
fig.savefig(out,facecolor=SURFACE,bbox_inches='tight'); print("wrote",out)
