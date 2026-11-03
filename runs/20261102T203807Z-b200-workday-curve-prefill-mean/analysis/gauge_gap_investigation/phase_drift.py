import json, bisect, math
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI=9.6
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
print("record-tick phase fitted per 300s window (exact matches of exported raw gauge, nonzero exported values only):")
for lo in range(0,1860,300):
    sub=[(t,v) for t,v,_ in rows if lo<=t<lo+300]
    best=[]
    for phi in [0.05*i for i in range(40)]:
        m=sum(1 for t,v in sub if v>0 and abs(at(T,L,phi+2*math.floor((PSI+10*math.floor((t-PSI)/10)-phi)/2))-v)<=1)
        best.append((m,phi))
    best.sort(reverse=True); nz=sum(1 for t,v in sub if v>0)
    print(f"  {lo:4d}-{lo+300:4d}: nonzero exported {nz:3d}; best phases {[(m,round(p,2)) for m,p in best[:4]]}")
