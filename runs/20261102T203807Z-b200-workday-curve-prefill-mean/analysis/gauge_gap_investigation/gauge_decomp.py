"""Decompose router gauge vs engine outstanding-prefill gap for one run."""
import json, sys, bisect, math, glob, os, statistics as st
R = sys.argv[1]
T = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
plot = json.load(open(f"{R}/analysis/plot_data.json")); t0 = plot["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); END = ev["periods"]["steady"]["end_s"]
manifest = json.load(open(f"{R}/manifest.json"))
import datetime
run_start = datetime.datetime.fromisoformat(manifest["started_at"]).timestamp()
run_end = datetime.datetime.fromisoformat(manifest["finished_at"]).timestamp()

# ---- gauge from scrapes
G = []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"): continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    for s in rec["samples"]:
        if s["name"] == "ray_serve_autoscaling_policy_metric" and s["labels"].get("metric") == "outstanding_effective_prefill_tokens" and "LLMServer" in s["labels"].get("deployment", ""):
            G.append((t, s["value"]))
G.sort()

# ---- traces of this run
reqs = {}   # rid -> dict
eng_deltas = []
files = [p for p in sorted(glob.glob(f"{T}/*.jsonl")) if run_start <= os.path.getmtime(p) <= run_end + 600]
for path in files:
    offset = None; chunks = {}; completed_at = {}; queued = {}; admitted = {}; first = {}; finished = {}; preempted = {}
    for line in open(path):
        try: e = json.loads(line)
        except ValueError: continue
        if offset is None and "wall_time_s" in e and "monotonic_s" in e: offset = e["wall_time_s"] - e["monotonic_s"]
        evn = e["event"]
        if evn == "batch_scheduled":
            for r in e["requests"]:
                if r["phase"] == "prefill": chunks.setdefault(r["request_id"], []).append((e["batch_index"], r["scheduled_tokens"]))
        elif evn == "batch_completed": completed_at[e["batch_index"]] = e["monotonic_s"]
        elif evn == "request_queued": queued[e["request_id"]] = e
        elif evn == "request_admitted": admitted[e["request_id"]] = e
        elif evn == "request_first_token": first[e["request_id"]] = e
        elif evn == "request_finished": finished[e["request_id"]] = e
        elif evn == "request_preempted": preempted.setdefault(e["request_id"], []).append((e["batch_index"], e["monotonic_s"]))
    if offset is None: continue
    rel = lambda mono: mono + offset - t0
    eng = os.path.basename(path).split(".")[0]
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs): continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            eng_deltas.append((rel(at), sum(t for b, t in cs if batch <= b < until)))
        eng_deltas.extend((rel(completed_at[b]), -t) for b, t in cs)
        adm = admitted.get(rid); f = first.get(rid)
        if adm is None or f is None: continue
        own = adm["prompt_tokens"] - adm["cached_prefix_tokens"]
        reqs[rid] = dict(eng=eng, q=rel(q["monotonic_s"]), adm=rel(adm["monotonic_s"]), ft=rel(f["monotonic_s"]),
                         fin=rel(finished[rid]["monotonic_s"]) if rid in finished else None,
                         own=own, isl=adm["prompt_tokens"], cached=adm["cached_prefix_tokens"],
                         pre_first_preempt=any(b <= f["batch_index"] for b, _ in preempted.get(rid, [])),
                         kind="cold" if adm["cached_prefix_tokens"] < 0.5 * adm["prompt_tokens"] else "warm")
print(f"trace files: {[os.path.basename(p) for p in files]}")
print(f"requests with queued+admitted+first_token: {len(reqs)}; in load phase: {sum(1 for r in reqs.values() if 0 <= r['q'] <= END)}")

def step(deltas):
    deltas = sorted(deltas); times = [t for t, _ in deltas]; levels = []; lv = 0
    for _, d in deltas: lv += d; levels.append(lv)
    return times, levels
def at(times, levels, t):
    i = bisect.bisect_right(times, t) - 1
    return levels[i] if i >= 0 else 0
def tmean(times, levels, a, b):
    """time-weighted mean of step fn over [a,b]"""
    i0 = bisect.bisect_right(times, a)
    pts = [(a, at(times, levels, a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, b))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(b, None)]):
        if t2 > t1: tot += v1 * (t2 - t1)
    return tot / (b - a)

def router_style(sel, a=0.0, b=0.0):
    d = []
    for r in sel:
        d.append((r["q"] - a, r["own"])); d.append((r["ft"] + b, -r["own"]))
    return step(d)

load = [r for r in reqs.values() if 0 <= r["q"] <= END]
Gt, Gv = step([(G[0][0], G[0][1])] + [(t, v - pv) for (t, v), (_, pv) in zip(G[1:], G)])
Et, El = step(eng_deltas)
Rt, Rl = router_style(reqs.values())
Rc = router_style([r for r in reqs.values() if r["kind"] == "cold"])
Rw = router_style([r for r in reqs.values() if r["kind"] == "warm"])
mG, mE, mR = tmean(Gt, Gv, 0, END), tmean(Et, El, 0, END), tmean(Rt, Rl, 0, END)
print(f"\nload-phase time means (0..{END:.0f}s): gauge {mG:.0f}  router-style rebuild {mR:.0f}  engine-style rebuild {mE:.0f}")
print(f"  gap gauge-engine {mG-mE:.0f}: whole-prompt-until-first-token (reason 1) {mR-mE:.0f} ({(mR-mE)/(mG-mE)*100:.0f}%), residual gauge-rebuild {mG-mR:.0f} ({(mG-mR)/(mG-mE)*100:.0f}%)")
print(f"  cold contribution to rebuild {tmean(*Rc, 0, END):.0f}, warm {tmean(*Rw, 0, END):.0f}")
tot_own = sum(r["own"] for r in load)
print(f"  sum own_uncached in load {tot_own:.0f}; each 100ms of extra hold adds {tot_own*0.1/END:.0f} to the mean; residual needs {(mG-mR)/tot_own*END*1000:.0f} ms of extra hold if it were lag alone")

# Does router-style match the gauge's shape? best shift
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else 0
Gs = [(t, v) for t, v in G if 0 <= t <= END]
best = None
for sh in [x/4 for x in range(-16, 17)]:
    rv = [at(Rt, Rl, t - sh) for t, _ in Gs]
    c = corr([v for _, v in Gs], rv)
    if best is None or c > best[1]: best = (sh, c)
print(f"\nbest time shift of gauge vs router-style rebuild: {best[0]:+.2f}s (corr {best[1]:.3f})")
sh = best[0]
# Regression G ~ a_c*Rc + a_w*Rw  (through origin), using scrapes
X = [(at(*Rc, t - sh), at(*Rw, t - sh)) for t, _ in Gs]; y = [v for _, v in Gs]
# normal equations
s11 = sum(a*a for a, _ in X); s22 = sum(b*b for _, b in X); s12 = sum(a*b for a, b in X); s1y = sum(a*v for (a, _), v in zip(X, y)); s2y = sum(b*v for (_, b), v in zip(X, y))
det = s11*s22 - s12*s12
ac = (s1y*s22 - s2y*s12)/det; aw = (s11*s2y - s12*s1y)/det
a1 = sum((a+b)*v for (a, b), v in zip(X, y)) / sum((a+b)**2 for a, b in X)
print(f"regression at that shift: gauge ~ {ac:.2f} x cold_rebuild + {aw:.2f} x warm_rebuild   (single slope: {a1:.2f})")
# time-weighted sanity: time-mean of gauge vs. a_c*cold + a_w*warm
print(f"  check: {ac:.2f}*{tmean(*Rc,0,END):.0f} + {aw:.2f}*{tmean(*Rw,0,END):.0f} = {ac*tmean(*Rc,0,END)+aw*tmean(*Rw,0,END):.0f} vs gauge mean {mG:.0f}")

# ---- single-request windows: scrapes where exactly one request is in router window for the preceding 3s
print("\nscrapes with exactly one request held (router-style) throughout [t-3, t]:")
byq = sorted(load, key=lambda r: r["q"])
rows = []
for t, v in Gs:
    inside = [r for r in load if r["q"] <= t - 3.0 and r["ft"] >= t]
    touching = [r for r in load if r["q"] <= t + 0.5 and r["ft"] >= t - 3.5]
    if len(inside) == 1 and len(touching) == 1:
        r = inside[0]; rows.append((t, v, r))
print(f"  n={len(rows)}")
for t, v, r in rows[:60]:
    print(f"  t={t:7.1f} gauge={v:8.0f} own_uncached={r['own']:7d} isl={r['isl']:7d} cached={r['cached']:7d} {r['kind']} ratio={v/r['own'] if r['own'] else float('nan'):.2f} engine={r['eng'][-6:]}")
exact = sum(1 for t, v, r in rows if abs(v - r["own"]) <= 64)
print(f"  gauge within one block (64) of own_uncached: {exact}/{len(rows)}; gauge within 64 of isl: {sum(1 for t,v,r in rows if abs(v-r['isl'])<=64)}/{len(rows)}")
json.dump({"reqs": {k: v for k, v in reqs.items()}, "G": G}, open(os.path.join(os.path.dirname(sys.argv[0]), "gauge_decomp_state.json"), "w"))
