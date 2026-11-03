import json, sys, bisect, math, statistics as st
R = sys.argv[1]; END = 1860.0
plot = json.load(open(f"{R}/analysis/plot_data.json")); t0 = plot["time_origin"]["profiling_start_ns"] / 1e9
S = json.load(open("gauge_decomp_state.json")); reqs = list(S["reqs"].values())
raw, pub, req = [], [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"): continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    d = {}
    for s in rec["samples"]:
        if "LLMServer" not in s["labels"].get("deployment", ""): continue
        if s["name"] == "ray_serve_autoscaling_policy_metric": d[s["labels"]["metric"]] = s["value"]
    if "outstanding_effective_prefill_tokens" in d: raw.append((t, d["outstanding_effective_prefill_tokens"]))
    if "prefill_required_replicas" in d: pub.append((t, d["prefill_required_replicas"] * 20000))
    if "required_replicas" in d: req.append((t, d["required_replicas"]))
def step(d):
    d = sorted(d); times = [t for t, _ in d]; lv = 0; levels = []
    for _, x in d: lv += x; levels.append(lv)
    return times, levels
def at(times, levels, t):
    i = bisect.bisect_right(times, t) - 1
    return levels[i] if i >= 0 else 0
def win_mean(times, levels, t, w=60.0):
    a = t - w; i0 = bisect.bisect_right(times, a)
    pts = [(a, at(times, levels, a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, t))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(t, None)]):
        if t2 > t1: tot += v1 * (t2 - t1)
    return tot / w
Rt, Rl = step([(r["q"], r["own"]) for r in reqs] + [(r["ft"], -r["own"]) for r in reqs])
pubL = [(t, v) for t, v in pub if 0 <= t <= END]
rawL = [(t, v) for t, v in raw if 0 <= t <= END]
print(f"published policy mean (what the policy divided by target): time-average over load {st.mean(v for _, v in pubL):.0f}")
print(f"raw exported gauge: sample average {st.mean(v for _, v in rawL):.0f}")
print(f"router-style rebuild from traces: 60s-mean averaged at the same times {st.mean(win_mean(Rt, Rl, t) for t, _ in pubL):.0f}")
# correlation/regression of published mean vs rebuild 60s-mean at shifts
def corr(x, y):
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy)
print("\nshift (s): corr(published mean, rebuild 60s-mean shifted), slope through origin")
for sh in [0, 2, 4, 6, 8, 10, 12, 16, 20]:
    x = [win_mean(Rt, Rl, t - sh) for t, _ in pubL]; y = [v for _, v in pubL]
    print(f"  {sh:3d}: corr {corr(x, y):.3f}  slope {sum(a*b for a, b in zip(x, y))/sum(a*a for a in x):.2f}")
# raw gauge vs rebuild instantaneous at shifts (wide)
print("\nraw exported gauge vs instantaneous rebuild: corr at shift")
for sh in [0, 2, 4, 6, 8, 10, 12, 14]:
    x = [at(Rt, Rl, t - sh) for t, _ in rawL]; y = [v for _, v in rawL]
    print(f"  {sh:3d}: corr {corr(x, y):.3f}")
# How does the published mean relate to the raw gauge samples? recompute 60s mean from raw samples (step) and compare pointwise
Gt, Gl = step([(raw[0][0], raw[0][1])] + [(t, v - pv) for (t, v), (_, pv) in zip(raw[1:], raw)])
diffs = [(t, v, win_mean(Gt, Gl, t)) for t, v in pubL]
print(f"\npublished mean vs 60s mean of raw scrapes: mean abs diff {st.mean(abs(v-w) for _, v, w in diffs):.0f}, mean signed (pub - raw) {st.mean(v-w for _, v, w in diffs):.0f}")
for sh in [0, 4, 8, 12]:
    x=[win_mean(Gt, Gl, t - sh) for t,_ in pubL]; y=[v for _,v in pubL]
    print(f"  shift {sh}: corr {corr(x,y):.3f} slope {sum(a*b for a,b in zip(x,y))/sum(a*a for a in x):.2f}")
