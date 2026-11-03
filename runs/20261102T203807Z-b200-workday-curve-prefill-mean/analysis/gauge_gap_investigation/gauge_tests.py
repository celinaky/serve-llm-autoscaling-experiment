import json, sys, bisect, math, random, statistics as st, itertools
S = json.load(open(sys.argv[1])); reqs = list(S["reqs"].values()); G = [tuple(x) for x in S["G"]]
END = 1860.0
Gs = [(t, v) for t, v in G if 0 <= t <= END]
load = [r for r in reqs if 0 <= r["q"] <= END]
print("requests", len(load), "own_uncached==0:", sum(r["own"] == 0 for r in load), "preempted before first token:", sum(r["pre_first_preempt"] for r in load))

def step(d):
    d = sorted(d); times = [t for t, _ in d]; lv = 0; levels = []
    for _, x in d: lv += x; levels.append(lv)
    return times, levels
def tmean(times, levels, a, b, f=lambda v: v):
    i0 = bisect.bisect_right(times, a)
    at = lambda t: (levels[bisect.bisect_right(times, t) - 1] if bisect.bisect_right(times, t) else 0)
    pts = [(a, at(a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, b))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(b, None)]):
        if t2 > t1: tot += f(v1) * (t2 - t1)
    return tot / (b - a)
def rebuild(sel, lag):
    return step([(r["q"], r["own"]) for r in sel] + [(r["ft"] + lag, -r["own"]) for r in sel])

gauge_busy = sum(v > 0 for _, v in Gs) / len(Gs)
gauge_mean = st.mean(v for _, v in Gs)
print(f"\ngauge: busy fraction (sample>0) {gauge_busy:.3f}, sample mean {gauge_mean:.0f}")
print("router-style rebuild from traces, extra hold per request (ms): busy fraction, time mean, cold/warm split")
for lag in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
    T, L = rebuild(load, lag)
    busy = tmean(T, L, 0, END, lambda v: 1.0 if v > 0 else 0.0)
    m = tmean(T, L, 0, END)
    Tc, Lc = rebuild([r for r in load if r["kind"] == "cold"], lag); Tw, Lw = rebuild([r for r in load if r["kind"] == "warm"], lag)
    print(f"  +{lag*1000:5.0f} ms: busy {busy:.3f}  mean {m:7.0f}  (cold {tmean(Tc,Lc,0,END):6.0f} warm {tmean(Tw,Lw,0,END):6.0f})")

# ---- isolated requests: no other request within 6s on either side
qs = sorted(load, key=lambda r: r["q"])
iso = []
for i, r in enumerate(qs):
    prev_ft = max((x["ft"] for x in qs[:i]), default=-1e9)
    nxt_q = qs[i + 1]["q"] if i + 1 < len(qs) else 1e9
    if r["q"] - prev_ft > 2 and nxt_q - r["ft"] > 2: iso.append(r)
print(f"\nisolated requests (gap > 6s both sides): {len(iso)}; cold {sum(r['kind']=='cold' for r in iso)} warm {sum(r['kind']=='warm' for r in iso)}")
# for each isolated request, scrapes in [q-1, ft+6] that are nonzero; value equality
nz_total = 0; hold_total = 0.0; eq = 0; neq = []; nz_by_kind = {"cold": [0, 0.0, 0], "warm": [0, 0.0, 0]}
for r in iso:
    samples = [(t, v) for t, v in Gs if r["q"] - 1 <= t <= r["ft"] + 6]
    nz = [(t, v) for t, v in samples if v > 0]
    nz_total += len(nz); hold_total += r["ft"] - r["q"]
    k = nz_by_kind[r["kind"]]; k[0] += len(nz); k[1] += r["ft"] - r["q"]; k[2] += 1
    for t, v in nz:
        if abs(v - r["own"]) <= 1: eq += 1
        else: neq.append((round(t, 1), v, r["own"], r["isl"], r["cached"], r["kind"], round(r["ft"] - r["q"], 2)))
print(f"  nonzero scrapes near them: {nz_total}; of which equal to own_uncached: {eq}; not equal: {len(neq)}")
if iso: print(f"  implied mean hold from scrapes: {nz_total*2/len(iso)*1000:.0f} ms vs engine queued->first token mean {hold_total/len(iso)*1000:.0f} ms  => extra hold ~{(nz_total*2-hold_total)/len(iso)*1000:.0f} ms")
for k, (n, h, c) in (nz_by_kind.items() if iso else []):
    if c: print(f"    {k}: n={c} scrapes-implied hold {n*2/c*1000:.0f} ms vs engine {h/c*1000:.0f} ms => extra ~{(n*2-h)/c*1000:.0f} ms")
print("  mismatches (t, gauge, own_uncached, isl, cached, kind, engine hold s):")
for x in neq[:40]: print("   ", x)

# ---- subset-sum test on all nonzero scrapes with few candidates
def subset_match(target, vals):
    # DP over sums (values are ints up to ~200K; candidates <= 14)
    sums = {0: ()}
    for i, v in enumerate(vals):
        new = {}
        for s, sub in sums.items():
            ns = s + v
            if ns <= target + 2 and ns not in sums and ns not in new: new[ns] = sub + (i,)
        sums.update(new)
    for s in (target, target - 1, target + 1):
        if s in sums: return sums[s]
    return None
def run_subset(reqset, label):
    tested = matched = 0; mism = []
    for t, v in Gs:
        if v <= 0: continue
        cands = [r for r in reqset if r["q"] - 1.0 <= t + 0.5 and r["ft"] + 2.5 >= t - 5.0]
        if not cands or len(cands) > 14: continue
        tested += 1
        sub = subset_match(int(round(v)), [r["own"] for r in cands])
        if sub is not None: matched += 1
        else: mism.append((round(t, 1), int(v), sorted((r["own"], r["kind"]) for r in cands)))
    print(f"\n{label}: nonzero scrapes with <=14 candidate requests: {tested}, gauge value == exact subset sum of own_uncached: {matched} ({matched/max(1,tested)*100:.0f}%)")
    return mism
mism = run_subset(load, "real own_uncached")
# control: shuffle own_uncached across requests
random.seed(1); owns = [r["own"] for r in load]; random.shuffle(owns)
shuf = [dict(r, own=o) for r, o in zip(load, owns)]
run_subset(shuf, "control (own_uncached shuffled across requests)")
print("sample mismatches (t, gauge, candidate (own,kind)):")
for x in mism[:25]: print("  ", x)
