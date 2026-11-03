import json, random, collections
S = json.load(open("gauge_decomp_state.json")); reqs = list(S["reqs"].values()); G = [tuple(x) for x in S["G"]]
Gs = [(t, v) for t, v in G if 0 <= t <= 1860]
# distinct exported states (collapse runs)
states = []
for t, v in Gs:
    if v > 0 and (not states or states[-1][1] != v or t - states[-1][2] > 4.5): states.append([t, v, t])
    elif v > 0: states[-1][2] = t
print("distinct nonzero exported states:", len(states))
def subset(target, vals, tol=1):
    sums = {0: ()}
    for i, v in enumerate(vals):
        new = {}
        for s, sub in sums.items():
            ns = s + v
            if ns <= target + tol and ns not in sums and ns not in new: new[ns] = sub + (i,)
        sums.update(new)
    for s in range(target - tol, target + tol + 1):
        if s in sums: return sums[s]
    return None
def run(rs, label, back=14.0, fwd=1.0, maxc=18):
    tested = matched = 0; mism = []
    for t, v, t2 in states:
        c = [r for r in rs if r["q"] <= t + fwd and r["ft"] >= t - back]
        if not c or len(c) > maxc: continue
        tested += 1
        sub = subset(int(round(v)), [r["own"] for r in c])
        if sub is not None: matched += 1
        else: mism.append((t, int(v), c))
    print(f"{label}: states with <= {maxc} candidates in [t-{back:.0f}, t+{fwd:.0f}]: {tested}; exact subset sum of engine own_uncached: {matched} ({matched/max(1,tested)*100:.0f}%)")
    return mism
mism = run(reqs, "real")
random.seed(3); owns = [r["own"] for r in reqs]; random.shuffle(owns); run([dict(r, own=o) for r, o in zip(reqs, owns)], "control shuffled")
# also try explaining mismatches with isl (router saw no overlap) for any candidate
print("\nunexplained states: try subset sums allowing each candidate to count as own OR isl")
expl2 = 0
for t, v, c in mism:
    vals = []
    for r in c: vals.append(r["own"])
    # alternative: each request contributes own or isl: encode as own plus optional extra (isl-own)
    extras = [r["isl"] - r["own"] for r in c if r["isl"] > r["own"]]
    sub = subset(int(round(v)), [r["own"] for r in c] + extras)
    if sub is not None: expl2 += 1
print(f"  {expl2}/{len(mism)} explained when a warm request may count as its full prompt (isl)")
print("\nfirst 30 unexplained states (t, value, candidates as (own, isl, kind, q, ft)):")
for t, v, c in mism[:30]:
    print(f"  t={t:7.1f} value={v:7d}  " + "; ".join(f"({r['own']},{r['isl']},{r['kind'][0]},q{r['q']:.1f},ft{r['ft']:.1f})" for r in sorted(c, key=lambda r: r['q'])))
