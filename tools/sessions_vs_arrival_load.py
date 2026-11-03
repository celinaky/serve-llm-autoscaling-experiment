#!/usr/bin/env python3
"""Does spreading sessions across replicas help each replica meet the target?

For every successful request, count the distinct root sessions (agent_depth 0
conversations) that sent a request to the same replica in the same window,
then bucket requests by that count and take the p95 of the uncached load they
found on arrival (own tokens excluded; analysis/load_requests.csv). If the p95
climbs with sessions per replica, fewer sessions per replica (spreading)
helps. Several runs can be pooled.

Windows within --exclude-downscale-s after a scale-down are dropped by
default, because the re-prefill wave that follows an eviction is a cost of
the move, not of the session count.

usage: python tools/sessions_vs_arrival_load.py runs/A [runs/B ...] [--window 60]
           [--exclude-downscale-s 180] [--min-requests 30] [--out PATH]
"""
import argparse, csv, glob, json, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, BAD, MEAN = "#1baf7a", "#e34948", "#9b9a95"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--window", type=float, default=60.0)
ap.add_argument("--exclude-downscale-s", type=float, default=180.0)
ap.add_argument("--min-requests", type=int, default=30, help="buckets with fewer requests are merged upward")
ap.add_argument("--out", default=None)
a = ap.parse_args()

per_bucket = defaultdict(list)   # sessions on replica -> loads (K)
target = None
dropped = kept = 0
for R in a.runs:
    R = R.rstrip("/")
    plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
    t0 = plot_data["time_origin"]["profiling_start_ns"]
    evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
    target = target or (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica")
    end = evaluation["periods"]["steady"]["end_s"]
    downs = [e["time_s"] for e in evaluation["scale_events"] if e["direction"] == "down"]
    meta = {}
    for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
        for line in open(path):
            m = json.loads(line)["metadata"]
            if m.get("benchmark_phase") == "profiling":
                meta[m["x_request_id"]] = ((m["request_start_ns"] - t0) / 1e9, m["conversation_id"], m.get("agent_depth", 0))
    rows = []
    sessions = defaultdict(set)  # (replica, window) -> root conversation ids
    for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
        m = meta.get(r["request_id"])
        if m is None or not 0 <= m[0] < end:
            continue
        pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
        k = int(m[0] // a.window)
        rows.append((m[0], pid, k, float(r["engine_load_tokens"]) / 1000))
        if m[2] == 0:
            sessions[(pid, k)].add(m[1])
    for t, pid, k, load in rows:
        if any(0 <= t - d <= a.exclude_downscale_s for d in downs):
            dropped += 1
            continue
        kept += 1
        per_bucket[len(sessions[(pid, k)])].append(load)

# Merge sparse buckets upward so every bar has at least --min-requests.
buckets = []
pending_label, pending = None, []
for n in sorted(per_bucket):
    pending.extend(per_bucket[n])
    pending_label = n if pending_label is None else pending_label
    if len(pending) >= a.min_requests:
        buckets.append((pending_label, n, pending))
        pending_label, pending = None, []
if pending:
    if buckets:
        lo, _, prev = buckets[-1]
        buckets[-1] = (lo, max(per_bucket), prev + pending)
    else:
        buckets.append((pending_label, max(per_bucket), pending))

print(f"requests kept {kept}, dropped within {a.exclude_downscale_s:.0f}s after a scale-down {dropped}")
print("sessions on replica | requests | p50 K | p95 K | share <= target | share arriving to empty engine")
stats = []
for lo, hi, loads in buckets:
    v = sorted(loads); n = len(v)
    p50, p95 = v[n // 2], v[min(n - 1, int(0.95 * n))]
    share = sum(1 for x in v if x * 1000 <= (target or 0)) / n
    empty = sum(1 for x in v if x == 0) / n
    label = f"{lo}" if lo == hi else f"{lo}–{hi}"
    stats.append((label, n, p50, p95, share, empty))
    print(f"{label:>19s} | {n:8d} | {p50:5.1f} | {p95:5.1f} | {share:6.1%} | {empty:6.1%}")

fig, ax = plt.subplots(figsize=(10, 5), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_axisbelow(True)
xs = range(len(stats))
ax.bar(xs, [p95 for _, _, _, p95, _, _ in stats], width=0.6,
       color=[GOOD if target and p95 * 1000 <= target else BAD for _, _, _, p95, _, _ in stats], lw=0)
ax.plot(xs, [p50 for _, _, p50, _, _, _ in stats], color=MEAN, marker="o", ms=4, lw=1.2)
if target:
    ax.axhline(target / 1000, color=INK2, lw=1, ls=":")
for x, (label, n, p50, p95, share, empty) in zip(xs, stats):
    ax.text(x, p95 + 1, f"n={n}\n{share:.0%} ≤ target", ha="center", va="bottom", fontsize=7.5, color=INK2)
ax.set_xticks(list(xs)); ax.set_xticklabels([s[0] for s in stats])
ax.set_xlabel("root sessions active on the request's replica in its window", color=INK2)
ax.set_ylabel("uncached prefill tokens at arrival (K)", color=INK2)
ax.set_ylim(0, max(p95 for _, _, _, p95, _, _ in stats) * 1.35)
ax.set_title(f"load at arrival vs sessions on the replica · p95 (bars, green = meets {target / 1000 if target else 0:.0f}K) and p50 (line) "
             f"· {len(a.runs)} run(s), {a.window:.0f}s windows, {dropped} requests after scale-downs excluded",
             loc="left", fontsize=9, color=INK)
out = a.out or f"{a.runs[0].rstrip('/')}/analysis/sessions_vs_arrival_load.png"
fig.tight_layout(); fig.savefig(out, facecolor=SURFACE); print("wrote", out)
