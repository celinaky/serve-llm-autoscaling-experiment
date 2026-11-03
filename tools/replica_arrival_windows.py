#!/usr/bin/env python3
"""Per-replica p95 of load at arrival per time window, with request counts.

Top: for each window, one bar per replica, the p95 of the uncached prefill
load that replica's requests found on arrival (own tokens excluded;
analysis/load_requests.csv from tools/load_breakdown.py). Replicas keep one
colour each in order of first request. A replica with fewer than
--min-samples requests in a window gets a small hollow marker instead of a
bar. Bottom: a grid of how many requests each replica served in each window.
Scale decisions, the target and the load phases are marked as in the other
window charts.

usage: python tools/replica_arrival_windows.py runs/<run_dir> [--window 60] [--min-samples 5]
           [--start 0] [--end <load end>] [--out PATH]
"""
import argparse, csv, glob, json, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, FOLD = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=60.0)
ap.add_argument("--min-samples", type=int, default=5)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--ymax", type=float, default=None, help="clip the p95 axis at this many K tokens (default 5x target); clipped bars are labelled")
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 0
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9

samples = defaultdict(lambda: defaultdict(list))  # window -> replica -> loads (K)
first_seen = {}
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is None or not a.start <= t < end:
        continue
    m = re.search(r"pid-(\d+)", r["engine"])
    pid = m.group(1) if m else r["engine"]
    first_seen[pid] = min(first_seen.get(pid, t), t)
    samples[int((t - a.start) // a.window)][pid].append(float(r["engine_load_tokens"]) / 1000)

replicas = sorted(first_seen, key=first_seen.get)
named = replicas[:len(PALETTE)]
folded = replicas[len(PALETTE):]
color = {pid: PALETTE[i] for i, pid in enumerate(named)}
slots = named + (["other"] if folded else [])
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)

fig, (ax, axc) = plt.subplots(2, 1, figsize=(max(14, n_windows * 0.55), 8.5), dpi=150, sharex=True,
                              gridspec_kw={"height_ratios": [3, 1.8], "hspace": 0.06})
fig.patch.set_facecolor(SURFACE)
for axis in (ax, axc):
    axis.set_facecolor(SURFACE)
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axis.spines[s].set_color(GRID)
    axis.tick_params(colors=INK2)
    axis.set_axisbelow(True)
ax.grid(axis="y", color=GRID, lw=0.6)

slot_w = a.window * 0.9 / len(slots)
ycap = a.ymax if a.ymax else (5 * target / 1000 if target else 100.0)
counts = defaultdict(dict)
for k in range(n_windows):
    ws = a.start + k * a.window
    per = samples.get(k, {})
    for j, slot in enumerate(slots):
        if slot == "other":
            loads = [y for pid in folded for y in per.get(pid, [])]
            col = FOLD
        else:
            loads = per.get(slot, [])
            col = color[slot]
        counts[slot][k] = len(loads)
        x = ws + a.window * 0.05 + (j + 0.5) * slot_w
        if len(loads) >= a.min_samples:
            v = sorted(loads)
            p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
            if p95 > ycap:
                ax.bar(x, ycap, width=slot_w * 0.9, color=col, lw=0)
                ax.text(x, ycap * 1.01, f"{p95:.0f}K", ha="center", va="bottom", fontsize=6.5, color=col, rotation=90)
            else:
                ax.bar(x, p95, width=slot_w * 0.9, color=col, lw=0)
        elif loads:
            ax.plot([x], [0], marker="o", ms=3, mfc="none", mec=col, mew=0.8, lw=0)

if target:
    ax.axhline(target / 1000, color=INK2, lw=1, ls=":")
ax.set_ylim(0, ycap * 1.3)
ymax = ax.get_ylim()[1]
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi <= lo:
        continue
    for axis in (ax, axc):
        axis.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
    ax.text((lo + hi) / 2, ymax * 0.995, f"{phase}: {p0['sessions']}→{p1['sessions']} sessions",
            ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    t = e["time_s"]
    if not a.start <= t <= end:
        continue
    up = e["direction"] == "up"
    for axis in (ax, axc):
        axis.axvline(t, color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
    ax.text(t + 3, ymax * 0.93, f"{e['from_replicas']:.0f}→{e['to_replicas']:.0f}", rotation=90,
            va="top", ha="left", fontsize=8, color=INK if up else INK2)

ax.set_ylabel("p95 of uncached prefill tokens at arrival (K)", color=INK2)
ax.set_title(f"{evaluation['experiment']} \u00b7 per replica, per {a.window:.0f}s window: p95 of load at arrival "
             f"(top; axis clipped at {ycap:.0f}K, clipped bars labelled) and requests served, stacked by replica (bottom) "
             f"\u00b7 hollow marker: fewer than {a.min_samples} requests", loc="left", fontsize=10, color=INK)

# Requests served per window, stacked by replica.
for k in range(n_windows):
    ws = a.start + k * a.window
    base = 0
    for slot in slots:
        n = counts[slot].get(k, 0)
        if not n:
            continue
        axc.bar(ws + a.window / 2, n, bottom=base, width=a.window * 0.9, color=FOLD if slot == "other" else color[slot], lw=0)
        base += n
    if base:
        axc.text(ws + a.window / 2, base + 1, str(base), ha="center", va="bottom", fontsize=6.5, color=INK2)
axc.set_ylim(bottom=0)
axc.set_ylabel("requests served, by replica", color=INK2)
axc.grid(axis="y", color=GRID, lw=0.6)
axc.set_xlabel("seconds since profiling start", color=INK2)
axc.grid(False)
ax.set_xlim(a.start, end)

handles = [Patch(color=color[p], label=f"replica {i + 1} (pid {p}, first request {first_seen[p]:.0f}s)") for i, p in enumerate(named)]
if folded:
    handles.append(Patch(color=FOLD, label=f"other: {len(folded)} replicas first seen after {min(first_seen[p] for p in folded):.0f}s"))
if target:
    handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K"))
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.06))
out = a.out or f"{R}/analysis/replica_arrival_windows{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
