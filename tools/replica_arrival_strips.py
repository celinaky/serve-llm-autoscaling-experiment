#!/usr/bin/env python3
"""Per replica, per window: the distribution of load at arrival, one dot per request.

Rows are replicas (first-request order, usual colours). In each window the
requests that landed on that replica are jittered across the window's width,
y = uncached prefill tokens already on the engine when the request arrived
(own tokens excluded; from analysis/load_requests.csv). A short dark bar
marks the window's p95, a lighter one the p50. Under each window: requests,
and the share that found nothing ahead. Target dotted; scale decisions as in
the other charts.

usage: python tools/replica_arrival_strips.py runs/<run_dir> [--window 120] [--ymax K]
           [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json, random, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, FOLD = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=120.0)
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")
random.seed(0)

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = ((evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000) / 1000
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
points = defaultdict(list)  # pid -> [(t, load_k)]
first_seen = {}
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is None or not a.start <= t < end:
        continue
    pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
    first_seen[pid] = min(first_seen.get(pid, t), t)
    points[pid].append((t, float(r["engine_load_tokens"]) / 1000))
replicas = sorted(first_seen, key=first_seen.get)
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
peak = max(v for pid in points for _, v in points[pid])
ycap = a.ymax if a.ymax else min(max(1.5 * target, peak * 1.03), 8 * target)
clipped = peak > ycap

def pct(v, q):
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]

fig, axes = plt.subplots(len(replicas), 1, figsize=(15, 2.6 * len(replicas) + 0.8), dpi=150, sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.12})
fig.patch.set_facecolor(SURFACE)
axes = list(axes) if len(replicas) > 1 else [axes]
for i, (ax, pid) in enumerate(zip(axes, replicas)):
    col = PALETTE[i] if i < len(PALETTE) else FOLD
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=INK2, labelsize=8)
    ax.grid(axis="y", color=GRID, lw=0.5)
    ax.set_axisbelow(True)
    for p0, p1 in zip(curve, curve[1:]):
        phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
        lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
        if hi > lo:
            ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
            if i == 0:
                ax.text((lo + hi) / 2, ycap * 1.1, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
    for e in evaluation["scale_events"]:
        if a.start <= e["time_s"] <= end:
            ready = e.get("running_matched_at_s")
            ax.axvline(e["time_s"], color=INK, lw=0.8, alpha=0.6)
            if ready is not None and ready > e["time_s"]:
                ax.axvspan(e["time_s"], min(ready, end), color=INK2, alpha=0.08, lw=0, zorder=0.5)
                ax.axvline(min(ready, end), color=INK2, lw=0.8, ls="--", alpha=0.6)
            if i == 0:
                ax.text(e["time_s"] + 4, ycap * 0.99, f"{int(e['from_replicas'])}{ARROW}{int(e['to_replicas'])}", ha="left", va="top", fontsize=8, color=INK, fontweight="bold")
    # window boundaries
    for k in range(1, n_windows):
        ax.axvline(a.start + k * a.window, color=GRID, lw=0.5)
    by_win = defaultdict(list)
    for t, v in points[pid]:
        by_win[int((t - a.start) // a.window)].append(v)
    xs, ys = [], []
    for k, vals in by_win.items():
        w0 = a.start + k * a.window
        w1 = min(end, w0 + a.window)
        span = w1 - w0
        for v in vals:
            xs.append(w0 + span * (0.12 + 0.76 * random.random()))
            ys.append(min(v, ycap))
        p95, p50 = pct(vals, 0.95), pct(vals, 0.50)
        ax.hlines(min(p95, ycap), w0 + span * 0.08, w1 - span * 0.08, color=INK, lw=1.6, zorder=4)
        ax.hlines(min(p50, ycap), w0 + span * 0.08, w1 - span * 0.08, color=INK2, lw=1.0, zorder=4, alpha=0.8)
        zero = sum(1 for v in vals if v <= 0.0) / len(vals)
        ax.text((w0 + w1) / 2, -ycap * 0.04, f"{len(vals)}\n{zero:.0%} at 0", ha="center", va="top", fontsize=6.5, color=INK2, linespacing=1.1)
    ax.scatter(xs, ys, s=9, color=col, alpha=0.45, lw=0, zorder=3)
    ax.axhline(target, color=INK2, lw=1, ls=":")
    ax.set_ylim(-ycap * 0.2, ycap * 1.12)
    ax.set_xlim(a.start, end)
    ax.text(0.005, 0.97, f"replica {i + 1} (first request {first_seen[pid]:.0f}s) {DOT} {len(points[pid])} requests", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color=INK)
axes[-1].set_xlabel("seconds since profiling start", color=INK2)
fig.text(0.04, 0.5, "uncached prefill tokens on the engine at arrival (K)" + (f", clipped at {ycap:.0f}K" if clipped else ""),
         rotation=90, va="center", ha="center", color=INK2, fontsize=9)
handles = [Line2D([], [], color=INK2, marker="o", ms=4, lw=0, alpha=0.6, label="one request (colour: replica)"),
           Line2D([], [], color=INK, lw=1.6, label="window p95"),
           Line2D([], [], color=INK2, lw=1.0, alpha=0.8, label="window p50"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K"),
           Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"),
           Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=6, bbox_to_anchor=(0.5, -0.01))
fig.suptitle(f"{evaluation['experiment']} {DOT} per replica, per {a.window:.0f}s window: every request at the uncached prefill load it found ahead on its engine {DOT} under each window: requests, share that found 0",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.subplots_adjust(top=0.93, bottom=0.1, left=0.08, right=0.99)
out = a.out or f"{R}/analysis/replica_arrival_strips{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
