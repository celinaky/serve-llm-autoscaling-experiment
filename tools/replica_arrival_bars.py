#!/usr/bin/env python3
"""Per replica, per window: stacked bars of requests by the load they found ahead.

Rows are replicas (first-request order). One bar per window, height = requests
that landed on that replica in the window, stacked by the uncached prefill
tokens already on the engine at arrival (own tokens excluded; from
analysis/load_requests.csv). Buckets go from pale (found nothing ahead) to
dark (found more than 4x the target). The bucket edge at the target is the
first red shade. Above each bar: the share of its requests over the target.

usage: python tools/replica_arrival_bars.py runs/<run_dir> [--window 120] [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=120.0)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = ((evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000) / 1000
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

# Buckets in units of the target: edges as fractions, colours light -> dark.
EDGES = [0.0, 0.2, 0.5, 1.0, 2.0, 4.0]
LABELS = ["0 (nothing ahead)", f"0{DOT}{0.2 * target:.0f}K", f"{0.2 * target:.0f}{DOT}{0.5 * target:.0f}K",
          f"{0.5 * target:.0f}{DOT}{target:.0f}K (under target)", f"{target:.0f}{DOT}{2 * target:.0f}K (over target)",
          f"{2 * target:.0f}{DOT}{4 * target:.0f}K", f"over {4 * target:.0f}K"]
COLORS = ["#dcdcd8", "#b8cfe8", "#8fb3dc", "#5e92cc", "#f2a37a", "#dc5d3e", "#8c1a0f"]

def bucket(v):
    if v <= 0:
        return 0
    for i, e in enumerate(EDGES[1:], start=1):
        if v <= e * target:
            return i
    return len(EDGES)

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
counts = defaultdict(lambda: defaultdict(lambda: [0] * len(COLORS)))  # pid -> window -> bucket counts
first_seen, total = {}, defaultdict(int)
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is None or not a.start <= t < end:
        continue
    pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
    first_seen[pid] = min(first_seen.get(pid, t), t)
    counts[pid][int((t - a.start) // a.window)][bucket(float(r["engine_load_tokens"]) / 1000)] += 1
    total[pid] += 1
replicas = sorted(first_seen, key=first_seen.get)
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
ymax = max(sum(c) for pid in counts for c in counts[pid].values()) * 1.25

fig, axes = plt.subplots(len(replicas), 1, figsize=(15, 2.5 * len(replicas) + 0.8), dpi=150, sharex=True, sharey=True,
                         gridspec_kw={"hspace": 0.12})
fig.patch.set_facecolor(SURFACE)
axes = list(axes) if len(replicas) > 1 else [axes]
for i, (ax, pid) in enumerate(zip(axes, replicas)):
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
                ax.text((lo + hi) / 2, ymax * 0.99, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
    for e in evaluation["scale_events"]:
        if a.start <= e["time_s"] <= end:
            ready = e.get("running_matched_at_s")
            ax.axvline(e["time_s"], color=INK, lw=0.8, alpha=0.6, zorder=5)
            if ready is not None and ready > e["time_s"]:
                ax.axvspan(e["time_s"], min(ready, end), color=INK2, alpha=0.08, lw=0, zorder=0.5)
                ax.axvline(min(ready, end), color=INK2, lw=0.8, ls="--", alpha=0.6, zorder=5)
            if i == 0:
                ax.text(e["time_s"] + 4, ymax * 0.88, f"{int(e['from_replicas'])}{ARROW}{int(e['to_replicas'])}", ha="left", va="top", fontsize=8, color=INK, fontweight="bold")
    for k in range(n_windows):
        c = counts[pid].get(k)
        if not c:
            continue
        w0 = a.start + k * a.window
        w1 = min(end, w0 + a.window)
        x, width = (w0 + w1) / 2, (w1 - w0) * 0.8
        bottom = 0
        for b, n in enumerate(c):
            if n:
                ax.bar(x, n, width=width, bottom=bottom, color=COLORS[b], lw=0, zorder=3)
                bottom += n
        n_all = sum(c)
        over = sum(c[bucket(target + 0.001):]) / n_all
        ax.text(x, bottom + ymax * 0.02, f"{over:.0%}", ha="center", va="bottom", fontsize=7,
                color=COLORS[-2] if over > 0.05 else INK2, fontweight="bold" if over > 0.05 else "normal")
    ax.set_ylim(0, ymax)
    ax.set_xlim(a.start, end)
    ax.text(0.005, 0.97, f"replica {i + 1} (first request {first_seen[pid]:.0f}s) {DOT} {total[pid]} requests", transform=ax.transAxes,
            ha="left", va="top", fontsize=8.5, color=INK)
axes[-1].set_xlabel("seconds since profiling start", color=INK2)
fig.text(0.04, 0.5, "requests in the window", rotation=90, va="center", ha="center", color=INK2, fontsize=9)
handles = [Patch(color=COLORS[b], label=LABELS[b]) for b in range(len(COLORS))]
handles += [Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"),
            Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=5, bbox_to_anchor=(0.5, -0.03),
           title="uncached prefill tokens already on the engine when the request arrived", title_fontsize=8)
fig.suptitle(f"{evaluation['experiment']} {DOT} per replica, per {a.window:.0f}s window: requests stacked by the uncached prefill load they found ahead {DOT} number over each bar: share over the {target:.0f}K target (bold if more than 5%, i.e. p95 over target)",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.subplots_adjust(top=0.93, bottom=0.1, left=0.08, right=0.99)
out = a.out or f"{R}/analysis/replica_arrival_bars{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
