#!/usr/bin/env python3
"""Per replica, per window: p95, p90 and mean of load at arrival, three panels of lines.

One faint line per replica, in the same colour order as the other per-replica
charts, over fixed time windows; a window with fewer than --min-samples
requests on that replica is left blank. The target is a dotted line, scale
decisions are vertical lines, load phases are shaded.

usage: python tools/replica_percentile_lines.py runs/<run_dir> [--window 60] [--min-samples 5]
           [--ymax K] [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json, re
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
ap.add_argument("--window", type=float, default=60.0)
ap.add_argument("--min-samples", type=int, default=5)
ap.add_argument("--ymax", type=float, default=None)
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

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
loads = defaultdict(lambda: defaultdict(list))
first_seen = {}
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is None or not a.start <= t < end:
        continue
    pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
    first_seen[pid] = min(first_seen.get(pid, t), t)
    loads[pid][int((t - a.start) // a.window)].append(float(r["engine_load_tokens"]) / 1000)
replicas = sorted(first_seen, key=first_seen.get)
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
centers = [a.start + (k + 0.5) * a.window for k in range(n_windows)]
def pct(v, q):
    if q is None:
        return sum(v) / len(v)
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]

# Axis fits the data (up to 5x target) unless --ymax is given.
if a.ymax:
    ycap = a.ymax
else:
    peak = max((pct(v, q) for pid in loads for v in loads[pid].values() if len(v) >= a.min_samples for q in (0.95, 0.90, None)), default=target)
    ycap = min(5 * target, max(1.5 * target, peak * 1.05))

fig, axes = plt.subplots(1, 3, figsize=(21, 5.2), dpi=150, sharey=True, gridspec_kw={"wspace": 0.05})
fig.patch.set_facecolor(SURFACE)
for ax, q, name in ((axes[0], 0.95, "p95"), (axes[1], 0.90, "p90"), (axes[2], None, "mean")):
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
            ax.text((lo + hi) / 2, ycap * 1.09, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
    # Solid line at the decision, labelled with the count change; shaded band
    # until the new replica is running, ending in a dashed line.
    for e in evaluation["scale_events"]:
        if a.start <= e["time_s"] <= end:
            up = e["direction"] == "up"
            ready = e.get("running_matched_at_s")
            ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.8, ls="-", alpha=0.6)
            if ready is not None and ready > e["time_s"]:
                ax.axvspan(e["time_s"], min(ready, end), color=INK2, alpha=0.08, lw=0, zorder=0.5)
                ax.axvline(min(ready, end), color=INK2, lw=0.8, ls="--", alpha=0.6)
            ax.text(e["time_s"] + 4, ycap * 0.99, f"{int(e['from_replicas'])}{ARROW}{int(e['to_replicas'])}", ha="left", va="top", fontsize=8, color=INK, fontweight="bold")
    for i, pid in enumerate(replicas):
        col = PALETTE[i] if i < len(PALETTE) else FOLD
        xs, ys = [], []
        for k in range(n_windows):
            v = loads[pid].get(k, [])
            if len(v) >= a.min_samples:
                xs.append(centers[k]); ys.append(min(pct(v, q), ycap))
            else:
                if xs and xs[-1] is not None:
                    xs.append(None); ys.append(None)
        # Break the line where windows are blank.
        seg_x, seg_y = [], []
        for x, y in zip(xs, ys):
            if x is None:
                if seg_x:
                    ax.plot(seg_x, seg_y, color=col, lw=1.4, alpha=0.55, marker="o", ms=3)
                seg_x, seg_y = [], []
            else:
                seg_x.append(x); seg_y.append(y)
        if seg_x:
            ax.plot(seg_x, seg_y, color=col, lw=1.4, alpha=0.55, marker="o", ms=3)
    ax.axhline(target, color=INK2, lw=1, ls=":")
    ax.set_ylim(0, ycap * 1.12)
    ax.set_xlim(a.start, end)
    ax.set_xlabel("seconds since profiling start", color=INK2)
    ax.set_title(f"{name} of load at arrival per replica, per {a.window:.0f}s window", loc="left", fontsize=9.5, color=INK)
axes[0].set_ylabel(f"uncached prefill tokens at arrival (K)" + (f", clipped at {ycap:.0f}K" if any(min(pct(v, q), ycap) < pct(v, q) for pid in loads for v in loads[pid].values() if len(v) >= a.min_samples for q in (0.95, 0.90, None)) else ""), color=INK2)
handles = [Line2D([], [], color=PALETTE[i] if i < len(PALETTE) else FOLD, lw=1.4, alpha=0.7, marker="o", ms=3,
                  label=f"replica {i + 1} (first {first_seen[p]:.0f}s)") for i, p in enumerate(replicas)]
handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K"))
handles.append(Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"))
handles.append(Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)"))
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=min(9, len(handles)), bbox_to_anchor=(0.5, -0.08))
fig.suptitle(f"{evaluation['experiment']} {DOT} per replica, per {a.window:.0f}s window: p95, p90 and mean of uncached prefill load at arrival, requests on that replica only {DOT} blank where a replica served fewer than {a.min_samples} requests in the window",
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/replica_percentiles{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
