#!/usr/bin/env python3
"""Over all requests, per fixed time window: p95, p90 and mean of load at arrival.

Load at arrival is the engine-side uncached prefill tokens already outstanding
on the replica the request landed on, its own tokens excluded (from
analysis/load_requests.csv, built by tools/load_breakdown.py). One line per
statistic, one point per window at the window centre; the request count sits
under each point. Target as a dotted line, scale decisions as vertical lines,
load phases shaded.

usage: python tools/arrival_stats_lines.py runs/<run_dir> [--window 120] [--min-samples 5]
           [--ymax K] [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
STATS = (("p95", 0.95, "#2a78d6"), ("p90", 0.90, "#6250d6"), ("mean", None, "#eb6834"))
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=120.0)
ap.add_argument("--min-samples", type=int, default=5)
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
ap.add_argument("--match", default=None, help="another run dir: use the same K-per-inch y scale as its chart, growing the figure height as needed")
a = ap.parse_args()
R = a.run.rstrip("/")

def load_run(R):
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
    loads = defaultdict(list)
    for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
        t = start_s.get(r["request_id"])
        if t is None or not a.start <= t < end:
            continue
        loads[int((t - a.start) // a.window)].append(float(r["engine_load_tokens"]) / 1000)
    n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
    centers = [a.start + (k + 0.5) * a.window for k in range(n_windows)]
    series = {name: [stat(loads[k], q) if len(loads.get(k, [])) >= a.min_samples else None for k in range(n_windows)]
              for name, q, _ in STATS}
    ycap = a.ymax if a.ymax else max(10.0, max(v for vs in series.values() for v in vs if v is not None) * 1.05)
    return evaluation, target, end, curve, loads, n_windows, centers, series, ycap

def stat(v, q):
    if q is None:
        return sum(v) / len(v)
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]

evaluation, target, end, curve, loads, n_windows, centers, series, ycap = load_run(R)
# Same K per inch as the matched run's chart: the axes grow taller instead of
# the scale compressing.
BASE_H = 5.6
height = BASE_H
ycap_ref = ycap  # the range the paddings and label offsets are a fraction of
if a.match:
    ycap_ref = load_run(a.match.rstrip("/"))[-1]
    height = BASE_H * ycap / ycap_ref
pad = 0.09 * ycap_ref

fig, ax = plt.subplots(figsize=(14, height), dpi=150)
# Title and legend sit a fixed distance (in inches) from the axes whatever the height.
fig.subplots_adjust(top=1 - 0.55 / height, bottom=1.15 / height)
fig.patch.set_facecolor(SURFACE)
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
        ax.text((lo + hi) / 2, ycap + pad, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
# A scale decision is a solid line labelled with the count change; the band
# after it is the startup, ending with a dashed line when the new replica is
# running (running_matched_at_s from the harness evaluation).
for e in evaluation["scale_events"]:
    if a.start <= e["time_s"] <= end:
        up = e["direction"] == "up"
        ready = e.get("running_matched_at_s")
        ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.8, ls="-", alpha=0.6)
        if ready is not None and ready > e["time_s"]:
            ax.axvspan(e["time_s"], min(ready, end), color=INK2, alpha=0.08, lw=0, zorder=0.5)
            ax.axvline(min(ready, end), color=INK2, lw=0.8, ls="--", alpha=0.6)
        ax.text(e["time_s"] + 4, ycap + 0.02 * ycap_ref, f"{int(e['from_replicas'])}{ARROW}{int(e['to_replicas'])}", ha="left", va="bottom", fontsize=8, color=INK, fontweight="bold")
for name, q, col in STATS:
    xs = [c for c, v in zip(centers, series[name]) if v is not None]
    ys = [v for v in series[name] if v is not None]
    ax.plot(xs, ys, color=col, lw=1.8 if name == "p95" else 1.4, marker="o", ms=3.5, label=name)
for k, c in enumerate(centers):
    n = len(loads.get(k, []))
    if n:
        ax.text(c, -0.4 * pad, str(n), ha="center", va="top", fontsize=7, color=INK2)
ax.axhline(target, color=INK2, lw=1, ls=":")
ax.set_ylim(-pad, ycap + 0.12 * ycap_ref)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel("uncached prefill tokens on the engine at arrival (K)", color=INK2)
handles = [Line2D([], [], color=col, lw=1.8 if name == "p95" else 1.4, marker="o", ms=3.5, label=f"{name} of load at arrival") for name, _, col in STATS]
handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K"))
handles.append(Line2D([], [], color=INK, lw=0.8, ls="-", alpha=0.6, label="scale decision (label: replicas before{}after)".format(ARROW)))
handles.append(Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)"))
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, 0.08 / height))
fig.suptitle(f"{evaluation['experiment']} {DOT} all requests, per {a.window:.0f}s window: p95, p90 and mean of uncached prefill load at arrival {DOT} number under each point: requests in the window",
             x=0.01, y=1 - 0.15 / height, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/arrival_stats{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
for k, c in enumerate(centers):
    n = len(loads.get(k, []))
    vals = " ".join(f"{name}={series[name][k]:.1f}K" if series[name][k] is not None else f"{name}=-" for name, _, _ in STATS)
    print(f"  {a.start + k * a.window:5.0f}-{min(end, a.start + (k + 1) * a.window):5.0f}s n={n:3d} {vals}")
