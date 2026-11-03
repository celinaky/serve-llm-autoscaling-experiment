#!/usr/bin/env python3
"""Uncached prefill load each request found on arrival, over time, with the replica count.

Top panel: one dot per successful profiling request at its start time, at the
uncached prefill tokens already queued or mid-prefill on the engine it was
routed to (its own tokens excluded; analysis/load_requests.csv from
tools/load_breakdown.py), coloured by that engine. A trailing-window p95 and
p50 over all requests show whether the tail is under the target. Bottom panel:
running and desired replicas. Scale decisions, load phases and session starts
are marked as in tools/replica_load_timeline.py.

usage: python tools/arrival_load_timeline.py runs/<run_dir> [--window 120] [--min-samples 20]
           [--start 0] [--end <load end>] [--out analysis/arrival_load.png]
"""
import argparse, bisect, csv, glob, json, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, FOLD = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=120.0, help="trailing window for the percentiles, seconds")
ap.add_argument("--min-samples", type=int, default=20, help="requests a window needs before its percentile is drawn")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica")
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

start_s, starts = {}, {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        m = rec["metadata"]
        if m.get("benchmark_phase") != "profiling":
            continue
        t = (m["request_start_ns"] - t0) / 1e9
        start_s[m["x_request_id"]] = t
        cid = m["conversation_id"]
        if cid not in starts or t < starts[cid][0]:
            starts[cid] = (t, m.get("agent_depth", 0))

points = []  # (t, load K, engine)
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is None or not a.start <= t <= end:
        continue
    m = re.search(r"pid-(\d+)", r["engine"])
    points.append((t, float(r["engine_load_tokens"]) / 1000, m.group(1) if m else r["engine"]))
points.sort()
times = [p[0] for p in points]
first_seen = {}
for t, _, pid in points:
    first_seen.setdefault(pid, t)
engines = sorted(first_seen, key=first_seen.get)
color = {pid: PALETTE[i] if i < len(PALETTE) else FOLD for i, pid in enumerate(engines)}

def rolling(q):
    xs, ys = [], []
    for i, t in enumerate(times):
        lo = bisect.bisect_left(times, t - a.window)
        window = sorted(p[1] for p in points[lo:i + 1])
        if len(window) < a.min_samples:
            continue
        xs.append(t)
        ys.append(window[min(len(window) - 1, int(q * len(window)))])
    return xs, ys

status = [s for s in plot_data["serve_status_samples"] if s.get("error") is None and a.start <= s["relative_time_s"] <= end]
desired = [p for p in (plot_data["serve_metric_series"].get("desired_replicas") or {}).get("points", [])
           if a.start <= p["relative_time_s"] <= end]

fig, (ax, axr) = plt.subplots(2, 1, figsize=(14, 7.6), dpi=150, sharex=True,
                              gridspec_kw={"height_ratios": [4, 1], "hspace": 0.08})
fig.patch.set_facecolor(SURFACE)
for axis in (ax, axr):
    axis.set_facecolor(SURFACE)
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axis.spines[s].set_color(GRID)
    axis.tick_params(colors=INK2)
    axis.grid(axis="y", color=GRID, lw=0.6)
    axis.set_axisbelow(True)

for i, pid in enumerate(engines):
    pts = [(t, y) for t, y, e in points if e == pid]
    ax.scatter([t for t, _ in pts], [y for _, y in pts], s=14, color=color[pid], alpha=0.75, lw=0,
               label=f"replica {i + 1} (pid {pid}, {len(pts)} requests)")
xs, ys = rolling(0.95)
ax.plot(xs, ys, color=INK, lw=1.8, label=f"p95 of load at arrival, trailing {a.window:.0f}s")
xs, ys = rolling(0.5)
ax.plot(xs, ys, color=INK2, lw=1.2, ls="--", label=f"p50, trailing {a.window:.0f}s")
if target:
    ax.axhline(target / 1000, color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K")

ax.set_ylim(bottom=0)
ymax = ax.get_ylim()[1]
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi <= lo:
        continue
    for axis in (ax, axr):
        axis.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
    ax.text((lo + hi) / 2, ymax * 0.995, f"{phase}: {p0['sessions']}→{p1['sessions']} sessions",
            ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    t = e["time_s"]
    if not a.start <= t <= end:
        continue
    up = e["direction"] == "up"
    for axis in (ax, axr):
        axis.axvline(t, color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
    ax.text(t + 3, ymax * 0.93, f"{e['from_replicas']:.0f}→{e['to_replicas']:.0f}", rotation=90,
            va="top", ha="left", fontsize=8, color=INK if up else INK2)

root = sorted(t for t, d in starts.values() if d == 0 and a.start <= t <= end)
sub = sorted(t for t, d in starts.values() if d > 0 and a.start <= t <= end)
tick_y = -ymax * 0.03
ax.plot(root, [tick_y] * len(root), marker="^", ls="none", ms=6, color=INK, clip_on=False,
        label=f"root session start ({len(root)})")
ax.plot(sub, [tick_y * 2.2] * len(sub), marker="^", ls="none", ms=4, color=INK2, alpha=0.6, clip_on=False,
        label=f"subagent start ({len(sub)})")
ax.set_ylim(bottom=tick_y * 3.2)
ax.set_ylabel("uncached prefill tokens on the engine at arrival (K)", color=INK2)
ax.set_title(f"{evaluation['experiment']} · load each request found on arrival, {a.start:.0f}–{end:.0f}s "
             f"· lines: desired-replica changes (solid up, dashed down)", loc="left", fontsize=10, color=INK)
ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.42))

axr.step([s["relative_time_s"] for s in status], [s["running_replicas"] for s in status], where="post",
         color=PALETTE[2], lw=1.6, label="running")
axr.step([p["relative_time_s"] for p in desired], [p["value"] for p in desired], where="post",
         color=INK2, lw=1.0, ls="--", label="desired")
axr.set_ylim(bottom=0)
axr.set_ylabel("replicas", color=INK2)
axr.set_xlabel("seconds since profiling start", color=INK2)
axr.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
ax.set_xlim(a.start, end)

out = a.out or f"{R}/analysis/arrival_load_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
loads = sorted(y for _, y, _ in points)
n = len(loads)
print(f"{n} requests; load at arrival p50 {loads[n // 2]:.0f}K p90 {loads[int(.9 * n)]:.0f}K p95 {loads[int(.95 * n)]:.0f}K; "
      f"share <= target: {sum(y * 1000 <= (target or 0) for y in loads) / n:.1%}")
xs, ys = rolling(0.95)
over = sum(1 for y in ys if target and y * 1000 > target)
print(f"trailing-{a.window:.0f}s p95 above target at {over} of {len(ys)} request times ({over / max(1, len(ys)):.0%})")
