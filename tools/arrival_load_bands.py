#!/usr/bin/env python3
"""Where each window's p95 comes from: the distribution of load at arrival.

Windows are fixed time (--window seconds) or fixed count (--per-requests N).
For every window, requests are classified by how much uncached prefill load
they found ahead of them on their engine (own tokens excluded;
analysis/load_requests.csv).

--style bands (default): a stacked bar from "empty engine" at the bottom to
the largest loads at the top, so the 95th-percentile request sits at the
dotted 95% line; whichever band crosses it is the window's p95 band. Bands
at or above the target are warm colours, so a window misses the target
exactly when warm colour reaches below the line. --zoom 0.7 starts the axis
at 70% to give the top of the stack room.

--style rate: one bar per window, the share of its requests that found more
than the target ahead (the collision rate), against the 5% line: the p95 is
over target exactly when this share is over 5%. The darker part is the share
over 2.5x target.

The p95 is printed above each bar (red when over target) and the request
count below. Scale decisions and load phases are marked.

usage: python tools/arrival_load_bands.py runs/<run_dir> [--window 60 | --per-requests 100]
           [--style bands|rate] [--zoom 0.7] [--min-samples 5] [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, BAD, DEEP = "#1baf7a", "#e34948", "#b4321c"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=60.0)
ap.add_argument("--per-requests", type=int, default=None)
ap.add_argument("--style", choices=["bands", "rate"], default="bands")
ap.add_argument("--zoom", type=float, default=0.0)
ap.add_argument("--min-samples", type=int, default=5)
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

BANDS = [
    (0.001, "empty engine (0)", "#dfe7f3"),
    (target / 2, f"under {target / 2:.0f}K", "#a9c3e6"),
    (target, f"{target / 2:.0f}-{target:.0f}K", "#5f93d4"),
    (target * 2.5, f"{target:.0f}-{target * 2.5:.0f}K (over target)", "#f3b88f"),
    (target * 5, f"{target * 2.5:.0f}-{target * 5:.0f}K", "#eb6834"),
    (float("inf"), f"over {target * 5:.0f}K", DEEP),
]

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
requests = sorted((start_s[r["request_id"]], float(r["engine_load_tokens"]) / 1000)
                  for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv"))
                  if r["request_id"] in start_s and a.start <= start_s[r["request_id"]] < end)
spans = []
if a.per_requests:
    for i in range(0, len(requests), a.per_requests):
        chunk = requests[i:i + a.per_requests]
        we = requests[i + a.per_requests][0] if i + a.per_requests < len(requests) else end
        spans.append((chunk[0][0], we, [y for _, y in chunk]))
    window_label = f"{a.per_requests}-request window"
else:
    k = 0
    while a.start + k * a.window < end:
        ws, we = a.start + k * a.window, min(a.start + (k + 1) * a.window, end)
        spans.append((ws, we, [y for t, y in requests if ws <= t < we]))
        k += 1
    window_label = f"{a.window:.0f}s window"

fig, ax = plt.subplots(figsize=(max(14, len(spans) * 0.5), 6.4), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2); ax.set_axisbelow(True)

ymin = a.zoom if a.style == "bands" else 0.0
if a.style == "bands":
    ytop = 1.0
else:
    ytop = max(0.3, max((sum(1 for y in v if y >= target) / len(v) for _, _, v in spans if v), default=0) * 1.2)
unit = ytop - ymin
met = missed = 0
for ws, we, v in spans:
    v = sorted(v); n = len(v); span = we - ws; mid = ws + span / 2
    if not n:
        continue
    p95 = v[min(n - 1, int(0.95 * n))]
    over = sum(1 for y in v if y >= target) / n
    far = sum(1 for y in v if y >= target * 2.5) / n
    if a.style == "bands":
        base = 0.0
        for upper, label, col in BANDS:
            share = sum(1 for y in v if y < upper) / n - base
            if share > 0 and base + share > ymin:
                lo_ = max(base, ymin)
                ax.bar(mid, base + share - lo_, bottom=lo_, width=span * 0.9, color=col, lw=0)
            base += share
    else:
        ax.bar(mid, over, width=span * 0.9, color=GOOD if over <= 0.05 else BAD, alpha=0.85, lw=0)
        ax.bar(mid, far, width=span * 0.9, color=DEEP, lw=0)
    if n >= a.min_samples:
        ok = p95 <= target
        met += ok; missed += not ok
        ax.text(mid, ytop + unit * 0.012, f"{p95:.0f}K", ha="center", va="bottom", fontsize=6.5,
                color=GOOD if ok else BAD, fontweight="bold" if not ok else "normal")
    ax.text(mid, ymin - unit * 0.012, str(n), ha="center", va="top", fontsize=6.5, color=INK2)

if a.style == "bands":
    ax.axhline(0.95, color=INK, lw=1, ls=":")
    ax.text(a.start + 2, 0.953, "95th percentile request", fontsize=7.5, color=INK, va="bottom")
    ticks = [t for t in (0, 0.25, 0.5, 0.75, 0.9, 0.95, 1.0) if t >= ymin]
    ax.set_yticks(ticks); ax.set_yticklabels([f"{t:.0%}" for t in ticks])
    ax.set_ylabel("share of the window's requests, by load found on arrival", color=INK2)
    handles = [Patch(color=col, label=label) for _, label, col in BANDS]
else:
    ax.axhline(0.05, color=INK, lw=1, ls=":")
    ax.text(a.start + 2, 0.052, "5%: above this the window's p95 is over target", fontsize=7.5, color=INK, va="bottom")
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("share of the window's requests that found more than the target ahead", color=INK2)
    handles = [Patch(color=GOOD, label=f"share over {target:.0f}K, window meets target"),
               Patch(color=BAD, label=f"share over {target:.0f}K, window misses"),
               Patch(color=DEEP, label=f"of which over {target * 2.5:.0f}K")]
ax.set_ylim(ymin - unit * 0.06, ytop + unit * 0.09)
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi > lo:
        ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
        ax.text((lo + hi) / 2, ytop + unit * 0.085, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions",
                ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    if a.start <= e["time_s"] <= end:
        up = e["direction"] == "up"
        ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
        ax.text(e["time_s"] + 3, ytop + unit * 0.07, f"{e['from_replicas']:.0f}{ARROW}{e['to_replicas']:.0f}", rotation=90,
                va="top", ha="left", fontsize=7.5, color=INK if up else INK2)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_title(f"{evaluation['experiment']} {DOT} per {window_label}: what requests found ahead of them {DOT} "
             f"p95 above each bar (red = over {target:.0f}K), requests below {DOT} {met} windows meet, {missed} miss",
             loc="left", fontsize=9.5, color=INK)
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=6, bbox_to_anchor=(0.5, -0.03))
tag = f"req{a.per_requests}" if a.per_requests else f"{int(a.window)}"
out = a.out or f"{R}/analysis/arrival_load_{a.style}{tag}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
