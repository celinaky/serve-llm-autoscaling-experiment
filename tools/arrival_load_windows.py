#!/usr/bin/env python3
"""Per-window p95 of the uncached prefill load requests found on arrival.

Windows are either fixed time (--window seconds) or fixed count (--per-requests
N consecutive requests by arrival time; each bar then spans from its first to
its last arrival). Each bar is the p95 of load at arrival, own tokens excluded
(analysis/load_requests.csv from tools/load_breakdown.py), green when it meets
target_prefill_tokens_per_replica and red when it does not. Under each bar:
the request count (grey) and the running replicas at the window's start and
end (blue). Scale decisions and load phases are marked as in
tools/replica_load_timeline.py. --mean adds the window mean; --style lines
draws points joined by a line instead of bars.

usage: python tools/arrival_load_windows.py runs/<run_dir> [--window 60 | --per-requests 100]
           [--min-samples 5] [--start 0] [--end <load end>] [--mean] [--style bars|lines] [--out PATH]
"""
import argparse, bisect, csv, glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GOOD, BAD, MEAN, REPLICAS = "#1baf7a", "#e34948", "#9b9a95", "#2a78d6"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--window", type=float, default=60.0, help="fixed time window, seconds")
ap.add_argument("--per-requests", type=int, default=None, help="fixed-count windows of this many requests instead")
ap.add_argument("--min-samples", type=int, default=5, help="time windows with fewer requests get no p95")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--mean", action="store_true", help="also show the mean over all requests in each window")
ap.add_argument("--style", choices=["bars", "lines"], default="bars")
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000  # runs without a token policy are scored against the usual 20K
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        m = rec["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9

requests = []  # (t, load K)
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    t = start_s.get(r["request_id"])
    if t is not None and a.start <= t < end:
        requests.append((t, float(r["engine_load_tokens"]) / 1000))
requests.sort()

def summarize(chunk):
    v = sorted(y for _, y in chunk)
    n = len(v)
    return n, v[min(n - 1, int(0.95 * n))] if n else None, (sum(v) / n if n else None)

rows = []  # (ws, we, n, p95, mean)
if a.per_requests:
    for i in range(0, len(requests), a.per_requests):
        chunk = requests[i:i + a.per_requests]
        ws = chunk[0][0]
        we = requests[i + a.per_requests][0] if i + a.per_requests < len(requests) else end
        n, p95, mean = summarize(chunk)
        rows.append((ws, we, n, p95 if n >= a.min_samples else None, mean))
    window_label = f"{a.per_requests}-request window"
else:
    k = 0
    while a.start + k * a.window < end:
        ws, we = a.start + k * a.window, min(a.start + (k + 1) * a.window, end)
        chunk = [p for p in requests if ws <= p[0] < we]
        n, p95, mean = summarize(chunk)
        rows.append((ws, we, n, p95 if n >= a.min_samples else None, mean))
        k += 1
    window_label = f"{a.window:.0f}s window"

status = [s for s in plot_data["serve_status_samples"] if s.get("error") is None and a.start <= s["relative_time_s"] <= end]
status_t = [s["relative_time_s"] for s in status]

def running_at(t):
    i = bisect.bisect_right(status_t, t) - 1
    return status[i]["running_replicas"] if i >= 0 else None

fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE)
ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)

met = missed = 0
for ws, we, n, p95, mean in rows:
    span = we - ws
    mid = ws + span / 2
    if p95 is not None:
        ok = p95 * 1000 <= target
        met += ok
        missed += not ok
        if a.style == "bars":
            if a.mean:
                ax.bar(ws + span * 0.28, p95, width=span * 0.42, color=GOOD if ok else BAD, lw=0)
            else:
                ax.bar(mid, p95, width=span * 0.8, color=GOOD if ok else BAD, lw=0)
        else:
            ax.plot([mid], [p95], marker="o", ms=5, color=GOOD if ok else BAD, lw=0, zorder=3)
    if mean is not None and a.mean and a.style == "bars":
        ax.bar(ws + span * 0.72, mean, width=span * 0.42, color=MEAN, lw=0)
    ax.text(mid, -0.5, str(n), ha="center", va="top", fontsize=6.5, color=INK2)
    r0, r1 = running_at(ws), running_at(we - 0.001)
    if r0 is not None:
        label = f"{r0}" if r1 is None or r1 == r0 else f"{r0}→{r1}"
        ax.text(mid, -0.5, "\n" + label, ha="center", va="top", fontsize=6.5, color=REPLICAS)
if a.style == "lines":
    pts = [((ws + we) / 2, p95) for ws, we, n, p95, mean in rows if p95 is not None]
    ax.plot([x for x, _ in pts], [y for _, y in pts], color=INK, lw=1.4, zorder=2)
    if a.mean:
        pts = [((ws + we) / 2, mean) for ws, we, n, p95, mean in rows if mean is not None]
        ax.plot([x for x, _ in pts], [y for _, y in pts], color=MEAN, lw=1.4, marker="o", ms=3)

if a.style == "bars":
    handles = [Patch(color=GOOD, label="p95 of load at arrival, window meets target"),
               Patch(color=BAD, label="p95 of load at arrival, window misses target")]
    if a.mean:
        handles.append(Patch(color=MEAN, label="mean of load at arrival, all requests"))
else:
    handles = [Line2D([], [], color=INK, lw=1.4, marker="o", mfc=GOOD, mec=GOOD, label="p95 of load at arrival, window meets target"),
               Line2D([], [], color=INK, lw=1.4, marker="o", mfc=BAD, mec=BAD, label="p95 of load at arrival, window misses target")]
    if a.mean:
        handles.append(Line2D([], [], color=MEAN, lw=1.4, marker="o", label="mean of load at arrival, all requests"))
if target:
    ax.axhline(target / 1000, color=INK2, lw=1, ls=":")
    handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K"))
handles.append(Line2D([], [], color=REPLICAS, lw=0, marker="$3$", ms=7, label="running replicas in the window (blue number under the bar)"))

ax.set_ylim(bottom=0)
ymax = ax.get_ylim()[1]
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi <= lo:
        continue
    ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
    ax.text((lo + hi) / 2, ymax * 0.995, f"{phase}: {p0['sessions']}→{p1['sessions']} sessions",
            ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    t = e["time_s"]
    if not a.start <= t <= end:
        continue
    up = e["direction"] == "up"
    ax.axvline(t, color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
    ax.text(t + 3, ymax * 0.93, f"{e['from_replicas']:.0f}→{e['to_replicas']:.0f}", rotation=90,
            va="top", ha="left", fontsize=8, color=INK if up else INK2)

ax.set_ylim(bottom=-ymax * 0.1)
ax.set_ylabel("uncached prefill tokens on the engine at arrival (K)", color=INK2)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_title(f"{evaluation['experiment']} · p95 of load at arrival per {window_label}: "
             f"{met} windows meet the target, {missed} miss · under each bar: requests, then running replicas",
             loc="left", fontsize=10, color=INK)
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=4, bbox_to_anchor=(0.5, -0.04))
ax.set_xlim(a.start, end)

tag = f"req{a.per_requests}" if a.per_requests else f"{int(a.window)}"
out = a.out or f"{R}/analysis/arrival_load_windows{tag}{'_lines' if a.style == 'lines' else ''}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
print(f"{window_label}s: {met} meet target, {missed} miss, {sum(1 for r in rows if r[3] is None)} too few requests")
for ws, we, n, p95, mean in rows:
    if p95 is not None and p95 * 1000 > target:
        print(f"  miss at {ws:.0f}-{we:.0f}s: n={n} p95={p95:.0f}K")
