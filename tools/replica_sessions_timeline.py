#!/usr/bin/env python3
"""Small multiples: per replica, requests served and sessions active per window.

One row per replica (fixed colour, in order of first request), all sharing
the x axis and the y axis so rows compare directly. In each row, bars are
the requests that replica served in each window and the black line is the
number of distinct root sessions (conversations with agent_depth 0) that
sent at least one request to it in that window. Subagent conversations are
counted in requests but not in sessions. Replicas beyond the palette are
summed into one "other" row. Scale decisions and load phases are marked.

usage: python tools/replica_sessions_timeline.py runs/<run_dir> [--window 60]
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
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

meta = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            meta[m["x_request_id"]] = ((m["request_start_ns"] - t0) / 1e9, m["conversation_id"], m.get("agent_depth", 0))

requests = defaultdict(lambda: defaultdict(int))        # replica -> window -> requests
sessions = defaultdict(lambda: defaultdict(set))        # replica -> window -> root conversation ids
first_seen = {}
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    m = meta.get(r["request_id"])
    if m is None or not a.start <= m[0] < end:
        continue
    pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
    first_seen[pid] = min(first_seen.get(pid, m[0]), m[0])
    k = int((m[0] - a.start) // a.window)
    requests[pid][k] += 1
    if m[2] == 0:
        sessions[pid][k].add(m[1])

replicas = sorted(first_seen, key=first_seen.get)
named = replicas[:len(PALETTE)]
folded = replicas[len(PALETTE):]
rows = named + (["other"] if folded else [])
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)

def row_series(slot):
    pids = folded if slot == "other" else [slot]
    req = [sum(requests[p].get(k, 0) for p in pids) for k in range(n_windows)]
    ses = [len(set().union(*(sessions[p].get(k, set()) for p in pids))) for k in range(n_windows)]
    return req, ses

fig, axes = plt.subplots(len(rows), 2, figsize=(16, 1.1 * len(rows) + 1.8), dpi=150, sharex=True,
                         gridspec_kw={"hspace": 0.14, "wspace": 0.08})
fig.patch.set_facecolor(SURFACE)
centers = [a.start + (k + 0.5) * a.window for k in range(n_windows)]
series = {slot: row_series(slot) for slot in rows}
req_max = max(max(r) for r, _ in series.values()) or 1
ses_max = max(max(s_) for _, s_ in series.values()) or 1
for i, slot in enumerate(rows):
    col = FOLD if slot == "other" else PALETTE[named.index(slot)]
    req, ses = series[slot]
    label = f"other ({len(folded)} replicas, first {min(first_seen[p] for p in folded):.0f}s)" if slot == "other" \
        else f"replica {named.index(slot) + 1} (first request {first_seen[slot]:.0f}s)"
    for j, (values, vmax) in enumerate(((req, req_max), (ses, ses_max))):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        for sp in ("left", "bottom"):
            ax.spines[sp].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        for p0, p1 in zip(curve, curve[1:]):
            phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
            lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
            if hi > lo:
                ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
        for e in evaluation["scale_events"]:
            if a.start <= e["time_s"] <= end:
                ax.axvline(e["time_s"], color=INK if e["direction"] == "up" else INK2, lw=0.7,
                           ls="-" if e["direction"] == "up" else "--", alpha=0.6)
        ax.bar(centers, values, width=a.window * 0.9, color=col, alpha=0.75 if j == 0 else 1.0, lw=0)
        ax.set_ylim(0, vmax * 1.25)
        ax.set_xlim(a.start, end)
        if j == 0:
            ax.text(0.005, 0.92, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)
axes[0][0].set_title("requests served per window", loc="left", fontsize=9, color=INK2)
axes[0][1].set_title("distinct root sessions with a request on the replica, per window", loc="left", fontsize=9, color=INK2)
for ax in axes[-1]:
    ax.set_xlabel("seconds since profiling start", color=INK2)
fig.suptitle(f"{evaluation['experiment']} \u00b7 per replica, per {a.window:.0f}s window \u00b7 vertical lines: scale decisions (solid up, dashed down)",
             x=0.01, ha="left", fontsize=10, color=INK)
fig.subplots_adjust(left=0.04, right=0.99, top=0.93, bottom=0.07)
out = a.out or f"{R}/analysis/replica_sessions{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
for slot in rows:
    req, ses = row_series(slot)
    print(f"{slot if slot == 'other' else 'replica ' + str(named.index(slot) + 1):10s} requests {sum(req):4d}  peak sessions in a window {max(ses)}")
