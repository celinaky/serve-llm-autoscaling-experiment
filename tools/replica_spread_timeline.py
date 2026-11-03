#!/usr/bin/env python3
"""Per-replica p95 of load at arrival (top) over the spread of requests (rows).

Top panel: per window, one bar per replica, the p95 of the uncached prefill
load that replica's requests found on arrival (own tokens excluded;
analysis/load_requests.csv), axis clipped at --ymax with clipped bars
labelled, hollow marker when the replica served fewer than --min-samples
requests. Below: one row per replica, same colour, the requests it served in
each window, all rows on one scale; the small number at the top of each bar
is the distinct root sessions active on it in that window. Scale decisions
and load phases are marked throughout.

usage: python tools/replica_spread_timeline.py runs/<run_dir> [--window 60] [--min-samples 5]
           [--ymax K] [--start 0] [--end <load end>] [--out PATH]
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
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
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

meta = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            meta[m["x_request_id"]] = ((m["request_start_ns"] - t0) / 1e9, m["conversation_id"], m.get("agent_depth", 0))

loads = defaultdict(lambda: defaultdict(list))   # replica -> window -> loads K
sessions = defaultdict(lambda: defaultdict(set)) # replica -> window -> root conversation ids
first_seen = {}
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    m = meta.get(r["request_id"])
    if m is None or not a.start <= m[0] < end:
        continue
    pid = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
    first_seen[pid] = min(first_seen.get(pid, m[0]), m[0])
    k = int((m[0] - a.start) // a.window)
    loads[pid][k].append(float(r["engine_load_tokens"]) / 1000)
    if m[2] == 0:
        sessions[pid][k].add(m[1])

replicas = sorted(first_seen, key=first_seen.get)
named = replicas[:len(PALETTE)]
folded = replicas[len(PALETTE):]
rows = named + (["other"] if folded else [])
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
ycap = a.ymax if a.ymax else (5 * target / 1000 if target else 100.0)
centers = [a.start + (k + 0.5) * a.window for k in range(n_windows)]

def slot_loads(slot, k):
    return [y for p in (folded if slot == "other" else [slot]) for y in loads[p].get(k, [])]

def slot_sessions(slot, k):
    return len(set().union(*(sessions[p].get(k, set()) for p in (folded if slot == "other" else [slot]))))

fig, axes = plt.subplots(1 + len(rows), 1, figsize=(max(14, n_windows * 0.5), 3.6 + 0.95 * len(rows)), dpi=150, sharex=True,
                         gridspec_kw={"height_ratios": [3.6] + [1] * len(rows), "hspace": 0.1})
fig.patch.set_facecolor(SURFACE)
top, rows_ax = axes[0], axes[1:]

def decorate(ax, label_phases=False, ytop=None):
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
            if label_phases:
                ax.text((lo + hi) / 2, ytop * 0.995, f"{phase}: {p0['sessions']}→{p1['sessions']} sessions",
                        ha="center", va="top", fontsize=8, color=INK2)
    for e in evaluation["scale_events"]:
        if a.start <= e["time_s"] <= end:
            up = e["direction"] == "up"
            ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.8, ls="-" if up else "--", alpha=0.7)
            if label_phases:
                ax.text(e["time_s"] + 3, ytop * 0.93, f"{e['from_replicas']:.0f}→{e['to_replicas']:.0f}", rotation=90,
                        va="top", ha="left", fontsize=8, color=INK if up else INK2)

# Top: grouped p95 bars.
slot_w = a.window * 0.9 / len(rows)
for k in range(n_windows):
    ws = a.start + k * a.window
    for j, slot in enumerate(rows):
        v = sorted(slot_loads(slot, k))
        col = FOLD if slot == "other" else PALETTE[named.index(slot)]
        x = ws + a.window * 0.05 + (j + 0.5) * slot_w
        if len(v) >= a.min_samples:
            p95 = v[min(len(v) - 1, int(0.95 * len(v)))]
            if p95 > ycap:
                top.bar(x, ycap, width=slot_w * 0.9, color=col, lw=0)
                top.text(x, ycap * 1.01, f"{p95:.0f}K", ha="center", va="bottom", fontsize=6.5, color=col, rotation=90)
            else:
                top.bar(x, p95, width=slot_w * 0.9, color=col, lw=0)
        elif v:
            top.plot([x], [0], marker="o", ms=3, mfc="none", mec=col, mew=0.8, lw=0)
if target:
    top.axhline(target / 1000, color=INK2, lw=1, ls=":")
top.set_ylim(0, ycap * 1.3)
decorate(top, label_phases=True, ytop=ycap * 1.3)
top.set_ylabel("p95 of load at arrival (K)", color=INK2)
top.set_title(f"{evaluation['experiment']} · per replica, per {a.window:.0f}s window · top: p95 of uncached prefill load at arrival "
              f"(axis clipped at {ycap:.0f}K; hollow = fewer than {a.min_samples} requests) · rows: requests served, number = root sessions active",
              loc="left", fontsize=9.5, color=INK)

# Rows: requests served per replica, one shared scale.
req = {slot: [len(slot_loads(slot, k)) for k in range(n_windows)] for slot in rows}
req_max = max(max(v) for v in req.values()) or 1
for ax, slot in zip(rows_ax, rows):
    col = FOLD if slot == "other" else PALETTE[named.index(slot)]
    decorate(ax)
    ax.bar(centers, req[slot], width=a.window * 0.9, color=col, alpha=0.8, lw=0)
    for k, n in enumerate(req[slot]):
        s_ = slot_sessions(slot, k)
        if s_:
            ax.text(centers[k], n + req_max * 0.04, str(s_), ha="center", va="bottom", fontsize=6, color=INK)
    ax.set_ylim(0, req_max * 1.4)
    label = f"other ({len(folded)} replicas, first {min(first_seen[p] for p in folded):.0f}s)" if slot == "other" \
        else f"replica {named.index(slot) + 1} (first request {first_seen[slot]:.0f}s)"
    ax.text(0.004, 0.9, label, transform=ax.transAxes, ha="left", va="top", fontsize=8, color=INK)
rows_ax[-1].set_xlabel("seconds since profiling start", color=INK2)
rows_ax[-1].set_xlim(a.start, end)
fig.text(0.012, 0.42, "requests served per window", rotation=90, va="center", ha="center", fontsize=9, color=INK2)
handles = [Patch(color=PALETTE[i], label=f"replica {i + 1}") for i in range(len(named))]
if folded:
    handles.append(Patch(color=FOLD, label=f"other ({len(folded)})"))
if target:
    handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K"))
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=min(10, len(handles)), bbox_to_anchor=(0.5, -0.01))
fig.subplots_adjust(left=0.05, right=0.99, top=0.95, bottom=0.07)
out = a.out or f"{R}/analysis/replica_spread{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
