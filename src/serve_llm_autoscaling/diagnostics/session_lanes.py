#!/usr/bin/env python3
"""Per AIPerf lane: which session was running, for how long, on which replica.

One row per lane (concurrency slot). Each bar is one root session (an
agent_depth 0 conversation tree), from its first request's start to its last
request's end, coloured by the first observed root replica and labelled with its
trace id, root turns and duration (above the bar when the bar is too short).
When every observed root matches one unique initial lane in stdout.log, use
those recorded lane IDs. Otherwise infer non-overlapping request-span rows;
those rows do not establish actual lane ownership or session completion.
Last observed request end is not session end: think time and pending subagents
can keep a slot occupied after the bar ends.

usage: python tools/session_lanes.py runs/<run_dir> [--out PATH]
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
HATCH = ["", "///", "...", "xxx", "\\\\\\", "ooo"]

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948",
           "#174a80", "#945019", "#007f83", "#a38400", "#a02867", "#9876be", "#637c32", "#7d363a"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--out"); a = ap.parse_args(); R = a.run.rstrip("/")
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"]
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); end = ev["periods"]["steady"]["end_s"]
engine = {r["request_id"]: r["engine"] for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv"))}
roots = defaultdict(list)
for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(p):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") != "profiling": continue
        roots[m["root_correlation_id"]].append(((m["request_start_ns"] - t0) / 1e9, (m["request_end_ns"] - t0) / 1e9,
                                                m.get("agent_depth", 0), m.get("source_trace_id", "")[-6:], engine.get(m["x_request_id"])))
sessions = sorted(roots.values(), key=lambda rs: min(r[0] for r in rs))
first = {}
for rs in sessions:
    for r in rs:
        if r[4]: first[r[4]] = min(first.get(r[4], 1e9), r[0])
engines = sorted(first, key=first.get)  # replica 1 = first to serve a request, as in the other charts
print("replicas:", ", ".join(f"{i + 1}=pid {e.split('pid-')[-1]} (first request {first[e]:.0f}s)" for i, e in enumerate(engines)))
initial_lanes = defaultdict(set)
for path in glob.glob(f"{R}/benchmark/*/stdout.log"):
    for line in open(path):
        match = re.search(r"lane=(\d+)\s+sample_time=.*?trace_id=(\S+)", line)
        if match:
            initial_lanes[match[2][-6:]].add(int(match[1]))
traces = [sorted(rs)[0][3] for rs in sessions]
recorded_lanes = (len(set(traces)) == len(traces) and bool(traces)
                  and all(len(initial_lanes[t]) == 1 for t in traces)
                  and len({next(iter(initial_lanes[t])) for t in traces}) == len(traces))
lanes, placed = [], []
if recorded_lanes:
    placed = [(next(iter(initial_lanes[sorted(rs)[0][3]])),
               min(r[0] for r in rs), max(r[1] for r in rs), rs) for rs in sessions]
    lanes = [0] * (max(p[0] for p in placed) + 1)
else:
    for rs in sessions:
        start, stop = min(r[0] for r in rs), max(r[1] for r in rs)
        for i, free_at in enumerate(lanes):
            if free_at <= start: lanes[i] = stop; placed.append((i, start, stop, rs)); break
        else: lanes.append(stop); placed.append((len(lanes) - 1, start, stop, rs))
lane_source = "recorded initial lane IDs" if recorded_lanes else "inferred request-span rows (not actual lane IDs)"
print("lane source:", lane_source)
print("Bar ends are last observed request ends, not verified session completion.")
n = len(lanes)
fig, ax = plt.subplots(figsize=(13, 0.4 * n + 2.0), dpi=150); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
plt.rcParams["hatch.linewidth"] = 0.6
seen_in_lane = defaultdict(int)
table = []
root_ids = {id(rs): root_id for root_id, rs in roots.items()}
for lane, start, stop, rs in placed:
    trace = sorted(rs)[0][3]
    pin = next((r[4] for r in sorted(rs) if r[2] == 0 and r[4]), None)
    rep = engines.index(pin) + 1 if pin in engines else None
    col = PALETTE[(rep - 1) % len(PALETTE)] if rep else "#9b9a95"
    k = seen_in_lane[lane]; seen_in_lane[lane] += 1
    ax.barh(lane, stop - start, left=start, height=0.62, facecolor=col, alpha=0.8, edgecolor=SURFACE if k == 0 else INK, lw=0.8, hatch=HATCH[k % len(HATCH)])
    turns = sum(1 for r in rs if r[2] == 0)
    root_engines = sorted({r[4] for r in rs if r[2] == 0 and r[4]})
    table.append(dict(lane=lane + 1, lane_source=lane_source, trace_id=trace,
        root_correlation_id=root_ids[id(rs)], first_request_s=round(start, 3),
        last_request_end_s=round(stop, 3), observed_span_s=round(stop - start, 3),
        root_turns=turns, replica=rep or '', engine=pin or '',
        root_replica_count=len(root_engines),
        root_replicas=';'.join(str(engines.index(e) + 1) for e in root_engines)))
    label = f"{trace} · {turns} turns · {stop - start:.0f}s"
    if stop - start >= 0.16 * end: ax.text(start + 4, lane, label, va="center", fontsize=6.8, color=INK, clip_on=True, bbox=dict(facecolor=SURFACE, alpha=0.85, lw=0, pad=1.2) if k else None)
    else: ax.text(start, lane - 0.36, label, va="bottom", ha="left" if start < 0.8 * end else "right", fontsize=6.3, color=INK2, clip_on=False)
ax.set_yticks(range(n)); ax.set_yticklabels([f"{'lane' if recorded_lanes else 'row'} {i + 1}" for i in range(n)], fontsize=8); ax.invert_yaxis()
ax.set_xlim(0, end); ax.set_xlabel("seconds since profiling start"); ax.grid(axis="x", color=GRID, lw=0.6)
for sp in ax.spines.values(): sp.set_visible(False)
name = os.path.basename(R).split("-", 1)[-1]
multi = sum(1 for i in range(n) if sum(1 for p in placed if p[0] == i) > 1)
handles = [Patch(facecolor=PALETTE[i % len(PALETTE)], alpha=0.8, label=f"first root on replica {i + 1}") for i in range(len(engines))]
if multi:
    handles += [Patch(facecolor="#dddbd6", edgecolor=INK, hatch="///", label="second observed span in row"), Patch(facecolor="#dddbd6", edgecolor=INK, hatch="...", label="third observed span in row")]
ax.legend(handles=handles, fontsize=7.5, ncol=min(6, len(handles)), loc="upper center", bbox_to_anchor=(0.5, -0.05 - 0.9 / n), frameon=False)
ax.set_title(f"{name} · {len(sessions)} sessions, {n} rows · {lane_source}\nBars: first to last observed request; bar ends do not establish session completion", fontsize=9.5, loc="left")
out = a.out or f"{R}/analysis/session_lanes_0_{int(end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
table_out = f"{R}/analysis/session_lanes.csv"
with open(table_out, "w", newline="") as fh:
    writer = csv.DictWriter(fh, fieldnames=list(table[0]))
    writer.writeheader(); writer.writerows(sorted(table, key=lambda r: (r['lane'], r['first_request_s'])))
print("wrote", table_out)
