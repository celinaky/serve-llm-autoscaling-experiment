#!/usr/bin/env python3
"""Several runs on one grid: rows are load at arrival, engine queue time and
client TTFT; columns are mean, p90 and p95; one line per run in every panel,
per fixed time window over all requests. Data from each run's
analysis/load_requests.csv (tools/load_breakdown.py). Each run's scale-ups are
thin vertical lines in that run's colour at the moment the new replica was
serving, labelled with the new count in the top row. Phase shading comes from the first run's curve.

usage: python tools/arrival_stats_compare.py runs/<run_a> runs/<run_b> [...] [--window 120]
           [--labels "a;b;..."] [--min-samples 5] [--start 0] [--end END] [--out PATH]
"""
import argparse, csv, glob, json, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
RUN_COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#eda100"]
ROWS = (("engine_load_tokens", 1 / 1000, "uncached prefill tokens on the engine at arrival (K)"),
        ("queue_ms", 1 / 1000, "engine queue time (s)"),
        ("client_ttft_ms", 1 / 1000, "client TTFT (s)"))
COLS = (("mean", None), ("p90", 0.90), ("p95", 0.95))
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--labels", default=None, help="legend labels, one per run, separated by ';'")
ap.add_argument("--window", type=float, default=120.0)
ap.add_argument("--min-samples", type=int, default=5)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()

def stat(v, q):
    if q is None:
        return sum(v) / len(v)
    v = sorted(v)
    return v[min(len(v) - 1, int(q * len(v)))]

def load_run(R):
    plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
    t0 = plot_data["time_origin"]["profiling_start_ns"]
    evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
    end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
    resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
    curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
    start_s = {}
    for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
        for line in open(path):
            m = json.loads(line)["metadata"]
            if m.get("benchmark_phase") == "profiling":
                start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
    values = {col: defaultdict(list) for col, _, _ in ROWS}
    for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
        t = start_s.get(r["request_id"])
        if t is None or not a.start <= t < end:
            continue
        k = int((t - a.start) // a.window)
        for col, scale, _ in ROWS:
            if r.get(col) not in (None, ""):
                values[col][k].append(float(r[col]) * scale)
    auto = (resolved.get("deployment") or {}).get("autoscaling") or {}
    policy = (auto.get("policy") or {}).get("name", "?")
    maxrep = max((int(e["to_replicas"]) for e in evaluation["scale_events"]), default=int(auto.get("initial_replicas") or 1))
    return dict(name=evaluation["experiment"], end=end, curve=curve, values=values, events=evaluation["scale_events"],
                policy=policy, maxrep=maxrep, target=((evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000) / 1000)

runs = [load_run(R.rstrip("/")) for R in a.runs]
labels = a.labels.split(";") if a.labels else [f"{r['name']} ({r['policy']} policy, up to {r['maxrep']} replica{'s' if r['maxrep'] != 1 else ''})" for r in runs]
end = min(r["end"] for r in runs)
n_windows = int((end - a.start) / a.window) + (1 if (end - a.start) % a.window else 0)
centers = [a.start + (k + 0.5) * a.window for k in range(n_windows)]
curve = runs[0]["curve"]
target = runs[0]["target"]

fig, axes = plt.subplots(len(ROWS), len(COLS), figsize=(6.2 * len(COLS), 3.9 * len(ROWS)), dpi=150, sharex=True,
                         gridspec_kw={"hspace": 0.14, "wspace": 0.16})
fig.patch.set_facecolor(SURFACE)
for i, (col, _, ylabel) in enumerate(ROWS):
    for j, (cname, q) in enumerate(COLS):
        ax = axes[i][j]
        ax.set_facecolor(SURFACE)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        for s in ("left", "bottom"):
            ax.spines[s].set_color(GRID)
        ax.tick_params(colors=INK2, labelsize=8)
        ax.grid(axis="y", color=GRID, lw=0.5)
        ax.set_axisbelow(True)
        ycap = 0.0
        for ri, run in enumerate(runs):
            ys = [stat(run["values"][col][k], q) if len(run["values"][col].get(k, [])) >= a.min_samples else None for k in range(n_windows)]
            xs = [c for c, v in zip(centers, ys) if v is not None]
            ys = [v for v in ys if v is not None]
            if ys:
                ycap = max(ycap, max(ys))
            ax.plot(xs, ys, color=RUN_COLORS[ri], lw=1.7, marker="o", ms=3, zorder=3)
        ycap = max(ycap * 1.06, 1e-6)
        for p0, p1 in zip(curve, curve[1:]):
            phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
            lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
            if hi > lo:
                ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
                if i == 0:
                    ax.text((lo + hi) / 2, ycap * 1.12, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']}", ha="center", va="top", fontsize=7.5, color=INK2)
        # Mark when the new replica was actually serving (running_matched_at_s),
        # not when the decision was made.
        for ri, run in enumerate(runs):
            for e in run["events"]:
                if e["direction"] != "up":
                    continue
                t_up = e.get("running_matched_at_s") or e["time_s"]
                if a.start <= t_up <= end:
                    ax.axvline(t_up, color=RUN_COLORS[ri], lw=0.8, alpha=0.55, zorder=1)
                    if i == 0:
                        ax.text(t_up + 3, ycap * 0.99, f"{int(e['to_replicas'])} up", ha="left", va="top", fontsize=7, color=RUN_COLORS[ri], fontweight="bold")
        if col == "engine_load_tokens":
            ax.axhline(target, color=INK2, lw=1, ls=":")
        ax.set_ylim(0, ycap * 1.15)
        ax.set_xlim(a.start, end)
        if j == 0:
            ax.set_ylabel(ylabel, color=INK2, fontsize=8.5)
        if i == 0:
            ax.set_title(cname, loc="left", fontsize=10, color=INK, fontweight="bold", pad=4)
        if i == len(ROWS) - 1:
            ax.set_xlabel("seconds since profiling start", color=INK2, fontsize=8.5)
handles = [Line2D([], [], color=RUN_COLORS[ri], lw=1.7, marker="o", ms=3, label=lab) for ri, lab in enumerate(labels)]
handles.append(Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K (top row)"))
handles.append(Line2D([], [], color=INK2, lw=0.8, alpha=0.55, label="new replica running, in the run's colour (label: replicas now serving)"))
fig.legend(handles=handles, loc="lower center", fontsize=8.5, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.02))
fig.suptitle(f"all requests, per {a.window:.0f}s window {DOT} rows: load at arrival, engine queue time, client TTFT {DOT} columns: mean, p90, p95 {DOT} one line per run",
             x=0.01, y=0.995, ha="left", fontsize=10, color=INK)
fig.subplots_adjust(top=0.94, bottom=0.1, left=0.06, right=0.99)
out = a.out or f"runs/compare_arrival_stats{int(a.window)}_{'_vs_'.join(os.path.basename(R.rstrip('/'))[:16] for R in a.runs)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
