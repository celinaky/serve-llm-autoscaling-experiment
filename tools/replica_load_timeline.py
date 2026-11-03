#!/usr/bin/env python3
"""Uncached prefill token load per engine over time, with scale decisions.

Rebuilds each engine's load from its vLLM scheduler trace exactly as
load_breakdown.py does (+uncached tokens when a request is queued, -chunk
when the step that computed it completes, recompute added back on
preemption), converts the trace's monotonic clock to profiling-relative
seconds through the wall_time_s stamps, and draws one line per engine.
Vertical lines mark every change of the autoscaler's desired replica count
from analysis/autoscaling_evaluation.json.

usage: python tools/replica_load_timeline.py runs/<run_dir> \
           [--traces DIR] [--start 0] [--end 990] [--out analysis/replica_load.png]

Engines whose first request lands after --fold-after seconds are summed
into one grey series, so a burst of empty replicas does not eat the palette.
"""
import argparse, bisect, glob, json, math, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURFACE, INK, INK2, GRID, FOLD = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df", "#9b9a95"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--fold-after", type=float, default=None, help="sum engines first seen after this many seconds")
ap.add_argument("--window", type=float, default=0.0,
                help="plot the trailing mean over this many seconds instead of the raw step function (0 = raw)")
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"] / 1e9
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = (evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica")
load_end = evaluation["periods"]["steady"]["end_s"]
end = a.end if a.end is not None else load_end
manifest = json.load(open(f"{R}/manifest.json"))
run_start = __import__("datetime").datetime.fromisoformat(manifest["started_at"]).timestamp()

# Engine id from the trace file name: k-<cluster>_pid-<pid>.<part>.jsonl, parts of one engine share a pid.
engines = defaultdict(list)
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    if os.path.getmtime(path) < run_start:
        continue
    m = re.search(r"pid-(\d+)", os.path.basename(path))
    engines[m.group(1) if m else os.path.basename(path)].append(path)

series = {}  # engine -> (times, levels) step function in profiling seconds
first_seen = {}
for pid, paths in engines.items():
    offset = None  # wall - monotonic
    chunks, completed_at, queued, preempted = defaultdict(list), {}, {}, defaultdict(list)
    for path in sorted(paths):
        for line in open(path):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if offset is None and "wall_time_s" in e and "monotonic_s" in e:
                offset = e["wall_time_s"] - e["monotonic_s"]
            ev = e["event"]
            if ev == "batch_scheduled":
                for r in e["requests"]:
                    if r["phase"] == "prefill":
                        chunks[r["request_id"]].append((e["batch_index"], r["scheduled_tokens"]))
            elif ev == "batch_completed":
                completed_at[e["batch_index"]] = e["monotonic_s"]
            elif ev == "request_queued":
                queued[e["request_id"]] = e
            elif ev == "request_preempted":
                preempted[e["request_id"]].append((e["batch_index"], e["monotonic_s"]))
    if not queued or offset is None:
        continue
    deltas = []
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs):
            continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            deltas.append((at, sum(t for b, t in cs if batch <= b < until)))
        deltas.extend((completed_at[b], -t) for b, t in cs)
    if not deltas:
        continue
    deltas.sort()
    times, levels, level = [], [], 0
    for t, d in deltas:
        level += d
        times.append(t + offset - t0)
        levels.append(level)
    series[pid] = (times, levels)
    first_seen[pid] = min(q["monotonic_s"] for q in queued.values()) + offset - t0

order = sorted((p for p in series if first_seen[p] <= end), key=lambda p: first_seen[p])
if a.fold_after is None:
    a.fold_after = math.inf if len(order) <= len(PALETTE) else first_seen[order[len(PALETTE) - 1]]
named = [p for p in order if first_seen[p] <= a.fold_after]
folded = [p for p in order if first_seen[p] > a.fold_after]

STEP = 0.5


def sample(pid, grid):
    """Load in K tokens at each grid time; with --window, its trailing time mean."""
    times, levels = series[pid]
    if a.window <= 0:
        out = []
        for t in grid:
            i = bisect.bisect_right(times, t)
            out.append(levels[i - 1] / 1000 if i else 0.0)
        return out
    # Trailing mean needs history before the window start: sample from grid[0] - window.
    n = int(round(a.window / STEP))
    pre = [grid[0] - (n - k) * STEP for k in range(n)]
    fine = []
    for t in pre + grid:
        i = bisect.bisect_right(times, t)
        fine.append(levels[i - 1] / 1000 if i else 0.0)
    acc, out = sum(fine[:n]), []
    for k in range(len(grid)):
        acc += fine[n + k]
        out.append(acc / (n + 1))
        acc -= fine[k]
    return out

grid = [a.start + i * STEP for i in range(int((end - a.start) / STEP) + 1)]
fig, ax = plt.subplots(figsize=(14, 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for k, pid in enumerate(named):
    ys = sample(pid, grid)
    ax.step(grid, ys, where="post", lw=1.6, color=PALETTE[k % len(PALETTE)],
            label=f"replica {k + 1} (engine pid {pid}, first request {first_seen[pid]:.0f}s)")
if folded:
    tot = [0.0] * len(grid)
    for pid in folded:
        for i, y in enumerate(sample(pid, grid)):
            tot[i] += y
    ax.step(grid, tot, where="post", lw=1.6, color=FOLD, ls="--",
            label=f"{len(folded)} replicas first seen after {min(first_seen[p] for p in folded):.0f}s, summed")
if target:
    ax.axhline(target / 1000, color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K per replica")

# Shade the configured load phases: each straight segment of the session curve
# is a ramp up, a hold or a ramp down, from resolved.yaml.
import yaml
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ymax = ax.get_ylim()[1]
for p0, p1 in zip(curve, curve[1:]):
    if p1["sessions"] > p0["sessions"]:
        phase = "ramp up"
    elif p1["sessions"] < p0["sessions"]:
        phase = "ramp down"
    else:
        phase = "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi <= lo:
        continue
    ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
    ax.text((lo + hi) / 2, ymax * 0.995, f"{phase}: {p0['sessions']}\u2192{p1['sessions']} sessions",
            ha="center", va="top", fontsize=8, color=INK2)
# Session starts: the first profiling request of each conversation, root
# sessions and subagents separately (both start cold).
starts = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        m = rec["metadata"]
        if m.get("benchmark_phase") != "profiling":
            continue
        t = m["request_start_ns"] / 1e9 - t0
        cid = m["conversation_id"]
        if cid not in starts or t < starts[cid][0]:
            starts[cid] = (t, m.get("agent_depth", 0))
root_starts = sorted(t for t, d in starts.values() if d == 0 and a.start <= t <= end)
sub_starts = sorted(t for t, d in starts.values() if d > 0 and a.start <= t <= end)
tick_y = -ymax * 0.03
ax.plot(root_starts, [tick_y] * len(root_starts), marker="^", ls="none", ms=6, color=INK,
        label=f"root session start ({len(root_starts)})", clip_on=False)
ax.plot(sub_starts, [tick_y * 2.2] * len(sub_starts), marker="^", ls="none", ms=4, color=INK2, alpha=0.6,
        label=f"subagent start ({len(sub_starts)})", clip_on=False)
for e in evaluation["scale_events"]:
    t = e["time_s"]
    if not a.start <= t <= end:
        continue
    up = e["direction"] == "up"
    ax.axvline(t, color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
    ax.text(t + 3, ymax * 0.93, f"{e['from_replicas']:.0f}→{e['to_replicas']:.0f}", rotation=90,
            va="top", ha="left", fontsize=8, color=INK if up else INK2)

ax.set_xlim(a.start, end)
ax.set_ylim(bottom=tick_y * 3.2)
ax.set_xlabel("seconds since profiling start", color=INK2)
mean_tag = "%.0fs trailing mean" % a.window if a.window else ""
ax.set_ylabel((mean_tag + " of " if mean_tag else "") + "uncached prefill tokens outstanding on the engine (K)", color=INK2)
ax.set_title(f"{evaluation['experiment']} · uncached prefill load per replica{", " + mean_tag if mean_tag else ""}, {a.start:.0f}–{end:.0f}s "
             f"· vertical lines: desired-replica changes (solid up, dashed down)", loc="left", fontsize=10, color=INK)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2)
ax.grid(axis="y", color=GRID, lw=0.6)
ax.set_axisbelow(True)
ax.legend(loc="upper center", fontsize=8, frameon=False, ncol=3, bbox_to_anchor=(0.5, -0.14))
suffix = "_mean%d" % a.window if a.window else ""
out = a.out or f"{R}/analysis/replica_load{suffix}_{int(a.start)}_{int(end)}.png"
fig.tight_layout()
fig.savefig(out, facecolor=SURFACE)
print("wrote", out)
for k, pid in enumerate(named):
    ys = sample(pid, grid)
    print(f"replica {k + 1} pid {pid}: first {first_seen[pid]:.0f}s, mean load {sum(ys) / len(ys):.1f}K, "
          f"share of time at 0: {sum(y == 0 for y in ys) / len(ys):.0%}, max {max(ys):.0f}K")
