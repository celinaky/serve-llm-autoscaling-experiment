#!/usr/bin/env python3
"""Which over-target arrivals were inevitable: one cold prompt, or a stack?

From the vLLM scheduler traces, rebuild for every engine the remaining
uncached prefill tokens of each in-flight request over time (+uncached when
queued, -chunk when the step that computed it completes, recompute re-added on
preemption). For every successful AIPerf request, look at the other requests
still ahead of it on its engine when it was queued and classify it:

  under target     total ahead at or under the target
  one cold prompt  the largest single request ahead alone exceeds the target:
                   a cold prefill cannot be split, so this wait could only be
                   avoided by giving that cold prompt a replica of its own
  stacked          no single request ahead exceeds the target, but together
                   they do: spreading those requests over more replicas would
                   have brought this arrival under target

Per window (fixed count or fixed time), a stacked bar of the share of
arrivals in each class, against the 5% line (above it the window's p95 is
over target). Writes analysis/collision_causes.csv with one row per request.

usage: python tools/arrival_collision_causes.py runs/<run_dir> [--traces DIR]
           [--per-requests 100 | --window 60] [--start 0] [--end END] [--out PATH]
"""
import argparse, bisect, csv, glob, json, math, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
from matplotlib.patches import Patch
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
UNDER, COLD, STACK = "#dfe7f3", "#eb6834", "#6250d6"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT = "→", "·"
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--per-requests", type=int, default=100)
ap.add_argument("--window", type=float, default=None, help="use fixed time windows of this many seconds instead")
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--simple", action="store_true", help="two classes only: behind one cold prompt that alone exceeds the target, or behind a pile of smaller requests")
ap.add_argument("--detail", action="store_true", help="split the avoidable waits into new-session placement and eviction, and flag idle replicas")
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target_k = ((evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000) / 1000
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
manifest = json.load(open(f"{R}/manifest.json"))
run_start = __import__("datetime").datetime.fromisoformat(manifest["started_at"]).timestamp()

aiperf = {}   # x_request_id -> (t_rel, conversation_id, agent_depth, turn_index)
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        m = rec["metadata"]
        if m.get("benchmark_phase") == "profiling" and not rec.get("error"):
            aiperf[m["x_request_id"]] = ((m["request_start_ns"] - t0) / 1e9, m["conversation_id"], m.get("agent_depth", 0), m["turn_index"])
engine_of = {}  # x_request_id -> engine file tag, to detect a conversation that moved engines
# Engine-side load at arrival per request, grouped by engine, for the "was the
# replica meeting the target when this session was placed on it" test.
arrivals_by_engine = defaultdict(list)  # tag -> sorted (t_rel, load_k)
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    if r["request_id"] in aiperf:
        tag_ = (re.search(r"pid-(\d+)", r["engine"]) or [None, r["engine"]])[1]
        arrivals_by_engine[tag_].append((aiperf[r["request_id"]][0], float(r["engine_load_tokens"]) / 1000))
for v in arrivals_by_engine.values():
    v.sort()
ap_window = ((evaluation.get("autoscaling_policy_kwargs") or {}).get("upscale_lookback_s") or 120)

def replica_p95_before(tag_, t):
    v = arrivals_by_engine.get(tag_, [])
    lo = bisect.bisect_left(v, (t - ap_window, -1.0)); hi = bisect.bisect_left(v, (t, -1.0))
    loads = sorted(y for _, y in v[lo:hi])
    if len(loads) < 10:
        return None
    return loads[min(len(loads) - 1, int(0.95 * len(loads)))]

rows = []  # (t, class, total_ahead_K, max_ahead_K, n_ahead, biggest_key, engine)
engines = {}  # engine tag -> (wall offset, remaining dict)
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    if os.path.getmtime(path) < run_start:
        continue
    tag = (re.search(r"pid-(\d+)", os.path.basename(path)) or [None, os.path.basename(path)])[1]
    chunks, completed_at, queued, preempted = defaultdict(list), {}, {}, defaultdict(list)
    offset = None
    for line in open(path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = e["event"]
        if offset is None and "wall_time_s" in e and "monotonic_s" in e:
            offset = e["wall_time_s"] - e["monotonic_s"]
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
    # Per request: a step function of remaining uncached tokens over monotonic time.
    remaining = {}  # rid -> (times, levels, t_queued, t_done)
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs):
            continue
        deltas = []
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            deltas.append((at, sum(t for b, t in cs if batch <= b < until)))
        deltas.extend((completed_at[b], -t) for b, t in cs)
        deltas.sort()
        times, levels, level = [], [], 0
        for t, d in deltas:
            level += d
            times.append(t); levels.append(level)
        remaining[rid] = (times, levels, q["monotonic_s"], times[-1])
    by_queue = sorted((v[2], rid) for rid, v in remaining.items())
    queue_times = [t for t, _ in by_queue]
    engines.setdefault(tag, [])
    engines[tag] = (offset, remaining)
    for rid in remaining:
        key = ENGINE_ID.match(rid)
        engine_of[key.group(1) if key else rid] = tag

    def remaining_at(rid, t):
        times, levels, _, _ = remaining[rid]
        i = bisect.bisect_right(times, t) - 1
        return levels[i] if i >= 0 else 0

    for rid, (times, levels, tq, tdone) in remaining.items():
        key = ENGINE_ID.match(rid)
        key = key.group(1) if key else rid
        if key not in aiperf:
            continue
        t_rel = aiperf[key][0]
        if not a.start <= t_rel < end:
            continue
        # Others queued strictly before this request and not yet done at tq.
        ahead = []
        for i in range(bisect.bisect_left(queue_times, tq)):
            other = by_queue[i][1]
            if remaining[other][3] > tq:
                rem = remaining_at(other, tq)
                if rem > 0:
                    ahead.append((rem / 1000, other))
        total = sum(x for x, _ in ahead)
        biggest, biggest_rid = max(ahead, default=(0.0, None))
        cls = "under" if total <= target_k else ("cold" if biggest > target_k else "stacked")
        bkey = ENGINE_ID.match(biggest_rid).group(1) if biggest_rid and ENGINE_ID.match(biggest_rid) else biggest_rid
        rows.append((t_rel, cls, total, biggest, len(ahead), bkey, tag, tq + offset))

# Sub-classify the "cold" arrivals by what the cold prompt ahead was and whether
# an idle replica existed at that moment.
conv_prev_engine = {}
for key, (t_rel, conv, depth, turn) in sorted(aiperf.items(), key=lambda kv: kv[1][0]):
    pass
prev_engine_of_conv = {}
moved = set()
for key, (t_rel, conv, depth, turn) in sorted(aiperf.items(), key=lambda kv: kv[1][0]):
    eng = engine_of.get(key)
    if eng is None:
        continue
    if conv in prev_engine_of_conv and prev_engine_of_conv[conv] != eng:
        moved.add(key)
    prev_engine_of_conv[conv] = eng

def idle_replica_elsewhere(wall, own_tag):
    for tag, (offset, remaining) in engines.items():
        if tag == own_tag:
            continue
        mono = wall - offset
        busy = False
        for rid, (times, levels, tq_, tdone) in remaining.items():
            if tq_ <= mono < tdone:
                i = bisect.bisect_right(times, mono) - 1
                if i >= 0 and levels[i] > 0:
                    busy = True
                    break
        if not busy and any(tq_ <= mono for _, (_, _, tq_, _) in remaining.items()):
            return True
    return False

# When did each conversation first land on each engine?
placed_at = {}
for key, (t_rel, conv, depth, turn) in aiperf.items():
    eng = engine_of.get(key)
    if eng is not None and (conv, eng) not in placed_at or (eng is not None and t_rel < placed_at[(conv, eng)]):
        placed_at[(conv, eng)] = t_rel
key_by_rid = {}
classified = []
for t_rel, cls, total, biggest, n_ahead, bkey, tag, wall in rows:
    sub = cls
    if cls == "cold":
        meta = aiperf.get(bkey)
        if bkey in moved:
            sub = "cold-moved"
        elif a.detail:
            if meta is not None and meta[3] == 0:
                sub = "cold-new-idle" if idle_replica_elsewhere(wall, tag) else "cold-new"
            else:
                sub = "cold-pinned"
        else:
            # The later-placed of the two colliding sessions: was this replica
            # meeting the target when it was placed here?
            waiter_key = next((k for k, m in aiperf.items() if m[0] == t_rel), None)
            waiter_conv = aiperf[waiter_key][1] if waiter_key else None
            cold_conv = meta[1] if meta else None
            placements = [placed_at.get((c, tag)) for c in (waiter_conv, cold_conv) if c is not None]
            placements = [x for x in placements if x is not None]
            later = max(placements) if placements else t_rel
            p95_then = replica_p95_before(tag, later)
            if p95_then is None or p95_then <= target_k:
                sub = "within-target"   # the replica was meeting the target when it took this session on
            else:
                sub = "over-target"     # the replica was already over target; the session needed somewhere else to go
    classified.append((t_rel, sub, total, biggest, n_ahead))
rows = sorted(classified)
if a.simple:
    rows = [(t, ("one-cold" if c not in ("under", "stacked") else c), *rest) for t, c, *rest in rows]
with open(f"{R}/analysis/collision_causes.csv", "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["t_s", "class", "total_ahead_k", "largest_ahead_k", "requests_ahead"])
    w.writerows(rows)

spans = []
if a.window:
    k = 0
    while a.start + k * a.window < end:
        ws, we = a.start + k * a.window, min(a.start + (k + 1) * a.window, end)
        spans.append((ws, we, [r for r in rows if ws <= r[0] < we])); k += 1
    window_label = f"{a.window:.0f}s window"
else:
    for i in range(0, len(rows), a.per_requests):
        chunk = rows[i:i + a.per_requests]
        we = rows[i + a.per_requests][0] if i + a.per_requests < len(rows) else end
        spans.append((chunk[0][0], we, chunk))
    window_label = f"{a.per_requests}-request window"

if a.simple:
    CLASSES = [
        ("one-cold", "#b4321c", f"behind one cold prompt that alone had more than {target_k:.0f}K left: cannot be split, so unavoidable without a replica per cold start"),
        ("stacked", "#6250d6", f"behind several requests, none over {target_k:.0f}K alone: spreading them over more replicas would have avoided it"),
    ]
elif a.detail:
    CLASSES = [
        ("cold-pinned", "#b4321c", "unavoidable: behind the cold turn of a session already pinned to this replica"),
        ("cold-new", "#eb6834", "avoidable: behind a new session's first prompt, no idle replica existed (scale earlier)"),
        ("cold-new-idle", "#eda100", "avoidable: behind a new session's first prompt while an idle replica existed (router choice)"),
        ("cold-moved", "#e87ba4", "avoidable: behind an evicted session re-prefilling (scale-down)"),
        ("stacked", "#6250d6", "avoidable: behind several smaller requests (spreading)"),
    ]
else:
    CLASSES = [
        ("within-target", "#b4321c", f"cost of the target: the replica was meeting {target_k:.0f}K at p95 when the later of the two sessions was placed on it"),
        ("over-target", "#eda100", f"autoscaler late: the replica was already over {target_k:.0f}K at p95 when the later session was placed on it"),
        ("cold-moved", "#e87ba4", "scale-down: the cold prompt was an evicted session re-prefilling"),
        ("stacked", "#6250d6", "several smaller requests ahead, none over the target alone"),
    ]
n_all = len(rows)
counts = {c: sum(1 for r in rows if r[1] == c) for c, _, _ in CLASSES}
counts["under"] = sum(1 for r in rows if r[1] == "under")
print(f"{n_all} requests placed, {counts['under'] / n_all:.1%} under target; over target by cause:")
for c, _, label in CLASSES:
    print(f"  {counts[c]:4d} ({counts[c] / n_all:5.1%})  {label}")

fig, ax = plt.subplots(figsize=(max(14, len(spans) * 0.5), 6.2), dpi=150)
fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
for s in ("left", "bottom"):
    ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2); ax.set_axisbelow(True); ax.grid(axis="y", color=GRID, lw=0.5)
ytop = max(0.25, max((sum(1 for r in v if r[1] != "under") / len(v) for _, _, v in spans if v), default=0) * 1.25)
for ws, we, v in spans:
    n = len(v)
    if not n:
        continue
    mid, span = ws + (we - ws) / 2, we - ws
    base = 0.0
    for c, col, _ in CLASSES:
        share = sum(1 for r in v if r[1] == c) / n
        if share:
            ax.bar(mid, share, bottom=base, width=span * 0.9, color=col, lw=0)
        base += share
    ax.text(mid, base + ytop * 0.01, f"{base:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK if base > 0.05 else INK2)
    ax.text(mid, -ytop * 0.015, str(n), ha="center", va="top", fontsize=6.5, color=INK2)
ax.axhline(0.05, color=INK, lw=1, ls=":")
ax.text(a.start + 2, 0.052, "5%: above this the window's p95 is over target", fontsize=7.5, color=INK, va="bottom")
ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax.set_ylim(-ytop * 0.06, ytop * 1.1)
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi > lo:
        ax.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)
        ax.text((lo + hi) / 2, ytop * 1.095, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    if a.start <= e["time_s"] <= end:
        up = e["direction"] == "up"
        ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.9, ls="-" if up else "--", alpha=0.8)
        ax.text(e["time_s"] + 3, ytop * 1.08, f"{e['from_replicas']:.0f}{ARROW}{e['to_replicas']:.0f}", rotation=90, va="top", ha="left",
                fontsize=7.5, color=INK if up else INK2)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel(f"share of the window's requests that found more than {target_k:.0f}K ahead", color=INK2)
ax.set_title(f"{evaluation['experiment']} {DOT} per {window_label}: arrivals that found more than {target_k:.0f}K ahead, by cause {DOT} "
             f"whole run: {counts['under'] / n_all:.1%} under target, " + ", ".join(f"{counts[c] / n_all:.1%} {c}" for c, _, _ in CLASSES if counts[c]),
             loc="left", fontsize=9, color=INK)
handles = [Patch(color=col, label=label) for c, col, label in CLASSES]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=1 if not a.detail else 2, bbox_to_anchor=(0.5, -0.08 if a.detail else -0.08))
tag = (f"{int(a.window)}" if a.window else f"req{a.per_requests}") + ("_detail" if a.detail else "_simple" if a.simple else "")
out = a.out or f"{R}/analysis/collision_causes{tag}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
