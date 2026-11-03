#!/usr/bin/env python3
"""What the mean policy observed vs what the engines held, with the accounting
difference isolated.

Three lookback means of the same quantity, total outstanding uncached prefill
tokens across all replicas:
  policy:  the router gauge's lookback mean as the policy published it
           (prefill_required_replicas x target), from the run's telemetry.
  rebuild: the router's accounting rebuilt from the vLLM scheduler traces with
           the engine's own uncached counts: a request's whole uncached prompt
           enters when the engine queues it and leaves at its first token.
  engine:  the engine's accounting from the same traces: the same tokens enter
           at the same time but each prefill chunk leaves when its batch
           completes, so the total is the work still to do.
rebuild - engine is the cost of holding the whole prompt instead of burning it
down; policy - rebuild is everything else the router does differently (hold
timing, its own uncached estimate).

usage: python tools/policy_vs_engine_rebuild.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH] [--hold-ms 0] [--shift 0]
--shift moves the two trace-side lines right by that many seconds: the policy
metric is exported through Ray's metrics agent every 10s, so the published
mean trails the traces by roughly 4-8s.
"""
import argparse, bisect, datetime, glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
POLICY, REBUILD, ENGINE = "#6250d6", "#c9480f", "#1baf7a"
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lookback", type=float, default=None, help="mean window; default: the policy's upscale_lookback_s")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--out", default=None)
ap.add_argument("--hold-ms", type=float, default=0.0, help="extra hold per request in the rebuild, after its first token")
ap.add_argument("--shift", type=float, default=0.0, help="seconds to move the trace-side lines right, to line up with the policy's export lag")
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"] / 1e9
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
target = (kwargs.get("target_prefill_tokens_per_replica") or 20000) / 1000
lookback = a.lookback or kwargs.get("upscale_lookback_s") or 60
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
manifest = json.load(open(f"{R}/manifest.json"))
run_start = datetime.datetime.fromisoformat(manifest["started_at"]).timestamp()
run_end = datetime.datetime.fromisoformat(manifest["finished_at"]).timestamp()

# Policy side: the published lookback mean, with the raw gauge as a fallback.
raw, pub = [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    if not a.start <= t <= end:
        continue
    d = {}
    for s in rec["samples"]:
        if "LLMServer" not in s["labels"].get("deployment", ""):
            continue
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            d[s["labels"]["metric"]] = s["value"]
    if "outstanding_effective_prefill_tokens" in d:
        raw.append((t, d["outstanding_effective_prefill_tokens"] / 1000))
    if "prefill_required_replicas" in d:
        pub.append((t, d["prefill_required_replicas"] * target))
ts = [t for t, _ in raw]

def step_mean(deltas, window, ts):
    """Time-weighted mean over the trailing window of a step function given as (t, delta)."""
    deltas.sort()
    times = [t for t, _ in deltas]
    levels, level = [], 0
    for _, d in deltas:
        level += d
        levels.append(level)
    def at(t):
        i = bisect.bisect_right(times, t) - 1
        return levels[i] if i >= 0 else 0
    out = []
    for t in ts:
        w0 = t - window
        i0 = bisect.bisect_right(times, w0)
        pts = [(w0, at(w0))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, t))]
        total = 0.0
        for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(t, None)]):
            if t2 > t1:
                total += v1 * (t2 - t1)
        out.append(total / window)
    return out

# Both trace-side accountings from this run's scheduler traces.
eng_deltas, reb_deltas = [], []
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    mtime = os.path.getmtime(path)
    if mtime < run_start or mtime > run_end + 600:
        continue
    chunks, completed_at, queued, first, preempted = {}, {}, {}, {}, {}
    offset = None
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
                    chunks.setdefault(r["request_id"], []).append((e["batch_index"], r["scheduled_tokens"]))
        elif ev == "batch_completed":
            completed_at[e["batch_index"]] = e["monotonic_s"]
        elif ev == "request_queued":
            queued[e["request_id"]] = e
        elif ev == "request_first_token":
            first[e["request_id"]] = e
        elif ev == "request_preempted":
            preempted.setdefault(e["request_id"], []).append((e["batch_index"], e["monotonic_s"]))
    if not queued or offset is None:
        continue
    rel = lambda mono: mono + offset - t0
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs):
            continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            eng_deltas.append((rel(at), sum(t for b, t in cs if batch <= b < until) / 1000))
        eng_deltas.extend((rel(completed_at[b]), -t / 1000) for b, t in cs)
        f = first.get(rid)
        if f is None:
            continue
        # Router accounting: the uncached prompt (its chunks up to the first
        # token) held whole from queueing until the first token.
        own = sum(t for b, t in cs if b <= f["batch_index"]) / 1000
        reb_deltas.append((rel(q["monotonic_s"]), own))
        reb_deltas.append((rel(f["monotonic_s"]) + a.hold_ms / 1000, -own))
engine_mean = step_mean(eng_deltas, lookback, ts)
rebuild_mean = step_mean(reb_deltas, lookback, ts)

if pub:
    pol_t, pol_v, pol_label = [t for t, _ in pub], [v for _, v in pub], "as published by the policy"
else:
    raw_deltas, prev = [], 0.0
    for t, v in raw:
        raw_deltas.append((t, v - prev)); prev = v
    pol_t, pol_v, pol_label = ts, step_mean(raw_deltas, lookback, ts), "recomputed from the raw gauge scrapes"

peak = max(pol_v + rebuild_mean + engine_mean) if pol_v else 1.0
ycap = a.ymax if a.ymax else max(min(peak * 1.05, 12 * target), 1.5 * target)

fig, ax = plt.subplots(figsize=(14, 5.8), dpi=150)
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
        ax.text((lo + hi) / 2, ycap * 1.07, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)

ax.plot(pol_t, [min(v, ycap) for v in pol_v], color=POLICY, lw=1.8)
tsh = [t + a.shift for t in ts]
ax.plot(tsh, [min(v, ycap) for v in rebuild_mean], color=REBUILD, lw=1.8, ls=(0, (4, 2)))
ax.plot(tsh, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)
ax.axhline(target, color=INK2, lw=1, ls=":")
ax.set_ylim(0, ycap * 1.1)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel("total outstanding uncached prefill tokens, all replicas (K)", color=INK2)

def avg(v):
    return sum(v) / len(v) if v else float("nan")
m_pol, m_reb, m_eng = avg(pol_v), avg(rebuild_mean), avg(engine_mean)
hold_note = f", +{a.hold_ms:.0f} ms hold" if a.hold_ms else ""
shift_note = f" {DOT} trace-side lines shifted right {a.shift:g}s to match the policy metric's export lag" if a.shift else ""
handles = [Line2D([], [], color=POLICY, lw=1.8, label=f"policy: {lookback:.0f}s mean of the router gauge, {pol_label} (load-phase average {m_pol:.1f}K)"),
           Line2D([], [], color=REBUILD, lw=1.8, ls=(0, (4, 2)), label=f"rebuild: router accounting from traces, whole uncached prompt held from queueing to first token{hold_note} ({m_reb:.1f}K)"),
           Line2D([], [], color=ENGINE, lw=1.8, label=f"engine: remaining uncached prefill from traces, chunks leave as batches complete ({m_eng:.1f}K){shift_note}"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.17))
fig.suptitle(f"{evaluation['experiment']} {DOT} policy vs engine, with the router's whole-prompt accounting rebuilt from the engine's own numbers",
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/policy_vs_engine_rebuild{int(lookback)}{f'_shift{a.shift:g}' if a.shift else ''}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
print(f"  load-phase averages: policy {m_pol:.1f}K  rebuild {m_reb:.1f}K  engine {m_eng:.1f}K")
print(f"  rebuild - engine (whole prompt held vs chunk burn-down): {m_reb - m_eng:.1f}K; policy - rebuild (hold timing, router estimate): {m_pol - m_reb:.1f}K")
