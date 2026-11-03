#!/usr/bin/env python3
"""What the mean policy observed vs what the engines held, over time.

Two accountings of the same quantity, total outstanding uncached prefill
tokens across all replicas:
  policy: the router gauge the policy scales on (a request's whole uncached
          prompt counts from routing until its first token), from the run's
          telemetry, and the policy's own published lookback mean of it
          (prefill_required_replicas x target).
  engine: rebuilt from the vLLM scheduler traces: a request's uncached tokens
          enter when it is queued and each chunk leaves when its batch
          completes, so the total is the work still to do. Same lookback mean.
Faint lines are the raw signals, solid lines the lookback means. Target
dotted; scale decisions as in the other charts.

usage: python tools/policy_vs_engine_mean.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH]
"""
import argparse, bisect, datetime, glob, json, math, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
POLICY, ENGINE = "#6250d6", "#1baf7a"
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lookback", type=float, default=None, help="mean window; default: the policy's upscale_lookback_s")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--out", default=None)
ap.add_argument("--raw", action="store_true", help="also draw the raw gauge and raw engine total under the means")
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
target = (kwargs.get("target_prefill_tokens_per_replica") or 20000) / 1000
lookback = a.lookback or kwargs.get("upscale_lookback_s") or 60
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

# Policy side: router gauge and the policy's published mean, from telemetry.
raw, pub, asks = [], [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0 / 1e9
    if not a.start <= t <= end:
        continue
    d = {}
    for s in rec["samples"]:
        if "LLMServer" not in s["labels"].get("deployment", ""):
            continue
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            d[s["labels"]["metric"]] = s["value"]
        elif s["name"] == "ray_serve_autoscaling_desired_replicas":
            d["desired"] = s["value"]
    if "required_replicas" in d and "desired" in d:
        asks.append((t, d["required_replicas"], d["desired"]))
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
    return out, times, levels

# Engine side from the scheduler traces of this run.
manifest = json.load(open(f"{R}/manifest.json"))
run_start = datetime.datetime.fromisoformat(manifest["started_at"]).timestamp()
deltas = []
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    if os.path.getmtime(path) < run_start:
        continue
    chunks, completed_at, queued, preempted = {}, {}, {}, {}
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
        elif ev == "request_preempted":
            preempted.setdefault(e["request_id"], []).append((e["batch_index"], e["monotonic_s"]))
    if not queued or offset is None:
        continue
    rel = lambda mono: mono + offset - t0 / 1e9
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs):
            continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            deltas.append((rel(at), sum(t for b, t in cs if batch <= b < until) / 1000))
        deltas.extend((rel(completed_at[b]), -t / 1000) for b, t in cs)
engine_mean, eng_times, eng_levels = step_mean(deltas, lookback, ts)
# Raw engine step, thinned to the plot range.
eng_raw = [(t, v) for t, v in zip(eng_times, eng_levels) if a.start <= t <= end]

# Policy-side mean recomputed from the raw gauge the same way, for runs whose
# policy did not publish one (p95 branch) or to confirm the published value.
raw_deltas, prev = [], 0.0
for t, v in raw:
    raw_deltas.append((t, v - prev)); prev = v
policy_mean = step_mean(raw_deltas, lookback, ts)[0]
pub_t = [t for t, _ in pub]
pub_v = [v for _, v in pub]

# Upscale-delay timer: Serve adds replicas only once the policy has asked for
# more than the current desired count for upscale_delay_s without a break.
# Rebuilt from the scrapes: an interval of consecutive asks above desired. It
# "passed" if a scale-up decision follows it directly, else it reset.
auto = (resolved.get("deployment") or {}).get("autoscaling") or {}
up_delay = float(auto.get("upscale_delay_s") or 0)
ups = sorted(e["time_s"] for e in evaluation["scale_events"] if e["direction"] == "up")
timers, cur = [], None
for i, (t, req, des) in enumerate(asks):
    on = req > des
    if on and cur is None:
        cur = [t, t]
    elif on:
        cur[1] = t
    elif cur is not None:
        timers.append(cur); cur = None
if cur is not None:
    timers.append(cur)
timer_spans = []
for t1, t2 in timers:
    gap = asks[1][0] - asks[0][0] if len(asks) > 1 else 2.0
    passed = [u for u in ups if t2 <= u <= t2 + 2 * gap + 1]
    timer_spans.append((t1, passed[0] if passed else t2, bool(passed)))

peak = max(([v for _, v in raw] + [v for _, v in eng_raw]) if a.raw else (pub_v or policy_mean) + engine_mean) if (raw or eng_raw) else 1.0
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
        ax.text((lo + hi) / 2, ycap * 1.09, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    if a.start <= e["time_s"] <= end:
        ready = e.get("running_matched_at_s")
        ax.axvline(e["time_s"], color=INK, lw=0.8, alpha=0.6)
        if ready is not None and ready > e["time_s"]:
            ax.axvspan(e["time_s"], min(ready, end), color=INK2, alpha=0.08, lw=0, zorder=0.5)
            ax.axvline(min(ready, end), color=INK2, lw=0.8, ls="--", alpha=0.6)
        ax.text(e["time_s"] + 4, ycap * 1.02, f"{int(e['from_replicas'])}{ARROW}{int(e['to_replicas'])}", ha="left", va="bottom", fontsize=8, color=INK, fontweight="bold")

if a.raw:
    ax.step([t for t, _ in raw], [min(v, ycap) for _, v in raw], where="post", color=POLICY, lw=0.7, alpha=0.3)
    ax.step([t for t, _ in eng_raw], [min(v, ycap) for _, v in eng_raw], where="post", color=ENGINE, lw=0.7, alpha=0.3)
if pub:
    ax.plot(pub_t, [min(v, ycap) for v in pub_v], color=POLICY, lw=1.8)
else:
    ax.plot(ts, [min(v, ycap) for v in policy_mean], color=POLICY, lw=1.8)
ax.plot(ts, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)
ax.axhline(target, color=INK2, lw=1, ls=":")
TIMER_PASS, TIMER_RESET = "#c9480f", "#e8b89a"
for t1, t2, passed in timer_spans:
    if t2 < a.start or t1 > end:
        continue
    ax.axvspan(max(t1, a.start), min(t2, end), ymin=0.0, ymax=0.035, color=TIMER_PASS if passed else TIMER_RESET, lw=0, zorder=2)
    if passed:
        ax.annotate("", xy=(t2, -ycap * 0.055), xytext=(t2 - 1, -ycap * 0.055), arrowprops=dict(arrowstyle="-|>", color=TIMER_PASS, lw=0.8, mutation_scale=7), annotation_clip=False)
    elif t2 - t1 >= 8:
        ax.text((t1 + t2) / 2, -ycap * 0.055, f"{t2 - t1:.0f}s", ha="center", va="top", fontsize=6.5, color=TIMER_PASS)
ax.set_ylim(-ycap * 0.06, ycap * 1.12)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel("total outstanding uncached prefill tokens, all replicas (K)", color=INK2)
handles = [Line2D([], [], color=POLICY, lw=1.8, label=f"policy: {lookback:.0f}s mean of the router gauge (the number the policy divides by the target)"),
           Line2D([], [], color=ENGINE, lw=1.8, label=f"engine: {lookback:.0f}s mean of the engines' remaining uncached prefill (traces)"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica"),
           Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"),
           Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)"),
           Patch(color=TIMER_PASS, label=f"on the x axis: upscale-delay timer running and passed ({up_delay:.0f}s of asking for more), ends at the decision"),
           Patch(color=TIMER_RESET, label="upscale-delay timer running but reset (the ask dropped back before the delay elapsed)")]
if a.raw:
    handles[1:1] = [Line2D([], [], color=POLICY, lw=0.7, alpha=0.4, label="policy: router gauge as scraped, every 2s (whole prompt counted until first token)")]
    handles[3:3] = [Line2D([], [], color=ENGINE, lw=0.7, alpha=0.4, label="engine: remaining uncached prefill as it changes, chunks leave as batches complete")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
sec = lambda v: f"{float(v):g}s" if v is not None else "?"
settings = (f"policy: ceil({lookback:.0f}s mean / {target:.0f}K) {DOT} Serve upscale delay {sec(auto.get('upscale_delay_s'))} {DOT} downscale delay {sec(auto.get('downscale_delay_s'))}"
            f" {DOT} replicas {auto.get('min_replicas')}-{auto.get('max_replicas')} {DOT} metrics every {sec(auto.get('metrics_interval_s'))}")
fig.text(0.5, -0.19, settings, ha="center", va="top", fontsize=8.5, color=INK2)
fig.suptitle(f"{evaluation['experiment']} {DOT} what the mean policy observed vs what the engines held: total outstanding uncached prefill tokens",
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/policy_vs_engine_mean{int(lookback)}{'_raw' if a.raw else ''}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)

# Summary: ratio of the two means where the engine mean is meaningful.
pairs = [(p, e) for p, e in zip(policy_mean, engine_mean) if e >= 1.0]
if pairs:
    ratios = sorted(p / e for p, e in pairs)
    print(f"  policy mean / engine mean where engine mean >= 1K: median {ratios[len(ratios) // 2]:.2f}, p10 {ratios[int(0.1 * len(ratios))]:.2f}, p90 {ratios[int(0.9 * len(ratios))]:.2f} (n={len(ratios)})")
    print(f"  time-average over the load phase: policy {sum(policy_mean) / len(policy_mean):.1f}K, engine {sum(engine_mean) / len(engine_mean):.1f}K")
    if pub:
        diffs = [abs(p - q) for p, q in zip(policy_mean, [pub_v[bisect.bisect_right(pub_t, t) - 1] for t in ts]) if True]
        print(f"  published policy mean vs recomputed from the gauge: max abs diff {max(diffs):.1f}K")
