#!/usr/bin/env python3
"""What the mean policy observed vs what the engines held, over time.

Two accountings of the same quantity, total outstanding uncached prefill
tokens across all replicas:
  policy: the policy's lookback mean: the max across KV router replicas of
          the load they report (serve_autoscaling_replica_metric; on runs
          from older Ray branches serve_autoscaling_router_metric, or the
          published prefill_required_replicas x target). The raw gauge
          reconstruction is used only when that published series is absent.
  engine: rebuilt from the vLLM scheduler traces: a request's uncached tokens
          enter when it is queued and each chunk leaves when its batch
          completes, so the total is the work still to do. Same lookback mean.
Faint lines are the raw signals, solid lines the lookback means. Target
dotted; scale decisions as in the other charts.

Run by ``autoscale-harness analyze``; by hand:
    python -m serve_llm_autoscaling.diagnostics.policy_vs_engine_mean runs/<run_dir> [--lookback 60] [--ymax K]
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
ap.add_argument("--lookback", type=float, default=None, help="mean window; default: the policy's upscale_lookback_s, else Serve's look_back_period_s")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--traces", default=None, help="scheduler trace directory; default: the run's analysis.scheduler_trace_dir, else <run>/telemetry/scheduler-traces")
ap.add_argument("--out", default=None)
ap.add_argument("--no-engine", action="store_true", help="omit the engine-side line (policy signal only)")
ap.add_argument("--no-timers", action="store_true", help="omit the upscale/downscale delay-timer bars on the x axis")
ap.add_argument("--raw", action="store_true", help="also draw the raw gauge and raw engine total under the means")
ap.add_argument("--mark-evictions", action="store_true", help="red ticks where a returning session arrived with its context evicted (turn >= 2, engine found <10%% cached, prompt not shorter than its previous turn)")
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
target = (kwargs.get("target_prefill_tokens_per_replica") or 20000) / 1000
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
# Older policies took their own window; the controller now averages over
# Serve's look_back_period_s.
lookback = (a.lookback or kwargs.get("upscale_lookback_s")
            or ((resolved.get("deployment") or {}).get("autoscaling") or {}).get("look_back_period_s")
            or 60)
a.traces = (a.traces or ((resolved.get("analysis") or {}).get("scheduler_trace_dir"))
            or f"{R}/telemetry/scheduler-traces")
from ..analysis import router_report
from .curve import phase_points
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
curve = phase_points(curve)

# Policy side: router gauge and the policy's published mean, from telemetry.
raw, pub, asks, scrape_ts = [], [], [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0 / 1e9
    if not a.start <= t <= end:
        continue
    scrape_ts.append(t)
    d = {}
    for s in rec["samples"]:
        s = router_report(s)  # one shape for the router's report across Ray versions
        if s.get("value") is None or "LLMServer" not in s["labels"].get("deployment", ""):
            continue
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            # Older branches: generic policy metric gauge, tagged by metric name.
            d[s["labels"]["metric"]] = s["value"]
        elif s["name"] == "ray_serve_autoscaling_desired_replicas":
            d["desired"] = s["value"]
        # Token policy gauges (PR 4): raw tokens, the lookback mean, the target.
        elif s["name"] == "ray_serve_llm_autoscaling_outstanding_prefill_tokens":
            d["outstanding_effective_prefill_tokens"] = s["value"]
        elif s["name"] == "ray_serve_llm_autoscaling_prefill_tokens_lookback_mean":
            d["lookback_mean_tokens"] = s["value"]
        elif s["name"] == "ray_serve_llm_autoscaling_target_prefill_tokens_per_replica":
            d["target_tokens"] = s["value"]
        # Controller gauge: each KV router's look-back mean, one series per
        # handle. The token policy scales on the max across handles.
        elif (s["name"] == "ray_serve_autoscaling_router_metric"
              and s["labels"].get("metric") == "outstanding_effective_prefill_tokens"):
            d["lookback_mean_tokens"] = max(d.get("lookback_mean_tokens", s["value"]), s["value"])
    if "target_tokens" in d:
        target = d["target_tokens"] / 1000
    if "lookback_mean_tokens" in d and "required_replicas" not in d:
        # PR 4 exports the mean, not the ask; the policy asks ceil(mean / target).
        d["prefill_required_replicas"] = d["lookback_mean_tokens"] / 1000 / target
        # Round first so float error from the K conversion can't add a replica.
        d["required_replicas"] = math.ceil(round(d["prefill_required_replicas"], 9))
    if "required_replicas" in d and "desired" in d:
        asks.append((t, d["required_replicas"], d["desired"]))
    if "outstanding_effective_prefill_tokens" in d:
        raw.append((t, d["outstanding_effective_prefill_tokens"] / 1000))
    if "prefill_required_replicas" in d:
        pub.append((t, d["prefill_required_replicas"] * target))
# Branches that do not publish the policy gauge (e.g. the 3-policy branch)
# still get the engine side, sampled at the scrape times.
ts = [t for t, _ in raw] or scrape_ts

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
if pub and not raw:
    # No raw gauge to recompute from (the controller exports only the mean):
    # the summary uses the published mean, sampled at the scrape times.
    policy_mean = [pub_v[i] if (i := bisect.bisect_right(pub_t, t) - 1) >= 0 else 0.0
                   for t in ts]

# Delay timers reconstructed from consecutive sampled recommendations above or
# below the accepted target. Bound recommendations as Serve does, so asking for
# zero at min_replicas=1 does not appear to start a downscale timer.
auto = (resolved.get("deployment") or {}).get("autoscaling") or {}
up_delay = float(auto.get("upscale_delay_s") or 0)
down_delay = float(auto.get("downscale_delay_s") or 0)
freeze_s = float(kwargs.get("post_downscale_cooldown_s") or 0)
# Decisions come from sampled target changes; window boundaries inherit the
# metric scrape resolution, rather than exact controller acceptance timestamps.
freeze_windows = [
    (e["time_s"], e["time_s"] + freeze_s)
    for e in evaluation["scale_events"]
    if e["direction"] == "down" and freeze_s > 0
]

def delay_timer_spans(direction):
    decisions = sorted(e["time_s"] for e in evaluation["scale_events"]
                       if e["direction"] == direction)
    timers, cur = [], None
    for t, req, des in asks:
        req = max(auto.get("min_replicas", 0), min(req, auto.get("max_replicas", math.inf)))
        # The policy returns the held target during a freeze, resetting both
        # timers even though it continues publishing workload recommendations.
        frozen = any(lo <= t < hi for lo, hi in freeze_windows)
        on = (req > des if direction == "up" else req < des) and not frozen
        if on and cur is None:
            cur = [t, t]
        elif on:
            cur[1] = t
        elif cur is not None:
            timers.append((*cur, False)); cur = None
    if cur is not None:
        timers.append((*cur, True))
    gap = asks[1][0] - asks[0][0] if len(asks) > 1 else 2.0
    spans = []
    for t1, t2, ongoing in timers:
        passed = [t for t in decisions if t2 <= t <= t2 + 2 * gap + 1]
        status = "passed" if passed else "running" if ongoing else "reset"
        spans.append((t1, passed[0] if passed else t2, status))
    return spans


timer_spans = delay_timer_spans("up")
down_timer_spans = delay_timer_spans("down")

_eng_peak = [] if a.no_engine else ([v for _, v in eng_raw] if a.raw else engine_mean)
peak = max(([v for _, v in raw] if a.raw else (pub_v or policy_mean)) + _eng_peak) if (raw or eng_raw) else 1.0
ycap = a.ymax if a.ymax else max(peak * 1.05, 1.5 * target)

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
for lo, hi in freeze_windows:
    left, right = max(lo, a.start), min(hi, end)
    if right > left:
        ax.axvspan(left, right, color="#e8b642", alpha=0.22, lw=0, zorder=0.6)
        if a.start <= hi <= end:
            ax.axvline(hi, color="#a77a12", lw=1, ls="--", alpha=0.9)
        ax.text((left + right) / 2, ycap * 0.96, f"{freeze_s:g}s freeze",
                ha="center", va="top", fontsize=8, color="#805b0a")
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
    if not a.no_engine:
        ax.step([t for t, _ in eng_raw], [min(v, ycap) for _, v in eng_raw], where="post", color=ENGINE, lw=0.7, alpha=0.3)
has_policy = bool(pub or raw)
if pub:
    ax.plot(pub_t, [min(v, ycap) for v in pub_v], color=POLICY, lw=1.8)
elif raw:
    ax.plot(ts, [min(v, ycap) for v in policy_mean], color=POLICY, lw=1.8)
if not a.no_engine:
    ax.plot(ts, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)
ax.axhline(target, color=INK2, lw=1, ls=":")
if a.mark_evictions:
    # Returning sessions whose context the engine had evicted: from the traces
    # (so failed requests count too), joined to AIPerf metadata for turn index.
    meta = {}
    for p_ in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
        for line in open(p_):
            m_ = json.loads(line)["metadata"]
            if m_.get("benchmark_phase") == "profiling":
                meta[m_["x_request_id"]] = m_
    adm = []
    for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
        qd, ad = {}, {}
        for line in open(path):
            try:
                e = json.loads(line)
            except ValueError:
                continue
            if e["event"] == "request_queued":
                qd[e["request_id"]] = e
            elif e["event"] == "request_admitted":
                ad[e["request_id"]] = e
        for rid, q in qd.items():
            m_ = meta.get(rid.split("chatcmpl-", 1)[-1].rsplit("-", 1)[0])
            if m_ and rid in ad:
                adm.append((q["arrival_time_s"] - t0 / 1e9, m_["conversation_id"], int(m_["turn_index"]), ad[rid]["prompt_tokens"], ad[rid]["cached_prefix_tokens"]))
    adm.sort()
    prev_prompt = {}
    evictions = []
    for t, cid, turn, prompt, cached in adm:
        prev = prev_prompt.get(cid)
        if turn >= 2 and cached < 0.1 * prompt and prev is not None and prompt >= prev and a.start <= t <= end:
            evictions.append((t, prompt - cached, cid.split("::")[0][-6:] + ("/sa" if "::" in cid else ""), turn))
        prev_prompt[cid] = prompt
    for i, (t, unc, name, turn) in enumerate(evictions):
        ax.axvline(t, color="#e34948", lw=1.1, alpha=0.9, zorder=3)
        ax.text(t + 2, ycap * (0.98 - 0.05 * (i % 6)), f"{name} t{turn} {unc // 1000}K", fontsize=6.3, color="#b5432f", va="top", zorder=4)
    print(f"evictions marked: {len(evictions)}")
    for t, unc, name, turn in evictions:
        print(f"  {t:7.1f}s {name} turn {turn} re-prefill {unc} tokens")
TIMER_PASS, TIMER_RESET = "#c9480f", "#e8b89a"
DOWN_TIMER_PASS, DOWN_TIMER_RESET = "#1766ad", "#9fc6e8"
for direction, spans, passed_color, pending_color in (
    ("up", timer_spans, TIMER_PASS, TIMER_RESET),
    ("down", down_timer_spans, DOWN_TIMER_PASS, DOWN_TIMER_RESET),
):
    for t1, t2, status in spans:
        if a.no_timers or t2 < a.start or t1 > end:
            continue
        passed = status == "passed"
        ax.axvspan(max(t1, a.start), min(t2, end), ymin=0.0, ymax=0.035,
                   color=passed_color if passed else pending_color, lw=0, zorder=2)
        if passed:
            ax.annotate("", xy=(t2, -ycap * 0.055), xytext=(t2 - 1, -ycap * 0.055), arrowprops=dict(arrowstyle="-|>", color=passed_color, lw=0.8, mutation_scale=7), annotation_clip=False)
        elif t2 - t1 >= 8:
            ax.text((t1 + t2) / 2, -ycap * 0.055, f"{t2 - t1:.0f}s", ha="center", va="top", fontsize=6.5, color=passed_color)
        print(f"  {direction}scale timer: {t1:.1f}-{t2:.1f}s ({status})")
ax.set_ylim(0 if a.no_timers else -ycap * 0.06, ycap * 1.12)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel("total outstanding uncached prefill tokens, all replicas (K)", color=INK2)
handles = [Line2D([], [], color=POLICY, lw=1.8, label=f"policy: {lookback:.0f}s mean of the router gauge (the number the policy divides by the target)"),
           Line2D([], [], color=ENGINE, lw=1.8, label=f"engine: {lookback:.0f}s mean of the engines' remaining uncached prefill (traces)"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica"),
           Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"),
           Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)"),
           Patch(color=TIMER_PASS, label=f"on the x axis: upscale-delay timer running and passed ({up_delay:.0f}s of asking for more), ends at the decision"),
           Patch(color=TIMER_RESET, label="upscale-delay timer running or reset before a scale decision"),
           Patch(color=DOWN_TIMER_PASS, label=f"on the x axis: downscale-delay timer passed ({down_delay:.0f}s of asking for fewer), ends at the decision"),
           Patch(color=DOWN_TIMER_RESET, label="downscale-delay timer running or reset before a scale decision")]
if a.no_timers:
    handles = [h for h in handles if not isinstance(h, Patch)]
if a.no_engine:
    handles = [h for h in handles if not (isinstance(h, Line2D) and h.get_color() == ENGINE)]
if freeze_windows:
    handles.append(Patch(facecolor="#e8b642", alpha=0.35,
                         label=f"{freeze_s:g}s freeze after accepted target decrease (sampled boundaries)"))
if a.mark_evictions:
    handles.append(Line2D([], [], color="#e34948", lw=1.1, label="returning session arrived with its context evicted by the engine (label: session, turn, tokens re-prefilled)"))
if a.raw:
    handles[1:1] = [Line2D([], [], color=POLICY, lw=0.7, alpha=0.4, label="policy: router gauge as scraped, every 2s (whole prompt counted until first token)")]
    handles[3:3] = [Line2D([], [], color=ENGINE, lw=0.7, alpha=0.4, label="engine: remaining uncached prefill as it changes, chunks leave as batches complete")]
if not has_policy:
    # No policy gauge was exported (e.g. the 3-policy branch): say so instead of drawing zeros.
    handles = [h for h in handles if getattr(h, "get_color", lambda: None)() != POLICY]
    fig.text(0.5, -0.215, "policy signal not exported by this Ray build: engine side only", ha="center", va="top", fontsize=8.5, color=POLICY)
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
sec = lambda v: f"{float(v):g}s" if v is not None else "?"
settings = (f"policy: ceil({lookback:.0f}s mean / {target:.0f}K) {DOT} Serve upscale delay {sec(auto.get('upscale_delay_s'))} {DOT} downscale delay {sec(auto.get('downscale_delay_s'))}"
            f" {DOT} replicas {auto.get('min_replicas')}-{auto.get('max_replicas')} {DOT} metrics every {sec(auto.get('metrics_interval_s'))}")
if freeze_s > 0:
    settings += f" {DOT} post-downscale freeze {freeze_s:g}s"
fig.text(0.5, -0.19, settings, ha="center", va="top", fontsize=8.5, color=INK2)
fig.suptitle(f"{evaluation['experiment']} {DOT} " + ("what the mean policy observed: total outstanding uncached prefill tokens" if a.no_engine else
             "what the mean policy observed vs what the engines held: total outstanding uncached prefill tokens"),
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or (f"{R}/analysis/policy_vs_engine_mean{int(lookback)}{'_raw' if a.raw else ''}{'_evictions' if a.mark_evictions else ''}"
                f"{'_policy_only' if a.no_engine else ''}{'_no_timers' if a.no_timers else ''}_{int(a.start)}_{int(end)}.png")
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)

# Summary: ratio of the two means where the engine mean is meaningful.
pairs = [(p, e) for p, e in zip(policy_mean, engine_mean) if e >= 1.0] if has_policy else []
if not has_policy and engine_mean:
    print(f"  no policy signal; time-average engine mean over the plot range: {sum(engine_mean) / len(engine_mean):.1f}K")
if pairs:
    ratios = sorted(p / e for p, e in pairs)
    print(f"  policy mean / engine mean where engine mean >= 1K: median {ratios[len(ratios) // 2]:.2f}, p10 {ratios[int(0.1 * len(ratios))]:.2f}, p90 {ratios[int(0.9 * len(ratios))]:.2f} (n={len(ratios)})")
    print(f"  time-average over the load phase: policy {sum(policy_mean) / len(policy_mean):.1f}K, engine {sum(engine_mean) / len(engine_mean):.1f}K")
    if pub and raw:
        diffs = [abs(p - q) for p, q in zip(policy_mean, [pub_v[bisect.bisect_right(pub_t, t) - 1] for t in ts]) if True]
        print(f"  published policy mean vs recomputed from the gauge: max abs diff {max(diffs):.1f}K")
