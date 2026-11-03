#!/usr/bin/env python3
"""What the arrival-p95 policy saw and decided, or would have, over time.

Three panels on one clock:
  1. load at arrival over the policy's window: the policy's own p95 (router
     gauge) when the run had the arrival-p95 policy, and from the engine traces
     (analysis/load_requests.csv) the p95, p90, p50 and mean over the same
     window, against the target;
  2. the number of arrival samples in the window, against the sample gate;
  3. replicas: running, Serve's desired count, the policy's fractional
     requirement (current x p95 / target), and the same rule with the window
     mean. For a run without the policy (e.g. a fixed replica count) the
     requirement is computed from the engine-side p95 instead, so the chart
     shows what the rule would have asked from that data.
Scale decisions and load phases are marked. --previous-policy overlays the
earlier policy's 60s mean of the total backlog gauge.

usage: python tools/policy_decision_timeline.py runs/<run_dir> [--ymax K] [--start 0] [--end END]
           [--target 20000] [--lookback 120] [--previous-policy] [--out PATH]
"""
import argparse, bisect, csv, glob, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.ticker
import matplotlib.pyplot as plt
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, ORANGE, AQUA, VIOLET, FOLD = "#2a78d6", "#eb6834", "#1baf7a", "#6250d6", "#9b9a95"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT, TIMES = "→", "·", "×"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--target", type=float, default=20000, help="target tokens when the run had no token policy")
ap.add_argument("--lookback", type=float, default=120, help="window when the run had no token policy")
ap.add_argument("--previous-policy", action="store_true")
ap.add_argument("--mean-lookback", type=float, default=60.0)
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--only-backlog", action="store_true", help="with --backlog: show only the rebuilt total-backlog signal (and its rule), not the per-request statistics")
ap.add_argument("--backlog", action="store_true", help="rebuild the previous policy's signal (total outstanding uncached tokens, 60s mean) from the traces, engine-side and router-like")
ap.add_argument("--per-requests", type=int, default=100, help="window size, in requests, for the middle panel's p95-vs-target bars")
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
target = kwargs.get("target_prefill_tokens_per_replica") or a.target
gate = kwargs.get("min_arrival_samples")
lookback = kwargs.get("upscale_lookback_s") or a.lookback
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
ycap = a.ymax if a.ymax else 5 * target / 1000

rows = []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    d = {"t": rec["scrape_started_at_ns"] / 1e9 - t0 / 1e9}
    for s in rec["samples"]:
        lab = s["labels"]
        if "LLMServer" not in lab.get("deployment", ""):
            continue
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            d[lab["metric"]] = s["value"]
        elif s["name"] == "ray_serve_autoscaling_desired_replicas":
            d["desired"] = s["value"]
    if ("desired" in d or "prefill_arrival_p95_tokens" in d) and a.start <= d["t"] <= end:
        rows.append(d)
have_policy = any("prefill_arrival_p95_tokens" in d for d in rows)
mean_policy = (not have_policy) and any("prefill_required_replicas" in d for d in rows)
have_desired = any("desired" in d for d in rows)
if not rows:
    rows = [{"t": a.start + 10 * i} for i in range(int((end - a.start) / 10) + 1)]
ts = [d["t"] for d in rows]
p95 = [d.get("prefill_arrival_p95_tokens", 0) / 1000 for d in rows]
samples = [d.get("prefill_arrival_samples", 0) for d in rows]
ratio = [d.get("prefill_required_replicas", 0) for d in rows]
desired = [d.get("desired", 0) for d in rows]

raw = [(d["t"], d.get("outstanding_effective_prefill_tokens", 0.0)) for d in rows]
raw_t = [x for x, _ in raw]
backlog_mean = []
for i, (t, _) in enumerate(raw):
    w0 = t - a.mean_lookback
    j = bisect.bisect_left(raw_t, w0)
    seg = raw[max(j - 1, 0):i + 1]
    total = 0.0
    for (t1, v1), (t2, _) in zip(seg, seg[1:] + [(t, None)]):
        lo_, hi_ = max(t1, w0), min(t2, t)
        if hi_ > lo_:
            total += v1 * (hi_ - lo_)
    span = t - max(w0, seg[0][0])
    backlog_mean.append(total / span if span > 0 else raw[i][1])
mean_wants = [max(1, -(-m // target)) for m in backlog_mean]

start_s = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling":
            start_s[m["x_request_id"]] = (m["request_start_ns"] - t0) / 1e9
arr = sorted((start_s[r["request_id"]], float(r["engine_load_tokens"]) / 1000)
             for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")) if r["request_id"] in start_s)
arr_t = [x for x, _ in arr]
eng_t, eng_p95, eng_p90, eng_p50, eng_mean, eng_n = [], [], [], [], [], []
for t in ts:
    lo = bisect.bisect_left(arr_t, t - lookback)
    hi = bisect.bisect_right(arr_t, t)
    v = sorted(y for _, y in arr[lo:hi])
    if len(v) >= 10:
        eng_t.append(t); eng_n.append(len(v))
        eng_p95.append(v[min(len(v) - 1, int(0.95 * len(v)))])
        eng_p90.append(v[min(len(v) - 1, int(0.9 * len(v)))])
        eng_p50.append(v[len(v) // 2]); eng_mean.append(sum(v) / len(v))

status = [s for s in plot_data["serve_status_samples"] if s.get("error") is None and a.start <= s["relative_time_s"] <= end]
status_t = [s["relative_time_s"] for s in status]

# Previous policy's signal from the traces: total outstanding uncached prefill
# tokens over all engines, as a step function, in two accountings.
backlog_engine = backlog_router = None
if a.backlog:
    import glob as _glob, math, os, re
    manifest = json.load(open(f"{R}/manifest.json"))
    run_start = __import__("datetime").datetime.fromisoformat(manifest["started_at"]).timestamp()
    deltas_engine, deltas_router = [], []  # (t_rel, delta tokens)
    for path in sorted(_glob.glob(f"{a.traces}/*.jsonl")):
        if os.path.getmtime(path) < run_start:
            continue
        chunks, completed_at, queued, preempted, first = {}, {}, {}, {}, {}
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
                first[e["request_id"]] = e["monotonic_s"]
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
                deltas_engine.append((rel(at), sum(t for b, t in cs if batch <= b < until)))
            deltas_engine.extend((rel(completed_at[b]), -t) for b, t in cs)
            # Router-like: the whole uncached prompt from queue until first token.
            own = sum(t for b, t in cs if b <= (min((b2 for b2, _ in cs if first.get(rid) is not None and completed_at.get(b2, math.inf) >= first[rid]), default=cs[-1][0])))
            own = max(own, 0)
            t_done = first.get(rid)
            if t_done is None:
                continue
            deltas_router.append((rel(q["monotonic_s"]), own))
            deltas_router.append((rel(t_done), -own))

    def step_mean(deltas, window):
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
    backlog_engine = step_mean(deltas_engine, a.mean_lookback)
    backlog_router = step_mean(deltas_router, a.mean_lookback)

def running_at(t):
    k = bisect.bisect_right(status_t, t) - 1
    return status[k]["running_replicas"] if k >= 0 else 1

fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(14, 9), dpi=150, sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1.5, 2], "hspace": 0.1})
fig.patch.set_facecolor(SURFACE)
for ax in (ax1, ax2, ax3):
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
    for e in evaluation["scale_events"]:
        if a.start <= e["time_s"] <= end:
            up = e["direction"] == "up"
            ax.axvline(e["time_s"], color=INK if up else INK2, lw=0.8, ls="-" if up else "--", alpha=0.7)

# 1. load at arrival
show_stats = not (a.backlog and a.only_backlog)
if have_policy and show_stats:
    ax1.step(ts, [min(y, ycap) for y in p95], where="post", color=BLUE, lw=1.6,
             label=f"policy p95 of load at arrival, last {lookback:.0f}s (router gauge)")
    last_label = None
    for t, y in zip(ts, p95):
        if y > ycap and round(y) != last_label:
            ax1.text(t, ycap * 1.02, f"{y:.0f}K", rotation=90, ha="center", va="bottom", fontsize=6.5, color=BLUE)
        last_label = round(y) if y > ycap else None
if show_stats:
    ax1.step(eng_t, [min(y, ycap) for y in eng_p95], where="post", color=FOLD if have_policy else BLUE, lw=1.2 if have_policy else 1.6,
             ls="--" if have_policy else "-", label="engine-side p95 over the same window (traces)" if have_policy else f"p95 of load at arrival, last {lookback:.0f}s (traces)")
    ax1.step(eng_t, [min(y, ycap) for y in eng_p90], where="post", color=VIOLET, lw=1.2, label="p90 over the same window (traces)")
    ax1.step(eng_t, [min(y, ycap) for y in eng_p50], where="post", color=AQUA, lw=1.2, label="p50 over the same window (traces)")
    ax1.step(eng_t, [min(y, ycap) for y in eng_mean], where="post", color=ORANGE, lw=1.4, label="mean over the same window (traces)")
if a.previous_policy or mean_policy:
    ax1.step(ts, [min(m / 1000, ycap) for m in backlog_mean], where="post", color=VIOLET, lw=1.4,
             label=f"the run's policy signal: {a.mean_lookback:.0f}s mean of total outstanding backlog (router gauge)" if mean_policy
             else f"previous policy's signal: {a.mean_lookback:.0f}s mean of total outstanding backlog (router gauge)")
if backlog_router is not None:
    ax1.step(ts, [min(m / 1000, ycap) for m in backlog_router], where="post", color=INK, lw=1.3,
             label=f"previous policy's signal rebuilt from traces: {a.mean_lookback:.0f}s mean of total outstanding, whole prompts until first token")
    ax1.step(ts, [min(m / 1000, ycap) for m in backlog_engine], where="post", color=FOLD, lw=1.3, ls="--",
             label=f"same, engine-side: chunks subtracted as they complete")
ax1.axhline(target / 1000, color=INK2, lw=1, ls=":", label=f"target {target / 1000:.0f}K")
ax1.set_ylim(0, ycap * 1.3)
ymax = ycap * 1.3
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi > lo:
        ax1.text((lo + hi) / 2, ymax * 0.995, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
for e in evaluation["scale_events"]:
    if a.start <= e["time_s"] <= end:
        ax1.text(e["time_s"] + 3, ymax * 0.93, f"{e['from_replicas']:.0f}{ARROW}{e['to_replicas']:.0f}", rotation=90, va="top", ha="left",
                 fontsize=8, color=INK if e["direction"] == "up" else INK2)
ax1.set_ylabel("load at arrival (K tokens)", color=INK2)
ax1.legend(loc="upper right", fontsize=7.5, frameon=False)
ax1.set_title(f"{evaluation['experiment']} {DOT} " + (f"the mean policy's signal (router gauge) against the same signal rebuilt from the traces, and what it decided" if (not show_stats and mean_policy)
              else f"the previous policy's signal rebuilt from the traces: {a.mean_lookback:.0f}s mean of total outstanding uncached tokens, and what its rule would have asked" if not show_stats
              else "what the arrival-p95 policy saw and decided" if have_policy
              else f"load at arrival over a {lookback:.0f}s window, and what the p95 rule would have asked"), loc="left", fontsize=10, color=INK)

# 2. per fixed-count window: the share of requests that found more than the
# target ahead of them. Over 5% means the window's p95 is over target.
GOOD, BAD, DEEP = "#1baf7a", "#e34948", "#b4321c"
met = missed = 0
wins = []
for i in range(0, len(arr), a.per_requests):
    chunk = arr[i:i + a.per_requests]
    ws = chunk[0][0]
    we = arr[i + a.per_requests][0] if i + a.per_requests < len(arr) else end
    n = len(chunk)
    over = sum(1 for _, y in chunk if y * 1000 > target) / n
    far = sum(1 for _, y in chunk if y * 1000 > 2.5 * target) / n
    wins.append((ws, we, n, over, far))
w_top = max(0.25, max(o for _, _, _, o, _ in wins) * 1.25)
for ws, we, n, over, far in wins:
    ok = over <= 0.05
    met += ok; missed += not ok
    mid = ws + (we - ws) / 2
    ax2.bar(mid, over, width=(we - ws) * 0.9, color=GOOD if ok else BAD, alpha=0.85, lw=0)
    ax2.bar(mid, far, width=(we - ws) * 0.9, color=DEEP, lw=0)
    ax2.text(mid, over + w_top * 0.015, f"{over:.0%}", ha="center", va="bottom", fontsize=6.5, color=INK if not ok else INK2)
    ax2.text(mid, -w_top * 0.03, str(n), ha="center", va="top", fontsize=6, color=INK2)
ax2.axhline(0.05, color=INK, lw=1, ls=":")
ax2.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
ax2.set_ylim(-w_top * 0.12, w_top * 1.15)
ax2.set_ylabel(f"share over target, per {a.per_requests} requests", color=INK2)
ax2.text(0.995, 0.95, f"share of requests that found more than {target / 1000:.0f}K ahead; darker: more than {2.5 * target / 1000:.0f}K "
         f"{DOT} over 5% = p95 over target {DOT} {met} windows meet, {missed} miss", transform=ax2.transAxes,
         ha="right", va="top", fontsize=8, color=INK)

# 3. replicas
ax3.step(status_t, [s["running_replicas"] for s in status], where="post", color=AQUA, lw=1.6, label="running replicas")
mean_rule = [running_at(t) * m * 1000 / target for t, m in zip(eng_t, eng_mean)]
p95_rule = [running_at(t) * y * 1000 / target for t, y in zip(eng_t, eng_p95)]
if have_desired:
    ax3.step(ts, desired, where="post", color=INK2, lw=1.0, ls="--", label="desired (Serve)")
if have_policy and show_stats:
    ax3.plot(ts, ratio, color=ORANGE, lw=1.2, label=f"policy requirement: current {TIMES} p95 / target (before rounding)")
elif mean_policy:
    ax3.plot(ts, ratio, color=VIOLET, lw=1.3, label=f"the run's mean policy requirement: {a.mean_lookback:.0f}s backlog mean / target (before rounding)")
elif show_stats:
    ax3.plot(eng_t, p95_rule, color=ORANGE, lw=1.2, label=f"what the p95 rule would ask: current {TIMES} p95 / target")
if show_stats:
    ax3.plot(eng_t, mean_rule, color=AQUA, lw=1.2, ls="--", label=f"same rule with the window mean: current {TIMES} mean / target")
if a.previous_policy and not mean_policy:
    ax3.step(ts, mean_wants, where="post", color=VIOLET, lw=1.2, label=f"previous mean policy: ceil({a.mean_lookback:.0f}s backlog mean / target)")
if backlog_router is not None:
    ax3.step(ts, [max(1, -(-m // target)) for m in backlog_router], where="post", color=INK, lw=1.2,
             label=f"previous policy's rule on the rebuilt signal: ceil({a.mean_lookback:.0f}s total backlog mean / target)")
    ax3.step(ts, [max(1, -(-m // target)) for m in backlog_engine], where="post", color=FOLD, lw=1.2, ls="--",
             label="same on the engine-side total")
cap = (resolved.get("deployment") or {}).get("autoscaling", {}).get("max_replicas")
if cap and cap > 1:
    ax3.axhline(cap, color=INK2, lw=0.8, ls=":", label=f"max_replicas {cap}")
top = max([cap or 0] + desired + (ratio + mean_rule + (p95_rule if not have_policy else []) if show_stats else [])
          + ([max(1, -(-m // target)) for m in backlog_router] if backlog_router is not None else []))
ax3.set_ylim(0, top * 1.15 + 0.5)
ax3.set_ylabel("replicas", color=INK2)
ax3.set_xlabel("seconds since profiling start", color=INK2)
ax3.legend(loc="upper right", fontsize=7.5, frameon=False, ncol=2)
ax3.set_xlim(a.start, end)
out = a.out or f"{R}/analysis/policy_decisions{'_backlog' if not show_stats else ''}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
