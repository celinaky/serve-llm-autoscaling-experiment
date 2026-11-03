#!/usr/bin/env python3
"""Total outstanding uncached prefill tokens over time, with replica changes.

Top: the router's raw gauge of outstanding uncached prefill tokens summed over
all replicas (scraped every 2s) and its 60s time-weighted mean, against the
target. Every change in the running replica count is marked: an up triangle
at the moment a new replica became RUNNING, a down triangle when one was
removed, labelled with the new count. Bottom: running and desired replicas.
Load phases are shaded.

usage: python tools/backlog_vs_replicas.py runs/<run_dir> [--ymax K] [--start 0] [--end END] [--out PATH]
"""
import argparse, bisect, json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
BLUE, AQUA, GOOD, BAD = "#2a78d6", "#1baf7a", "#1baf7a", "#e34948"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--out", default=None)
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"] / 1e9
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
target = ((evaluation.get("autoscaling_policy_kwargs") or {}).get("target_prefill_tokens_per_replica") or 20000) / 1000
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []
routing = (resolved.get("deployment") or {}).get("routing", {}).get("policy")

raw, desired = [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    if not a.start <= t <= end:
        continue
    for s in rec["samples"]:
        lab = s["labels"]
        if "LLMServer" not in lab.get("deployment", ""):
            continue
        if s["name"] == "ray_serve_autoscaling_policy_metric" and lab.get("metric") == "outstanding_effective_prefill_tokens":
            raw.append((t, s["value"] / 1000))
        elif s["name"] == "ray_serve_autoscaling_desired_replicas":
            desired.append((t, s["value"]))
raw.sort(); desired.sort()
raw_t = [x for x, _ in raw]
mean = []
for i, (t, _) in enumerate(raw):
    w0 = t - 60
    j = bisect.bisect_left(raw_t, w0)
    seg = raw[max(j - 1, 0):i + 1]
    total = 0.0
    for (t1, v1), (t2, _) in zip(seg, seg[1:] + [(t, None)]):
        lo_, hi_ = max(t1, w0), min(t2, t)
        if hi_ > lo_:
            total += v1 * (hi_ - lo_)
    span = t - max(w0, seg[0][0])
    mean.append(total / span if span > 0 else raw[i][1])

status = [s for s in plot_data["serve_status_samples"] if s.get("error") is None and a.start <= s["relative_time_s"] <= end]
changes = []  # (t, new running count, +1/-1)
for prev, cur in zip(status, status[1:]):
    if cur["running_replicas"] != prev["running_replicas"]:
        changes.append((cur["relative_time_s"], cur["running_replicas"], 1 if cur["running_replicas"] > prev["running_replicas"] else -1))

ycap = a.ymax if a.ymax else max(y for _, y in raw) * 1.05
fig, (ax, axr) = plt.subplots(2, 1, figsize=(14, 7), dpi=150, sharex=True, gridspec_kw={"height_ratios": [3.2, 1], "hspace": 0.08})
fig.patch.set_facecolor(SURFACE)
for axis in (ax, axr):
    axis.set_facecolor(SURFACE)
    for s in ("top", "right"):
        axis.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        axis.spines[s].set_color(GRID)
    axis.tick_params(colors=INK2, labelsize=8)
    axis.grid(axis="y", color=GRID, lw=0.5)
    axis.set_axisbelow(True)
    for p0, p1 in zip(curve, curve[1:]):
        phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
        lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
        if hi > lo:
            axis.axvspan(lo, hi, color=PHASE_TINT[phase], zorder=0, lw=0)

ax.step(raw_t, [min(y, ycap) for _, y in raw], where="post", color=BLUE, lw=0.9, alpha=0.6, label="outstanding uncached prefill tokens, all replicas (router gauge, 2s)")
ax.step(raw_t, [min(y, ycap) for y in mean], where="post", color=BLUE, lw=1.8, label="60s time-weighted mean of it (the mean policy's signal)")
ax.axhline(target, color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica")
for t, n, d in changes:
    ax.axvline(t, color=GOOD if d > 0 else BAD, lw=0.9, ls="-" if d > 0 else "--", alpha=0.8)
    ax.plot([t], [ycap * 1.06], marker="^" if d > 0 else "v", ms=7, color=GOOD if d > 0 else BAD, clip_on=False)
    ax.text(t + 3, ycap * 1.05, str(n), fontsize=7.5, color=GOOD if d > 0 else BAD, va="center", ha="left")
ax.set_ylim(0, ycap * 1.12)
ax.set_ylabel("outstanding uncached prefill tokens (K)", color=INK2)
for p0, p1 in zip(curve, curve[1:]):
    phase = "ramp up" if p1["sessions"] > p0["sessions"] else "ramp down" if p1["sessions"] < p0["sessions"] else "hold"
    lo, hi = max(p0["time_s"], a.start), min(p1["time_s"], end)
    if hi > lo:
        ax.text((lo + hi) / 2, ycap * 0.99, f"{phase}: {p0['sessions']}{ARROW}{p1['sessions']} sessions", ha="center", va="top", fontsize=8, color=INK2)
ax.legend(loc="upper right", fontsize=8, frameon=False, bbox_to_anchor=(1, 0.93))
ax.set_title(f"{evaluation['experiment']} {DOT} routing {routing} {DOT} outstanding uncached prefill, with every replica that became running (green, up) or was removed (red, down)",
             loc="left", fontsize=9.5, color=INK)
axr.step([s["relative_time_s"] for s in status], [s["running_replicas"] for s in status], where="post", color=AQUA, lw=1.6, label="running")
axr.step([t for t, _ in desired], [v for _, v in desired], where="post", color=INK2, lw=1.0, ls="--", label="desired")
for t, n, d in changes:
    axr.axvline(t, color=GOOD if d > 0 else BAD, lw=0.9, ls="-" if d > 0 else "--", alpha=0.8)
axr.set_ylim(bottom=0)
axr.set_ylabel("replicas", color=INK2)
axr.set_xlabel("seconds since profiling start", color=INK2)
axr.legend(loc="upper left", fontsize=8, frameon=False, ncol=2)
axr.set_xlim(a.start, end)
out = a.out or f"{R}/analysis/backlog_vs_replicas_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
