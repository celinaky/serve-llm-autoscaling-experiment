#!/usr/bin/env python3
"""Zoom on one stretch: what the policy's mean was made of, against what arriving requests saw.

Top: the engines' outstanding uncached prefill tokens rebuilt from the
scheduler traces (each request's prompt minus cached prefix counts from the
moment it is queued until its first token), stacked per request so the big
cold prompts are visible, with the router gauge the policy actually read
(telemetry) and the policy's 60s time-weighted mean of it on top.
Bottom: what each arriving request found ahead of it on its replica (load
at arrival from load_requests.csv, own tokens excluded), one dot per request,
with the 180s-window p95 and mean the arrival-stats chart reports.

usage: python tools/gauge_vs_arrival_zoom.py runs/<run_dir> --start 380 --end 560
           [--engines pid1,pid2] [--lookback 60] [--traces DIR] [--out PATH]
"""
import argparse, csv, glob, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
ap.add_argument("--engines", default=""); ap.add_argument("--lookback", type=float, default=60); ap.add_argument("--traces", default=TRACES); ap.add_argument("--out")
a = ap.parse_args(); R = a.run.rstrip("/")
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"] / 1e9
meta = {}
for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(p):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling": meta[m["x_request_id"]] = m
rows = {r["request_id"]: r for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv"))}
engines = a.engines.split(",") if a.engines else sorted({r["engine"].split("pid-")[-1].split(".")[0] for r in rows.values()})
# rebuild per-request outstanding uncached spans from the traces
spans = []  # (queued, first_token, uncached, label, engine)
for e in engines:
    path = glob.glob(f"{a.traces}/*pid-{e}*.jsonl")
    if not path: continue
    off = None; queued, admitted, first = {}, {}, {}
    for line in open(path[0]):
        d = json.loads(line); k = d["event"]
        if off is None and "wall_time_s" in d: off = d["wall_time_s"] - d["monotonic_s"]
        if k == "request_queued": queued[d["request_id"]] = d
        elif k == "request_admitted": admitted[d["request_id"]] = d
        elif k == "request_first_token": first[d["request_id"]] = d
    for rid, q in queued.items():
        if rid not in first or rid not in admitted: continue
        x = rid.split("chatcmpl-", 1)[-1].rsplit("-", 1)[0]; m = meta.get(x)
        unc = admitted[rid]["prompt_tokens"] - admitted[rid]["cached_prefix_tokens"]
        lab = (m["conversation_id"].split("::")[0][-6:] + ("/sa" if "::" in m["conversation_id"] else "") + f" t{m['turn_index']}") if m else "?"
        spans.append((q["arrival_time_s"] - t0, first[rid]["monotonic_s"] + off - t0, unc, lab, e))
spans.sort()
ts = sorted({t for s in spans for t in (s[0], s[1]) if a.start - a.lookback - 5 <= t <= a.end + 1}); ts = [a.start - a.lookback - 5] + ts + [a.end + 1]
def gauge(t): return sum(u for q, f, u, _, _ in spans if q <= t < f)
g = [gauge(t) for t in ts]
# time-weighted trailing mean of the rebuilt gauge
def mean_at(t):
    lo = t - a.lookback; acc = 0.0
    for i in range(len(ts) - 1):
        s, e = max(ts[i], lo), min(ts[i + 1], t)
        if e > s: acc += g[i] * (e - s)
    return acc / a.lookback
grid = [a.start + i * 0.5 for i in range(int((a.end - a.start) * 2) + 1)]
mean_rebuilt = [mean_at(t) for t in grid]
# router gauge + published mean from telemetry
router, published = [], []
tgt = 20000
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    d = json.loads(line); t = d["timestamp_ns"] / 1e9 - t0
    if not a.start - 5 <= t <= a.end + 5: continue
    for s in d.get("samples") or []:
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            if s["labels"].get("metric") == "outstanding_effective_prefill_tokens": router.append((t, s["value"]))
            if s["labels"].get("metric") == "prefill_required_replicas": published.append((t, s["value"] * tgt))
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.6, 1], "hspace": 0.08}); fig.patch.set_facecolor(SURFACE)
# stacked per-request spans in the window
inwin = [s for s in spans if s[1] > a.start and s[0] < a.end]
import colorsys
base = 0
# draw as stacked step areas: at each ts, order requests by queue time
order = sorted(inwin, key=lambda s: s[0])
cols = {}
for i, s in enumerate(order): cols[s] = colorsys.hsv_to_rgb((i * 0.618) % 1, 0.55, 0.85)
stack_ts = [t for t in ts if a.start <= t <= a.end]; stack_ts = [a.start] + stack_ts + [a.end]
bottoms = [0.0] * len(stack_ts)
for s in order:
    q, f, u, lab, e = s
    tops = [b + (u if q <= t < f else 0) for b, t in zip(bottoms, stack_ts)]
    ax.fill_between(stack_ts, [b / 1000 for b in bottoms], [tp / 1000 for tp in tops], step="post", color=cols[s], alpha=0.85, lw=0)
    if u >= 15000:
        mid = (max(q, a.start) + min(f, a.end)) / 2; i = max(k for k, t in enumerate(stack_ts) if t <= mid)
        ax.text(mid, (bottoms[i] + tops[i]) / 2000, f"{lab} {u // 1000}K", ha="center", va="center", fontsize=6.5, color=INK)
    bottoms = tops
ax.plot([t for t, _ in router], [v / 1000 for _, v in router], color="#6250d6", lw=1, alpha=0.6, label="router gauge the policy reads (telemetry, 2s scrapes)")
ax.plot(grid, [v / 1000 for v in mean_rebuilt], color=INK, lw=1.6, label=f"{a.lookback:.0f}s time-weighted mean of the rebuilt gauge")
ax.plot([t for t, _ in published], [v / 1000 for _, v in published], color="#6250d6", lw=1.6, ls="--", label="policy's published mean (prefill_required_replicas × 20K)")
ax.axhline(20, color=INK, ls=":", lw=1); ax.axhline(40, color=INK, ls=":", lw=0.6)
ax.text(a.end - 1, 20.5, "20K = 1 replica", ha="right", fontsize=7); ax.text(a.end - 1, 40.5, "40K = 2 replicas", ha="right", fontsize=7)
ax.set_ylabel("outstanding uncached prefill tokens, all replicas (K)"); ax.legend(fontsize=7.5, loc="upper left", frameon=False); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
ax.set_title("stacked: each request's uncached prompt, from queued to first token (rebuilt from the scheduler traces; labels on prompts over 15K)", fontsize=8.5, loc="left", color=INK2)
for sp in ax.spines.values(): sp.set_visible(False)
# arrivals
arr = []
for x, r in rows.items():
    m = meta.get(x)
    if not m: continue
    t = m["request_start_ns"] / 1e9 - t0
    if a.start <= t <= a.end: arr.append((t, float(r["engine_load_tokens"]), float(r["own_uncached"]), r["engine"].split("pid-")[-1].split(".")[0]))
ecol = {e: c for e, c in zip(engines, ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"])}
for t, ahead, own, e in arr:
    ax2.scatter(t, ahead / 1000, s=12 + own / 1500, color=ecol.get(e, INK2), alpha=0.85, edgecolor=SURFACE, lw=0.4, zorder=3)
# 180s-window stats as in the arrival-stats chart
w = 180
for k in range(int(a.start // w), int(a.end // w) + 1):
    lo, hi = k * w, (k + 1) * w
    vals = sorted(ahead for t, ahead, _, _ in arr if lo <= t < hi)
    allv = sorted(float(r["engine_load_tokens"]) for x, r in rows.items() if x in meta and lo <= meta[x]["request_start_ns"] / 1e9 - t0 < hi)
    if len(allv) >= 5:
        p95 = allv[min(len(allv) - 1, int(0.95 * len(allv)))]; mean = sum(allv) / len(allv)
        ax2.hlines(p95 / 1000, max(lo, a.start), min(hi, a.end), color="#2a78d6", lw=1.2, ls="--"); ax2.hlines(mean / 1000, max(lo, a.start), min(hi, a.end), color="#eb6834", lw=1.2, ls="--")
        ax2.text(max(lo, a.start) + 2, p95 / 1000 + 0.8, f"{lo}-{hi}s window: {len(allv)} requests, p95 {p95/1000:.0f}K, mean {mean/1000:.1f}K", fontsize=7, color=INK2)
ax2.axhline(20, color=INK, ls=":", lw=1)
ax2.set_ylabel("uncached tokens ahead at arrival (K)"); ax2.set_xlabel("seconds since profiling start"); ax2.grid(color=GRID, lw=0.6); ax2.set_facecolor(SURFACE)
ax2.set_title("dots: what each arriving request found ahead on its replica (own prompt excluded; dot size = own uncached tokens; colour = replica) · dashed: the 180s-window p95 and mean", fontsize=8.5, loc="left", color=INK2)
for sp in ax2.spines.values(): sp.set_visible(False)
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
for x in (ax, ax2):
    for e in ev.get("scale_events", []):
        if a.start <= e["time_s"] <= a.end: x.axvline(e["time_s"], color=INK2, lw=0.9); x.text(e["time_s"] + 1, x.get_ylim()[1] * 0.97, f"{int(e['from_replicas'])}→{int(e['to_replicas'])}", fontsize=8, fontweight="bold", va="top")
ax.set_xlim(a.start, a.end)
fig.suptitle(f"{os.path.basename(R).split('-', 1)[-1]} · {a.start:.0f}-{a.end:.0f}s: what the policy's mean was made of vs what arriving requests saw", fontsize=10, x=0.01, ha="left")
out = a.out or f"{R}/analysis/gauge_vs_arrival_{int(a.start)}_{int(a.end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
# numbers: contribution of each request to the mean at its peak
pk = max(range(len(grid)), key=lambda i: mean_rebuilt[i]); tpk = grid[pk]
print(f"rebuilt mean peaks at {tpk:.0f}s = {mean_rebuilt[pk]/1000:.1f}K; contributions over [{tpk-a.lookback:.0f},{tpk:.0f}]s:")
contrib = []
for q, f, u, lab, e in spans:
    ov = max(0, min(f, tpk) - max(q, tpk - a.lookback))
    if ov > 0: contrib.append((u * ov / a.lookback, lab, u, q, f, e))
for c, lab, u, q, f, e in sorted(contrib, reverse=True)[:8]: print(f"  {c/1000:5.1f}K  {lab:14s} uncached {u:6d} queued {q:6.1f}s first token {f:6.1f}s ({f-q:4.1f}s) replica pid {e}")
