#!/usr/bin/env python3
"""Three plain panels: the policy's mean, how it splits per replica, and what arrivals saw.

1. Total outstanding uncached prefill tokens across replicas, as the 60s
   time-weighted mean, rebuilt from the scheduler traces (each request's
   uncached prompt counts from queued to first token), next to the mean the
   policy published. Horizontal lines mark 1, 2, 3... replicas' worth of the
   target.
2. The same 60s mean per replica, so you can see which replica the total
   lives on. Dotted line: the per-replica target.
3. Per window of --window seconds: the mean uncached tokens that arriving
   requests found ahead of them on their replica (own prompt excluded), with
   the number of requests under each bar. The line is the engine-side
   counterpart as a time average: the 60s time-weighted mean of uncached
   prefill still to do (chunks leave as their batches complete), summed over
   replicas. Bars below the line mean arrivals came when queues were empty.

usage: python tools/mean_signal_breakdown.py runs/<run_dir> [--window 60] [--lookback 60]
           [--target 20000] [--start 0] [--end END] [--traces DIR] [--out PATH]
"""
import argparse, csv, glob, json, os
from collections import defaultdict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

PALETTE = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#6250d6", "#008300", "#e34948"]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--window", type=float, default=60); ap.add_argument("--lookback", type=float, default=60)
ap.add_argument("--target", type=float, default=20000); ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float); ap.add_argument("--traces", default=TRACES); ap.add_argument("--out")
a = ap.parse_args(); R = a.run.rstrip("/")
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); end = a.end or ev["periods"]["steady"]["end_s"]
meta = {}
for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(p):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling": meta[m["x_request_id"]] = m
rows = list(csv.DictReader(open(f"{R}/analysis/load_requests.csv")))
engines = []
for r in rows:
    e = r["engine"].split("pid-")[-1].split(".")[0]
    if e not in engines: engines.append(e)
spans = defaultdict(list); deltas = []
for e in engines:
    path = glob.glob(f"{a.traces}/*pid-{e}*.jsonl")
    if not path: continue
    off = None; q, ad, ft = {}, {}, {}; chunks = defaultdict(list); done = {}
    for line in open(path[0]):
        d = json.loads(line); k = d["event"]
        if off is None and "wall_time_s" in d: off = d["wall_time_s"] - d["monotonic_s"]
        if k == "request_queued": q[d["request_id"]] = d
        elif k == "request_admitted": ad[d["request_id"]] = d
        elif k == "request_first_token": ft[d["request_id"]] = d
        elif k == "batch_scheduled":
            for r in d["requests"]:
                if r["phase"] == "prefill": chunks[r["request_id"]].append((d["batch_index"], r["scheduled_tokens"]))
        elif k == "batch_completed": done[d["batch_index"]] = d["monotonic_s"] + off - t0
    for rid, qq in q.items():
        if rid in ad and rid in ft: spans[e].append((qq["arrival_time_s"] - t0, ft[rid]["monotonic_s"] + off - t0, ad[rid]["prompt_tokens"] - ad[rid]["cached_prefix_tokens"]))
        cs = chunks.get(rid)
        if cs and all(b in done for b, _ in cs):
            deltas.append((qq["arrival_time_s"] - t0, sum(tk for _, tk in cs)))
            deltas.extend((done[b], -tk) for b, tk in cs)
grid = [a.start + i * 2 for i in range(int((end - a.start) / 2) + 1)]
def mean_series(sp):
    out = []
    for t in grid:
        lo = t - a.lookback; acc = 0.0
        for qs, fs, u in sp:
            ov = min(fs, t) - max(qs, lo)
            if ov > 0: acc += u * ov
        out.append(acc / a.lookback / 1000)
    return out
per = {e: mean_series(spans[e]) for e in engines}
deltas.sort(); dts = [t for t, _ in deltas]; levels = []; lv = 0.0
for _, dd in deltas: lv += dd; levels.append(lv)
def remaining_mean(t):
    lo = t - a.lookback; acc = 0.0; prev_t, prev_l = lo, 0.0
    import bisect
    i = bisect.bisect_right(dts, lo) - 1
    prev_l = levels[i] if i >= 0 else 0.0
    j = i + 1
    while j < len(dts) and dts[j] < t:
        acc += prev_l * (dts[j] - prev_t); prev_t, prev_l = dts[j], levels[j]; j += 1
    acc += prev_l * (t - prev_t)
    return acc / a.lookback / 1000
remaining = [remaining_mean(t) for t in grid]
total = [sum(v[i] for v in per.values()) for i in range(len(grid))]
pub = []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    d = json.loads(line); t = d["timestamp_ns"] / 1e9 - t0
    for s in d.get("samples") or []:
        if s["name"] == "ray_serve_autoscaling_policy_metric" and s["labels"].get("metric") == "prefill_required_replicas": pub.append((t, s["value"] * a.target / 1000))
arr = defaultdict(list)
for r in rows:
    m = meta.get(r["request_id"])
    if not m: continue
    t = m["request_start_ns"] / 1e9 - t0
    if a.start <= t < end: arr[int((t - a.start) // a.window)].append(float(r["engine_load_tokens"]) / 1000)
fig, axes = plt.subplots(3, 1, figsize=(14, 11), dpi=150, sharex=True, gridspec_kw={"hspace": 0.12}); fig.patch.set_facecolor(SURFACE)
ax = axes[0]
ax.plot(grid, total, color=INK, lw=1.8, label=f"PROMPTS IN PREFILL: {a.lookback:.0f}s mean of the uncached prompt tokens of every request between routing and prefill complete, own tokens included (engine traces)")
ax.plot([t for t, _ in pub], [v for _, v in pub], color="#6250d6", lw=1.4, ls="--", label="the same thing as the policy published it (router gauge mean, what it divided by the target)")
top = max(max(total), max(v for _, v in pub) if pub else 0)
for k in range(1, int(top / (a.target / 1000)) + 2):
    ax.axhline(k * a.target / 1000, color=INK, ls=":", lw=0.7); ax.text(end - 2, k * a.target / 1000 + 0.6, f"{k} replica{'s' if k > 1 else ''}", ha="right", fontsize=7, color=INK2)
ax.set_ylabel("K prompt tokens in prefill, all replicas"); ax.legend(fontsize=8, loc="upper left", frameon=False); ax.set_title("1. PROMPTS IN PREFILL, the policy's signal: how many uncached prompt tokens were being prefilled, averaged over time", fontsize=9.5, loc="left")
ax = axes[1]
for i, e in enumerate(engines):
    ax.plot(grid, per[e], color=PALETTE[i % len(PALETTE)], lw=1.5, label=f"replica {i + 1}")
ax.axhline(a.target / 1000, color=INK, ls=":", lw=0.9); ax.text(end - 2, a.target / 1000 + 0.6, f"target {a.target/1000:.0f}K per replica", ha="right", fontsize=7, color=INK2)
ax.set_ylabel("K prompt tokens in prefill, per replica"); ax.legend(fontsize=8, loc="upper left", frameon=False, ncol=len(engines)); ax.set_title("2. PROMPTS IN PREFILL split by replica", fontsize=9.5, loc="left")
ax = axes[2]
for k, vals in sorted(arr.items()):
    lo = a.start + k * a.window; mean = sum(vals) / len(vals)
    ax.bar(lo + a.window / 2, mean, width=a.window * 0.9, color="#eb6834", alpha=0.8)
    ax.text(lo + a.window / 2, -0.04 * max(1, max(sum(v) / len(v) for v in arr.values())), str(len(vals)), ha="center", va="top", fontsize=6.5, color=INK2)
ax.plot(grid, remaining, color=INK, lw=1.6, label=f"QUEUED WORK, time average: {a.lookback:.0f}s mean of prompt tokens still waiting to be prefilled, all replicas (engine traces)")
ax.bar([0], [0], width=0, color="#eb6834", alpha=0.8, label=f"QUEUED AHEAD AT ARRIVAL: tokens already waiting on the replica when a request arrived, own tokens excluded, mean per {a.window:.0f}s window")
ax.axhline(a.target / 1000, color=INK, ls=":", lw=0.9); ax.legend(fontsize=8, loc="upper left", frameon=False)
ax.set_ylabel("K tokens queued"); ax.set_xlabel("seconds since profiling start"); ax.set_title("3. QUEUED WORK vs QUEUED AHEAD AT ARRIVAL: the same queue averaged over time, and sampled when requests arrived (requests per window under each bar)", fontsize=9.5, loc="left")
for x in axes:
    x.grid(color=GRID, lw=0.6); x.set_facecolor(SURFACE); x.set_xlim(a.start, end)
    for sp in x.spines.values(): sp.set_visible(False)
    for e in ev.get("scale_events", []): x.axvline(e["time_s"], color=INK2, lw=0.8)
for e in ev.get("scale_events", []): axes[0].text(e["time_s"] + 2, axes[0].get_ylim()[1] * 0.95, f"{int(e['from_replicas'])}→{int(e['to_replicas'])}", fontsize=8, fontweight="bold", va="top")
fig.suptitle(f"{os.path.basename(R).split('-', 1)[-1]} · three different measurements, all 60s means: prompts in prefill (the policy signal), queued work, queued ahead at arrival", fontsize=10, x=0.01, ha="left")
out = a.out or f"{R}/analysis/mean_signal_breakdown{int(a.window)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
