#!/usr/bin/env python3
"""One graph: the router's published mean against the engine's actual signal, same definition, same lookback.

Signal: total uncached prefill tokens outstanding across all replicas, where a
request's uncached prompt counts from the moment it is queued until its first
token (that is how the router's tracker counts it: added at routing, removed
at prefill complete).
  grey  : the engine's actual value at each instant, rebuilt from the
          scheduler traces with the engine's measured uncached tokens
  black : its time-weighted mean over the last --lookback seconds
  purple: the mean the router/policy published over the same lookback
          (prefill_required_replicas x target, from telemetry)

usage: python tools/router_vs_engine_signal.py runs/<run_dir> [--lookback 60] [--target 20000]
           [--start 0] [--end END] [--traces DIR] [--out PATH]
"""
import argparse, csv, glob, json, os, bisect
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--lookback", type=float, default=60); ap.add_argument("--target", type=float, default=20000)
ap.add_argument("--start", type=float, default=0); ap.add_argument("--end", type=float); ap.add_argument("--traces", default=TRACES); ap.add_argument("--out")
a = ap.parse_args(); R = a.run.rstrip("/")
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); end = a.end or ev["periods"]["steady"]["end_s"]
engines = []
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    e = r["engine"].split("pid-")[-1].split(".")[0]
    if e not in engines: engines.append(e)
deltas = []
for e in engines:
    path = glob.glob(f"{a.traces}/*pid-{e}*.jsonl")
    if not path: continue
    off = None; q, ad, ft = {}, {}, {}
    for line in open(path[0]):
        d = json.loads(line); k = d["event"]
        if off is None and "wall_time_s" in d: off = d["wall_time_s"] - d["monotonic_s"]
        if k == "request_queued": q[d["request_id"]] = d
        elif k == "request_admitted": ad[d["request_id"]] = d
        elif k == "request_first_token": ft[d["request_id"]] = d
    for rid, qq in q.items():
        if rid in ad and rid in ft:
            u = ad[rid]["prompt_tokens"] - ad[rid]["cached_prefix_tokens"]
            deltas.append((qq["arrival_time_s"] - t0, u)); deltas.append((ft[rid]["monotonic_s"] + off - t0, -u))
deltas.sort(); dts = [t for t, _ in deltas]; levels = []; lv = 0
for _, dd in deltas: lv += dd; levels.append(lv)
def level_at(t):
    i = bisect.bisect_right(dts, t) - 1; return levels[i] if i >= 0 else 0
def mean_at(t):
    lo = t - a.lookback; acc = 0.0
    i = bisect.bisect_right(dts, lo) - 1; prev_t = lo; prev_l = levels[i] if i >= 0 else 0; j = i + 1
    while j < len(dts) and dts[j] < t:
        acc += prev_l * (dts[j] - prev_t); prev_t, prev_l = dts[j], levels[j]; j += 1
    acc += prev_l * (t - prev_t); return acc / a.lookback
grid = [a.start + i for i in range(int(end - a.start) + 1)]
inst = [level_at(t) / 1000 for t in grid]; mean = [mean_at(t) / 1000 for t in grid]
pub = []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    d = json.loads(line); t = d["timestamp_ns"] / 1e9 - t0
    for s in d.get("samples") or []:
        if s["name"] == "ray_serve_autoscaling_policy_metric" and s["labels"].get("metric") == "prefill_required_replicas": pub.append((t, s["value"] * a.target / 1000))
fig, ax = plt.subplots(figsize=(14, 5.5), dpi=150); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.step(grid, inst, where="post", color="#9b9a95", lw=0.7, alpha=0.8, label="engine, actual value at each instant (rebuilt from the scheduler traces)")
ax.plot(grid, mean, color=INK, lw=1.8, label=f"engine, {a.lookback:.0f}s time-weighted mean of that")
ax.plot([t for t, _ in pub], [v for _, v in pub], color="#6250d6", lw=1.8, ls="--", label=f"router, the {a.lookback:.0f}s mean it published (prefill_required_replicas × {a.target/1000:.0f}K)")
top = max(max(mean), max(v for _, v in pub) if pub else 0) * 1.15
for k in range(1, int(top / (a.target / 1000)) + 1):
    ax.axhline(k * a.target / 1000, color=INK, ls=":", lw=0.6); ax.text(end - 2, k * a.target / 1000 + 0.6, f"{k} replica{'s' if k > 1 else ''}", ha="right", fontsize=7, color=INK2)
for e in ev.get("scale_events", []):
    ax.axvline(e["time_s"], color=INK2, lw=0.8); ax.text(e["time_s"] + 2, top * 0.97, f"{int(e['from_replicas'])}→{int(e['to_replicas'])}", fontsize=8, fontweight="bold", va="top")
ax.set_xlim(a.start, end); ax.set_ylim(0, top); ax.set_xlabel("seconds since profiling start"); ax.set_ylabel("K uncached prefill tokens outstanding, all replicas")
ax.legend(fontsize=8, loc="upper left", frameon=False); ax.grid(color=GRID, lw=0.6)
for sp in ax.spines.values(): sp.set_visible(False)
ax.set_title(f"{os.path.basename(R).split('-', 1)[-1]} · total uncached prefill tokens outstanding (each request's uncached prompt, from queued to first token): router vs engine", fontsize=9.5, loc="left")
out = a.out or f"{R}/analysis/router_vs_engine_signal{int(a.lookback)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
# summary numbers
pairs = [(t, v, mean_at(t) / 1000) for t, v in pub if a.start + a.lookback < t < end]
import statistics
rat = [v / m for t, v, m in pairs if m > 5]
print(f"router/engine mean ratio where engine mean > 5K: median {statistics.median(rat):.2f}, p10 {sorted(rat)[len(rat)//10]:.2f}, p90 {sorted(rat)[9*len(rat)//10]:.2f}")
worst = sorted(pairs, key=lambda x: x[2] - x[1], reverse=True)[:5]
print("largest engine-above-router gaps:", [(round(t), round(v), round(m)) for t, v, m in worst])
