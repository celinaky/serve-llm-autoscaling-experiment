#!/usr/bin/env python3
"""How full one replica's KV cache was, and with what: running requests vs cached prefixes.

Two stacked areas as percent of the cache: what running requests held (the
scheduler trace's kv_cache_usage at every batch) and the reconstructed cached
prefixes above it. vLLM logs no evictions, so the cached part is rebuilt: a
conversation's context is cached when its request finishes and is evicted,
least recently used first, when the cache minus the running share cannot fit
it. Red ticks on top mark arrivals the engine measured cold (context gone) at
turn 2 or later, i.e. re-prefills after eviction, with their size.

usage: python tools/replica_kv_fill.py runs/<run_dir> --engine <pid suffix>
           [--kv-tokens 1257989] [--traces DIR] [--out PATH]
"""
import argparse, glob, json, os
from collections import OrderedDict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--engine", required=True); ap.add_argument("--kv-tokens", type=int, default=1257989)
ap.add_argument("--traces", default=TRACES); ap.add_argument("--out"); a = ap.parse_args(); R = a.run.rstrip("/"); KV = a.kv_tokens
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); end = ev["periods"]["steady"]["end_s"]
meta = {}
for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(p):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling": meta[m["x_request_id"]] = m
path = glob.glob(f"{a.traces}/*pid-{a.engine}*.jsonl")[0]
queued, admitted, finished, usage, ctx_after, off = {}, {}, {}, [], {}, None
for line in open(path):
    e = json.loads(line); k = e["event"]
    if off is None and "wall_time_s" in e: off = e["wall_time_s"] - e["monotonic_s"]
    if k == "request_queued": queued[e["request_id"]] = e
    elif k == "request_admitted": admitted[e["request_id"]] = e
    elif k == "request_finished": finished[e["request_id"]] = e
    elif k == "batch_scheduled":
        usage.append((e["monotonic_s"] + off - t0, e["kv_cache_usage"]))
        for r in e["requests"]: ctx_after[r["request_id"]] = r["context_tokens_after"]
def conv(rid):
    m = meta.get(rid.split("chatcmpl-", 1)[-1].rsplit("-", 1)[0]); return (m["conversation_id"], m) if m else (None, None)
events = []
for rid, q in queued.items():
    c, m = conv(rid)
    if not c or rid not in admitted: continue
    ad = admitted[rid]; events.append((q["arrival_time_s"] - t0, "arrive", c, ad["prompt_tokens"], ad["cached_prefix_tokens"], m))
    if rid in finished: events.append((finished[rid]["monotonic_s"] + off - t0, "finish", c, ctx_after.get(rid, ad["prompt_tokens"]), None, m))
events.sort(key=lambda x: x[0]); usage.sort(); ui = 0
def running_at(t):
    global ui
    while ui + 1 < len(usage) and usage[ui + 1][0] <= t: ui += 1
    return usage[ui][1] if usage else 0
resident = OrderedDict(); ts, cached_pct, run_pct, cold = [], [], [], []
for t, kind, c, size, cached, m in events:
    if kind == "arrive":
        if cached < 0.5 * size and int(m.get("turn_index", 0)) >= 2 and size >= 20000: cold.append((t, size))
        if c in resident: resident.move_to_end(c)
    else:
        resident.pop(c, None); resident[c] = size
        while sum(resident.values()) > KV * (1 - running_at(t)) and len(resident) > 1: resident.popitem(last=False)
    ts.append(t); run_pct.append(running_at(t) * 100); cached_pct.append(sum(resident.values()) / KV * 100)
fig, ax = plt.subplots(figsize=(13, 4.8), dpi=150); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
ax.fill_between(ts, 0, run_pct, step="post", color="#eda100", alpha=0.75, lw=0, label="held by running requests (vLLM kv_cache_usage)")
ax.fill_between(ts, run_pct, [r + c for r, c in zip(run_pct, cached_pct)], step="post", color="#2a78d6", alpha=0.55, lw=0, label="cached prefixes of idle conversations (reconstructed, LRU)")
ax.axhline(100, color=INK, ls=":", lw=1)
for t, size in cold:
    ax.plot([t, t], [101, 109], color="#e34948", lw=1.6); ax.text(t, 110, f"{size // 1000}K", ha="center", fontsize=6.5, color="#b5432f")
ax.plot([], [], color="#e34948", lw=1.6, label="re-prefill: a returning conversation found its context evicted (label: tokens)")
for e in ev.get("scale_events", []): ax.axvline(e["time_s"], color=INK2, lw=0.8)
ax.set_xlim(0, end); ax.set_ylim(0, 118); ax.set_ylabel("% of the KV cache"); ax.set_xlabel("seconds since profiling start")
ax.legend(fontsize=7.5, loc="lower right", frameon=True, facecolor=SURFACE, edgecolor=GRID); ax.grid(axis="y", color=GRID, lw=0.6)
for sp in ax.spines.values(): sp.set_visible(False)
name = os.path.basename(R).split("-", 1)[-1]
ax.set_title(f"{name} · replica pid {a.engine}: what filled the {KV/1e6:.2f}M-token KV cache", fontsize=9.5, loc="left")
out = a.out or f"{R}/analysis/kv_fill_{a.engine}_0_{int(end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
