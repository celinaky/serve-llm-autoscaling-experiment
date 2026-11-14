#!/usr/bin/env python3
"""What one replica's KV cache held over time: contexts resident vs evicted, and usage.

vLLM does not log evictions, so residency is reconstructed: a conversation's
context (its last prompt plus output) enters the prefix cache when its request
finishes and stays until the cache needs the space, when the conversation
least recently used is evicted first (vLLM frees cached blocks LRU). The
budget is the KV cache minus what running requests hold at that moment (the
scheduler trace's kv_cache_usage). The reconstruction is checked against what
the engine actually reported at each admission (cached_prefix_tokens in the
trace): a marker at each arrival shows measured warm (most of the prompt
cached) or cold, and a hollow marker where the reconstruction disagreed.

Top: stacked area of resident context per conversation (one colour each, the
largest conversations named), with arrivals marked. Bottom: KV cache usage
from the trace (running requests) and the reconstructed resident-prefix share,
both in percent of the cache.

usage: python tools/replica_kv_residency.py runs/<run_dir> --engine <pid suffix>
           [--kv-tokens 1257989] [--traces DIR] [--out PATH]
"""
import argparse, csv, glob, json, os
from collections import defaultdict, OrderedDict
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--engine", required=True); ap.add_argument("--kv-tokens", type=int, default=1257989)
ap.add_argument("--traces", default=TRACES); ap.add_argument("--out"); a = ap.parse_args(); R = a.run.rstrip("/"); KV = a.kv_tokens
pd_ = json.load(open(f"{R}/analysis/plot_data.json")); t0 = pd_["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); end = ev["periods"]["steady"]["end_s"]
meta = {}
for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(p):
        m = json.loads(line)["metadata"]
        if m.get("benchmark_phase") == "profiling": meta[m["x_request_id"]] = m
path = [p for p in glob.glob(f"{a.traces}/*pid-{a.engine}*.jsonl")][0]
queued, admitted, finished, usage, ctx_after = {}, {}, {}, [], {}
mono_to_wall = None
for line in open(path):
    e = json.loads(line); evn = e["event"]
    if mono_to_wall is None and "wall_time_s" in e and "monotonic_s" in e: mono_to_wall = e["wall_time_s"] - e["monotonic_s"]
    if evn == "request_queued": queued[e["request_id"]] = e
    elif evn == "request_admitted": admitted[e["request_id"]] = e
    elif evn == "request_finished": finished[e["request_id"]] = e
    elif evn == "batch_scheduled":
        usage.append((e["monotonic_s"] + mono_to_wall - t0, e["kv_cache_usage"]))
        for r in e["requests"]: ctx_after[r["request_id"]] = r["context_tokens_after"]
def conv(rid):
    x = rid.split("chatcmpl-", 1)[-1].rsplit("-", 1)[0]; m = meta.get(x)
    return (m["conversation_id"], m) if m else (None, None)
events = []  # (time, kind, conv, size, measured_cached, prompt)
for rid, q in queued.items():
    c, m = conv(rid)
    if not c or rid not in admitted: continue
    ad = admitted[rid]; t_arr = q["arrival_time_s"] - t0
    events.append((t_arr, "arrive", c, ad["prompt_tokens"], ad["cached_prefix_tokens"], m))
    if rid in finished: events.append((finished[rid]["monotonic_s"] + mono_to_wall - t0, "finish", c, ctx_after.get(rid, ad["prompt_tokens"]), None, m))
events.sort(key=lambda x: x[0])
usage.sort(); ui = 0
def running_at(t):
    global ui
    while ui + 1 < len(usage) and usage[ui + 1][0] <= t: ui += 1
    return usage[ui][1] * KV if usage else 0
resident = OrderedDict()  # conv -> tokens, ordered by last use (oldest first)
series = defaultdict(list); times = []; snaps = []; arrivals = []; agree = disagree = 0
def snapshot(t):
    times.append(t); snaps.append(dict(resident))
for t, kind, c, size, cached, m in events:
    if kind == "arrive":
        predicted_warm = c in resident and resident[c] >= 0.5 * size
        measured_warm = cached >= 0.5 * size
        ok = predicted_warm == measured_warm; agree += ok; disagree += (not ok)
        arrivals.append((t, c, measured_warm, ok, size, cached))
        if c in resident: resident.move_to_end(c)
    else:
        resident.pop(c, None); resident[c] = size
        budget = KV - running_at(t)
        while sum(resident.values()) > budget and len(resident) > 1:
            resident.popitem(last=False)
    snapshot(t)
convs = sorted({c for s in snaps for c in s}, key=lambda c: max(s.get(c, 0) for s in snaps), reverse=True)
import colorsys
cols = {c: colorsys.hsv_to_rgb((i * 0.618) % 1, 0.45 + 0.3 * (i % 2), 0.78 if i % 3 else 0.6) for i, c in enumerate(convs)}
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 9.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [2.6, 1.2], "hspace": 0.08})
fig.patch.set_facecolor(SURFACE)
stack = [[s.get(c, 0) / 1e6 for s in snaps] for c in convs]
ax.stackplot(times, stack, colors=[cols[c] for c in convs], step="post", lw=0.3, edgecolor=SURFACE)
ax.axhline(KV / 1e6, color=INK, ls=":", lw=1); ax.text(end - 2, KV / 1e6 + 0.02, f"KV cache {KV/1e6:.2f}M tokens", ha="right", fontsize=8)
tot = [sum(s.values()) / 1e6 for s in snaps]; ax.step(times, tot, where="post", color=INK, lw=0.9)
# name the largest conversations at their peak
for c in convs[:8]:
    i = max(range(len(snaps)), key=lambda k: snaps[k].get(c, 0))
    if snaps[i].get(c, 0) > 60000:
        base = sum(snaps[i].get(d, 0) for d in convs[:convs.index(c)]) / 1e6
        ax.text(times[i], base + snaps[i][c] / 2e6, c[-6:] if "::" not in c else c.split("::")[0][-6:] + "/sa", fontsize=6.5, color=INK, va="center")
for t, c, warm, ok, size, cached in arrivals:
    if size < 20000: continue
    y = KV / 1e6 * 1.12 if warm else KV / 1e6 * 1.2
    ax.scatter(t, y, s=10 + size / 4000, facecolor=("#2a78d6" if warm else "#e34948") if ok else SURFACE, edgecolor="#2a78d6" if warm else "#e34948", lw=0.9, zorder=5)
ax.set_ylim(0, KV / 1e6 * 1.3); ax.set_ylabel("resident context, million tokens"); ax.grid(axis="y", color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
for sp in ax.spines.values(): sp.set_visible(False)
handles = [Line2D([0], [0], marker="o", ls="", color="#2a78d6", label="arrival measured warm (engine found most of the prompt cached)"),
           Line2D([0], [0], marker="o", ls="", color="#e34948", label="arrival measured cold (context gone, full prefill)"),
           Line2D([0], [0], marker="o", ls="", markerfacecolor=SURFACE, color=INK2, label="hollow: reconstruction disagreed with the engine"),
           Line2D([0], [0], color=INK, lw=0.9, label="total reconstructed resident context")]
ax.legend(handles=handles, fontsize=7.5, loc="upper left", frameon=False)
ax2.plot([t for t, _ in usage], [u * 100 for _, u in usage], color="#eda100", lw=0.7, label="KV cache usage by running requests (vLLM kv_cache_usage, every batch)")
ax2.step(times, [v / KV * 100 for v in [sum(s.values()) for s in snaps]], where="post", color=INK, lw=0.9, label="reconstructed cached-prefix share")
ax2.axhline(100, color=INK, ls=":", lw=1); ax2.set_ylim(0, 115); ax2.set_ylabel("% of KV cache"); ax2.set_xlabel("seconds since profiling start")
ax2.legend(fontsize=7.5, loc="upper left", frameon=False); ax2.grid(color=GRID, lw=0.6); ax2.set_facecolor(SURFACE)
for sp in ax2.spines.values(): sp.set_visible(False)
for x in (ax, ax2):
    for e in ev.get("scale_events", []): x.axvline(e["time_s"], color=INK2, lw=0.8)
ax.set_xlim(0, end)
name = os.path.basename(R).split("-", 1)[-1]
fig.suptitle(f"{name} · replica pid {a.engine}: contexts resident in the KV cache (reconstructed, LRU) and cache usage · reconstruction matched the engine's cached/uncached outcome on {agree} of {agree+disagree} arrivals", fontsize=9.5, x=0.01, ha="left")
out = a.out or f"{R}/analysis/kv_residency_{a.engine}_0_{int(end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
print(f"arrivals {agree+disagree}: reconstruction agreed on {agree}, disagreed on {disagree}")
for t, c, warm, ok, size, cached in arrivals:
    if not ok and size >= 20000: print(f"  disagree t={t:6.0f} {c[-8:]} prompt {size} cached {cached} measured {'warm' if warm else 'cold'}")
