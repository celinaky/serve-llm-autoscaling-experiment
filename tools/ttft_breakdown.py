#!/usr/bin/env python3
"""Client TTFT split into pre-engine, queue, prefill and post-engine time.

Joins AIPerf's per-request records to the vLLM scheduler trace by request ID,
buckets requests by client start time and plots the mean of each component
per bucket, cold and warm turns separately, above the running replica count.
With --normalize, each bucket shows component time per 1K prompt tokens (isl)
or per 1K prompt tokens not served from the prefix cache (uncached), computed
as a ratio of sums so short prompts' fixed overheads do not dominate.

usage: python tools/ttft_breakdown.py runs/<run_dir> \
           [--traces /mnt/cluster_storage/.../vllm-scheduler-traces] [--step 30] \
           [--normalize {none,isl,uncached}]

Writes analysis/ttft_breakdown[_per_<normalize>].{png,csv} in the run.
"""
import argparse, csv, glob, json, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
# vLLM turns X-Request-ID <id> into chatcmpl-<id>-<8 random hex>.
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
COMPONENTS = [  # stacked bottom to top; colors are categorical slots 1-4
    ("prefill_ms", "Prefill (engine)", "#2a78d6"),
    ("queue_ms", "Queue wait (engine)", "#eb6834"),
    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--traces", default=DEFAULT_TRACES)
ap.add_argument("--step", type=float, default=30)
ap.add_argument("--normalize", choices=["none", "isl", "uncached"], default="none")
a = ap.parse_args()
suffix = "" if a.normalize == "none" else f"_per_{a.normalize}"
unit = {"none": "seconds", "isl": "ms per 1K prompt tokens",
        "uncached": "ms per 1K uncached prompt tokens"}[a.normalize]
R = a.run.rstrip("/")

pd = json.load(open(f"{R}/analysis/plot_data.json"))
origin_ns = pd["time_origin"]["profiling_start_ns"]

aiperf = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        meta, m = rec["metadata"], rec.get("metrics") or {}
        if meta.get("benchmark_phase") != "profiling" or rec.get("error") or "time_to_first_token" not in m:
            continue
        isl = m.get("input_sequence_length", {}).get("value")
        cached = m.get("usage_prompt_cache_read_tokens", {}).get("value")
        aiperf[meta["x_request_id"]] = {
            "start_ns": meta["request_start_ns"],
            "ttft_ms": m["time_to_first_token"]["value"],
            "turn": meta.get("turn_index"),
            "isl": isl,
            "uncached": None if not isl or cached is None else isl - cached,
            # Same rule as the harness: cold when under half the prompt was cached.
            "cold": None if not isl or cached is None else cached < 0.5 * isl,
        }

# Only request_queued and request_first_token are needed; skip parsing the rest.
queued, first = {}, {}
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    for line in open(path):
        if '"request_queued"' not in line and '"request_first_token"' not in line:
            continue
        try:
            e = json.loads(line)
        except ValueError:
            continue
        m = ENGINE_ID.match(e["request_id"])
        rid = m.group(1) if m else e["request_id"]
        if rid in aiperf:
            (queued if e["event"] == "request_queued" else first)[rid] = e

rows = []
for rid, r in aiperf.items():
    q, f = queued.get(rid), first.get(rid)
    if not q or not f or f.get("queue_wait_ms") is None or r["cold"] is None:
        continue
    pre = (q["wall_time_s"] * 1e9 - r["start_ns"]) / 1e6
    rows.append({
        "t_s": (r["start_ns"] - origin_ns) / 1e9, "cold": r["cold"], "turn": r["turn"],
        "isl": r["isl"], "uncached": r["uncached"],
        "ttft_ms": r["ttft_ms"], "pre_ms": pre, "queue_ms": f["queue_wait_ms"],
        "prefill_ms": f["prefill_span_ms"],
        "post_ms": r["ttft_ms"] - pre - f["engine_ttft_ms"],
    })
print(f"joined {len(rows)} of {len(aiperf)} successful AIPerf requests")

buckets = defaultdict(list)
for row in rows:
    buckets[(int(row["t_s"] // a.step), row["cold"])].append(row)
table = []
for (b, cold), rs in sorted(buckets.items()):
    out = {"bucket_start_s": b * a.step, "kind": "cold" if cold else "warm", "requests": len(rs),
           "mean_isl": sum(r["isl"] for r in rs) / len(rs),
           "mean_uncached": sum(r["uncached"] for r in rs) / len(rs)}
    # Plotted value: seconds of mean time, or ms per 1K (uncached) prompt tokens.
    denom = len(rs) * 1000 if a.normalize == "none" else sum(r[a.normalize] for r in rs) / 1000
    for key in ["ttft_ms"] + [c[0] for c in COMPONENTS]:
        out[f"mean_{key}"] = sum(r[key] for r in rs) / len(rs)
        out[f"plot_{key}"] = sum(r[key] for r in rs) / denom if denom else 0.0
    table.append(out)
with open(f"{R}/analysis/ttft_breakdown{suffix}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

status = pd["serve_status_samples"]
load_end = pd["experiment"].get("duration_s")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(3, 1, figsize=(13, 11.5), sharex=True,
                         gridspec_kw={"height_ratios": [3, 3, 1.3]}, facecolor=SURFACE)
width = a.step * 0.86
ymax = max(t["plot_ttft_ms"] for t in table) * 1.12
for ax, kind in zip(axes[:2], ("cold", "warm")):
    ts = [t for t in table if t["kind"] == kind]
    xs = [t["bucket_start_s"] + a.step / 2 for t in ts]
    bottom = [0.0] * len(ts)
    for key, label, color in COMPONENTS:
        vals = [max(t[f"plot_{key}"], 0) for t in ts]
        ax.bar(xs, vals, width, bottom=bottom, color=color, label=label,
               edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, t, top in zip(xs, ts, bottom):
        ax.text(x, top + ymax * 0.01, f"n={t['requests']}", ha="center", va="bottom",
                fontsize=7, color=INK2)
    n = sum(t["requests"] for t in ts)
    ax.set_title(f"{kind.capitalize()} turns ({'<' if kind == 'cold' else '≥'}50% of prompt cached, "
                 f"{n} requests): client TTFT per {a.step:.0f}s bucket"
                 + ("" if a.normalize == "none" else f", {unit}"),
                 loc="left", fontsize=11, color=INK)
    ax.set_ylabel(unit)
    ax.set_ylim(0, ymax)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), frameon=False,
           ncol=4, fontsize=9)

ax = axes[2]
st_t = [s["relative_time_s"] for s in status]
ax.step(st_t, [s["running_replicas"] for s in status], where="post", color=INK, linewidth=2,
        label="running")
ax.step(st_t, [s["running_replicas"] + s["starting_replicas"] for s in status], where="post",
        color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), label="running + starting")
ax.set_ylabel("replicas")
ax.set_title("Replicas", loc="left", fontsize=11, color=INK)
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_ylim(0, max(s["running_replicas"] + s["starting_replicas"] for s in status) + 1)

xmax = max(t["bucket_start_s"] for t in table) + a.step
for ax in axes:
    ax.tick_params(labelbottom=True)
    ax.set_xlabel("seconds since first request")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if load_end:
        ax.axvline(load_end, color=INK2, linewidth=1, linestyle=":")
    ax.set_xlim(0, xmax)
axes[0].text(load_end, ymax * 0.98, " load ends", fontsize=8, color=INK2, va="top")
fig.suptitle(f"{pd['experiment'].get('name', os.path.basename(R))}: where client TTFT goes",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{R}/analysis/ttft_breakdown{suffix}.png", dpi=130, facecolor=SURFACE)
print(f"wrote {R}/analysis/ttft_breakdown{suffix}.png and .csv")
