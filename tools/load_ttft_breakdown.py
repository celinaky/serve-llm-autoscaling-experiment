#!/usr/bin/env python3
"""Client TTFT against the engine state on arrival, pooled over runs.

Column 1 of tools/state_breakdown.py at finer load bins (0, then 20K-wide up to
--max-load, then above it) and pooled over several runs' load_requests.csv
(from tools/load_breakdown.py). One row per band of the request's own uncached
tokens, with requests under 4K split at 704 uncached tokens: with 1568-token
blocks a 16,384-token step leaves 704 tokens beside a block-aligned chunk, which such a request can use whole, so it queues less at
the same load. Each bar is the mean client TTFT of its requests, split into its
components; a faint dot per run marks that run's mean TTFT in the cell (cells
with at least --min-n requests), so runs that disagree stand out. Requests
preempted before their first token are left out: their TTFT and load include
recompute.

--measure picks the x axis, one of state_breakdown.py's three columns:
  load        uncached prefill token load (default; bins as above)
  sequences   ongoing sequences, running plus waiting (4-wide bins)
  kv          KV context in use (250K-token bins, with % of capacity)
  policy      the default autoscaling policy's input as of arrival: Serve's
              ray_serve_autoscaling_total_requests (queued + ongoing requests,
              averaged over the look-back window and refreshed every few
              seconds), the last scrape before the request started
  inflight    the instantaneous count that metric averages: requests started
              by any client (AIPerf and side probes) and not yet finished when
              the request started, itself excluded
policy and inflight read the raw files load_breakdown.py and the telemetry
leave in each run (telemetry/serve_metrics.jsonl,
benchmark/*/profile_export.jsonl), which are not kept in git.
--bands original keeps state_breakdown.py's four rows (<4K undivided).

usage: python tools/load_ttft_breakdown.py --out <dir> runs/A [runs/B ...] [--measure load]
           [--bands split] [--min-n 3] [--max-load 600000]

Writes <out>/<measure>_ttft_breakdown[_4rows].{png,csv} and
<out>/<measure>_queue_by_size[_4rows].png.
"""
import argparse, bisect, csv, glob, json, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = [  # as in state_breakdown.py
    ("prefill_ms", "Prefill steps (engine)", "#2a78d6"),
    ("queue_ms", "Queue (engine)", "#eb6834"),
    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
]
SPLIT_BANDS = [(0, 705, "≤704"), (705, 4e3, "705–4K"), (4e3, 16e3, "4K–16K"), (16e3, 48e3, "16K–48K"),
               (48e3, math.inf, "≥48K")]
ORIGINAL_BANDS = [(0, 4e3, "<4K"), *SPLIT_BANDS[2:]]
BAND_COLORS = {"≤704": "#c23b80", "705–4K": "#2a78d6", "<4K": "#2a78d6", "4K–16K": "#eb6834",
               "16K–48K": "#1baf7a", "≥48K": "#8a5cd1"}
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--min-n", type=int, default=3, help="fewest requests for a per-run dot")
ap.add_argument("--max-load", type=float, default=600e3, help="20K bins up to this load, then one open bin")
ap.add_argument("--measure", choices=["load", "sequences", "kv", "policy", "inflight"], default="load")
ap.add_argument("--bands", choices=["split", "original"], default="split")
a = ap.parse_args()
SIZE_BANDS = SPLIT_BANDS if a.bands == "split" else ORIGINAL_BANDS
out_name = {"load": "load", "sequences": "seqs", "kv": "kv", "policy": "policy", "inflight": "inflight"}[a.measure] + "_{}" + ("" if a.bands == "split" else "_4rows")

rows = []
for R in a.runs:
    name = os.path.basename(R.rstrip("/"))
    for r in csv.DictReader(open(f"{R.rstrip('/')}/analysis/load_requests.csv")):
        if int(float(r["preemptions_before_first_token"])):
            continue
        for k, v in r.items():
            try:
                r[k] = float(v)
            except ValueError:
                pass
        r["run"] = name
        rows.append(r)
runs = [os.path.basename(R.rstrip("/")) for R in a.runs]


def serve_requests(R):
    """Request starts and ends from every client, and the policy metric's scrapes."""
    spans = {}
    for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
        for line in open(path):
            meta = json.loads(line)["metadata"]
            if meta.get("request_start_ns") and meta.get("request_end_ns"):
                spans[meta["x_request_id"]] = (meta["request_start_ns"], meta["request_end_ns"])
    scrapes = []
    for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
        e = json.loads(line)
        for m in e["samples"]:
            if m["normalized_name"] == "serve_autoscaling_total_requests":
                scrapes.append((e["scrape_started_at_ns"], m["value"]))
    return spans, sorted(scrapes)


if a.measure in ("policy", "inflight"):
    by_run = {}
    for R in a.runs:
        spans, scrapes = serve_requests(R.rstrip("/"))
        starts = sorted(s for s, _ in spans.values())
        ends = sorted(e for _, e in spans.values())
        by_run[os.path.basename(R.rstrip("/"))] = (spans, scrapes, starts, ends)
    kept = []
    for r in rows:
        spans, scrapes, starts, ends = by_run[r["run"]]
        if r["request_id"] not in spans:
            continue
        t = spans[r["request_id"]][0]
        # Started before it (strictly) and not finished by its start.
        r["inflight"] = bisect.bisect_left(starts, t) - bisect.bisect_right(ends, t)
        i = bisect.bisect_right(scrapes, (t, math.inf))
        r["policy_total_requests"] = scrapes[i - 1][1] if i else 0.0
        kept.append(r)
    print(f"{len(rows) - len(kept)} requests without client start/end times left out")
    rows = kept

# Bins as (lo, hi]; the first also includes lo.
if a.measure == "load":
    FIELD, NOUN, XLABEL = "engine_load_tokens", "uncached prefill token load", \
        "uncached prefill token load on the engine at arrival (tokens)"
    top = int(a.max_load // 20e3)
    BINS = [(0, 0, "0")] + [(i * 20e3, (i + 1) * 20e3, f"{i * 20}–{(i + 1) * 20}K" if i else "≤20K")
                            for i in range(top)] + [(top * 20e3, math.inf, f">{top * 20}K")]
elif a.measure == "sequences":
    FIELD, NOUN, XLABEL = "ongoing_sequences", "ongoing sequences", \
        "ongoing sequences on the engine at arrival (running + waiting)"
    BINS = [(0, 3, "0–3")] + [(4 * i - 1, 4 * i + 3, f"{4 * i}–{4 * i + 3}") for i in range(1, 22)] + \
        [(87, math.inf, "≥88")]
elif a.measure in ("policy", "inflight"):
    FIELD, NOUN, XLABEL = ("policy_total_requests", "the autoscaling policy's total requests",
                           "ray_serve_autoscaling_total_requests at arrival (queued + ongoing, look-back average)") \
        if a.measure == "policy" else \
        ("inflight", "requests in flight", "requests in flight at arrival (started and not finished, all clients)")
    BINS = [(0, 3, "0–3")] + [(4 * i - 1, 4 * i + 3, f"{4 * i}–{4 * i + 3}") for i in range(1, 22)] + \
        [(87, math.inf, "≥88")]
else:
    # KV usage is reported as a fraction of capacity; the context tokens track it exactly.
    usage = sorted(r["kv_context_tokens"] / r["kv_cache_usage"] for r in rows if r["kv_cache_usage"] > 0.05)
    capacity = usage[len(usage) // 2]
    FIELD, NOUN, XLABEL = "kv_context_tokens", "KV context in use", \
        f"KV context in use on the engine at arrival (tokens, % of the {capacity / 1e6:.2f}M capacity)"
    BINS = [(i * 250e3, (i + 1) * 250e3, f"{i * 250 / 1e3:g}–{(i + 1) * 250 / 1e3:g}M "
             f"({100 * i * 250e3 / capacity:.0f}–{100 * (i + 1) * 250e3 / capacity:.0f}%)") for i in range(16)] + \
        [(4e6, math.inf, f">4M (>{100 * 4e6 / capacity:.0f}%)")]


def bin_of(v):
    return next(i for i, (lo, hi, _) in enumerate(BINS) if (lo <= v <= hi if i == 0 else lo < v <= hi))


for r in rows:
    r["bin"] = bin_of(r[FIELD])

table = []
for s_lo, s_hi, s_name in SIZE_BANDS:
    for i, (*_, label) in enumerate(BINS):
        rs = [r for r in rows if s_lo <= r["own_uncached"] < s_hi and r["bin"] == i]
        if not rs:
            continue
        out = {"size_band": s_name, "bin": label, "bin_index": i, "requests": len(rs),
               "mean_own_uncached": sum(r["own_uncached"] for r in rs) / len(rs),
               "mean_client_ttft_ms": sum(r["client_ttft_ms"] for r in rs) / len(rs)}
        for key, *_ in COMPONENTS:
            out[f"mean_{key}"] = sum(r[key] for r in rs) / len(rs)
        table.append(out)
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/{out_name.format('ttft_breakdown')}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(len(SIZE_BANDS), 1, figsize=(22, 4.2 * len(SIZE_BANDS)), facecolor=SURFACE)
for ax, (s_lo, s_hi, s_name) in zip(axes, SIZE_BANDS):
    ts = [t for t in table if t["size_band"] == s_name]
    xs = [t["bin_index"] for t in ts]
    bottom = [0.0] * len(ts)
    for key, name, color in COMPONENTS:
        vals = [max(t[f"mean_{key}"], 0) / 1000 for t in ts]
        ax.bar(xs, vals, 0.72, bottom=bottom, color=color, label=name, edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    band_rows = [r for r in rows if s_lo <= r["own_uncached"] < s_hi]
    for run in runs:
        px, py = [], []
        for i in range(len(BINS)):
            sel = [r["client_ttft_ms"] for r in band_rows if r["run"] == run and r["bin"] == i]
            if len(sel) >= a.min_n:
                px.append(i)
                py.append(sum(sel) / len(sel) / 1000)
        ax.scatter(px, py, s=16, color=INK, alpha=0.35, zorder=3, linewidths=0)
    ymax = max(bottom) * 1.15
    for x, t, top in zip(xs, ts, bottom):
        ax.text(x, top + ymax * 0.01, f"{t['requests']}", ha="center", va="bottom", fontsize=7, color=INK2)
    ax.set_title(f"{s_name} own uncached ({len(band_rows)} requests; bar labels are request counts)",
                 loc="left", fontsize=10, color=INK)
    ax.set_xticks(range(len(BINS)), [label for *_, label in BINS], fontsize=8, rotation=60, ha="right")
    ax.set_xlim(-0.6, len(BINS) - 0.4)
    ax.set_xlabel(XLABEL, fontsize=9)
    ax.set_ylim(0, ymax)
    ax.set_ylabel("mean client TTFT (s)")
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
handles, labels = axes[0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK, alpha=0.35, marker="o", markersize=4, linestyle=""))
labels.append(f"one run's mean TTFT (cells with ≥{a.min_n} requests)")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.968), frameon=False, ncol=5, fontsize=9)
fig.suptitle(f"Client TTFT by {NOUN} on arrival: {len(rows)} requests pooled over "
             f"{len(runs)} runs (y axis per row)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{a.out}/{out_name.format('ttft_breakdown')}.png", dpi=120, facecolor=SURFACE)
print(f"{len(rows)} requests; wrote {a.out}/{out_name.format('ttft_breakdown')}.png and .csv")

# Engine queue time per bin, one line per size band on a shared axis.
fig, ax = plt.subplots(figsize=(18, 8), facecolor=SURFACE)
for _, _, s_name in SIZE_BANDS:
    color = BAND_COLORS[s_name]
    ts = [t for t in table if t["size_band"] == s_name]
    xs = [t["bin_index"] for t in ts]
    ys = [t["mean_queue_ms"] / 1000 for t in ts]
    ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label=f"{s_name} own uncached")
ax.set_xticks(range(len(BINS)), [label for *_, label in BINS], fontsize=8, rotation=60, ha="right")
ax.set_xlim(-0.4, len(BINS) - 0.6)
ax.set_xlabel(XLABEL, fontsize=9)
ax.set_ylabel("mean engine queue time (s)")
ax.set_ylim(0)
ax.set_facecolor(SURFACE)
ax.grid(axis="y", color=GRID, linewidth=0.8)
ax.set_axisbelow(True)
for side in ("top", "right"):
    ax.spines[side].set_visible(False)
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title(f"Engine queue time by {NOUN} on arrival, per request size ({len(rows)} requests, {len(runs)} runs)",
             loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(f"{a.out}/{out_name.format('queue_by_size')}.png", dpi=120, facecolor=SURFACE)
print(f"wrote {a.out}/{out_name.format('queue_by_size')}.png")
