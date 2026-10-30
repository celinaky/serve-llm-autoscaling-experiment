#!/usr/bin/env python3
"""Observed client TTFT against the engine state each request found on arrival.

Reads the per-request table from tools/load_breakdown.py and bins requests by
three measures of the engine that served them, taken when they arrived (their
own work excluded):

  uncached prefill token load   uncached tokens still to prefill, queued or
                                mid-prefill
  ongoing sequences             running plus waiting requests
  KV context in use             context tokens of every sequence in the most
                                recent step: what attention covers, so what
                                drives decode step length

Columns are the three measures; rows are bands of the request's own uncached
tokens, so request size is held roughly fixed within a row instead of being
divided out. Each bar is the mean client TTFT of its requests in seconds,
split into its components. The measures move together (sequences with KV
especially), so each column shows association, not the effect of one measure
with the others held fixed. Requests from every replica are pooled; each is
placed by its own engine's state. --engine limits the figure to engines whose
trace file name contains the given text.

A second figure draws the same cells as lines: mean client TTFT, queue and
prefill steps, bold for all engines pooled and faint per engine in the same
color, so engines that follow the same trend read as one group.

usage: python tools/state_breakdown.py runs/<run_dir> [--engine pid-79255] [--min-n 2]

Writes analysis/state_breakdown[_<engine>].{png,csv} and
analysis/state_breakdown_lines[_<engine>].png in the run.
"""
import argparse, csv, math, os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COMPONENTS = [  # stacked bottom to top; categorical slots 1-4, as in load_breakdown
    ("prefill_ms", "Prefill steps (engine)", "#2a78d6"),
    ("queue_ms", "Queue (engine)", "#eb6834"),
    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
]
SIZE_BANDS = [(0, 4e3), (4e3, 16e3), (16e3, 48e3), (48e3, math.inf)]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--engine", help="only engines whose trace file name contains this")
ap.add_argument("--min-n", type=int, default=2,
                help="fewest requests for a per-engine point in the line figure")
a = ap.parse_args()
R = a.run.rstrip("/")
suffix = f"_{a.engine}" if a.engine else ""

rows = list(csv.DictReader(open(f"{R}/analysis/load_requests.csv")))
for r in rows:
    for k, v in r.items():
        try:
            r[k] = float(v)
        except ValueError:
            pass
# vLLM reports KV usage as a fraction of capacity; KV context tokens track it
# exactly, so their ratio is the engine's KV capacity in tokens.
usage = sorted(r["kv_context_tokens"] / r["kv_cache_usage"] for r in rows if r["kv_cache_usage"] > 0.05)
capacity = usage[len(usage) // 2]
if a.engine:
    rows = [r for r in rows if a.engine in r["engine"]]
engines = sorted({r["engine"] for r in rows})


def fmt_k(v):
    return f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1e3:g}K"


def range_label(lo, hi):
    if hi == math.inf:
        return f">{fmt_k(lo)}"
    return f"{fmt_k(lo)}–{fmt_k(hi)}" if lo else f"≤{fmt_k(hi)}"


def size_label(lo, hi):
    return (f"≥{fmt_k(lo)}" if hi == math.inf else f"{fmt_k(lo)}–{fmt_k(hi)}" if lo else f"<{fmt_k(hi)}") + \
        " own uncached"


# (field, axis label, bins as (lo, hi]; the first bin includes lo, tick label)
MEASURES = [
    ("engine_load_tokens", "uncached prefill token load on the engine (tokens)",
     [(0, 0, "0"), *((lo, hi, range_label(lo, hi)) for lo, hi in
                     [(0, 16e3), (16e3, 32e3), (32e3, 64e3), (64e3, math.inf)])]),
    ("ongoing_sequences", "ongoing sequences on the engine (running + waiting)",
     [(0, 3, "0–3"), (3, 7, "4–7"), (7, 11, "8–11"), (11, 15, "12–15"), (15, math.inf, "≥16")]),
    ("kv_context_tokens", f"KV context in use on the engine (tokens; capacity ≈{capacity / 1e6:.1f}M)",
     [(lo, hi, range_label(lo, hi) + (f"\n({100 * lo / capacity:.0f}–{100 * hi / capacity:.0f}%)"
                                      if hi != math.inf else f"\n(>{100 * lo / capacity:.0f}%)"))
      for lo, hi in [(0, 250e3), (250e3, 500e3), (500e3, 750e3), (750e3, 1e6), (1e6, math.inf)]]),
]


def in_bin(v, lo, hi, first):
    return lo <= v <= hi if first else lo < v <= hi


table = []
for s_lo, s_hi in SIZE_BANDS:
    band = [r for r in rows if s_lo <= r["own_uncached"] < s_hi]
    for field, _, bins in MEASURES:
        for i, (lo, hi, label) in enumerate(bins):
            rs = [r for r in band if in_bin(r[field], lo, hi, i == 0)]
            if not rs:
                continue
            out = {"size_band": size_label(s_lo, s_hi), "measure": field, "bin": label.replace("\n", " "),
                   "bin_index": i, "requests": len(rs),
                   "mean_own_uncached": sum(r["own_uncached"] for r in rs) / len(rs),
                   "mean_client_ttft_ms": sum(r["client_ttft_ms"] for r in rs) / len(rs)}
            for key, *_ in COMPONENTS:
                out[f"mean_{key}"] = sum(r[key] for r in rs) / len(rs)
            table.append(out)
with open(f"{R}/analysis/state_breakdown{suffix}.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
print(f"{len(rows)} requests from {len(engines)} engine(s)")
for t in table:
    print(f"{t['size_band']:22} {t['measure']:19} {t['bin']:22} n={t['requests']:4}  "
          f"TTFT {t['mean_client_ttft_ms']:6.0f}  queue {t['mean_queue_ms']:5.0f}  prefill {t['mean_prefill_ms']:5.0f}  "
          f"pre {t['mean_pre_ms']:4.0f}  post {t['mean_post_ms']:3.0f}  (ms)")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(len(SIZE_BANDS), len(MEASURES), figsize=(15, 16), sharey=True, facecolor=SURFACE)
ymax = max(t["mean_client_ttft_ms"] for t in table) / 1000 * 1.15
for row, (s_lo, s_hi) in enumerate(SIZE_BANDS):
    band = size_label(s_lo, s_hi)
    n_band = sum(s_lo <= r["own_uncached"] < s_hi for r in rows)
    for col, (field, xlabel, bins) in enumerate(MEASURES):
        ax = axes[row, col]
        ts = [t for t in table if t["size_band"] == band and t["measure"] == field]
        xs = [t["bin_index"] for t in ts]
        bottom = [0.0] * len(ts)
        for key, name, color in COMPONENTS:
            vals = [max(t[f"mean_{key}"], 0) / 1000 for t in ts]
            ax.bar(xs, vals, 0.72, bottom=bottom, color=color, label=name, edgecolor=SURFACE, linewidth=2)
            bottom = [b + v for b, v in zip(bottom, vals)]
        for x, t, top in zip(xs, ts, bottom):
            ax.text(x, top + ymax * 0.01, f"n={t['requests']}", ha="center", va="bottom", fontsize=8, color=INK2)
        ax.set_title(f"{band} ({n_band} requests)", loc="left", fontsize=10, color=INK)
        ax.set_xticks(range(len(bins)), [label for *_, label in bins], fontsize=8.5)
        ax.set_xlim(-0.6, len(bins) - 0.4)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelleft=True)
    axes[row, 0].set_ylabel("mean client TTFT (s)")
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.968), frameon=False, ncol=4, fontsize=9)
scope = f"engine {a.engine}" if a.engine else f"all {len(engines)} replicas pooled"
fig.suptitle(f"{os.path.basename(R)}: observed client TTFT by the engine state each request found on arrival "
             f"({scope})", x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{R}/analysis/state_breakdown{suffix}.png", dpi=120, facecolor=SURFACE)
print(f"wrote {R}/analysis/state_breakdown{suffix}.png and .csv")

# Lines: the two engine-side components and their total, pooled and per engine.
LINES = [("client_ttft_ms", "Mean client TTFT", INK),
         ("queue_ms", "Queue (engine)", COMPONENTS[1][2]),
         ("prefill_ms", "Prefill steps (engine)", COMPONENTS[0][2])]


def cell_means(rs, bins, field, key, min_n):
    out = []
    for i, (lo, hi, _) in enumerate(bins):
        sel = [r[key] for r in rs if in_bin(r[field], lo, hi, i == 0)]
        out.append(sum(sel) / len(sel) / 1000 if len(sel) >= min_n else float("nan"))
    return out


fig, axes = plt.subplots(len(SIZE_BANDS), len(MEASURES), figsize=(15, 16), sharey=True, facecolor=SURFACE)
for row, (s_lo, s_hi) in enumerate(SIZE_BANDS):
    band_rows = [r for r in rows if s_lo <= r["own_uncached"] < s_hi]
    for col, (field, xlabel, bins) in enumerate(MEASURES):
        ax = axes[row, col]
        xs = list(range(len(bins)))
        for key, name, color in LINES:
            for engine in engines:
                ys = cell_means([r for r in band_rows if r["engine"] == engine], bins, field, key, a.min_n)
                ax.plot(xs, ys, color=color, linewidth=1, alpha=0.3, marker="o", markersize=3)
            ys = cell_means(band_rows, bins, field, key, 1)
            ax.plot(xs, ys, color=color, linewidth=2.2, marker="o", markersize=7,
                    markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{name}, all engines")
        counts = [sum(in_bin(r[field], lo, hi, i == 0) for r in band_rows) for i, (lo, hi, _) in enumerate(bins)]
        for x, n in zip(xs, counts):
            if n:
                ax.text(x, ymax * 0.97, f"n={n}", ha="center", va="top", fontsize=8, color=INK2)
        ax.set_title(f"{size_label(s_lo, s_hi)} ({len(band_rows)} requests)", loc="left", fontsize=10, color=INK)
        ax.set_xticks(xs, [label for *_, label in bins], fontsize=8.5)
        ax.set_xlim(-0.4, len(bins) - 0.6)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_ylim(0, ymax)
        ax.tick_params(labelleft=True)
    axes[row, 0].set_ylabel("seconds")
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
handles, labels = axes[0, 0].get_legend_handles_labels()
handles.append(plt.Line2D([], [], color=INK2, linewidth=1, alpha=0.4, marker="o", markersize=3))
labels.append(f"faint: one engine each (cells with ≥{a.min_n} requests)")
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.968), frameon=False, ncol=4, fontsize=9)
fig.suptitle(f"{os.path.basename(R)}: TTFT, queue and prefill by the engine state on arrival ({scope})",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.955))
fig.savefig(f"{R}/analysis/state_breakdown_lines{suffix}.png", dpi=120, facecolor=SURFACE)
print(f"wrote {R}/analysis/state_breakdown_lines{suffix}.png")
