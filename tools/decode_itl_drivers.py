#!/usr/bin/env python3
"""Per-request ITL against three candidate drivers of decode step time.

A decoding request's ITL is the mean step time of the steps it decoded in.
Each candidate is read from the engine's scheduler traces and averaged,
time-weighted, over the request's own decode phase (engine first token ->
engine finish), so every x value describes the batches its tokens shared:

  decodes      requests in the decode phase of each step (batch_scheduled's
               decode_requests), the count the kv-aware router reports and the
               token policy divides by target_decode_requests_per_replica;
               the request itself is excluded
  kv           KV cache utilization at each step (batch_scheduled's
               kv_cache_usage, a fraction of capacity)
  load         uncached prefill token load: uncached prompt tokens queued or
               mid-prefill, less chunks already computed, as load_breakdown.py
               defines it; the backlog that chunked prefill feeds into decode
               steps
  prefill_tok  prefill tokens actually scheduled in each step (the chunk a
               decode step carried), a check on `load`

ITL is AIPerf's per-request mean inter-token latency; requests with fewer
than --min-output-tokens output tokens are left out. Request size does not
enter: sizes are pooled.

usage: python tools/decode_itl_drivers.py --out <dir> runs/A [runs/B ...]
           [--traces <dir>] [--min-output-tokens 8] [--min-n 10] [--itl-slo-ms 50]

Writes <out>/decode_itl_drivers.png (three panels: ITL by decodes, by KV
utilization, by uncached prefill load), <out>/decode_itl_by_decodes_and_load.png
(ITL by decodes, one line per load band, to separate the two), and
<out>/decode_itl_requests.csv with every request's values.
"""
import argparse, bisect, csv, glob, json, math, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
ACCENT = "#2a78d6"
LOAD_BAND_COLORS = ["#2a78d6", "#1baf7a", "#eb6834", "#8c5ad6"]

ap = argparse.ArgumentParser()
ap.add_argument("runs", nargs="+")
ap.add_argument("--out", required=True)
ap.add_argument("--traces", default=DEFAULT_TRACES)
ap.add_argument("--min-output-tokens", type=int, default=8)
ap.add_argument("--min-n", type=int, default=10, help="fewest requests for a plotted cell")
ap.add_argument("--itl-slo-ms", type=float, default=50)
a = ap.parse_args()


def aiperf_id(engine_id):
    m = ENGINE_ID.match(engine_id)
    return m.group(1) if m else engine_id


class StepFn:
    """Right-continuous step function with a prefix integral, for window means."""

    def __init__(self, points):
        points = sorted(points)
        self.t = [t for t, _ in points]
        self.v = [v for _, v in points]
        self.integral, area = [], 0.0
        for i, (t, v) in enumerate(points):
            if i:
                area += self.v[i - 1] * (t - self.t[i - 1])
            self.integral.append(area)

    def _cum(self, t):
        i = bisect.bisect_right(self.t, t) - 1
        if i < 0:
            return 0.0
        return self.integral[i] + self.v[i] * (t - self.t[i])

    def at(self, t):
        i = bisect.bisect_right(self.t, t) - 1
        return self.v[i] if i >= 0 else 0.0

    def mean(self, lo, hi):
        if hi <= lo:
            return self.at(lo)
        return (self._cum(hi) - self._cum(lo)) / (hi - lo)


def load_aiperf(R):
    out = {}
    for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
        for line in open(path):
            rec = json.loads(line)
            meta, m = rec["metadata"], rec.get("metrics") or {}
            if meta.get("benchmark_phase") != "profiling" or rec.get("error"):
                continue
            itl = m.get("inter_token_latency", {}).get("value")
            out_tokens = m.get("output_token_count", {}).get("value")
            if itl is None or out_tokens is None:
                continue
            out[meta["x_request_id"]] = {"itl_ms": itl, "output_tokens": out_tokens,
                                         "isl": m.get("input_sequence_length", {}).get("value")}
    return out


def engines_of(R):
    with open(f"{R}/analysis/load_requests.csv") as fh:
        return sorted({r["engine"] for r in csv.DictReader(fh)})


rows = []
for R in a.runs:
    R = R.rstrip("/")
    name = os.path.basename(R)
    aiperf = load_aiperf(R)
    for engine in engines_of(R):
        paths = sorted(glob.glob(f"{a.traces}/{engine}.*.jsonl"))
        if not paths:
            print(f"{name}: no traces for engine {engine}")
            continue
        steps = []  # (t, kv usage, decode requests, prefill tokens scheduled)
        completed_at = {}
        chunks = defaultdict(list)  # engine request id -> [(batch, tokens)]
        queued, first, finished = {}, {}, {}
        preempted = defaultdict(list)
        external = {}
        for path in paths:
            for line in open(path):
                try:
                    e = json.loads(line)
                except ValueError:
                    continue
                ev = e["event"]
                if ev == "batch_scheduled":
                    steps.append((e["monotonic_s"], e["kv_cache_usage"], e["decode_requests"], e["prefill_tokens"]))
                    for r in e["requests"]:
                        if r["phase"] == "prefill":
                            chunks[r["request_id"]].append((e["batch_index"], r["scheduled_tokens"]))
                elif ev == "batch_completed":
                    completed_at[e["batch_index"]] = e["monotonic_s"]
                    for rid in e.get("finished_request_ids") or []:
                        finished.setdefault(rid, e["monotonic_s"])
                elif ev == "request_queued":
                    queued[e["request_id"]] = e["monotonic_s"]
                    external[e["request_id"]] = e.get("external_request_id", e["request_id"])
                elif ev == "request_first_token":
                    first[e["request_id"]] = e["monotonic_s"]
                elif ev == "request_finished":
                    finished[e["request_id"]] = e["monotonic_s"]
                elif ev == "request_preempted":
                    preempted[e["request_id"]].append((e["batch_index"], e["monotonic_s"]))
        if not steps:
            continue
        # Uncached prefill load as a step function, as in load_breakdown.py.
        deltas = []
        for rid, at_queue in queued.items():
            cs = chunks.get(rid)
            if not cs or any(b not in completed_at for b, _ in cs):
                continue
            starts = [(0, at_queue)] + sorted(preempted.get(rid, []))
            for k, (batch, at) in enumerate(starts):
                until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
                deltas.append((at, sum(t for b, t in cs if batch <= b < until)))
            deltas.extend((completed_at[b], -t) for b, t in cs)
        deltas.sort()
        level, load_points = 0, []
        for t, d in deltas:
            level += d
            load_points.append((t, level))
        load_fn = StepFn(load_points)
        kv_fn = StepFn([(t, kv) for t, kv, _, _ in steps])
        dec_fn = StepFn([(t, d) for t, _, d, _ in steps])
        ptok_fn = StepFn([(t, p) for t, _, _, p in steps])

        n = 0
        for rid, t_first in first.items():
            t_end = finished.get(rid)
            key = aiperf_id(external.get(rid, rid))
            r = aiperf.get(key)
            if t_end is None or r is None or r["output_tokens"] < a.min_output_tokens or t_end <= t_first:
                continue
            rows.append({
                "run": name, "engine": engine, "request_id": key, "itl_ms": r["itl_ms"],
                "output_tokens": r["output_tokens"], "isl": r["isl"],
                "decode_phase_s": t_end - t_first,
                "decodes": max(0.0, dec_fn.mean(t_first, t_end) - 1),
                "kv_usage": kv_fn.mean(t_first, t_end),
                "load_tokens": load_fn.mean(t_first, t_end),
                "prefill_tokens_per_step": ptok_fn.mean(t_first, t_end),
                "preempted": len(preempted.get(rid, [])),
            })
            n += 1
        print(f"{name} / {engine}: {len(steps)} steps, {len(first)} first tokens, {n} ITL samples")

if not rows:
    raise SystemExit("no requests")
os.makedirs(a.out, exist_ok=True)
with open(f"{a.out}/decode_itl_requests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
    w.writeheader()
    w.writerows(rows)

DRIVERS = [
    ("decodes", "other requests decoding (mean over its decode phase)",
     [(0, 3, "0–3")] + [(4 * i - 1, 4 * i + 3, f"{4 * i}–{4 * i + 3}") for i in range(1, 16)] + [(63, math.inf, "≥64")]),
    ("kv_usage", "KV cache utilization (mean over its decode phase)",
     [(i / 10, (i + 1) / 10, f"{10 * i}–{10 * (i + 1)}%") for i in range(10)]),
    ("load_tokens", "uncached prefill token load on the engine (mean over its decode phase)",
     [(0, 10e3, "≤10K")] + [(i * 20e3, (i + 1) * 20e3, f"{20 * i}–{20 * (i + 1)}K") if i else (10e3, 20e3, "10–20K")
                             for i in range(10)] + [(200e3, math.inf, ">200K")]),
]


def bin_of(v, bins):
    for i, (lo, hi, _) in enumerate(bins):
        if (lo <= v <= hi) if i == 0 else (lo < v <= hi):
            return i
    return len(bins) - 1


def pct(values, p):
    values = sorted(values)
    k = (len(values) - 1) * p
    lo, hi = math.floor(k), math.ceil(k)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


def style(ax, xlabel, bins):
    ax.set_xticks(range(len(bins)), [label for *_, label in bins], fontsize=8, rotation=60, ha="right")
    ax.set_xlim(-0.4, len(bins) - 0.6)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylim(0)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)


# Figure 1: one panel per driver, pooled over sizes.
fig, axes = plt.subplots(1, 3, figsize=(20, 6.5), facecolor=SURFACE)
summary = []
for ax, (field, xlabel, bins) in zip(axes, DRIVERS):
    cells = defaultdict(list)
    for r in rows:
        cells[bin_of(r[field], bins)].append(r["itl_ms"])
    xs = [i for i in range(len(bins)) if len(cells[i]) >= a.min_n]
    means = [sum(cells[i]) / len(cells[i]) for i in xs]
    p90s = [pct(cells[i], 0.9) for i in xs]
    ax.plot(xs, means, color=ACCENT, linewidth=2, marker="o", markersize=7, markeredgecolor=SURFACE,
            markeredgewidth=1.5, label="mean ITL")
    ax.plot(xs, p90s, color=INK2, linewidth=1.5, linestyle="--", label="p90 ITL")
    ax.axhline(a.itl_slo_ms, color=INK2, linewidth=1, linestyle=":", label=f"ITL SLO {a.itl_slo_ms:g}ms")
    for i, m in zip(xs, means):
        ax.annotate(str(len(cells[i])), (i, m), textcoords="offset points", xytext=(0, 7), ha="center",
                    fontsize=7, color=INK2)
    style(ax, xlabel, bins)
    ax.set_ylabel("ITL (ms)")
    ax.legend(frameon=False, fontsize=9, loc="upper left")
    for i in xs:
        summary.append({"driver": field, "bin": bins[i][2], "requests": len(cells[i]),
                        "mean_itl_ms": sum(cells[i]) / len(cells[i]), "p50_itl_ms": pct(cells[i], 0.5),
                        "p90_itl_ms": pct(cells[i], 0.9)})
fig.suptitle(f"ITL by what the decode steps carried: {len(rows)} requests with ≥{a.min_output_tokens} output "
             f"tokens, {len(a.runs)} runs (labels are request counts)", x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{a.out}/decode_itl_drivers.png", dpi=120, facecolor=SURFACE)
with open(f"{a.out}/decode_itl_drivers.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(summary[0].keys()))
    w.writeheader()
    w.writerows(summary)
print(f"{len(rows)} requests; wrote {a.out}/decode_itl_drivers.png and .csv")

# Figure 2: ITL by decodes, one line per uncached-load band, to separate the two.
LOAD_BANDS = [(0, 10e3, "≤10K load"), (10e3, 50e3, "10–50K load"), (50e3, 150e3, "50–150K load"),
              (150e3, math.inf, ">150K load")]
field, xlabel, bins = DRIVERS[0]
fig, ax = plt.subplots(figsize=(14, 7), facecolor=SURFACE)
for (lo, hi, label), color in zip(LOAD_BANDS, LOAD_BAND_COLORS):
    cells = defaultdict(list)
    for r in rows:
        if lo <= r["load_tokens"] < hi:
            cells[bin_of(r[field], bins)].append(r["itl_ms"])
    xs = [i for i in range(len(bins)) if len(cells[i]) >= a.min_n]
    ax.plot(xs, [sum(cells[i]) / len(cells[i]) for i in xs], color=color, linewidth=2, marker="o", markersize=7,
            markeredgecolor=SURFACE, markeredgewidth=1.5, label=f"{label} ({sum(len(c) for c in cells.values())} req)")
ax.axhline(a.itl_slo_ms, color=INK2, linewidth=1, linestyle=":", label=f"ITL SLO {a.itl_slo_ms:g}ms")
style(ax, xlabel, bins)
ax.set_ylabel("mean ITL (ms)")
ax.legend(frameon=False, fontsize=9, loc="upper left")
ax.set_title("Mean ITL by active decodes, split by the uncached prefill load during the same decode phase",
             loc="left", fontsize=12)
fig.tight_layout()
fig.savefig(f"{a.out}/decode_itl_by_decodes_and_load.png", dpi=120, facecolor=SURFACE)
print(f"wrote {a.out}/decode_itl_by_decodes_and_load.png")
