#!/usr/bin/env python3
"""Client TTFT components against the engine's uncached prefill token load.

The uncached prefill token load of an engine at time t is the uncached prompt
tokens it still has to prefill: summed over every request queued or mid-prefill
on it, the request's uncached tokens minus the prefill chunks that have
completed by t. Each request gets two load values, one per mechanism:

  load at arrival   just before it arrived, its own tokens excluded. vLLM
                    admits first come, first served, so this is the work its
                    queue time waits behind; its own tokens cannot delay its
                    admission and would put every cold turn in the top bins.
  load in its steps mean load at the start of each step it had tokens in, its
                    own remaining tokens included, since its chunks are part
                    of what makes those steps slow.

Its client TTFT is split into

  before engine  client start -> scheduler receives it (proxy, router, tokenize)
  queue          arrival -> its first step with tokens starts executing:
                 admission, the step in flight at admission (async
                 scheduling) and any step it sat in without tokens
  prefill        the steps it had tokens in: steps x mean step time
  after engine   first token produced -> client receives it

usage: python tools/load_breakdown.py runs/<run_dir> \
           [--traces /mnt/cluster_storage/.../vllm-scheduler-traces]

Writes analysis/load_breakdown.{png,csv}, load_breakdown_steps.csv and
load_requests.csv.
"""
import argparse, bisect, csv, glob, json, math, os, re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
COMPONENTS = [  # stacked bottom to top; categorical slots 1-4
    ("prefill_ms", "Prefill steps (engine)", "#2a78d6"),
    ("queue_ms", "Queue (engine)", "#eb6834"),
    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def aiperf_id(engine_id):
    m = ENGINE_ID.match(engine_id)
    return m.group(1) if m else engine_id


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--traces", default=DEFAULT_TRACES)
a = ap.parse_args()
R = a.run.rstrip("/")

aiperf = {}
for path in glob.glob(f"{R}/benchmark/*/profile_export.jsonl"):
    for line in open(path):
        rec = json.loads(line)
        meta, m = rec["metadata"], rec.get("metrics") or {}
        if meta.get("benchmark_phase") != "profiling" or rec.get("error") or "time_to_first_token" not in m:
            continue
        isl = m.get("input_sequence_length", {}).get("value")
        cached = m.get("usage_prompt_cache_read_tokens", {}).get("value")
        if not isl or cached is None:
            continue
        aiperf[meta["x_request_id"]] = {
            "start_ns": meta["request_start_ns"], "ttft_ms": m["time_to_first_token"]["value"],
            "kind": "cold" if cached < 0.5 * isl else "warm", "isl": isl,
        }

rows = []
budget = 0
# Non-final chunks end on a block boundary; their gcd is the block size.
block = 0
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    scheduled_at, completed_at = {}, {}
    chunks = defaultdict(list)  # engine request id -> [(batch, tokens)]
    queued, admitted, first = {}, {}, {}
    preempted = defaultdict(list)  # engine request id -> [(batch, time)]
    step_state = []  # (scheduled at, KV context tokens, KV cache usage) per step
    for line in open(path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = e["event"]
        if ev == "batch_scheduled":
            scheduled_at[e["batch_index"]] = e["monotonic_s"]
            # KV in use, in tokens: every scheduled sequence's context after
            # this step, the work its attention covers.
            step_state.append((e["monotonic_s"], sum(r["context_tokens_after"] for r in e["requests"]),
                               e["kv_cache_usage"]))
            for r in e["requests"]:
                if r["phase"] == "prefill":
                    chunks[r["request_id"]].append((e["batch_index"], r["scheduled_tokens"]))
                    budget = max(budget, r["scheduled_tokens"])
                    if r["remaining_prefill_after"] > 0:
                        block = math.gcd(block, r["computed_before"] + r["scheduled_tokens"])
        elif ev == "batch_completed":
            completed_at[e["batch_index"]] = e["monotonic_s"]
        elif ev == "request_queued":
            queued[e["request_id"]] = e
        elif ev == "request_admitted":
            admitted[e["request_id"]] = e
        elif ev == "request_first_token":
            first[e["request_id"]] = e
        elif ev == "request_preempted":
            preempted[e["request_id"]].append((e["batch_index"], e["monotonic_s"]))
    if not queued:
        continue

    # Load as a step function: +uncached when queued, -chunk when its step
    # completes. A preemption discards the request's KV, so the chunks that
    # recompute it are added back when it is preempted, not when it arrived.
    # Every request on the engine counts, joined to AIPerf or not.
    deltas = []
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs):
            continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            deltas.append((at, sum(t for b, t in cs if batch <= b < until)))
        deltas.extend((completed_at[b], -t) for b, t in cs)
    deltas.sort()
    times = [t for t, _ in deltas]
    level, levels = 0, []
    for _, d in deltas:
        level += d
        levels.append(level)

    def load_before(t):
        i = bisect.bisect_left(times, t)  # strictly before t: excludes a request's own +
        return levels[i - 1] if i else 0

    def load_at(t):
        i = bisect.bisect_right(times, t)
        return levels[i - 1] if i else 0

    step_times = [t for t, _, _ in step_state]

    def step_state_before(t):
        i = bisect.bisect_left(step_times, t)
        return step_state[i - 1][1:] if i else (0, 0.0)

    duration, started_at = {}, {}
    for i, done in completed_at.items():
        started_at[i] = max(completed_at.get(i - 1, scheduled_at.get(i, done)), scheduled_at.get(i, done))
        duration[i] = (done - started_at[i]) * 1000
    for rid, q in queued.items():
        key, adm, f = aiperf_id(rid), admitted.get(rid), first.get(rid)
        if key not in aiperf or adm is None or f is None or not chunks.get(rid):
            continue
        span = range(adm["batch_index"], f["batch_index"] + 1)
        # Its own steps end at its first token; later prefill chunks recompute
        # it after a preemption during decode and do not delay its TTFT.
        own_chunks = [(b, t) for b, t in chunks[rid] if b <= f["batch_index"]]
        steps = sorted({b for b, _ in own_chunks})
        own_uncached = adm["prompt_tokens"] - adm["cached_prefix_tokens"]
        if any(i not in duration for i in span):
            continue
        r = aiperf[key]
        steps_ms = sum(duration[i] for i in steps)
        pre = (q["wall_time_s"] * 1e9 - r["start_ns"]) / 1e6
        rows.append({
            "request_id": key, "engine": os.path.basename(path).split(".")[0], "kind": r["kind"],
            "engine_load_tokens": load_before(q["monotonic_s"]),
            # request_queued counts the request itself among the waiting.
            "ongoing_sequences": q["running_requests"] + q["waiting_requests"] - 1,
            "kv_context_tokens": step_state_before(q["monotonic_s"])[0],
            "kv_cache_usage": step_state_before(q["monotonic_s"])[1],
            "step_load_tokens": sum(load_at(started_at[i]) for i in steps) / len(steps),
            "idle_steps": len(span) - len(steps),
            "own_uncached": own_uncached, "isl": r["isl"],
            "preemptions_before_first_token": sum(b <= f["batch_index"] for b, _ in preempted.get(rid, [])),
            "recompute_tokens_before_first_token": max(0, sum(t for _, t in own_chunks) - own_uncached),
            "steps": len(steps), "step_ms": steps_ms / len(steps),
            "client_ttft_ms": r["ttft_ms"], "pre_ms": pre,
            "queue_ms": f["engine_ttft_ms"] - steps_ms, "prefill_ms": steps_ms,
            "post_ms": r["ttft_ms"] - pre - f["engine_ttft_ms"],
        })
# Steps the request needs alone: block-aligned chunks of at most the budget,
# then a partial-block tail in its own step (vLLM's Mamba "align" cache mode).
for r in rows:
    tail = r["own_uncached"] % block if block else 0
    r["min_steps"] = max(1, math.ceil((r["own_uncached"] - tail) / budget) + (1 if tail else 0))
    r["chunk_budget"] = budget
print(f"placed {len(rows)} of {len(aiperf)} successful AIPerf requests; "
      f"chunk budget {budget} tokens, block {block} tokens")
print(f"requests preempted before their first token: "
      f"{sum(r['preemptions_before_first_token'] > 0 for r in rows)} "
      f"({sum(r['recompute_tokens_before_first_token'] for r in rows):.0f} tokens recomputed)")
print(f"requests that sat in a step without tokens after admission: {sum(r['idle_steps'] > 0 for r in rows)} "
      f"({sum(r['idle_steps'] for r in rows)} steps in all)")
with open(f"{R}/analysis/load_requests.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

loads = sorted(r["engine_load_tokens"] for r in rows)
print("engine uncached prefill load at arrival, tokens: " + "  ".join(
    f"p{p} {loads[min(len(loads) - 1, len(loads) * p // 100)]:.0f}" for p in (10, 25, 50, 75, 90, 95, 99, 100)))
print(f"requests arriving to an engine with no uncached prefill load: {sum(b == 0 for b in loads)}")

# Bins in thousands of tokens.
edges = [0, 16, 32, 64, math.inf]
labels = ["0", "≤16K", "16–32K", "32–64K", ">64K"]


def bin_of(load):
    if load == 0:
        return 0
    for i in range(1, len(edges)):
        if load / 1000 <= edges[i]:
            return i
    return len(edges) - 1


table = []
for kind in ("cold", "warm"):
    for i, label in enumerate(labels):
        rs = [r for r in rows if r["kind"] == kind and bin_of(r["engine_load_tokens"]) == i]
        if not rs:
            continue
        out = {"kind": kind, "engine_load": label, "requests": len(rs),
               "mean_engine_load_tokens": sum(r["engine_load_tokens"] for r in rs) / len(rs),
               "mean_own_uncached": sum(r["own_uncached"] for r in rs) / len(rs),
               "mean_steps": sum(r["steps"] for r in rs) / len(rs),
               # Request-weighted: the step time a request in this bin sat through.
               "mean_step_ms": sum(r["prefill_ms"] for r in rs) / sum(r["steps"] for r in rs)}
        for key in ["client_ttft_ms"] + [c[0] for c in COMPONENTS]:
            out[f"mean_{key}"] = sum(r[key] for r in rs) / len(rs)
        table.append(out)
with open(f"{R}/analysis/load_breakdown.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)

# Step time against the load in the steps themselves, own tokens included,
# so there is no empty bin: the request's own chunks are always there.
step_edges = [16, 32, 64, 128, math.inf]
step_labels = ["≤16K", "16–32K", "32–64K", "64–128K", ">128K"]


def step_bin_of(load):
    return next(i for i, edge in enumerate(step_edges) if load / 1000 <= edge)


step_table = []
for kind in ("cold", "warm"):
    for i, label in enumerate(step_labels):
        rs = [r for r in rows if r["kind"] == kind and step_bin_of(r["step_load_tokens"]) == i]
        if rs:
            step_table.append({
                "kind": kind, "step_load": label, "requests": len(rs),
                "mean_step_load_tokens": sum(r["step_load_tokens"] for r in rs) / len(rs),
                "mean_step_ms": sum(r["prefill_ms"] for r in rs) / sum(r["steps"] for r in rs)})
with open(f"{R}/analysis/load_breakdown_steps.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(step_table[0]))
    w.writeheader()
    w.writerows(step_table)
print(f"{'kind':5} {'step load':>9} {'n':>4} {'step ms':>7}")
for t in step_table:
    print(f"{t['kind']:5} {t['step_load']:>9} {t['requests']:>4} {t['mean_step_ms']:7.0f}")

print(f"{'kind':5} {'load':>7} {'n':>4} {'TTFT':>6} {'queue':>6} {'prefill':>7} {'pre':>5} {'post':>4} "
      f"{'steps':>5} {'step':>5} {'own unc':>7}   (ms; steps = own steps; step = mean step ms)")
for t in table:
    print(f"{t['kind']:5} {t['engine_load']:>7} {t['requests']:>4} {t['mean_client_ttft_ms']:6.0f} "
          f"{t['mean_queue_ms']:6.0f} {t['mean_prefill_ms']:7.0f} {t['mean_pre_ms']:5.0f} {t['mean_post_ms']:4.0f} "
          f"{t['mean_steps']:5.2f} {t['mean_step_ms']:5.0f} {t['mean_own_uncached']:7.0f}")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharey="row",
                         gridspec_kw={"height_ratios": [3, 1.6, 1.6]}, facecolor=SURFACE)
for col, kind in enumerate(("cold", "warm")):
    ts = [t for t in table if t["kind"] == kind]
    xs = [labels.index(t["engine_load"]) for t in ts]
    ax = axes[0, col]
    bottom = [0.0] * len(ts)
    for key, label, color in COMPONENTS:
        vals = [max(t[f"mean_{key}"], 0) / 1000 for t in ts]
        ax.bar(xs, vals, 0.72, bottom=bottom, color=color, label=label, edgecolor=SURFACE, linewidth=2)
        bottom = [b + v for b, v in zip(bottom, vals)]
    for x, t, top in zip(xs, ts, bottom):
        ax.text(x, top, f"n={t['requests']}", ha="center", va="bottom", fontsize=8, color=INK2)
    n = sum(t["requests"] for t in ts)
    ax.set_title(f"{kind.capitalize()} turns ({n}): mean client TTFT", loc="left", fontsize=11, color=INK)
    sts = [t for t in step_table if t["kind"] == kind]
    for row, rows_, xs_, key, title, scale, fmt in (
            (1, sts, [step_labels.index(t["step_load"]) for t in sts], "mean_step_ms",
             "Mean duration of the steps it had tokens in", 1, "{:.0f} ms"),
            (2, ts, xs, "mean_own_uncached", "Its own uncached tokens (confounder check)", 1000, "{:.1f}K")):
        ax = axes[row, col]
        vals = [t[key] / scale for t in rows_]
        ax.plot(xs_, vals, color=INK, linewidth=2, marker="o", markersize=7,
                markeredgecolor=SURFACE, markeredgewidth=1.5)
        for x, v, t in zip(xs_, vals, rows_):
            label = fmt.format(v) + (f" (n={t['requests']})" if row == 1 else "")
            ax.annotate(label, (x, v), textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=8, color=INK2)
        ax.set_title(title, loc="left", fontsize=10, color=INK)
        ax.set_ylim(0, max(t[key] / scale for t in (step_table if row == 1 else table)) * 1.3)
axes[0, 0].set_ylabel("seconds")
axes[1, 0].set_ylabel("ms per step")
axes[2, 0].set_ylabel("K tokens")
for ax in (*axes[0], *axes[2]):
    ax.set_xticks(range(len(labels)), labels)
    ax.set_xlim(-0.6, len(labels) - 0.4)
    ax.set_xlabel("uncached prefill token load on the engine at arrival (excl. the request)")
for ax in axes[1]:
    ax.set_xticks(range(len(step_labels)), step_labels)
    ax.set_xlim(-0.6, len(step_labels) - 0.4)
    ax.set_xlabel("uncached prefill token load on the engine during its steps (incl. the request)")
for ax in axes.flat:
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
handles, hl = axes[0, 0].get_legend_handles_labels()
fig.legend(handles, hl, loc="upper left", bbox_to_anchor=(0.01, 0.965), frameon=False, ncol=4, fontsize=9)
fig.suptitle(f"{os.path.basename(R)}: client TTFT by uncached prefill token load on the engine",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{R}/analysis/load_breakdown.png", dpi=130, facecolor=SURFACE)
print(f"wrote {R}/analysis/load_breakdown.png, load_breakdown.csv, load_breakdown_steps.csv, load_requests.csv")
