#!/usr/bin/env python3
"""Engine TTFT as queue time, step time and step inflation, per time bucket.

Joins AIPerf's per-request records to the vLLM scheduler trace and, for each
request, measures

  queue time       request_queued -> its first step starts executing: the
                   wait to be admitted plus, with async scheduling, the wait
                   for the step in flight at admission to finish
  steps S          engine steps from admission to the first-token step,
                   inclusive, whether or not the request got tokens in each
  step time T      mean duration of those S steps
  step inflation   S / S_min, where S_min is the steps the request needs
                   alone: ceil(block-aligned uncached tokens / chunk budget),
                   plus one for a partial-block tail. Hybrid (Mamba) models in
                   vLLM's "align" cache mode end every chunk but the prompt's
                   last on a block boundary, so the tail always runs in its
                   own step. 1 means the request prefilled as fast as alone.

so that engine TTFT = queue + S x T exactly and, per uncached token,
engine TTFT ~ queue/uncached + (S/S_min) x T / budget.

usage: python tools/step_breakdown.py runs/<run_dir> \
           [--traces /mnt/cluster_storage/.../vllm-scheduler-traces] [--step 30]

Writes analysis/step_breakdown.{png,csv} in the run.
"""
import argparse, csv, glob, json, math, os, re
from functools import reduce
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

DEFAULT_TRACES = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
KINDS = [("cold", "Cold turns (<50% cached)", "#2a78d6"),
         ("warm", "Warm turns (≥50% cached)", "#eb6834")]
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"


def aiperf_id(engine_id):
    m = ENGINE_ID.match(engine_id)
    return m.group(1) if m else engine_id


ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--traces", default=DEFAULT_TRACES)
ap.add_argument("--step", type=float, default=30)
a = ap.parse_args()
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
        if not isl or cached is None:
            continue
        aiperf[meta["x_request_id"]] = {
            "t_s": (meta["request_start_ns"] - origin_ns) / 1e9,
            "kind": "cold" if cached < 0.5 * isl else "warm",
        }

# One pass per trace file (one engine process): per-step durations, the
# requests' admission and first-token steps, and the largest single chunk.
requests = []
budget = 0
# Non-final chunks end on a block boundary; their gcd is the block size.
block = 0
for path in sorted(glob.glob(f"{a.traces}/*.jsonl")):
    scheduled_at, completed_at = {}, {}
    admitted, first, preempted = {}, {}, set()
    for line in open(path):
        try:
            e = json.loads(line)
        except ValueError:
            continue
        ev = e["event"]
        if ev == "batch_scheduled":
            scheduled_at[e["batch_index"]] = e["monotonic_s"]
            for r in e["requests"]:
                if r["phase"] == "prefill":
                    budget = max(budget, r["scheduled_tokens"])
                    if r["remaining_prefill_after"] > 0:
                        block = math.gcd(block, r["computed_before"] + r["scheduled_tokens"])
        elif ev == "batch_completed":
            completed_at[e["batch_index"]] = e["monotonic_s"]
        elif ev in ("request_admitted", "request_first_token", "request_preempted"):
            rid = aiperf_id(e["request_id"])
            if rid not in aiperf:
                continue
            if ev == "request_admitted":
                admitted[rid] = e
            elif ev == "request_first_token":
                first[rid] = e
            else:
                preempted.add(rid)
    if not admitted:
        continue
    # A step lasts from the previous step's output (or its own scheduling,
    # if the engine was idle in between) to its own output. With async
    # scheduling, dispatch_to_output_ms overlaps the next step instead.
    duration = {}
    for i, done in completed_at.items():
        start = max(completed_at.get(i - 1, scheduled_at.get(i, done)), scheduled_at.get(i, done))
        duration[i] = (done - start) * 1000
    for rid, adm in admitted.items():
        f = first.get(rid)
        if f is None:
            continue
        steps = range(adm["batch_index"], f["batch_index"] + 1)
        if any(i not in duration for i in steps):
            continue
        uncached = max(1, adm["prompt_tokens"] - adm["cached_prefix_tokens"])
        requests.append({
            **aiperf[rid], "rid": rid, "uncached": uncached, "steps": len(steps),
            "steps_ms": sum(duration[i] for i in steps), "preempted": rid in preempted,
            "admission_wait_ms": f["queue_wait_ms"],
        })
        requests[-1]["queue_ms"] = f["engine_ttft_ms"] - requests[-1]["steps_ms"]

for r in requests:
    tail = r["uncached"] % block if block else 0
    r["min_steps"] = max(1, math.ceil((r["uncached"] - tail) / budget) + (1 if tail else 0))
print(f"chunk budget (largest single prefill chunk): {budget} tokens; block size {block} tokens")
print(f"measured {len(requests)} of {len(aiperf)} successful AIPerf requests; "
      f"{sum(r['preempted'] for r in requests)} preempted")
print(f"queue time below the admission wait (should be 0): "
      f"{sum(r['queue_ms'] < r['admission_wait_ms'] - 1 for r in requests)}")

buckets = defaultdict(list)
for r in requests:
    buckets[(int(r["t_s"] // a.step), r["kind"])].append(r)
table = []
for (b, kind), rs in sorted(buckets.items()):
    table.append({
        "bucket_start_s": b * a.step, "kind": kind, "requests": len(rs),
        "mean_uncached": sum(r["uncached"] for r in rs) / len(rs),
        # Ratios of sums so large prompts are not drowned by small ones.
        "queue_ms_per_1k_uncached": sum(r["queue_ms"] for r in rs) / sum(r["uncached"] for r in rs) * 1000,
        "mean_step_ms": sum(r["steps_ms"] for r in rs) / sum(r["steps"] for r in rs),
        "step_inflation": sum(r["steps"] for r in rs) / sum(r["min_steps"] for r in rs),
        "mean_steps": sum(r["steps"] for r in rs) / len(rs),
        "mean_queue_ms": sum(r["queue_ms"] for r in rs) / len(rs),
        "mean_admission_wait_ms": sum(r["admission_wait_ms"] for r in rs) / len(rs),
    })
with open(f"{R}/analysis/step_breakdown.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=list(table[0]))
    w.writeheader()
    w.writerows(table)
for kind, *_ in KINDS:
    rs = [r for r in requests if r["kind"] == kind]
    print(f"{kind}: n={len(rs)}  queue {sum(r['queue_ms'] for r in rs)/sum(r['uncached'] for r in rs)*1000:.1f} ms/1K uncached  "
          f"step {sum(r['steps_ms'] for r in rs)/sum(r['steps'] for r in rs):.0f} ms  "
          f"inflation {sum(r['steps'] for r in rs)/sum(r['min_steps'] for r in rs):.2f}  "
          f"(mean steps {sum(r['steps'] for r in rs)/len(rs):.2f}, mean min steps {sum(r['min_steps'] for r in rs)/len(rs):.2f}, "
          f"mean queue {sum(r['queue_ms'] for r in rs)/len(rs):.0f} ms of which admission {sum(r['admission_wait_ms'] for r in rs)/len(rs):.0f} ms)")

plt.rcParams.update({"font.size": 10, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                     "xtick.color": INK2, "ytick.color": INK2, "text.color": INK})
fig, axes = plt.subplots(4, 1, figsize=(13, 13), sharex=True,
                         gridspec_kw={"height_ratios": [2, 2, 2, 1.2]}, facecolor=SURFACE)
panels = [
    ("queue_ms_per_1k_uncached", "Queue time: arrival until the request's first step starts",
     "ms per 1K uncached tokens", None),
    ("mean_step_ms", "Mean duration of the steps a request's prefill spanned", "ms per step", None),
    ("step_inflation", f"Step inflation: steps taken / steps needed alone (chunk budget {budget}, block {block} tokens)",
     "S / S_min", 1.0),
]
for ax, (key, title, ylabel, ref) in zip(axes, panels):
    for kind, label, color in KINDS:
        ts = [t for t in table if t["kind"] == kind]
        xs = [t["bucket_start_s"] + a.step / 2 for t in ts]
        ys = [t[key] for t in ts]
        ax.plot(xs, ys, color=color, linewidth=2, marker="o", markersize=6,
                markeredgecolor=SURFACE, markeredgewidth=1.5, label=label)
    if ref is not None:
        ax.axhline(ref, color=INK2, linewidth=1, linestyle=(0, (4, 3)))
        ax.text(2, ref, " alone", fontsize=8, color=INK2, va="bottom")
    ax.set_title(title, loc="left", fontsize=11, color=INK)
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)

ax = axes[3]
status = pd["serve_status_samples"]
st_t = [s["relative_time_s"] for s in status]
ax.step(st_t, [s["running_replicas"] for s in status], where="post", color=INK, linewidth=2, label="running")
ax.step(st_t, [s["running_replicas"] + s["starting_replicas"] for s in status], where="post",
        color=INK2, linewidth=1.5, linestyle=(0, (4, 3)), label="running + starting")
ax.set_title("Replicas", loc="left", fontsize=11, color=INK)
ax.set_ylabel("replicas")
ax.legend(loc="upper left", frameon=False, fontsize=9)
ax.set_ylim(0, max(s["running_replicas"] + s["starting_replicas"] for s in status) + 1)

load_end = pd["experiment"].get("duration_s")
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
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper left", bbox_to_anchor=(0.01, 0.965), frameon=False, ncol=2, fontsize=9)
fig.suptitle(f"{os.path.basename(R)}: engine TTFT ≈ queue + steps × step time, per {a.step:.0f}s bucket",
             x=0.01, ha="left", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.95))
fig.savefig(f"{R}/analysis/step_breakdown.png", dpi=130, facecolor=SURFACE)
print(f"wrote {R}/analysis/step_breakdown.png and .csv")
