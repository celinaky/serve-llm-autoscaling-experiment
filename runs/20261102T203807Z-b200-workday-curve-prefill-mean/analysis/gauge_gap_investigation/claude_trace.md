# Claude Code trace: why the KV router's outstanding-prefill gauge runs above the engines

Run `20261102T203807Z-b200-workday-curve-prefill-mean`, session 2026-11-02. Exported from the Claude Code session
transcript; tool outputs are trimmed to a few KB each. Scripts referenced from the scratchpad are copied into this
directory. Memory and system reminders are stripped.

## User  (2026-11-03 00:20:01)

read the last message from the latest other claude session for context

## Claude  (2026-11-03 00:20:05)

I'll look in the Claude Code transcript directory for the most recent session other than this one and pull its last message.

**Bash** — List most recent Claude session transcripts

```bash
ls -t /home/ray/.claude/projects/-home-ray-default/*.jsonl 2>/dev/null | grep -v 0f3c0e18-6c3d-439a-b85c-5218c786df8a | head -5; echo ---; ls -t /home/ray/.claude/projects/*/*.jsonl 2>/dev/null | grep -v 0f3c0e18-6c3d-439a-b85c-5218c786df8a | head -3
```

<details><summary>Bash result</summary>

```text
/home/ray/.claude/projects/-home-ray-default/c79a47ff-421c-436f-a2a9-451dc9ea4832.jsonl
/home/ray/.claude/projects/-home-ray-default/71dc0a34-decc-41fd-899b-15a26e2bd991.jsonl
/home/ray/.claude/projects/-home-ray-default/6202a627-40e9-4427-ab51-058078fa2df2.jsonl
/home/ray/.claude/projects/-home-ray-default/cecce1b8-a608-4966-b16c-0e0973823747.jsonl
/home/ray/.claude/projects/-home-ray-default/35e6da1a-35a1-448c-87e1-4eb321105f8b.jsonl
---
/home/ray/.claude/projects/-home-ray-default/c79a47ff-421c-436f-a2a9-451dc9ea4832.jsonl
/home/ray/.claude/projects/-home-ray-default/71dc0a34-decc-41fd-899b-15a26e2bd991.jsonl
/home/ray/.claude/projects/-home-ray-default/6202a627-40e9-4427-ab51-058078fa2df2.jsonl
```

</details>

**Bash** — Print the last user and assistant messages from the most recent other session

```bash
f=/home/ray/.claude/projects/-home-ray-default/c79a47ff-421c-436f-a2a9-451dc9ea4832.jsonl; stat -c '%y' "$f"; python3 -I - "$f" <<'EOF'
import json,sys
msgs=[]
for line in open(sys.argv[1]):
    try: d=json.loads(line)
    except: continue
    if d.get('type') not in ('user','assistant'): continue
    c=d.get('message',{}).get('content')
    if isinstance(c,str): text=c
    else:
        text='\n'.join(b.get('text','') for b in c if b.get('type')=='text')
    if text.strip(): msgs.append((d['type'],d.get('timestamp'),text))
print(len(msgs),'messages')
# last user message and last assistant message
for role in ('user','assistant'):
    for t,ts,tx in reversed(msgs):
        if t==role:
            print('=====',role,ts); print(tx[:4000]); break
EOF
```

<details><summary>Bash result</summary>

```text
2026-11-02 16:18:45.736596287 -0800
510 messages
===== user 2026-11-03T00:17:56.345Z
give me the file path and background of the mean token based policy autoscaling run and the plot of the router gauage vs the actual engine mean, i want to share it with another claude agent to research why the router reports higher than the engine
===== assistant 2026-11-03T00:18:42.639Z
Here's a handoff note you can paste to the other agent. Everything in it is a real path on this machine.

---

**Run to study:** `/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/`

**The plot:** `analysis/policy_vs_engine_mean60_0_1860.png` in that directory. A variant with the raw 2s samples underneath is `analysis/policy_vs_engine_mean60_raw_0_1860.png`. Both come from `tools/policy_vs_engine_mean.py` in the harness repo at `/home/ray/default/serve-llm-autoscaling-experiment`. Run it with the run directory as the only argument. It prints the ratio summary at the end.

**What the plot shows.** Purple is the 60s time-weighted mean of the router's gauge of total outstanding uncached prefill tokens across all replicas, as published by the autoscaling policy every 2s. Green is the same quantity rebuilt from the vLLM scheduler traces: a request's uncached tokens are added when the engine queues it, and each prefill chunk's tokens are removed when its batch completes. Over the 1860s load phase the purple mean averages 17.9K and the green 8.2K. Where the engine mean is at least 1K, the purple to green ratio has median 2.0, p10 0.7, p90 4.2. The two lines have the same shape, the purple is just higher, and the gap is widest during cold-prompt bursts, up to 55K vs 24K at 840s.

**Setup.** Ray Serve LLM, 8 B200s, Qwen3.6-27B-FP8, 1 to 3 replicas, KV-aware router with session pinning, chunked prefill at 16384 tokens per step. Workload is an AgentX session replay ramping 1 to 32 sessions over 930s then holding. Autoscaling policy is `ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy` with target 20000, lookback 60s. Ray checkout is `/mnt/local_storage/ray` on branch `celina/prefill-token-autoscaling-policy` at commit `ed547dd6dc`. The site-packages `ray/serve` and `ray/llm` symlink into it.

**Where the gauge lives.** `python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py` in that checkout. The counter is `_outstanding_effective_prefill_tokens`. It is incremented by the request's full `effective_prefill_tokens` at admission in the method around line 644, decremented in `on_prefill_complete` at line 693 and in `on_request_completed` at line 723. Lifecycle events arrive from replicas in batches through `on_lifecycle_events`. The policy reads it through `AutoscalingContext.raw_router_metrics` in `python/ray/serve/experimental/llm_token_autoscaling_policy.py`.

**Known reasons for the gap, to confirm or rule out:**
1. The gauge holds the whole uncached prompt until prefill completes, while the engine burns it down chunk by chunk at about 16K tokens per step. A 150K prompt sits at 150K on the gauge for its entire 9 or 10 step prefill, but the engine's remaining work decreases linearly. On its own this accounts for roughly 2x on long prompts.
2. `effective_prefill_tokens` is the router's estimate of uncached tokens from its own prefix index, not the engine's actual cache hit. If the engine hits more cache than the router predicted, the gauge over-counts.
3. The decrement waits for the replica's `on_prefill_complete` event, which is sent after the first token, batched. Any delivery lag keeps completed prefills on the gauge.
4. Requests pinned to a replica and re-routed, or requests that fail or time out, may miss their decrement.

**Data for checking.** Scheduler traces for the run are in `/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/*.jsonl`, files with mtime after the run's `manifest.json` `started_at`. Events include `request_queued`, `batch_scheduled` with per-request `scheduled_tokens` and `phase`, `batch_completed`, `request_first_token`. The tracer is `python/ray/llm/_internal/serve/engines/vllm/scheduler_trace.py`. Router-side scrapes are in the run's `telemetry/serve_metrics.jsonl`, metric name `ray_serve_autoscaling_policy_metric` with label `metric=outstanding_effective_prefill_tokens`. Per-reques
```

</details>

## Claude  (2026-11-03 00:20:13)

The assistant's note got cut off at my print limit. Let me get the rest.

**Bash** — Print the remainder of the last assistant message

```bash
f=/home/ray/.claude/projects/-home-ray-default/c79a47ff-421c-436f-a2a9-451dc9ea4832.jsonl; python3 -I - "$f" <<'EOF'
import json,sys
last=None
for line in open(sys.argv[1]):
    try: d=json.loads(line)
    except: continue
    if d.get('type')!='assistant': continue
    c=d['message']['content']
    text='\n'.join(b.get('text','') for b in c if b.get('type')=='text') if not isinstance(c,str) else c
    if text.strip(): last=text
print(last[3900:])
EOF
```

<details><summary>Bash result</summary>

```text
erve_autoscaling_policy_metric` with label `metric=outstanding_effective_prefill_tokens`. Per-request engine timings with the router's request ids are in `analysis/load_requests.csv`, columns `own_uncached`, `isl`, `queue_ms`, `prefill_ms`, `steps`. Comparing `own_uncached` from the engine against the router's `effective_prefill_tokens` per request would settle reason 2 directly, but the router's per-request value is not logged in this run, so it would need a debug log line or a second run.

**Related earlier chart** with both accountings rebuilt from traces, router-style (whole prompt until first token) and engine-style: `analysis/policy_decisions_0_1860.png` in the same run, from `tools/policy_decision_timeline.py <run> --backlog`. The router-style rebuild from traces matches the real gauge closely, which points at reason 1 as the main cause and the others as secondary.
```

</details>

## Claude  (2026-11-03 00:20:35)

The latest other session (ended about 16:18 today) closed with a handoff note written for exactly this purpose. The user's last request there was for the file path, background, and plot of the mean-token-based autoscaling run so another Claude agent could research why the router gauge reports higher than the engine mean. Here is that final message in full:

---

**Run to study:** `/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/`

**The plot:** `analysis/policy_vs_engine_mean60_0_1860.png` in that directory. A variant with the raw 2s samples underneath is `analysis/policy_vs_engine_mean60_raw_0_1860.png`. Both come from `tools/policy_vs_engine_mean.py` in the harness repo at `/home/ray/default/serve-llm-autoscaling-experiment`. Run it with the run directory as the only argument. It prints the ratio summary at the end.

**What the plot shows.** Purple is the 60s time-weighted mean of the router's gauge of total outstanding uncached prefill tokens across all replicas, as published by the autoscaling policy every 2s. Green is the same quantity rebuilt from the vLLM scheduler traces: a request's uncached tokens are added when the engine queues it, and each prefill chunk's tokens are removed when its batch completes. Over the 1860s load phase the purple mean averages 17.9K and the green 8.2K. Where the engine mean is at least 1K, the purple to green ratio has median 2.0, p10 0.7, p90 4.2. The two lines have the same shape, the purple is just higher, and the gap is widest during cold-prompt bursts, up to 55K vs 24K at 840s.

**Setup.** Ray Serve LLM, 8 B200s, Qwen3.6-27B-FP8, 1 to 3 replicas, KV-aware router with session pinning, chunked prefill at 16384 tokens per step. Workload is an AgentX session replay ramping 1 to 32 sessions over 930s then holding. Autoscaling policy is `ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy` with target 20000, lookback 60s. Ray checkout is `/mnt/local_storage/ray` on branch `celina/prefill-token-autoscaling-policy` at commit `ed547dd6dc`. The site-packages `ray/serve` and `ray/llm` symlink into it.

**Where the gauge lives.** `python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py` in that checkout. The counter is `_outstanding_effective_prefill_tokens`. It is incremented by the request's full `effective_prefill_tokens` at admission in the method around line 644, decremented in `on_prefill_complete` at line 693 and in `on_request_completed` at line 723. Lifecycle events arrive from replicas in batches through `on_lifecycle_events`. The policy reads it through `AutoscalingContext.raw_router_metrics` in `python/ray/serve/experimental/llm_token_autoscaling_policy.py`.

**Known reasons for the gap, to confirm or rule out:**
1. The gauge holds the whole uncached prompt until prefill completes, while the engine burns it down chunk by chunk at about 16K tokens per step. A 150K prompt sits at 150K on the gauge for its entire 9 or 10 step prefill, but the engine's remaining work decreases linearly. On its own this accounts for roughly 2x on long prompts.
2. `effective_prefill_tokens` is the router's estimate of uncached tokens from its own prefix index, not the engine's actual cache hit. If the engine hits more cache than the router predicted, the gauge over-counts.
3. The decrement waits for the replica's `on_prefill_complete` event, which is sent after the first token, batched. Any delivery lag keeps completed prefills on the gauge.
4. Requests pinned to a replica and re-routed, or requests that fail or time out, may miss their decrement.

**Data for checking.** Scheduler traces for the run are in `/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/*.jsonl`, files with mtime after the run's `manifest.json` `started_at`. Events include `request_queued`, `batch_scheduled` with per-request `scheduled_tokens` and `phase`, `batch_completed`, `request_first_token`. The tracer is `python/ray/llm/_internal/serve/engines/vllm/scheduler_trace.py`. Router-side scrapes are in the run's `telemetry/serve_metrics.jsonl`, metric name `ray_serve_autoscaling_policy_metric` with label `metric=outstanding_effective_prefill_tokens`. Per-request engine timings with the router's request ids are in `analysis/load_requests.csv`, columns `own_uncached`, `isl`, `queue_ms`, `prefill_ms`, `steps`. Comparing `own_uncached` from the engine against the router's `effective_prefill_tokens` per request would settle reason 2 directly, but the router's per-request value is not logged in this run, so it would need a debug log line or a second run.

**Related earlier chart** with both accountings rebuilt from traces, router-style (whole prompt until first token) and engine-style: `analysis/policy_decisions_0_1860.png` in the same run, from `tools/policy_decision_timeline.py <run> --backlog`. The router-style rebuild from traces matches the real gauge closely, which points at reason 1 as the main cause and the others as secondary.

---

I now have this context loaded. The implied next step is investigating why the router gauge runs about 2x above the engine's actual outstanding prefill tokens, with reason 1 (whole-prompt accounting until prefill completes) as the leading hypothesis. Tell me if you want me to start on that research or do something else with this.

## User  (2026-11-03 00:27:21)

can you research and find evidence for what is the difference coming from?

## Claude  (2026-11-03 00:27:24)

I'll start by reading the gauge code and the existing analysis tool, then quantify each of the four hypotheses against the trace data.

**Bash** — Inspect run directory layout and manifest

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment && ls runs/20261102T203807Z-b200-workday-curve-prefill-mean/ runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis runs/20261102T203807Z-b200-workday-curve-prefill-mean/telemetry 2>&1 | head -60; echo ---; cat runs/20261102T203807Z-b200-workday-curve-prefill-mean/manifest.json | head -60; echo ---; wc -l tools/policy_vs_engine_mean.py tools/policy_decision_timeline.py
```

<details><summary>Bash result</summary>

```text
runs/20261102T203807Z-b200-workday-curve-prefill-mean/:
analysis
benchmark
deployment
input.yaml
manifest.json
resolved.yaml
telemetry

runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis:
arrival_load_windowsreq100_0_1860.png
arrival_load_windowsreq100_lines_0_1860.png
arrival_stats120_0_1860.png
arrival_stats_stack120_0_1860.png
autoscaling_evaluation.json
autoscaling_timeline.png
backlog_vs_replicas_0_1860.png
collision_causes.csv
collision_causesreq100_simple_0_1860.png
concurrency_qps.json
load_breakdown.csv
load_breakdown.png
load_breakdown_steps.csv
load_requests.csv
metrics_inventory.json
plot_data.json
policy_decisions_0_1860.png
policy_vs_engine_mean60_0_1860.png
policy_vs_engine_mean60_raw_0_1860.png
replica_arrival_bars120_0_1860.png
replica_arrival_strips120_0_1860.png
replica_load_mean60_0_1860.png
replica_percentiles120_0_1860.png
replica_percentiles60_0_1860.png
replica_sessions60_0_1860.png
replica_spread60_0_1860.png
request_timeseries.json

runs/20261102T203807Z-b200-workday-curve-prefill-mean/telemetry:
serve_metrics.jsonl
serve_status.jsonl
---
{
  "analysis": {
    "plot_error": null,
    "warnings": [
      "phase_manifest.json unavailable; time origin is the first profiling credit"
    ]
  },
  "autoscaling": {
    "policy": "token",
    "policy_function": "ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy",
    "policy_kwargs": {
      "target_prefill_tokens_per_replica": 20000,
      "upscale_aggregation": "mean",
      "upscale_lookback_s": 60
    },
    "verified_deployments": [
      "LLMServer:Qwen--Qwen3_6-27B-FP8"
    ]
  },
  "environment": {
    "aiperf_source": "git+https://github.com/celinaky/aiperf@1c61dfbb7644b26c70ce21163f0b4cda893438b0",
    "aiperf_version": "0.13.0",
    "cluster_resources": {
      "CPU": 184.0,
      "GPU": 8.0,
      "accelerator_type:B200": 1.0,
      "anyscale/accelerator_shape:8xB200-180G": 1.0,
      "anyscale/node-group:head": 1.0,
      "anyscale/provider:generic": 1.0,
      "anyscale/region:no-sys2-osl1": 1.0,
      "memory": 2061584302080.0,
      "node:10.240.13.26": 1.0,
      "node:__internal_head__": 1.0,
      "object_store_memory": 200000000000.0
    },
    "python_executable": "/home/ray/anaconda3/bin/python",
    "python_version": "3.12.15",
    "ray_path": "/home/ray/anaconda3/lib/python3.12/site-packages/ray/__init__.py",
    "ray_version": "3.0.0.dev0"
  },
  "error": null,
  "failure_stage": null,
  "finished_at": "2026-11-02T21:16:36.840855+00:00",
  "git_revision": "d2199b37c4538f751b460014d06f873efe1ea489",
  "name": "b200-workday-curve-prefill-mean",
  "readiness_s": 0.018422137014567852,
  "routing": {
    "ai_dynamo_version": "1.6.1",
    "deployments": [
      "LLMRouter",
      "LLMServer:Qwen--Qwen3_6-27B-FP8"
    ],
    "policy": "kv_aware",
    "request_router_class": "ray.serve.llm.request_router.KVAwareRouter",
    "topology": "direct_streaming"
  },
  "started_at": "2026-11-02T20:38:07.341965+00:00",
  "status": "succeeded",
  "telemetry": {
    "serve_metrics.jsonl": {
---
  259 tools/policy_vs_engine_mean.py
  340 tools/policy_decision_timeline.py
  599 total
```

</details>

**Bash** — Read the policy vs engine comparison tool

```bash
cat tools/policy_vs_engine_mean.py
```

**Bash** — Read the token tracker increment and decrement code

```bash
cd /mnt/local_storage/ray && git log --oneline -1 && sed -n 560,760p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py
```

**Bash** — Outline the token tracker module

```bash
cd /mnt/local_storage/ray && grep -n "effective_prefill\|on_prefill_complete\|on_request_completed\|on_lifecycle_events\|class \|def " python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py | head -120
```

**Bash** — Find lifecycle event producers and read the scheduler tracer

```bash
cd /mnt/local_storage/ray && grep -rn "on_prefill_complete\|prefill_complete\|LifecycleEvent\|lifecycle_event" python/ray/llm/_internal/serve --include=*.py | grep -v test | head -40; echo ---; cat python/ray/llm/_internal/serve/engines/vllm/scheduler_trace.py | head -150
```

**Bash** — Peek at per-request CSV, trace files, and router metric samples

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean && head -3 analysis/load_requests.csv; wc -l analysis/load_requests.csv; echo ---; ls -la --time-style=full-iso /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/ | tail -15; echo ---; grep -m1 outstanding_effective_prefill_tokens telemetry/serve_metrics.jsonl | head -c 1500
```

<details><summary>Bash result</summary>

```text
#!/usr/bin/env python3
"""What the mean policy observed vs what the engines held, over time.

Two accountings of the same quantity, total outstanding uncached prefill
tokens across all replicas:
  policy: the router gauge the policy scales on (a request's whole uncached
          prompt counts from routing until its first token), from the run's
          telemetry, and the policy's own published lookback mean of it
          (prefill_required_replicas x target).
  engine: rebuilt from the vLLM scheduler traces: a request's uncached tokens
          enter when it is queued and each chunk leaves when its batch
          completes, so the total is the work still to do. Same lookback mean.
Faint lines are the raw signals, solid lines the lookback means. Target
dotted; scale decisions as in the other charts.

usage: python tools/policy_vs_engine_mean.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH]
"""
import argparse, bisect, datetime, glob, json, math, os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
POLICY, ENGINE = "#6250d6", "#1baf7a"
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lookback", type=float, default=None, help="mean window; default: the policy's upscale_lookback_s")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces")
ap.add_argument("--out", default=None)
ap.add_argument("--raw", action="store_true", help="also draw the raw gauge and raw engine total under the means")
a = ap.parse_args()
R = a.run.rstrip("/")

plot_data = json.load(open(f"{R}/analysis/plot_data.json"))
t0 = plot_data["time_origin"]["profiling_start_ns"]
evaluation = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
target = (kwargs.get("target_prefill_tokens_per_replica") or 20000) / 1000
lookback = a.lookback or kwargs.get("upscale_lookback_s") or 60
end = a.end if a.end is not None else evaluation["periods"]["steady"]["end_s"]
resolved = yaml.safe_load(open(f"{R}/resolved.yaml"))
curve = ((resolved.get("benchmark") or {}).get("workload") or {}).get("concurrency_series") or []

# Policy side: router gauge and the policy's published mean, from telemetry.
raw, pub, asks = [], [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"):
        continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0 / 1e9
    if not a.start <= t <= end:
... [7864 chars trimmed] ...
D([], [], color=POLICY, lw=1.8, label=f"policy: {lookback:.0f}s mean of the router gauge (the number the policy divides by the target)"),
           Line2D([], [], color=ENGINE, lw=1.8, label=f"engine: {lookback:.0f}s mean of the engines' remaining uncached prefill (traces)"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica"),
           Line2D([], [], color=INK, lw=0.8, alpha=0.6, label=f"scale decision (replicas before{ARROW}after)"),
           Line2D([], [], color=INK2, lw=0.8, ls="--", alpha=0.6, label="new replica running (shaded: startup)"),
           Patch(color=TIMER_PASS, label=f"on the x axis: upscale-delay timer running and passed ({up_delay:.0f}s of asking for more), ends at the decision"),
           Patch(color=TIMER_RESET, label="upscale-delay timer running but reset (the ask dropped back before the delay elapsed)")]
if a.raw:
    handles[1:1] = [Line2D([], [], color=POLICY, lw=0.7, alpha=0.4, label="policy: router gauge as scraped, every 2s (whole prompt counted until first token)")]
    handles[3:3] = [Line2D([], [], color=ENGINE, lw=0.7, alpha=0.4, label="engine: remaining uncached prefill as it changes, chunks leave as batches complete")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=2, bbox_to_anchor=(0.5, -0.15))
sec = lambda v: f"{float(v):g}s" if v is not None else "?"
settings = (f"policy: ceil({lookback:.0f}s mean / {target:.0f}K) {DOT} Serve upscale delay {sec(auto.get('upscale_delay_s'))} {DOT} downscale delay {sec(auto.get('downscale_delay_s'))}"
            f" {DOT} replicas {auto.get('min_replicas')}-{auto.get('max_replicas')} {DOT} metrics every {sec(auto.get('metrics_interval_s'))}")
fig.text(0.5, -0.19, settings, ha="center", va="top", fontsize=8.5, color=INK2)
fig.suptitle(f"{evaluation['experiment']} {DOT} what the mean policy observed vs what the engines held: total outstanding uncached prefill tokens",
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/policy_vs_engine_mean{int(lookback)}{'_raw' if a.raw else ''}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)

# Summary: ratio of the two means where the engine mean is meaningful.
pairs = [(p, e) for p, e in zip(policy_mean, engine_mean) if e >= 1.0]
if pairs:
    ratios = sorted(p / e for p, e in pairs)
    print(f"  policy mean / engine mean where engine mean >= 1K: median {ratios[len(ratios) // 2]:.2f}, p10 {ratios[int(0.1 * len(ratios))]:.2f}, p90 {ratios[int(0.9 * len(ratios))]:.2f} (n={len(ratios)})")
    print(f"  time-average over the load phase: policy {sum(policy_mean) / len(policy_mean):.1f}K, engine {sum(engine_mean) / len(engine_mean):.1f}K")
    if pub:
        diffs = [abs(p - q) for p, q in zip(policy_mean, [pub_v[bisect.bisect_right(pub_t, t) - 1] for t in ts]) if True]
        print(f"  published policy mean vs recomputed from the gauge: max abs diff {max(diffs):.1f}K")
```

</details>

<details><summary>Bash result</summary>

```text
ed547dd6dc [serve][llm] Make the token autoscaling policy prefill-only
        lora_name: Optional[str] = None,
        session_id: Optional[str] = None,
    ) -> WorkerSelection:
        """Score the allowed workers for a request based on KV-cache overlap and
        load and pick the best one.

        Session requests reuse their selected worker while it remains eligible
        and session affinity hasn't expired. Every request still reserves load.

        Args:
            request_id: Unique identifier for the request being routed.
            token_ids: Prompt token ids used to compute KV-cache overlap.
            allowed_worker_ids: Candidate worker ids the router may select from.
            expected_output_tokens: The request's output-token cap. With
                select-time reservation this lets selection service decay decode
                load without per-token progress events.
            lora_name: The adapter name vLLM stamps on the request's KV-cache
                events. The selection service salts its KV hashes with it, so
                omitting it on a LoRA request scores against the wrong
                namespace and never matches.
            session_id: The request's session, if any.

        Returns:
            The selected worker (see ``WorkerSelection``).
        """
        if token_ids is None or len(token_ids) == 0:
            raise ValueError("KV aware routing requires non-empty token_ids.")

        if self._svc is None:
            # ai-dynamo is not installed, so this deployment cannot score requests.
            # Fail fast and surface RuntimeError to the client as a 503 via LLMRouter.
            raise RuntimeError(
                "KV-aware routing is unavailable because ai-dynamo is not "
                "installed in the deployment's environment."
            )
        await self._evict_stale_requests()
        request = {
            "model_name": _MODEL_NAME,
            "routing_group": _ROUTING_GROUP,
            "selection_id": request_id,
            "token_ids": token_ids,
            "allowed_worker_ids": allowed_worker_ids,
            "expected_output_tokens": expected_output_tokens,
            "lora_name": lora_name,
            "session_id": session_id,
        }
        selection = await self._svc.select_and_reserve(request)
        self._track_request_state(
            request_id,
            selection["worker_id"],
            len(token_ids),
            expected_output_tokens,
            selection["effective_prefill_tokens"],
        )
        self._publish_autoscaling_snapshot()
        if self._replica_forwarder is not None:
            self._replica_forwarder.report(
                "load_reservation",
                {
                    "model_name": _MODEL_NAME,
                    "routing_group": _ROUTING_GROUP,
                    "selection_id": request_id,
                    "worker_id": selection["worker_id"],
                    "dp_rank": selection["dp_rank"],
  
... [3188 chars trimmed] ...
te.prefill_completed:
            return
        state.prefill_completed = True
        self._outstanding_effective_prefill_tokens -= state.effective_prefill_tokens
        self._active_decode_requests += 1
        self._publish_autoscaling_snapshot()
        await self._svc.prefill_complete(request_id)

    async def on_decode_progress(
        self, request_id: str, cumulative_output_tokens: int
    ) -> None:
        """Advance ``request_id`` to an exact cumulative output-token count,
        booking one decode block in the selection service per crossed boundary.
        """
        state = self._requests.get(request_id)
        if state is None:
            return
        state.output_tokens = cumulative_output_tokens
        new_total_blocks = math.ceil(
            (state.prompt_tokens + cumulative_output_tokens) / self._block_size
        )
        decay_fraction = self._get_decay_fraction(state)
        while new_total_blocks > state.total_blocks:
            state.total_blocks += 1
            self._svc.add_output_block(request_id, decay_fraction=decay_fraction)

    async def on_request_completed(self, request_id: str) -> None:
        """Free ``request_id`` from the selection service's active load and the
        local view."""
        state = self._requests.pop(request_id, None)
        self._mark_request_completed(request_id)
        if state is None:
            return
        self._remove_autoscaling_load(state)
        self._untrack_worker_request(request_id, state.worker_id)
        self._publish_autoscaling_snapshot()
        await self._svc.free_reservation(request_id)

    # TODO: could change this name to not be autoscaling specific
    def _remove_autoscaling_load(self, state: RequestLifecycle) -> None:
        """Remove one request's contribution from its current lifecycle phase."""
        if state.prefill_completed:
            self._active_decode_requests -= 1
        else:
            self._outstanding_effective_prefill_tokens -= state.effective_prefill_tokens

    async def get_autoscaling_snapshot(self) -> Dict[str, int]:
        """Return this router replica's latest global LLM work estimate."""
        await self._evict_stale_requests()
        return self._get_autoscaling_snapshot()

    def _mark_request_completed(self, request_id: str) -> None:
        """Remember completions that beat reservation admission on this ingress."""
        self._completed_request_ids.pop(request_id, None)
        self._completed_request_ids[request_id] = time.monotonic()

    def _untrack_worker_request(self, request_id: str, worker_id: int) -> None:
        """Drop a request from the per-worker reverse index, keeping it in
        lockstep with ``_requests``."""
        request_ids = self._request_ids_by_worker.get(worker_id)
        if request_ids is not None:
            request_ids.discard(request_id)
            if not request_ids:
                del self._request_ids_by_worker[worker_id]
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
48:# Hooks a replica may invoke through ``KVTokenTracker.on_lifecycle_events``.
51:        "on_prefill_complete",
53:        "on_request_completed",
58:def get_worker_id(replica_unique_id: str) -> int:
66:class RequestLifecycle:
73:    effective_prefill_tokens: int = 0
86:class WorkerSelection(TypedDict):
96:    effective_prefill_tokens: int
99:class PeerUpdateForwarder:
112:    def __init__(self, handle: Any, source_writer_id: int):
123:    def report(self, kind: str, event: Dict[str, Any]) -> None:
130:    async def _deliver(self) -> None:
156:    async def flush(self) -> None:
160:    def close(self) -> None:
166:class KVTokenTracker:
192:    def __init__(
222:        self._outstanding_effective_prefill_tokens = 0
251:    def get_block_size(self) -> int:
255:    def start_replica_broadcast(self, handle: Any) -> None:
266:    async def _forward_session_bindings(self) -> None:
272:    async def on_replica_updates(
285:    async def _apply_replica_updates(self) -> None:
310:    def _apply_session_binding(self, binding: Dict[str, Any]) -> None:
316:    async def _apply_reservation(self, booking: Dict[str, Any]) -> None:
335:                booking["effective_prefill_tokens"],
340:    def set_autoscaling_metrics_sink(
347:    def _get_autoscaling_snapshot(self) -> Dict[str, int]:
350:                self._outstanding_effective_prefill_tokens
355:    def _publish_autoscaling_snapshot(self) -> None:
363:    def _defer_autoscaling_snapshot(self) -> None:
368:    def _flush_autoscaling_snapshot(self) -> None:
376:    def _create_selection_service(self) -> None:
403:    def _start_replica_tracking(self) -> None:
421:    def _schedule(self, awaitable) -> None:
426:        async def _run():
433:    def _register_block_size(self, block_size: int, replica_id: str) -> None:
456:    def _on_deployment_targets(self, target_info: DeploymentTargetInfo) -> None:
497:    def remove_worker(self, worker_id: int) -> None:
513:    async def _upsert_worker(
554:    async def select_worker(
612:            selection["effective_prefill_tokens"],
627:                    "effective_prefill_tokens": selection["effective_prefill_tokens"],
635:            "effective_prefill_tokens": selection["effective_prefill_tokens"],
638:    def _track_request_state(
644:        effective_prefill_tokens: int,
659:            effective_prefill_tokens=effective_prefill_tokens,
663:        self._outstanding_effective_prefill_tokens += effective_prefill_tokens
666:    async def on_lifecycle_events(self, events: List[tuple]) -> None:
693:    async def on_prefill_complete(self, request_id: str) -> None:
700:        self._outstanding_effective_prefill_tokens -= state.effective_prefill_tokens
705:    async def on_decode_progress(
723:    async def on_request_completed(self, request_id: str) -> None:
736:    def _remove_autoscaling_load(self, state: RequestLifecycle) -> None:
741:            self._outstanding_effective_prefill_tokens -= state.effective_prefill_tokens
743:    async def get_autoscaling_snapshot(self) -> Dict[str, int]:
748:    def _mark_request_completed(self, request_id: str) -> None:
753:    def _untrack_worker_request(self, request_id: str, worker_id: int) -> None:
762:    async def _evict_stale_requests(self) -> None:
791:    def _get_decay_fraction(self, state: RequestLifecycle) -> Optional[float]:
802:def set_kv_token_tracker(tracker: "KVTokenTracker") -> None:
807:def get_kv_token_tracker() -> Optional["KVTokenTracker"]:
812:# name override -> the class name). Engine replicas RPC its ``on_lifecycle_events``
817:def get_llm_router_handle():
819:    its ``on_lifecycle_events`` method. Resolved in the current Serve app.
825:def build_kv_token_tracker(
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
(eval):1: no matches found: --include=*.py
---
"""Opt-in request and batch tracing for the vLLM V1 scheduler.

Set ``RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR`` in the LLM deployment's runtime
environment to enable tracing. The optional ``*_FILE_PREFIX``,
``*_MAX_QUEUE_SIZE``, ``*_MAX_FILE_BYTES``, and ``*_FLUSH_INTERVAL_S``
environment variables tune output naming and buffering. The scheduler hot path
only performs a nonblocking queue write; JSON encoding and file I/O run on a
background thread.
"""

import json
import os
import queue
import re
import socket
import threading
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Iterable, Mapping, Optional, Tuple

from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.scheduler import Scheduler

from ray.llm._internal.serve.observability.logging import get_logger

logger = get_logger(__name__)

SCHEDULER_TRACE_DIR_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR"
SCHEDULER_TRACE_FILE_PREFIX_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_FILE_PREFIX"
SCHEDULER_TRACE_MAX_QUEUE_SIZE_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_MAX_QUEUE_SIZE"
SCHEDULER_TRACE_MAX_FILE_BYTES_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_MAX_FILE_BYTES"
SCHEDULER_TRACE_FLUSH_INTERVAL_S_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_FLUSH_INTERVAL_S"

_DEFAULT_MAX_QUEUE_SIZE = 10_000
_DEFAULT_MAX_FILE_BYTES = 256 * 1024 * 1024
_DEFAULT_FLUSH_INTERVAL_S = 1.0
_TRACE_SCHEMA_VERSION = 1
_STOP = object()

_SYNC_SCHEDULER_PATH = (
    "ray.llm._internal.serve.engines.vllm.scheduler_trace.TracingScheduler"
)
_ASYNC_SCHEDULER_PATH = (
    "ray.llm._internal.serve.engines.vllm.scheduler_trace.TracingAsyncScheduler"
)


def scheduler_trace_enabled() -> bool:
    return bool(os.environ.get(SCHEDULER_TRACE_DIR_ENV, "").strip())


def configure_scheduler_trace(vllm_config: Any) -> bool:
    """Install the tracing scheduler selected for the configured execution mode."""
    if not scheduler_trace_enabled():
        return False

    scheduler_config = vllm_config.scheduler_config
    current_cls = scheduler_config.scheduler_cls
    tracing_classes = {
        _SYNC_SCHEDULER_PATH,
        _ASYNC_SCHEDULER_PATH,
        TracingScheduler,
        TracingAsyncScheduler,
    }
    if current_cls is not None and current_cls not in tracing_classes:
        raise ValueError(
            f"{SCHEDULER_TRACE_DIR_ENV} cannot be combined with a custom "
            f"vLLM scheduler_cls ({current_cls!r})."
        )

    scheduler_config.scheduler_cls = (
        _ASYNC_SCHEDULER_PATH
        if scheduler_config.async_scheduling
        else _SYNC_SCHEDULER_PATH
    )
    return True


def _positive_int_from_env(name: str, default: int) -> int:
    value = int(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _positive_float_from_env(name: str, default: float) -> float:
    value = float(os.environ.get(name, str(default)))
    if value <= 0:
        raise ValueError(f"{name} must be positive, got {value}")
    return value


def _safe_filename_component(value: str) -> str:
    value = re.sub(r"[^A-Za-z0-9_.-]+", "-", value).strip("-.")
    return value or "engine"


class _JsonlTraceWriter:
    """A bounded, nonblocking producer with one background JSONL writer."""

    def __init__(
        self,
        trace_dir: str,
        *,
        file_prefix: Optional[str] = None,
        max_queue_size: int = _DEFAULT_MAX_QUEUE_SIZE,
        max_file_bytes: int = _DEFAULT_MAX_FILE_BYTES,
        flush_interval_s: float = _DEFAULT_FLUSH_INTERVAL_S,
    ) -> None:
        self._trace_dir = Path(trace_dir)
        self._trace_dir.mkdir(parents=True, exist_ok=True)
        default_prefix = f"{socket.gethostname()}_pid-{os.getpid()}"
        if file_prefix:
            default_prefix = f"{file_prefix}_{default_prefix}"
        self._file_prefix = _safe_filename_component(default_prefix)
        self._max_file_bytes = max_file_bytes
        self._flush_interval_s = flush_interval_s
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=max_queue_size)
        self._dropped_events = 0
        self._drop_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="vllm-scheduler-trace-writer",
            daemon=True,
        )
        self._thread.start()

    def emit(self, event: Dict[str, Any]) -> None:
        if self._closed:
            return

        with self._drop_lock:
            dropped_events = self._dropped_events
            self._dropped_events = 0

        if dropped_events:
            drop_event = {
                "schema_version": _TRACE_SCHEMA_VERSION,
                "event": "trace_events_dropped",
                "monotonic_s": time.monotonic(),
                "wall_time_s": time.time(),
                "count": dropped_events,
            }
            try:
                self._queue.put_nowait(drop_event)
            except queue.Full:
                with self._drop_lock:
                    self._dropped_events += dropped_events
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
request_id,engine,kind,engine_load_tokens,ongoing_sequences,kv_context_tokens,kv_cache_usage,step_load_tokens,idle_steps,own_uncached,isl,preemptions_before_first_token,recompute_tokens_before_first_token,steps,step_ms,client_ttft_ms,pre_ms,queue_ms,prefill_ms,post_ms,min_steps,chunk_budget
f7fcefe8-2956-4075-8982-8768ce95888a,k-5be017850000_pid-151035,cold,0,0,0,0.0,665.0,0,665,665,0,0,1,42.49587398953736,140.119171,95.221504,0.43502310290932655,42.49587398953736,1.9667699075533136,1,15680
04562d55-a87b-43b5-a9bf-19a60ca31b8e,k-5be017850000_pid-151035,cold,0,0,684,0.0014561339643247573,32364.0,0,63724,63724,0,0,5,654.9183895811439,3425.4791259999997,123.12704,0.523966271430254,3274.5919479057193,27.236171822850338,5,15680
1585 analysis/load_requests.csv
---
-rw-r--r-- 1 ray users 123113466 2026-10-29 11:55:30.534404142 -0700 k-f2a796a80000_pid-107271.0000.jsonl
-rw-r--r-- 1 ray users  47357390 2026-10-29 11:54:25.466949002 -0700 k-f2a796a80000_pid-126723.0000.jsonl
-rw-r--r-- 1 ray users  27500380 2026-10-29 11:55:01.412354217 -0700 k-f2a796a80000_pid-137405.0000.jsonl
-rw-r--r-- 1 ray users 268434901 2026-10-29 13:56:56.734236797 -0700 k-f2a796a80000_pid-314276.0000.jsonl
-rw-r--r-- 1 ray users  42692320 2026-10-29 14:01:36.827348228 -0700 k-f2a796a80000_pid-314276.0001.jsonl
-rw-r--r-- 1 ray users 262469893 2026-10-29 14:02:09.688729803 -0700 k-f2a796a80000_pid-329231.0000.jsonl
-rw-r--r-- 1 ray users 268435352 2026-10-29 14:01:49.184697502 -0700 k-f2a796a80000_pid-332291.0000.jsonl
-rw-r--r-- 1 ray users    904490 2026-10-29 14:01:55.714375281 -0700 k-f2a796a80000_pid-332291.0001.jsonl
-rw-r--r-- 1 ray users 243338199 2026-10-29 14:01:49.754698400 -0700 k-f2a796a80000_pid-337202.0000.jsonl
-rw-r--r-- 1 ray users 163235689 2026-10-29 13:58:05.064887524 -0700 k-f2a796a80000_pid-348080.0000.jsonl
-rw-r--r-- 1 ray users         0 2026-10-29 13:44:42.122855050 -0700 k-f2a796a80000_pid-354957.0000.jsonl
-rw-r--r-- 1 ray users  23763349 2026-10-29 13:49:58.399577784 -0700 k-f2a796a80000_pid-362063.0000.jsonl
-rw-r--r-- 1 ray users  10145144 2026-10-29 13:49:58.397577781 -0700 k-f2a796a80000_pid-363688.0000.jsonl
-rw-r--r-- 1 ray users   9293013 2026-10-29 13:54:35.056013608 -0700 k-f2a796a80000_pid-377095.0000.jsonl
-rw-r--r-- 1 ray users 203317489 2026-10-29 11:54:06.066259335 -0700 k-f2a796a80000_pid-79255.0000.jsonl
---
{"node_id": "33c8be3ae1e891d03c293636681cadf5daaff6dd64aa71a95bc02323", "node_ip": "10.240.13.26", "endpoint": "http://10.240.13.26:8085/metrics", "scrape_started_at_ns": 1793652004833193404, "samples": [{"name": "ray_serve_autoscaling_target_ongoing_requests", "normalized_name": "serve_autoscaling_target_ongoing_requests", "type": "gauge", "labels": {"Component": "core_worker", "NodeAddress": "10.240.13.26", "SessionName": "session_2026-10-29_21-07-14_629089_584", "Version": "3.0.0.dev0", "WorkerId": "9837a047025640413c340746f04c412f937c40ecb887b245d5f4dd32", "application": "ttft-benchmark", "deployment": "LLMServer:Qwen--Qwen3_6-27B-FP8"}, "value": 8.0}, {"name": "ray_serve_autoscaling_total_requests", "normalized_name": "serve_autoscaling_total_requests", "type": "gauge", "labels": {"Component": "core_worker", "NodeAddress": "10.240.13.26", "SessionName": "session_2026-10-29_21-07-14_629089_584", "Version": "3.0.0.dev0", "WorkerId": "9837a047025640413c340746f04c412f937c40ecb887b245d5f4dd32", "application": "ttft-benchmark", "deployment": "LLMServer:Qwen--Qwen3_6-27B-FP8"}, "value": 0.0}, {"name": "ray_serve_autoscaling_desired_replicas", "normalized_name": "serve_autoscaling_desired_replicas", "type": "gauge", "labels": {"Component": "core_worker", "NodeAddress": "10.240.13.26", "SessionName": "session_2026-10-29_21-07-14_629089_584", "Version": "3.0.0.dev0", "WorkerId": "9837a047025640413c340746f04c412f937c40ecb887b245d5f4dd32", "application": "ttft-benchmark", "deploymen
```

</details>

**Bash** — Read the token autoscaling policy

```bash
cd /mnt/local_storage/ray && cat python/ray/serve/experimental/llm_token_autoscaling_policy.py
```

**Bash** — Read tracker init, peer reservation application, snapshot publishing, and stale eviction

```bash
cd /mnt/local_storage/ray && sed -n 166,260p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo ======; sed -n 266,380p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo =====; sed -n 760,800p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py
```

**Bash** — Find where replicas emit lifecycle events

```bash
cd /mnt/local_storage/ray && grep -rn "on_prefill_complete\|on_lifecycle_events\|on_request_completed" python/ray/llm/_internal/serve python/ray/serve --include='*.py' | grep -v "/tests/" 
```

**Bash** — Find this run's trace files, inspect serve status format, and show deployment config

```bash
find /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces -name '*.jsonl' -newermt '2026-11-02T20:38:07Z' -exec ls -la --time-style=full-iso {} \; ; echo ---; python3 -I - <<'EOF'
import json
# router replica count over time
seen=set()
for line in open('telemetry/serve_status.jsonl'):
    try: d=json.loads(line)
    except: continue
    s=json.dumps(d)
    import re
    for m in re.finditer(r'"LLMRouter[^"]*"[^{}]{0,400}', s):
        pass
    # crude: print keys first time
    if not seen:
        print(list(d.keys())); seen.add(1)
        print(s[:2500])
    break
EOF
echo ---; grep -n -i "router\|num_replicas\|autoscaling\|lookback\|target\|upscale\|downscale\|metrics_interval" resolved.yaml | head -60
```

**Bash** — Summarize per-request CSV by kind and find column definitions

```bash
python3 -I - <<'EOF'
import csv,statistics as st
rows=list(csv.DictReader(open('analysis/load_requests.csv')))
from collections import Counter
print(Counter(r['kind'] for r in rows))
for k in ('cold','warm','session','pinned'):
    rs=[r for r in rows if r['kind']==k]
    if not rs: continue
    own=[float(r['own_uncached']) for r in rs]; isl=[float(r['isl']) for r in rs]
    print(k, len(rs), 'own_uncached sum %.0f mean %.0f median %.0f | isl sum %.0f mean %.0f median %.0f'%(sum(own),st.mean(own),st.median(own),sum(isl),st.mean(isl),st.median(isl)))
    q=[float(r['queue_ms']) for r in rs]; p=[float(r['prefill_ms']) for r in rs]; pre=[float(r['pre_ms']) for r in rs]; post=[float(r['post_ms']) for r in rs]
    print('   queue_ms med %.0f mean %.0f | prefill_ms med %.0f mean %.0f | pre_ms med %.0f mean %.0f | post_ms med %.0f mean %.0f | steps mean %.1f'%(st.median(q),st.mean(q),st.median(p),st.mean(p),st.median(pre),st.mean(pre),st.median(post),st.mean(post),st.mean([float(r['steps']) for r in rs])))
print('total own_uncached', sum(float(r['own_uncached']) for r in rows), 'total isl', sum(float(r['isl']) for r in rows))
EOF
grep -rn "pre_ms\|post_ms\|own_uncached\|kind" /home/ray/default/serve-llm-autoscaling-experiment/tools/*.py | grep -i "def \|=\s*\|#" | head -30
```

**Bash** — Locate the dynamo selection service and how effective_prefill_tokens is computed

```bash
python3 -c "import dynamo,os;print(os.path.dirname(dynamo.__file__))" 2>/dev/null; pip show ai-dynamo 2>/dev/null | head -3; python3 - <<'EOF'
import glob,os
for p in glob.glob('/home/ray/anaconda3/lib/python3.12/site-packages/dynamo*'):
    print(p)
EOF
grep -rln "effective_prefill_tokens" /home/ray/anaconda3/lib/python3.12/site-packages/dynamo* 2>/dev/null | head; grep -rn "effective_prefill_tokens" /home/ray/anaconda3/lib/python3.12/site-packages/dynamo*/ --include='*.py' --include='*.pyi' 2>/dev/null | head -20
```

<details><summary>Bash result</summary>

```text
import math
import time
from typing import Any, Dict, List, Literal, Tuple

from ray.serve._private.constants import SERVE_AUTOSCALING_POLICY_METRICS_KEY
from ray.serve.config import AutoscalingContext

OUTSTANDING_EFFECTIVE_PREFILL_TOKENS = "outstanding_effective_prefill_tokens"

# Policy metrics exported by the controller as serve_autoscaling_policy_metric.
PREFILL_REQUIRED_REPLICAS = "prefill_required_replicas"
REQUIRED_REPLICAS = "required_replicas"

_UpscaleAggregation = Literal["max", "mean"]
_Snapshot = Tuple[float, float]


def _now(ctx: AutoscalingContext) -> float:
    return ctx.current_time if ctx.current_time is not None else time.time()


def _fresh_snapshots_by_source(
    ctx: AutoscalingContext,
) -> Dict[str, List[_Snapshot]]:
    """Return fresh ``(timestamp, prefill)`` series per reporting source."""
    raw_metrics = ctx.raw_router_metrics or {}
    prefill_by_source = raw_metrics.get(OUTSTANDING_EFFECTIVE_PREFILL_TOKENS, {})
    cutoff = _now(ctx) - ctx.config.look_back_period_s
    snapshots: Dict[str, List[_Snapshot]] = {}

    for source_id, prefill_series in prefill_by_source.items():
        if not prefill_series:
            continue
        if prefill_series[-1].timestamp < cutoff:
            continue

        source_snapshots = [
            (sample.timestamp, sample.value)
            for sample in prefill_series
            if sample.timestamp >= cutoff
        ]
        if source_snapshots:
            snapshots[source_id] = source_snapshots

    return snapshots


def _time_weighted_mean(
    samples: List[Tuple[float, float]], window_start: float, window_end: float
) -> float:
    """Average a right-continuous gauge over its observed part of a window."""
    observed_start = max(window_start, samples[0][0])
    if window_end <= observed_start:
        return samples[-1][1]

    weighted_sum = 0.0
    for index, (timestamp, value) in enumerate(samples):
        interval_start = max(timestamp, observed_start)
        interval_end = samples[index + 1][0] if index + 1 < len(samples) else window_end
        interval_end = min(interval_end, window_end)
        if interval_end > interval_start:
            weighted_sum += value * (interval_end - interval_start)

    return weighted_sum / (window_end - observed_start)


class LLMTokenAutoscalingPolicy:
    """Scale an aggregated LLM deployment on outstanding uncached prefill tokens.

    Both directions follow the time-weighted mean over ``upscale_lookback_s``
    of the router's outstanding effective (uncached) prefill tokens divided by
    ``target_prefill_tokens_per_replica``. Router replicas report the same
    global snapshot, so per-reporter means are maximized rather than summed.
    The latest raw snapshot and the mean are published under
    ``SERVE_AUTOSCALING_POLICY_METRICS_KEY`` for the controller to export. Asymmetry comes from Serve's ``upscale_delay_s``
    and ``downscale_delay_s``.

    ``upscale_aggregation`` and ``scale_down_sensitivity`` are validat
... [1586 chars trimmed] ...
w - self._upscale_lookback_s
        capacity_by_source = []
        for snapshots in snapshots_by_source.values():
            capacity_samples = [
                (timestamp, self._required_capacity(prefill))
                for timestamp, prefill in snapshots
            ]
            samples = [
                sample for sample in capacity_samples if sample[0] >= window_start
            ]
            if not samples:
                continue
            # Metrics are gauges. Carry the last observation before the window
            # boundary forward so the mean covers the full window, provided a
            # new observation confirms that the source is live.
            samples_before_window = [
                sample for sample in capacity_samples if sample[0] < window_start
            ]
            if samples_before_window:
                samples.insert(0, samples_before_window[-1])
            capacity_by_source.append(_time_weighted_mean(samples, window_start, now))

        return max(capacity_by_source, default=0.0)

    def __call__(self, ctx: AutoscalingContext) -> Tuple[int, Dict[str, Any]]:
        if self._upscale_lookback_s > ctx.config.look_back_period_s:
            raise ValueError(
                "upscale_lookback_s must not exceed the autoscaling config's "
                "look_back_period_s"
            )

        snapshots_by_source = _fresh_snapshots_by_source(ctx)
        # A missing or stale observation is not evidence of zero work.
        if not snapshots_by_source:
            return ctx.target_num_replicas, {}

        now = _now(ctx)
        window_start = now - self._upscale_lookback_s
        if not any(
            timestamp >= window_start
            for snapshots in snapshots_by_source.values()
            for timestamp, _ in snapshots
        ):
            # No observation in the lookback window is also fail-closed.
            return ctx.target_num_replicas, {}

        prefill_required_replicas = self._mean_required_capacity(
            snapshots_by_source, now
        )
        required_replicas = math.ceil(prefill_required_replicas)
        if ctx.total_queued_requests > 0:
            required_replicas = max(1, required_replicas)

        latest_snapshots = [snapshots[-1] for snapshots in snapshots_by_source.values()]
        outstanding_prefill_tokens = max(
            max(0, prefill) for _, prefill in latest_snapshots
        )
        state: Dict[str, Any] = {
            OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
            REQUIRED_REPLICAS: required_replicas,
            "lookback_s": self._upscale_lookback_s,
            SERVE_AUTOSCALING_POLICY_METRICS_KEY: {
                OUTSTANDING_EFFECTIVE_PREFILL_TOKENS: outstanding_prefill_tokens,
                PREFILL_REQUIRED_REPLICAS: prefill_required_replicas,
                REQUIRED_REPLICAS: required_replicas,
            },
        }
        return required_replicas, state
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
Exit code 1
class KVTokenTracker:
    """Tracks per-replica KV-cache overlap, token load, and session affinity
    inside the LLMRouter ingress replica.

    Built by the LLMRouter ingress replica via ``build_kv_token_tracker``, so
    ``select_worker`` is a local call on the ingress event loop.

    1. Owns a router-local Dynamo ``SelectionService``.
    2. Tracks live replicas via a ``LongPollClient`` on ``DEPLOYMENT_TARGETS``,
       mapping each running replica to a Dynamo worker id.
    3. The ``SelectionService`` maintains a global KV index radix tree, fed by
       every replica's KV events; each node records which workers hold that KV block.
    4. Scoring (``select_worker``) atomically ranks candidate workers by
       KV-cache overlap and current token load, reserves the chosen worker, and
       records local lifecycle state. A request with a session id reuses its
       session's bound worker while that worker remains a candidate; otherwise
       the selected worker becomes the session's new binding. A separate
       select-then-reserve flow causes herding because concurrent requests can
       select the same worker from stale load state before any reservation is
       visible.
    5. Load reservations and session affinity bindings are replicated to peer
       ingress replicas in the background (see ``PeerUpdateForwarder``), so
       every selection service applies the engine's subsequent lifecycle
       events and pins a session's later turns to the same worker.
    """

    def __init__(
        self,
        indexer_threads: int = DEFAULT_KV_INDEXER_THREADS,
        serve_deployment_id: Optional[Any] = None,
        session_affinity_ttl_s: float = DEFAULT_SESSION_AFFINITY_TTL_S,
    ):
        # The tracked LLMServer deployment id, passed in by the LLMRouter
        # that builds the tracker.
        self._serve_deployment_id = serve_deployment_id
        # KV-cache block size, learned once from the first replica's reported
        # engine config and passed to the selection service, which uses it to
        # track the worker's active load and index its KV blocks for overlap.
        self._block_size: Optional[int] = None
        self._indexer_threads = indexer_threads
        self._session_affinity_ttl_s = session_affinity_ttl_s
        # _replica_id_by_worker maps a Dynamo worker id to the running replica's full
        # id string, kept in sync with the deployment's live replicas over LongPoll.
        # NOTE (jeffreywang): _replica_id_by_worker is later used by select_worker
        # to get candidate workers to route among.
        self._replica_id_by_worker: Dict[int, str] = {}
        # Per-request state that the lifecycle hooks need, keyed by request id, serves
        # the following purposes:
        #   1. Block cursor: Turn cumulative decode tokens into add_output_block deltas.
        #   2. expected_output_tokens for decode-block decay weighting.
        #   3. In-flight request set: Free reservation exactly once.
        # Ordered oldest-first so the TTL sweep pops stale entries off the front.
        self._requests: "OrderedDict[str, RequestLifecycle]" = OrderedDict()
        # Global work snapshots exported to the LLM deployment autoscaler.
        # Keep these incrementally rather than scanning every live request on
        # each metrics collection interval.
        self._outstanding_effective_prefill_tokens = 0
        self._active_decode_requests = 0
        self._autoscaling_metrics_sink: Optional[
            Callable[[Dict[str, int]], None]
        ] = None
        self._autoscaling_publish_defer_depth = 0
        self._autoscaling_publish_dirty = False
        # Reverse index of in-flight request ids per worker, kept in lockstep with
        # _requests, so remove_worker is O(k) in the worker's requests, not O(N).
        self._request_ids_by_worker: Dict[int, Set[str]] = {}
        # Request ids whose completion arrived before their reservation broadcast.
        # Ordered oldest-first so the stale sweep can bound memory.
        self._completed_request_ids: "OrderedDict[str, float]" = OrderedDict()
        self._pending_tasks: Set[asyncio.Task] = set()
        # Identifies this ingress: tags its broadcasts so it skips its own, and
        # versions the session affinity bindings it writes.
        self._replication_writer_id = get_worker_id(uuid.uuid4().hex)
        # Sends this ingress's load reservations and session affinity bindings to peers.
        self._replica_forwarder: Optional[PeerUpdateForwarder] = None
        # Streams session affinity bindings from the selection service to the forwarder.
        self._session_affinity_task: Optional[asyncio.Task] = None
        # Peer load reservations and session affinity bindings awaiting apply.
        self._replica_updates: asyncio.Queue = asyncio.Queue()
        # Applies queued peer updates off the RPC path.
        self._replica_apply_task: Optional[asyncio.Task] = None
        self._long_poll_client: Optional[LongPollClient] = None
        self._create_selection_service()
        self._start_replica_tracking()

    def get_block_size(self) -> int:
        """Return the KV-cache block size used for decode-block accounting."""
        return self._block_size

    def start_replica_broadcast(self, handle: Any) -> None:
        """Broadcast load reservations and session affinity bindings to peers."""
        if self._svc is not None:
            self._replica_forwarder = PeerUpdateForwarder(
                handle, self._replication_writer_id
            )
(eval):1: ===== not found
```

</details>

<details><summary>Bash result</summary>

```text
python/ray/llm/_internal/serve/serving_patterns/prefill_decode/pd_server.py:308:                    # context (so on_request_completed fires once).
python/ray/llm/_internal/serve/core/ingress/router.py:150:        # engine-facing on_lifecycle_events method can book load into it.
python/ray/llm/_internal/serve/core/ingress/router.py:298:    async def on_lifecycle_events(self, batch):
python/ray/llm/_internal/serve/core/ingress/router.py:305:        return await self._kv_token_tracker.on_lifecycle_events(batch)
python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:61:                    "on_lifecycle_events", batch
python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:125:            self._forwarder.report("on_prefill_complete", self._request_id)
python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:135:            self._forwarder.report("on_request_completed", self._request_id)
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:48:# Hooks a replica may invoke through ``KVTokenTracker.on_lifecycle_events``.
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:51:        "on_prefill_complete",
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:53:        "on_request_completed",
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:666:    async def on_lifecycle_events(self, events: List[tuple]) -> None:
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:693:    async def on_prefill_complete(self, request_id: str) -> None:
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:723:    async def on_request_completed(self, request_id: str) -> None:
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:812:# name override -> the class name). Engine replicas RPC its ``on_lifecycle_events``
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:819:    its ``on_lifecycle_events`` method. Resolved in the current Serve app.
python/ray/serve/experimental/capacity_queue_router.py:280:    def on_request_completed(
python/ray/serve/handle.py:1239:        exits. The router fires ``on_request_completed`` exactly once per dispatched
python/ray/serve/_private/request_router/replica_wrapper.py:437:    # choose_replica's finally to decide whether to fire on_request_completed
python/ray/serve/_private/router.py:984:            self._active_request_router.on_request_completed(
python/ray/serve/_private/router.py:1181:                self._active_request_router.on_request_completed(
python/ray/serve/_private/router.py:1410:                # Every dispatched request must fire on_request_completed exactly
python/ray/serve/_private/router.py:1420:                    self._active_request_router.on_request_completed(
python/ray/serve/_private/router.py:1500:        # The done-callback registered above will fire on_request_completed
python/ray/serve/_private/request_router/request_router.py:1524:    def on_request_completed(
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
-rw-r--r-- 1 ray users 268434833 2026-11-02 12:57:05.848808025 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-151035.0000.jsonl
-rw-r--r-- 1 ray users 178973301 2026-11-02 13:11:52.447714065 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-151035.0001.jsonl
-rw-r--r-- 1 ray users 228855770 2026-11-02 13:12:28.708170682 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-173074.0000.jsonl
-rw-r--r-- 1 ray users 218945425 2026-11-02 13:11:20.271753818 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-180495.0000.jsonl
-rw-r--r-- 1 ray users 268434676 2026-11-02 14:56:52.782812795 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-361781.0000.jsonl
-rw-r--r-- 1 ray users 190672836 2026-11-02 15:11:08.526728455 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-361781.0001.jsonl
-rw-r--r-- 1 ray users 268434545 2026-11-02 15:09:52.067826160 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-379492.0000.jsonl
-rw-r--r-- 1 ray users 10394342 2026-11-02 15:11:08.506728481 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-379492.0001.jsonl
-rw-r--r-- 1 ray users 237771954 2026-11-02 15:09:46.422833374 -0800 /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-387289.0000.jsonl
---
Traceback (most recent call last):
  File "<stdin>", line 4, in <module>
FileNotFoundError: [Errno 2] No such file or directory: 'telemetry/serve_status.jsonl'
---
ugrep: warning: resolved.yaml: No such file or directory
```

</details>

<details><summary>Bash result</summary>

```text
Traceback (most recent call last):
  File "<stdin>", line 2, in <module>
FileNotFoundError: [Errno 2] No such file or directory: 'analysis/load_requests.csv'
/home/ray/default/serve-llm-autoscaling-experiment/tools/bucket_counts.py:19:        n = sum(lo <= float(r["own_uncached"]) < hi and
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_model.py:33:    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_model.py:34:    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_model.py:51:L, U, I = col("engine_load_tokens") / 1000, col("own_uncached") / 1000, col("isl") / 1000
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_breakdown.py:30:    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_breakdown.py:31:    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_breakdown.py:103:    out = {"bucket_start_s": b * a.step, "kind": "cold" if cold else "warm", "requests": len(rs),
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_breakdown.py:127:    ts = [t for t in table if t["kind"] == kind]
/home/ray/default/serve-llm-autoscaling-experiment/tools/ttft_breakdown.py:139:    ax.set_title(f"{kind.capitalize()} turns ({'<' if kind == 'cold' else '≥'}50% of prompt cached, "
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_ttft_breakdown.py:51:    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_ttft_breakdown.py:52:    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_ttft_breakdown.py:183:        rs = [r for r in rows if s_lo <= r["own_uncached"] < s_hi and r["bin"] == i]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_ttft_breakdown.py:209:    band_rows = [r for r in rows if s_lo <= r["own_uncached"] < s_hi]
/home/ray/default/serve-llm-autoscaling-experiment/tools/step_breakdown.py:162:    rs = [r for r in requests if r["kind"] == kind]
/home/ray/default/serve-llm-autoscaling-experiment/tools/step_breakdown.py:163:    print(f"{kind}: n={len(rs)}  queue {sum(r['queue_ms'] for r in rs)/sum(r['uncached'] for r in rs)*1000:.1f} ms/1K uncached  "
/home/ray/default/serve-llm-autoscaling-experiment/tools/step_breakdown.py:182:        ts = [t for t in table if t["kind"] == kind]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:44:    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:45:    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:166:        own_uncached = adm["prompt_tokens"] - adm["cached_prefix_tokens"]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:192:    tail = r["own_uncached"] % block if block else 0
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:193:    r["min_steps"] = max(1, math.ceil((r["own_uncached"] - tail) / budget) + (1 if tail else 0))
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:229:        rs = [r for r in rows if r["kind"] == kind and bin_of(r["engine_load_tokens"]) == i]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:232:        out = {"kind": kind, "engine_load": label, "requests": len(rs),
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:259:        rs = [r for r in rows if r["kind"] == kind and step_bin_of(r["step_load_tokens"]) == i]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:285:    ts = [t for t in table if t["kind"] == kind]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:296:    ax.set_title(f"{kind.capitalize()} turns ({n}): mean client TTFT", loc="left", fontsize=11, color=INK)
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:297:    sts = [t for t in step_table if t["kind"] == kind]
/home/ray/default/serve-llm-autoscaling-experiment/tools/state_breakdown.py:42:    ("pre_ms", "Before engine (proxy, router, tokenize)", "#1baf7a"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/state_breakdown.py:43:    ("post_ms", "After engine (detokenize, stream)", "#eda100"),
/home/ray/default/serve-llm-autoscaling-experiment/tools/state_breakdown.py:108:    band = [r for r in rows if s_lo <= r["own_uncached"] < s_hi]
```

</details>

<details><summary>Bash result</summary>

```text
Name: ai-dynamo
Version: 1.6.1
Summary: Distributed Inference Framework
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo
```

</details>

**Bash** — Read the replica-side lifecycle event emitter

```bash
cd /mnt/local_storage/ray && cat python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py
```

**Bash** — Read peer forwarder, reservation application, snapshot publishing, and stale eviction

```bash
cd /mnt/local_storage/ray && sed -n 99,165p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo '#####'; sed -n 266,380p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo '#####'; sed -n 760,800p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py
```

**Bash** — Show deployment config and serve status format

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; ls $R/telemetry $R/deployment; grep -n -i "router\|num_replicas\|autoscaling\|lookback\|target\|upscale\|downscale\|metrics_interval\|block\|chunk\|max_num" $R/resolved.yaml | head -60; echo '#####'; python3 -I - $R <<'EOF'
import json,sys,re
R=sys.argv[1]
n=0
for line in open(f'{R}/telemetry/serve_status.jsonl'):
    try: d=json.loads(line)
    except: continue
    s=json.dumps(d)
    if n==0: print(s[:3000]); 
    n+=1
print('lines',n)
EOF
```

**Bash** — Summarize per-request CSV by kind and show how its columns are derived

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import csv,statistics as st,sys
R=sys.argv[1]
rows=list(csv.DictReader(open(f'{R}/analysis/load_requests.csv')))
from collections import Counter
print(Counter(r['kind'] for r in rows), Counter(r['engine'] for r in rows))
for k in sorted(set(r['kind'] for r in rows)):
    rs=[r for r in rows if r['kind']==k]
    own=[float(r['own_uncached']) for r in rs]; isl=[float(r['isl']) for r in rs]
    print(k, len(rs), 'own_uncached sum %.0f mean %.0f median %.0f | isl sum %.0f mean %.0f median %.0f'%(sum(own),st.mean(own),st.median(own),sum(isl),st.mean(isl),st.median(isl)))
    q=[float(r['queue_ms']) for r in rs]; p=[float(r['prefill_ms']) for r in rs]; pre=[float(r['pre_ms']) for r in rs]; post=[float(r['post_ms']) for r in rs]
    print('   queue_ms med %.0f mean %.0f | prefill_ms med %.0f mean %.0f | pre_ms med %.0f mean %.0f | post_ms med %.0f mean %.0f | steps mean %.1f'%(st.median(q),st.mean(q),st.median(p),st.mean(p),st.median(pre),st.mean(pre),st.median(post),st.mean(post),st.mean([float(r['steps']) for r in rs])))
print('total own_uncached', sum(float(r['own_uncached']) for r in rows), 'total isl', sum(float(r['isl']) for r in rows))
EOF
sed -n 140,200p /home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py
```

**Bash** — Show scheduler trace event schema samples

```bash
T=/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-151035.0000.jsonl; head -c 3000 $T; echo; echo '#####'; python3 -I - $T <<'EOF'
import json,sys
seen={}
for i,line in enumerate(open(sys.argv[1])):
    try: e=json.loads(line)
    except: continue
    ev=e.get('event')
    if ev not in seen:
        seen[ev]=line[:1500]
    if len(seen)>12 and i>200000: break
for k,v in seen.items(): print('==',k); print(v)
EOF
```

**Bash** — Locate the dynamo selection service implementation

```bash
D=/home/ray/anaconda3/lib/python3.12/site-packages/dynamo; find $D -maxdepth 2 | head -40; grep -rl "effective_prefill" $D 2>/dev/null | head; find $D -name '*.so' | head; for f in $(find $D -name '*.so' | head -3); do strings $f | grep -i "effective_prefill" | head -5; done
```

<details><summary>Bash result</summary>

```text
import asyncio
from typing import Optional, Type

from vllm.outputs import RequestOutput
from vllm.sampling_params import RequestOutputKind
from vllm.v1.engine.async_llm import AsyncLLM

from ray import serve
from ray.llm._internal.serve.observability.logging import get_logger
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (
    get_llm_router_handle,
    get_worker_id,
)
from ray.llm._internal.serve.utils.server_utils import get_serve_request_id
from ray.serve.handle import DeploymentHandle

logger = get_logger(__name__)


class LifecycleEventForwarder:
    """Ordered, non-blocking bridge from the engine to every LLMRouter
    replica's KVTokenTracker, which maintains per-replica token load
    statistics. Each batch is broadcast to all ingress replicas, whose
    selection services received the reservation through the ingress-side
    background broadcast.

    ``report`` only enqueues locally, so generation never blocks on delivery.
    A single delivery task per replica drains the queue to the trackers,
    awaiting one broadcast at a time so events arrive everywhere in the order
    they were reported. Events that pile up during a call are sent together in
    the next one.
    """

    def __init__(self, handle: DeploymentHandle, worker_id: int):
        self.handle = handle
        self.worker_id = worker_id
        self._events: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None

    def report(self, method_name: str, *args) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._events.put_nowait((method_name, args))

    async def _deliver(self) -> None:
        while True:
            # Wait for the next event, then drain whatever queued up behind it
            # into the same batch.
            batch = [await self._events.get()]
            while not self._events.empty():
                batch.append(self._events.get_nowait())
            try:
                # return_exceptions so one failed replica cannot mask delivery
                # to the rest; failures are logged and dropped below.
                results = await self.handle.broadcast(
                    "on_lifecycle_events", batch
                ).results_async(
                    timeout_s=LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(
                        "KV lifecycle events dropped on %d/%d ingress replicas: %s",
                        len(errors),
                        len(results),
                        errors[0],
                  
... [2417 chars trimmed] ...
  self._forwarder.report("on_request_completed", self._request_id)


def enable_token_tracking(
    engine_cls: Type[AsyncLLM], report_decode_progress: bool = False
) -> Type[AsyncLLM]:
    """Decorator adding KV-router request lifecycle tracking."""

    class TokenTrackingEngine(engine_cls):
        _lifecycle_forwarder: Optional[LifecycleEventForwarder] = None
        _resolve_warned: bool = False
        _report_decode_progress: bool = report_decode_progress

        def _resolve_lifecycle_forwarder(self) -> Optional[LifecycleEventForwarder]:
            if self._lifecycle_forwarder is None:
                try:
                    # The KVTokenTracker lives inside the LLMRouter, so
                    # forward lifecycle events to its deployment handle method.
                    handle = get_llm_router_handle()
                    worker_id = get_worker_id(
                        serve.get_replica_context().replica_id.unique_id
                    )
                    self._lifecycle_forwarder = LifecycleEventForwarder(
                        handle, worker_id
                    )
                except Exception as e:
                    # Warn once: resolution is retried per request until it succeeds.
                    if not self._resolve_warned:
                        self._resolve_warned = True
                        logger.warning("KV token tracking disabled: %s", e)
            return self._lifecycle_forwarder

        def shutdown(self, *args, **kwargs):
            if self._lifecycle_forwarder is not None:
                self._lifecycle_forwarder.close()
            return super().shutdown(*args, **kwargs)

        async def generate(self, prompt, sampling_params, request_id, *args, **kwargs):
            stream = super().generate(
                prompt, sampling_params, request_id, *args, **kwargs
            )
            forwarder = self._resolve_lifecycle_forwarder()
            # CUMULATIVE repeats output-so-far per chunk; our accounting sums
            # deltas, so skip it rather than over-count. vLLM's OpenAI layer only
            # uses DELTA/FINAL_ONLY (*Request.to_sampling_params):
            # https://github.com/vllm-project/vllm/tree/main/vllm/entrypoints/openai
            if forwarder is None or (
                sampling_params.output_kind == RequestOutputKind.CUMULATIVE
            ):
                async for output in stream:
                    yield output
                return

            lifecycle_request_id = get_serve_request_id() or request_id
            tracker = RequestTokenTracker(
                forwarder,
                lifecycle_request_id,
                report_decode_progress=self._report_decode_progress,
            )
            try:
                async for output in stream:
                    tracker.on_output(output)
                    yield output
            finally:
                tracker.finish()

    return TokenTrackingEngine
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
class PeerUpdateForwarder:
    """Broadcast this ingress's routing state to its peer ingress replicas.

    Each update is one of:
      1. ``load_reservation``: the booking for a request this ingress routed,
         so every peer counts the same token load on the selected worker.
      2. ``session_affinity``: a session binding this ingress committed or
         refreshed, so every peer pins the session's later turns to that worker.

    ``report`` only enqueues. A background task batches pending updates into
    one broadcast, off the request's selection and dispatch path.
    """

    def __init__(self, handle: Any, source_writer_id: int):
        self._handle = handle
        # Breaks ties when ingresses bind one session concurrently (newest
        # (sequence, writer_id) wins); receivers also skip their own broadcasts.
        self._source_writer_id = source_writer_id
        if getattr(handle, "is_initialized", True) is False:
            # Keep broadcast routing off the ingress replica's request loop.
            handle._init(_run_router_in_separate_loop=True)
        self._updates: asyncio.Queue = asyncio.Queue()
        self._delivery_task: Optional[asyncio.Task] = None

    def report(self, kind: str, event: Dict[str, Any]) -> None:
        if self._delivery_task is None or self._delivery_task.done():
            self._delivery_task = asyncio.get_running_loop().create_task(
                self._deliver()
            )
        self._updates.put_nowait({"kind": kind, "event": event})

    async def _deliver(self) -> None:
        while True:
            batch = [await self._updates.get()]
            while not self._updates.empty():
                batch.append(self._updates.get_nowait())
            try:
                results = await self._handle.broadcast(
                    "on_replica_updates", self._source_writer_id, batch
                ).results_async(
                    timeout_s=LIFECYCLE_EVENT_BROADCAST_TIMEOUT_S,
                    return_exceptions=True,
                )
                errors = [r for r in results if isinstance(r, Exception)]
                if errors:
                    logger.warning(
                        "KV peer broadcasts dropped on %d/%d ingress replicas: %s",
                        len(errors),
                        len(results),
                        errors[0],
                    )
            except Exception as e:
                logger.warning("Dropping selection service peer broadcast: %s", e)
            finally:
                for _ in batch:
                    self._updates.task_done()

    async def flush(self) -> None:
        """Wait until every reported replica broadcast has been attempted."""
        await self._updates.join()

    def close(self) -> None:
        if self._delivery_task is not None:
            self._delivery_task.cancel()
            self._delivery_task = None


#####
    async def _forward_session_bindings(self) -> None:
        # One consumer per s
... [3988 chars trimmed] ...
attr(self, "_autoscaling_publish_defer_depth", 0) > 0:
            self._autoscaling_publish_dirty = True
            return
        sink = getattr(self, "_autoscaling_metrics_sink", None)
        if sink is not None:
            sink(self._get_autoscaling_snapshot())

    def _defer_autoscaling_snapshot(self) -> None:
        self._autoscaling_publish_defer_depth = (
            getattr(self, "_autoscaling_publish_defer_depth", 0) + 1
        )

    def _flush_autoscaling_snapshot(self) -> None:
        self._autoscaling_publish_defer_depth -= 1
        if self._autoscaling_publish_defer_depth == 0 and getattr(
            self, "_autoscaling_publish_dirty", False
        ):
            self._autoscaling_publish_dirty = False
            self._publish_autoscaling_snapshot()

    def _create_selection_service(self) -> None:
        """Create the router-local Dynamo selection service for this deployment."""
        # Imported here, not at module scope, to keep Dynamo's pyo3 extension off
        # the import path of every process that imports this module; only the
        # ingress replica that builds the tracker needs it.
#####
                del self._request_ids_by_worker[worker_id]

    async def _evict_stale_requests(self) -> None:
        """Backstop for a lost completion on a live replica: evict requests tracked
        past ``REQUEST_TRACKING_TTL_S``, freeing their reservations.
        """
        cutoff = time.monotonic() - REQUEST_TRACKING_TTL_S
        while self._completed_request_ids:
            request_id, completed_at = next(iter(self._completed_request_ids.items()))
            if completed_at > cutoff:
                break
            self._completed_request_ids.pop(request_id, None)
        autoscaling_load_changed = False
        while self._requests:
            request_id, state = next(iter(self._requests.items()))
            if state.created_at > cutoff:
                break
            self._requests.popitem(last=False)
            self._remove_autoscaling_load(state)
            autoscaling_load_changed = True
            self._untrack_worker_request(request_id, state.worker_id)
            logger.warning(
                "Evicting stale KV request %s (tracked > %ds without completion); "
                "freeing its reservation.",
                request_id,
                REQUEST_TRACKING_TTL_S,
            )
            await self._svc.free_reservation(request_id)
        if autoscaling_load_changed:
            self._publish_autoscaling_snapshot()

    def _get_decay_fraction(self, state: RequestLifecycle) -> Optional[float]:
        """Fraction of output still expected, or ``None`` without an estimate;
        weights each decode block by how much generation remains."""
        if not state.expected_output_tokens:
            return None
        return max(0.0, 1.0 - state.output_tokens / state.expected_output_tokens)


_KV_TOKEN_TRACKER: Optional["KVTokenTracker"] = None
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/deployment:
autoscaling.json
config.json
events.jsonl
initial_status.json
ready_status.json
routing.json

/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/telemetry:
serve_metrics.jsonl
serve_status.jsonl
14:    request_router_kwargs: {}
19:    max_num_seqs: 1024
20:    max_num_batched_tokens: 16384
25:  autoscaling:
29:        target_prefill_tokens_per_replica: 20000
30:        upscale_lookback_s: 60
31:        upscale_aggregation: mean
35:    target_ongoing_requests: 8.0
36:    upscale_delay_s: 40.0
37:    downscale_delay_s: 1800.0
38:    metrics_interval_s: 2
59:    target_concurrency: 32
90:  metrics_interval_s: 2.0
#####
{"timestamp_ns": 1793652004832052738, "application": "ttft-benchmark", "application_status": "RUNNING", "deployments": {"LLMServer:Qwen--Qwen3_6-27B-FP8": {"status": "HEALTHY", "status_trigger": "CONFIG_UPDATE_COMPLETED", "replica_states": {"RUNNING": 1}}, "LLMRouter": {"status": "HEALTHY", "status_trigger": "UPSCALE_COMPLETED", "replica_states": {"RUNNING": 1}}}, "available": true, "error": null, "tick": 0}
lines 2172
```

</details>

<details><summary>Bash result</summary>

```text
Counter({'warm': 1436, 'cold': 148}) Counter({'k-5be017850000_pid-151035': 702, 'k-5be017850000_pid-180495': 457, 'k-5be017850000_pid-173074': 425})
cold 148 own_uncached sum 4640272 mean 31353 median 27016 | isl sum 4847248 mean 32752 median 28230
   queue_ms med 13 mean 146 | prefill_ms med 1405 mean 1832 | pre_ms med 77 mean 129 | post_ms med 15 mean 20 | steps mean 3.5
warm 1436 own_uncached sum 6313313 mean 4396 median 2419 | isl sum 121653825 mean 84717 median 78864
   queue_ms med 15 mean 88 | prefill_ms med 198 mean 336 | pre_ms med 162 mean 201 | post_ms med 31 mean 34 | steps mean 1.8
total own_uncached 10953585.0 total isl 126501073.0
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
```

</details>

<details><summary>Bash result</summary>

```text
{"async_scheduling":true,"event":"trace_started","max_num_batched_tokens":16384,"max_num_seqs":1024,"monotonic_s":1543886.288601501,"scheduler_class":"TracingAsyncScheduler","schema_version":1,"wall_time_s":1793652000.8454103}
{"arrival_time_s":1793652009.2320743,"event":"request_queued","external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","max_tokens":20,"monotonic_s":1543894.68289622,"priority":0,"prompt_tokens":665,"queued_after_batch":0,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","running_requests":0,"schema_version":1,"waiting_requests":1,"wall_time_s":1793652009.239705}
{"batch_index":1,"decode_context_tokens":0,"decode_requests":0,"decode_tokens":0,"event":"batch_scheduled","first_scheduled_request_ids":["chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef"],"kv_cache_usage":0.0014561339643247573,"max_num_scheduled_tokens":16384,"monotonic_s":1543894.683436043,"preempted_request_ids":[],"prefill_requests":1,"prefill_tokens":665,"remaining_token_budget":15719,"requests":[{"cached_prefix_tokens":0,"completed_or_cached_before":0,"computed_before":0,"context_tokens_after":665,"external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","in_flight_before":0,"locally_computed_before":0,"num_preemptions":0,"phase":"prefill","prompt_tokens":665,"remaining_prefill_after":0,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","scheduled_tokens":665}],"running_requests_after":1,"running_requests_before":0,"scheduler_step":1,"scheduler_time_ms":0.47454098239541054,"schema_version":1,"total_scheduled_tokens":665,"waiting_constraint":null,"waiting_requests_after":0,"waiting_requests_before":1,"wall_time_s":1793652009.2402446}
{"batch_index":1,"batches_waited":0,"cached_prefix_tokens":0,"event":"request_admitted","external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","monotonic_s":1543894.683447231,"prompt_tokens":665,"queue_wait_ms":0.5590228829532862,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","scheduled_tokens":665,"schema_version":1,"wall_time_s":1793652009.2402558}
{"batch_index":2,"decode_context_tokens":666,"decode_requests":1,"decode_tokens":1,"event":"batch_scheduled","first_scheduled_request_ids":[],"kv_cache_usage":0.0014561339643247573,"max_num_scheduled_tokens":16384,"monotonic_s":1543894.683944918,"preempted_request_ids":[],"prefill_requests":0,"prefill_tokens":0,"remaining_token_budget":16383,"requests":[{"cached_prefix_tokens":0,"completed_or_cached_before":0,"computed_before":665,"context_tokens_after":666,"external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","in_flight_before":665,"locally_computed_before":0,"num_preemptions":0,"phase":"decode","prompt_tokens":665,"remaining_prefill_after":0,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","scheduled_tokens":1}],"running_requests_after":1,"running_requests_before":1,"scheduler_step":2,"scheduler_time_ms"
... [588 chars trimmed] ...
ning_requests":0,"schema_version":1,"waiting_requests":1,"wall_time_s":1793652009.239705}

== batch_scheduled
{"batch_index":1,"decode_context_tokens":0,"decode_requests":0,"decode_tokens":0,"event":"batch_scheduled","first_scheduled_request_ids":["chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef"],"kv_cache_usage":0.0014561339643247573,"max_num_scheduled_tokens":16384,"monotonic_s":1543894.683436043,"preempted_request_ids":[],"prefill_requests":1,"prefill_tokens":665,"remaining_token_budget":15719,"requests":[{"cached_prefix_tokens":0,"completed_or_cached_before":0,"computed_before":0,"context_tokens_after":665,"external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","in_flight_before":0,"locally_computed_before":0,"num_preemptions":0,"phase":"prefill","prompt_tokens":665,"remaining_prefill_after":0,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","scheduled_tokens":665}],"running_requests_after":1,"running_requests_before":0,"scheduler_step":1,"scheduler_time_ms":0.47454098239541054,"schema_version":1,"total_scheduled_tokens":665,"waiting_constraint":null,"waiting_requests_after":0,"waiting_requests_before":1,"wall_time_s":1793652009.2402446}

== request_admitted
{"batch_index":1,"batches_waited":0,"cached_prefix_tokens":0,"event":"request_admitted","external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","monotonic_s":1543894.683447231,"prompt_tokens":665,"queue_wait_ms":0.5590228829532862,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","scheduled_tokens":665,"schema_version":1,"wall_time_s":1793652009.2402558}

== request_first_token
{"batch_index":1,"batches_waited":0,"engine_ttft_ms":42.930897092446685,"event":"request_first_token","external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","monotonic_s":1543894.725918673,"prefill_span_ms":42.3718742094934,"queue_wait_ms":0.5590228829532862,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","schema_version":1,"wall_time_s":1793652009.2827275}

== batch_completed
{"batch_index":1,"dispatch_to_output_ms":42.3532749991864,"event":"batch_completed","finished_request_ids":[],"first_token_request_ids":["chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef"],"monotonic_s":1543894.725931917,"request_ids":["chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef"],"schema_version":1,"wall_time_s":1793652009.2827406}

== scheduler_idle_step
{"event":"scheduler_idle_step","monotonic_s":1543894.889551577,"running_requests":1,"scheduler_step":21,"scheduler_time_ms":0.03033597022294998,"schema_version":1,"waiting_requests":0,"wall_time_s":1793652009.44636}

== request_finished
{"batch_index":20,"event":"request_finished","external_request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","finish_reason":"length","monotonic_s":1543894.898641082,"request_id":"chatcmpl-f7fcefe8-2956-4075-8982-8768ce95888a-aeca6eef","schema_version":1,"wall_time_s":1793652009.4554496}
```

</details>

<details><summary>Bash result</summary>

```text
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/llm
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/experimental
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/__pycache__
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/squeeze_evolve
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/py.typed
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.pyi
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/health_check.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/kv_state_agent
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/select_service
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/nixl_connect
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/router
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/mocker
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/slot_tracker
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/common
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/vllm
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_internal
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/profiler
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/kv_dc_relay
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/replay
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/indexer
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/triton
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/logits_processing
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/global_planner
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/frontend
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.abi3.so
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/prometheus_metrics.pyi
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/planner
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/global_router
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/thunderagent_router
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/runtime
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/tokenspeed
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/prometheus_names.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/trtllm
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/sglang
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/llm/__pycache__
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/llm/_unary.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/llm/__init__.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/llm/exceptions.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/experimental/__pycache__
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/planner/tests/unit/test_planner_config.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/planner/core/base.py
/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/planner/core/perf_model/engine_query.py
/home/ray/an
... [17944 chars trimmed] ...
endpointskv_event_source_modetrack_prefill_tokensoverlapeffective_prefill_tokenspotential_decode_blocksdecode_busyworker_loadkv_hintincompleteschedulableunschedulableloadspending_countpending_isl_tokensactive_prefill_tokensprefill_token_capacityprefill_busylifecyclenot_schedulable_reasonsuser_messagetool_resultotherSelectionSessionContextparent_session_idsession_finalinput_triggeragent_headersdynamo/residency-owner/cache-owner/v1string salt derivation is infallibleBlockExtraInfomm_objectsPrefillLoadHintexpected_prefill_durationstoredremovedclearedparent_hashstart_positionblocksBlockMmObjectInfomm_hashoffsetsKvCacheRemoveDataworkerrouter_idWorkerAffinityTargetpreferredKvCacheStoredBlockDatablock_hashtokens_hashmm_extra_infoActiveSequenceEventDataAddRequesttoken_sequenceprefill_load_hintMarkPrefillCompletedRouterEventstorage_tiereventstate_sourceStorageTierdevicehost_pinnedKvCacheEventpotential_prefill_tokensactive_requestsa tuple of size 2a sequencestruct KvHinta tuple of size 3struct KvHintActionstruct ActiveLoadenum ResetScopestruct RouterEventstruct KvCacheEventstruct SchedulerLoadstruct BlockExtraInfostruct variant SchedulerGroup::ReplicaGroupstruct variant SchedulerGroup::Standalonestruct PrefillLoadHintvariant identifierenum KvCacheEventDatastruct KvCacheStoreDatastruct WorkerWithDpRankstruct BlockMmObjectInfostruct KvCacheRemoveDatastruct RoutingConstraintsstruct ActiveSequenceEventstruct WorkerAffinityTargetstruct KvCacheStoredBlockDataenum ActiveSequenceEventDatastruct ActiveSequenceEventBatchstruct RouterHintSourceMetadatavariant index 0 <= i < 3a mapstruct WorkerRequestdefault_policy_familypolicy_classesuncached_isl_bucketsstruct SelectRequestWirepolicy_familycache_bucketqueue_policyquantumprefill_busy_thresholdprefill_busy_threshold_fracrequest_queue_limit_per_workerraw_isl_token_queue_limit_per_workercached_token_queue_limit_per_workermin_tokensbucketstruct OutputBlockRequeststruct WorkerPatchRequestroutermodelsrequest_classifierstruct ReservationRequestWirestruct SelectionSessionContextstruct OverlapScoresRequestWirestruct SelectAndReserveRequestWireaggregatedprefillencodeinstancesparametersstruct variant ActiveSequenceEventData::AddRequestdescription() is deprecated; use Displaycalled `Result::unwrap()` on an `Err` valueErrorinnerIoErrorWireErrorReflectMessageNotInitializedBufferHasNotEnoughCapacityIncompatibleProtobufTypeAndRuntimeTypeGroupIsNotImplementedMmInfoTokenBlockSaltSerializationMissingFieldDeviceHostPinnedLocalBlockHashWorkerCacheOwnerParentBlockNotFoundBlockNotFoundInvalidBlockSequenceCapacityExhaustedOwnershipDegreeOverflowIndexerInvariantViolationUnsupportedResidencyDomainFailed to find parent blockFailed to find blockInvalid block sequenceIndexer capacity exhaustedIndexer allocation failedIndexer ownership degree overflowIndexer invariant violatedUnsupported residency domainoptionExternalSequenceBlockHashcache_ownerunexpected end of sequenceunexpected end of mappinga YAML tag starting with '!'a residency-domain string
```

</details>

**Bash** — Read selection service construction, constants, and the router metrics sink

```bash
cd /mnt/local_storage/ray && sed -n 376,460p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo '#####'; cat python/ray/llm/_internal/serve/routing_policies/kv_aware/constants.py; echo '#####'; sed -n 120,175p python/ray/llm/_internal/serve/core/ingress/router.py; sed -n 285,310p python/ray/llm/_internal/serve/core/ingress/router.py
```

**Bash** — Find how router metrics are timestamped and the selection service API

```bash
cd /mnt/local_storage/ray && grep -rn "raw_router_metrics\|record_router_metric\|router_metrics" python/ray/serve/_private/replica.py python/ray/serve/_private/autoscaling_state.py python/ray/serve/_private/metrics_utils.py python/ray/serve/_private/router.py 2>/dev/null | head -30; grep -n "select_and_reserve\|effective_prefill\|class SelectionService\|def select" -A3 /home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.pyi | head -80
```

**Bash** — Check Serve logs for stale evictions and dropped lifecycle events

```bash
ls -d /tmp/ray/session_*/ 2>/dev/null; ls /tmp/ray/session_latest/logs/serve/ 2>/dev/null | head -40; cd /tmp/ray/session_latest/logs/serve/ 2>/dev/null && ls -la --time-style=full-iso | grep -i "router\|LLMServer" | head; grep -l "Evicting stale\|Dropping KV lifecycle\|lifecycle events dropped\|peer broadcasts dropped" * 2>/dev/null | head; grep -c "Evicting stale" replica_LLMRouter* 2>/dev/null
```

**Bash** — Check router replica count over time and the gauge floor and tail

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import json,sys
R=sys.argv[1]
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
# router replica counts over time
prev=None
for line in open(f'{R}/telemetry/serve_status.jsonl'):
    d=json.loads(line); t=d['timestamp_ns']/1e9-t0
    st={k:v['replica_states'] for k,v in d['deployments'].items()}
    if st!=prev: print('%.0f'%t, st); prev=st
# gauge over time: first/last nonzero, negative values, floor after load
raw=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='outstanding_effective_prefill_tokens' and 'LLMServer' in s['labels'].get('deployment',''):
            raw.append((t,s['value']))
print('scrapes',len(raw),'t range %.0f..%.0f'%(raw[0][0],raw[-1][0]))
print('negative samples',sum(v<0 for _,v in raw),'min',min(v for _,v in raw))
nz=[t for t,v in raw if v>0]; print('first nonzero %.0f last nonzero %.0f'%(nz[0],nz[-1]))
ev=json.load(open(f'{R}/analysis/autoscaling_evaluation.json'))
print('periods',ev['periods']); print('scale events',[(e['time_s'],e['from_replicas'],e['to_replicas']) for e in ev['scale_events']])
# gauge in tail after steady end
end=ev['periods']['steady']['end_s']
tail=[(t,v) for t,v in raw if t>end]
print('tail samples',len(tail),'values after end:', [ (round(t),v) for t,v in tail[::max(1,len(tail)//25)]])
import collections
print('zero-fraction during load', sum(v==0 for t,v in raw if 0<=t<=end)/max(1,sum(1 for t,v in raw if 0<=t<=end)))
EOF
```

**Bash** — Show routing and autoscaling deployment settings for the run

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; grep -n "report_decode_progress\|enable_token_tracking" -r /mnt/local_storage/ray/python/ray/llm/_internal/serve --include='*.py' | grep -v tests | head; echo '#####'; cat $R/deployment/routing.json | head -50; echo; cat $R/deployment/autoscaling.json | head -40; echo '#####'; sed -n 1,60p $R/resolved.yaml
```

<details><summary>Bash result</summary>

```text
def _create_selection_service(self) -> None:
        """Create the router-local Dynamo selection service for this deployment."""
        # Imported here, not at module scope, to keep Dynamo's pyo3 extension off
        # the import path of every process that imports this module; only the
        # ingress replica that builds the tracker needs it.
        try:
            from dynamo.llm import SelectionService
        except ImportError:
            self._svc = None
            logger.warning(
                "ai-dynamo is not installed; KV-aware routing requires ai-dynamo."
            )
            return

        self._svc = SelectionService(
            indexer_threads=self._indexer_threads,
            session_affinity_ttl_secs=self._session_affinity_ttl_s or None,
            session_affinity_mode=SESSION_AFFINITY_MODE,
            session_replication_writer_id=(
                self._replication_writer_id if self._session_affinity_ttl_s else None
            ),
        )
        logger.info(
            "Dynamo SelectionService created (indexer threads %d).",
            self._indexer_threads,
        )

    def _start_replica_tracking(self) -> None:
        """Subscribe to this deployment's running replicas via LongPollClient."""
        deployment_id = self._serve_deployment_id
        controller = ray.get_actor(SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
        self._long_poll_client = LongPollClient(
            controller,
            {
                (
                    LongPollNamespace.DEPLOYMENT_TARGETS,
                    deployment_id,
                ): self._on_deployment_targets,
            },
            # Built inside the LLMRouter ingress replica's async __init__, so this
            # binds LongPoll callbacks to that event loop.
            call_in_event_loop=asyncio.get_running_loop(),
            client_id=f"{type(self).__name__}:{deployment_id}",
        )

    def _schedule(self, awaitable) -> None:
        """Schedule an awaitable (coroutine or future) on the ingress replica's
        event loop, holding a reference until it completes.
        """

        async def _run():
            await awaitable

        task = asyncio.create_task(_run())
        self._pending_tasks.add(task)
        task.add_done_callback(self._pending_tasks.discard)

    def _register_block_size(self, block_size: int, replica_id: str) -> None:
        """Pin the deployment's KV-cache block size from the first replica's
        reported engine config.
        """
        if self._block_size is None:
            self._block_size = block_size
            logger.info("KV router block size set to %d.", block_size)
        elif block_size != self._block_size:
            # Replicas of a deployment are expected to resolve the same block
            # size, so a mismatch is unexpected. We still register the worker so
            # the selection service spawns its KV-event listener, but the indexer
            # only ingests blocks whose s
... [4482 chars trimmed] ...
efault
    # keeps the guard safe before __init__ runs.
    _warned_no_routing_key: bool = False
    _warned_no_token_endpoint: bool = False
    _base_model_id: Optional[str] = None

    async def __init__(
        self,
        server: DeploymentHandle,
        llm_config: Optional["LLMConfig"] = None,
        base_model_id: Optional[str] = None,
    ):
        self._handle: DeploymentHandle = server
        self._base_model_id = base_model_id
        self._tokenizer = None
        self._token_sender = None
        # Holds the KVTokenTracker (KV-aware deployments only) so the
        # engine-facing on_lifecycle_events method can book load into it.
        self._kv_token_tracker = None
        # A non-None llm_config signals pre-routing tokenization, which the
        # builder binds only for a KV-aware request router.
        if llm_config is not None:
            # Build the tracker before _handle._init() below, which initializes
            # the KVAwareRouter that looks it up. server.deployment_id is the
            # tracked LLMServer deployment.
            from ray.llm._internal.serve.routing_policies.kv_aware.kv_token_tracker import (  # noqa: E501
                build_kv_token_tracker,
                get_llm_router_handle,
            )

            self._kv_token_tracker = build_kv_token_tracker(
                llm_config, server.deployment_id
            )
            self._kv_token_tracker.start_replica_broadcast(get_llm_router_handle())
            # Lazy import: this module pulls in vLLM's renderer;
            # keep it off the non-KV ingress import path.
            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (  # noqa: E501
                Tokenizer,
            )

            self._tokenizer = await asyncio.to_thread(Tokenizer, llm_config)
            # Lazy import: pyzmq is not a Ray runtime dependency, so keep it off
            # the non-KV ingress import path.

        return self._handle.options(multiplexed_model_id=multiplexed_model_id)

    @router_app.get("/health")
    async def health(self):
        return {"status": "ok"}

    def __del__(self) -> None:
        """Close the token channel ZMQ sockets upon cleanup."""
        token_sender = getattr(self, "_token_sender", None)
        if token_sender is not None:
            token_sender.close()

    async def on_lifecycle_events(self, batch):
        """Engine-facing intake for request lifecycle events.

        Engine replicas broadcast each batch to every LLMRouter replica to
        book request load; this applies it to the KVTokenTracker on this
        ingress replica's event loop.
        """
        return await self._kv_token_tracker.on_lifecycle_events(batch)

    async def on_replica_updates(self, source_writer_id, batch):
        """Accept load and affinity updates from peer ingress replicas."""
        return await self._kv_token_tracker.on_replica_updates(source_writer_id, batch)
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
python/ray/serve/_private/autoscaling_state.py:157:    router_metrics: Dict[str, MetricSamples]
python/ray/serve/_private/autoscaling_state.py:636:                router_metrics=_router_metric_samples(payload),
python/ray/serve/_private/autoscaling_state.py:768:            raw_router_metrics=self._get_raw_router_metrics,
python/ray/serve/_private/autoscaling_state.py:1080:    def _get_raw_router_metrics(self) -> Dict[str, Dict[str, TimeSeries]]:
python/ray/serve/_private/autoscaling_state.py:1085:            router_metrics = getattr(
python/ray/serve/_private/autoscaling_state.py:1087:                "router_metrics",
python/ray/serve/_private/autoscaling_state.py:1090:            for metric_name, timeseries in router_metrics.items():
python/ray/serve/_private/autoscaling_state.py:1101:            for metric_name, samples in columnar_report.router_metrics.items():
python/ray/serve/_private/router.py:197:        self._router_metrics_store = InMemoryMetricsStore()
python/ray/serve/_private/router.py:198:        self._latest_custom_router_metrics: Dict[str, float] = {}
python/ray/serve/_private/router.py:393:    def set_custom_router_metrics(
python/ray/serve/_private/router.py:411:        self._latest_custom_router_metrics = validated
python/ray/serve/_private/router.py:516:        if self._latest_custom_router_metrics:
python/ray/serve/_private/router.py:517:            self._router_metrics_store.add_metrics_point(
python/ray/serve/_private/router.py:518:                cast(Dict[Any, float], self._latest_custom_router_metrics), timestamp
python/ray/serve/_private/router.py:530:        self._router_metrics_store.prune_keys_and_compact_data(start_timestamp)
python/ray/serve/_private/router.py:539:        self._router_metrics_store.prune_keys_and_compact_data(
python/ray/serve/_private/router.py:562:            router_metrics={
python/ray/serve/_private/router.py:565:                    metric_name: self._router_metrics_store.data.get(
python/ray/serve/_private/router.py:568:                    for metric_name, value in self._latest_custom_router_metrics.items()
python/ray/serve/_private/router.py:873:        self._metrics_manager.set_custom_router_metrics(metrics_dict)
772:class SelectionService:
773-    """
774-    In-process handle to a runtime-free Dynamo selection core.
775-    """
--
853:    async def select(self, request: JsonLike) -> JsonLike:
854-        """Select the best worker by KV-overlap + load, without booking."""
855-        ...
856-
857:    async def select_and_reserve(self, request: JsonLike) -> JsonLike:
858-        """Select the best worker and book its load."""
859-        ...
860-
--
3298:class SelectionServiceError(DynamoException):
3299-    """
3300-    Raised by `SelectionService` for selector failures that are not malformed
3301-    input.
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
/tmp/ray/session_2026-10-29_21-07-14_629089_584/
/tmp/ray/session_latest/
controller_106885.log
controller_149281.log
controller_154006.log
controller_155748.log
controller_160064.log
controller_1927568.log
controller_1963444.log
controller_202368.log
controller_2112693.log
controller_2138454.log
controller_2159667.log
controller_3246542.log
controller_360226.log
controller_3889931.log
controller_399098.log
controller_545986.log
controller_625885.log
controller_703738.log
proxy_10.240.13.26.log
replica_ttft-benchmark_LLMRouter_18y80odu.log
replica_ttft-benchmark_LLMRouter_1x5xlckn.log
replica_ttft-benchmark_LLMRouter_3jybiw18.log
replica_ttft-benchmark_LLMRouter_4eibuykc.log
replica_ttft-benchmark_LLMRouter_4hklg2ky.log
replica_ttft-benchmark_LLMRouter_5d4pstsm.log
replica_ttft-benchmark_LLMRouter_94rs2hzp.log
replica_ttft-benchmark_LLMRouter_9y9ieyng.log
replica_ttft-benchmark_LLMRouter_cmqnji7s.log
replica_ttft-benchmark_LLMRouter_dgz35pge.log
replica_ttft-benchmark_LLMRouter_hzmthft2.log
replica_ttft-benchmark_LLMRouter_jxhnt8e3.log
replica_ttft-benchmark_LLMRouter_m2fxg0pg.log
replica_ttft-benchmark_LLMRouter_ox2gp8j3.log
replica_ttft-benchmark_LLMRouter_req26gft.log
replica_ttft-benchmark_LLMRouter_rizdk6y1.log
replica_ttft-benchmark_LLMRouter_rk7bdfgw.log
replica_ttft-benchmark_LLMRouter_t993temu.log
replica_ttft-benchmark_LLMRouter_uj68h71a.log
replica_ttft-benchmark_LLMRouter_xhfh6064.log
replica_ttft-benchmark_LLMRouter_y21fdlg1.log
-rw-r--r-- 1 ray users      530 2026-10-30 01:52:17.540746537 -0700 replica_ttft-benchmark_LLMRouter_18y80odu.log
-rw-r--r-- 1 ray users      530 2026-10-30 02:33:43.565631295 -0700 replica_ttft-benchmark_LLMRouter_1x5xlckn.log
-rw-r--r-- 1 ray users      530 2026-11-02 12:38:29.711991188 -0800 replica_ttft-benchmark_LLMRouter_3jybiw18.log
-rw-r--r-- 1 ray users      530 2026-10-30 17:07:32.371175305 -0700 replica_ttft-benchmark_LLMRouter_4eibuykc.log
-rw-r--r-- 1 ray users      530 2026-10-31 15:45:23.929070229 -0700 replica_ttft-benchmark_LLMRouter_4hklg2ky.log
-rw-r--r-- 1 ray users      530 2026-10-31 16:01:42.962849136 -0700 replica_ttft-benchmark_LLMRouter_5d4pstsm.log
-rw-r--r-- 1 ray users      530 2026-10-31 15:21:52.118830814 -0700 replica_ttft-benchmark_LLMRouter_94rs2hzp.log
-rw-r--r-- 1 ray users      530 2026-10-31 15:43:09.993237290 -0700 replica_ttft-benchmark_LLMRouter_9y9ieyng.log
-rw-r--r-- 1 ray users      530 2026-10-30 03:15:17.480670386 -0700 replica_ttft-benchmark_LLMRouter_cmqnji7s.log
-rw-r--r-- 1 ray users      530 2026-10-30 15:31:52.050842994 -0700 replica_ttft-benchmark_LLMRouter_dgz35pge.log
(eval):1: no matches found: replica_LLMRouter*
```

</details>

<details><summary>Bash result</summary>

```text
-4 {'LLMServer:Qwen--Qwen3_6-27B-FP8': {'RUNNING': 1}, 'LLMRouter': {'RUNNING': 1}}
599 {'LLMServer:Qwen--Qwen3_6-27B-FP8': {'RUNNING': 1, 'STARTING': 1}, 'LLMRouter': {'RUNNING': 1}}
698 {'LLMServer:Qwen--Qwen3_6-27B-FP8': {'RUNNING': 2}, 'LLMRouter': {'RUNNING': 1}}
835 {'LLMServer:Qwen--Qwen3_6-27B-FP8': {'RUNNING': 2, 'STARTING': 1}, 'LLMRouter': {'RUNNING': 1}}
935 {'LLMServer:Qwen--Qwen3_6-27B-FP8': {'RUNNING': 3}, 'LLMRouter': {'RUNNING': 1}}
scrapes 543 t range -4..2164
negative samples 0 min 0.0
first nonzero 20 last nonzero 1796
periods {'steady': {'end_s': 1860.0, 'start_s': 0.0}, 'whole_run': {'end_s': 2166.738573837, 'start_s': 0.0}}
scale events [(602.32503288, 1.0, 2.0), (842.215691455, 2.0, 3.0)]
tail samples 76 values after end: [(1864, 0.0), (1876, 0.0), (1888, 0.0), (1900, 0.0), (1912, 0.0), (1924, 0.0), (1936, 0.0), (1948, 0.0), (1960, 0.0), (1972, 0.0), (1984, 0.0), (1996, 0.0), (2008, 0.0), (2020, 0.0), (2032, 0.0), (2044, 0.0), (2056, 0.0), (2068, 0.0), (2080, 0.0), (2092, 0.0), (2104, 0.0), (2116, 0.0), (2128, 0.0), (2140, 0.0), (2152, 0.0), (2164, 0.0)]
zero-fraction during load 0.5849462365591398
```

</details>

<details><summary>Bash result</summary>

```text
/mnt/local_storage/ray/python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:77:    enable_token_tracking,
/mnt/local_storage/ray/python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:649:            engine_cls = enable_token_tracking(
/mnt/local_storage/ray/python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:651:                report_decode_progress=RAY_SERVE_LLM_ENABLE_DECODE_BLOCK_PROGRESS,
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:102:        report_decode_progress: bool = False,
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:109:        self._report_decode_progress = report_decode_progress
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:126:        if self._report_decode_progress:
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:138:def enable_token_tracking(
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:139:    engine_cls: Type[AsyncLLM], report_decode_progress: bool = False
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:146:        _report_decode_progress: bool = report_decode_progress
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/token_tracking.py:192:                report_decode_progress=self._report_decode_progress,
#####
{
  "direct_streaming": true,
  "environment": {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE": "8388608",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1"
  },
  "forward_request_body": true,
  "haproxy_request_buffer_bytes": 8388608,
  "policy": "kv_aware",
  "request_router_class": "ray.serve.llm.request_router.KVAwareRouter",
  "request_router_kwargs": {},
  "resolved": {
    "ai_dynamo_version": "1.6.1",
    "policy": "kv_aware",
    "request_router_class": "ray.serve.llm.request_router.KVAwareRouter"
  },
  "verified": {
    "deployments": [
      "LLMRouter",
      "LLMServer:Qwen--Qwen3_6-27B-FP8"
    ],
    "topology": "direct_streaming"
  }
}

{
  "policy": "token",
  "policy_function": "ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy",
  "policy_kwargs": {
    "target_prefill_tokens_per_replica": 20000,
    "upscale_aggregation": "mean",
    "upscale_lookback_s": 60
  },
  "verified_deployments": [
    "LLMServer:Qwen--Qwen3_6-27B-FP8"
  ]
}
#####
name: b200-workday-curve-prefill-mean
deployment:
  application_name: ttft-benchmark
  model_id: Qwen/Qwen3.6-27B-FP8
  model_source: Qwen/Qwen3.6-27B-FP8
  tokenizer: Qwen/Qwen3.6-27B-FP8
  accelerator_type: B200
  route_prefix: /
  routing:
    policy: kv_aware
    direct_streaming: true
    forward_request_body: true
    haproxy_request_buffer_bytes: 8388608
    request_router_kwargs: {}
  engine:
    tensor_parallel_size: 1
    max_model_len: 262144
    gpu_memory_utilization: 0.92
    max_num_seqs: 1024
    max_num_batched_tokens: 16384
    enable_prefix_caching: true
    enable_prompt_tokens_details: true
    kv_cache_dtype: fp8
    language_model_only: true
  autoscaling:
    policy:
      name: token
      kwargs:
        target_prefill_tokens_per_replica: 20000
        upscale_lookback_s: 60
        upscale_aggregation: mean
    min_replicas: 1
    initial_replicas: 1
    max_replicas: 8
    target_ongoing_requests: 8.0
    upscale_delay_s: 40.0
    downscale_delay_s: 1800.0
    metrics_interval_s: 2
    look_back_period_s: 60
  max_ongoing_requests: null
  experimental_configs: {}
benchmark:
  generator: aiperf
  mode: agentx_concurrency_ramp
  levels:
  - 1.0
  - 2.0
  - 4.0
  - 8.0
  duration_s: 1860.0
  grace_period_s: 120.0
  post_load_observation_s: 180.0
  warmup:
    enabled: true
    duration_s: 30.0
  workload:
    type: agentx
    public_dataset: semianalysis_cc_traces_weka_062126_256k
    target_concurrency: 32
    concurrency_ramp_duration_s: null
```

</details>

**Bash** — Check how often the router records the gauge into its time series

```bash
cd /mnt/local_storage/ray && sed -n 393,420p python/ray/serve/_private/router.py; echo '#####'; sed -n 500,545p python/ray/serve/_private/router.py; echo '#####'; grep -rn "ROUTER_METRICS\|METRICS_RECORD\|record_interval\|push_interval\|RECORD_METRICS_INTERVAL" python/ray/serve/_private/router.py python/ray/serve/_private/constants.py | head
```

**Bash** — Check the Serve-side KV-aware router and the vLLM KV event publisher configuration

```bash
cd /mnt/local_storage/ray && ls python/ray/llm/_internal/serve/routing_policies/kv_aware/; grep -n "on_request_completed\|on_request_routed\|select_worker\|def \|class " python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_aware_router.py 2>/dev/null | head -40; grep -rn "kv_events\|KVEventsConfig\|publisher\|enable_kv_cache_events" python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py | head -20
```

**Bash** — Look for dropped lifecycle event or stale eviction warnings in this run's Serve logs

```bash
cd /tmp/ray/session_latest/logs/serve/ && ls -la --time-style=full-iso | awk '$6=="2026-11-02" && $7>"12:30" && $7<"13:20"' ; echo '#####'; cat replica_ttft-benchmark_LLMRouter_3jybiw18.log; echo '#####'; for f in $(ls -t replica_ttft-benchmark_LLMServer* 2>/dev/null | head -8); do echo "== $f $(stat -c %y $f)"; grep -c "Dropping KV lifecycle\|lifecycle events dropped\|Evicting stale\|peer broadcasts dropped" $f; done; grep -l "Evicting stale KV request" * 2>/dev/null | head
```

**Bash** — Look for session ids in the benchmark output

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; ls $R/benchmark | head; ls $R/benchmark/* | head -30; grep -rn "session\|conversation" $R/benchmark/*.json 2>/dev/null | head -5
```

**Bash** — Check trace coverage, max KV cache usage, and dropped trace events per engine

```bash
T=/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces; python3 -I - $T <<'EOF'
import json,sys,glob
T=sys.argv[1]
files=[f'{T}/k-5be017850000_pid-151035.0000.jsonl',f'{T}/k-5be017850000_pid-151035.0001.jsonl',f'{T}/k-5be017850000_pid-173074.0000.jsonl',f'{T}/k-5be017850000_pid-180495.0000.jsonl']
for f in files:
    mx=0; n=0; nq=0; dropped=0; first=None; last=None
    for line in open(f):
        try: e=json.loads(line)
        except: continue
        n+=1
        if first is None: first=e.get('wall_time_s')
        last=e.get('wall_time_s',last)
        if e['event']=='batch_scheduled': mx=max(mx,e['kv_cache_usage'])
        elif e['event']=='request_queued': nq+=1
        elif e['event']=='trace_events_dropped': dropped+=e['count']
    print(f.split('/')[-1], 'events',n,'queued',nq,'max kv usage %.3f'%mx,'dropped',dropped,'wall %.0f..%.0f'%(first or 0,last or 0))
EOF
```

<details><summary>Bash result</summary>

```text
def set_custom_router_metrics(
        self, metrics_dict: Dict[str, Union[int, float]]
    ) -> None:
        """Atomically replace the latest router-owned autoscaling snapshot."""
        validated: Dict[str, float] = {}
        for name, value in metrics_dict.items():
            if not isinstance(name, str):
                raise TypeError("Router autoscaling metric names must be strings.")
            if name in (ROUTER_QUEUED_REQUESTS_KEY, RUNNING_REQUESTS_KEY):
                raise ValueError(
                    f"Router autoscaling metric name '{name}' is reserved."
                )
            if not isinstance(value, (int, float)) or not math.isfinite(value):
                raise TypeError(
                    f"Router autoscaling metric '{name}' must be a finite number."
                )
            validated[name] = float(value)

        self._latest_custom_router_metrics = validated

    def inc_num_queued_requests(self):
        self.num_queued_requests += 1
        if not self._cached_metrics_enabled:
            self.num_queued_requests_gauge.set(self.num_queued_requests)

    def dec_num_queued_requests(self):
        self.num_queued_requests -= 1
        if not self._cached_metrics_enabled:
#####
        │                          │ Push metrics                         │
        │                          └─────────────────> Controller         │
        │                                                                 │
        └─────────────────────────────────────────────────────────────────┘

        :::{note}
        The long-term plan is to deprecate handle-based metrics collection in favor of
        replica-based collection. Replica-based collection will become the default in a
        future release. Queued requests will be continues to be tracked at the handle.
        :::
        """

        timestamp = time.time()
        self.metrics_store.add_metrics_point(
            {QUEUED_REQUESTS_KEY: self.num_queued_requests}, timestamp
        )
        if self._latest_custom_router_metrics:
            self._router_metrics_store.add_metrics_point(
                cast(Dict[Any, float], self._latest_custom_router_metrics), timestamp
            )
        if RAY_SERVE_COLLECT_AUTOSCALING_METRICS_ON_HANDLE:
            self.metrics_store.add_metrics_point(
                cast(Dict[Any, float], self.num_requests_sent_to_replicas), timestamp
            )

        # Prevent in memory metrics store memory from growing
        autoscaling_config = self.autoscaling_config
        assert autoscaling_config is not None
        start_timestamp = time.time() - autoscaling_config.look_back_period_s
        self.metrics_store.prune_keys_and_compact_data(start_timestamp)
        self._router_metrics_store.prune_keys_and_compact_data(start_timestamp)

    def _get_metrics_report(self) -> HandleMetricReport:
        timestamp = time.time()
        running_requests = dict()
        autoscaling_config = self.autoscaling_config
        assert autoscaling_config is not None
        look_back_period = autoscaling_config.look_back_period_s
        self.metrics_store.prune_keys_and_compact_data(time.time() - look_back_period)
        self._router_metrics_store.prune_keys_and_compact_data(
            time.time() - look_back_period
        )
        queued_requests = self.metrics_store.data.get(
            QUEUED_REQUESTS_KEY, [TimeStampedValue(timestamp, self.num_queued_requests)]
        )
        if RAY_SERVE_COLLECT_AUTOSCALING_METRICS_ON_HANDLE:
#####
python/ray/serve/_private/constants.py:1053:RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED = get_env_bool(
python/ray/serve/_private/constants.py:1054:    "RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED", "0"
python/ray/serve/_private/constants.py:1059:# RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED is true.
python/ray/serve/_private/constants.py:1159:if RAY_SERVE_INGRESS_REQUEST_ROUTER_METRICS_ENABLED:
python/ray/serve/_private/router.py:339:            record_interval_s = (
python/ray/serve/_private/router.py:346:                min(record_interval_s, autoscaling_config.metrics_interval_s),
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
__init__.py
__pycache__
constants.py
kv_aware_router.py
kv_token_tracker.py
token_channel.py
utils.py
vllm
26:def _get_expected_output_tokens(pending_request: PendingRequest) -> Optional[int]:
38:class KVAwareRouter(MultiplexMixin, RequestRouter):
53:    def initialize_state(
74:    async def choose_replicas(
82:        ``KVTokenTracker`` to rank them via ``select_worker``, and
132:        selection = await self._kv_token_tracker.select_worker(
146:def is_kv_aware(llm_config: LLMConfig) -> bool:
python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:65:from ray.llm._internal.serve.routing_policies.kv_aware.vllm.kv_events import (
python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:66:    assign_replica_kv_events_endpoint,
python/ray/llm/_internal/serve/engines/vllm/vllm_engine.py:326:        assign_replica_kv_events_endpoint(self.llm_config)
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
-rw-r--r-- 1 ray users    48257 2026-11-02 13:17:09.514322336 -0800 controller_149281.log
-rw-r--r-- 1 ray users      530 2026-11-02 12:38:29.711991188 -0800 replica_ttft-benchmark_LLMRouter_3jybiw18.log
-rw-r--r-- 1 ray users      754 2026-11-02 12:38:46.851974378 -0800 replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_2age6oh7.log
-rw-r--r-- 1 ray users      754 2026-11-02 12:50:30.228284519 -0800 replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_nx9jlpa8.log
-rw-r--r-- 1 ray users      754 2026-11-02 12:54:26.180004985 -0800 replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_wwhfhx0n.log
#####
INFO 2026-11-02 12:38:29,689 ttft-benchmark_LLMRouter 3jybiw18 -- Started initializing replica.
INFO 2026-11-02 12:38:29,707 ttft-benchmark_LLMRouter 3jybiw18 -- Dynamo SelectionService created (indexer threads 4).
INFO 2026-11-02 12:38:29,713 ttft-benchmark_LLMRouter 3jybiw18 -- Started <ray.serve._private.router.SharedRouterLongPollClient object at 0x734b13f41b20>.
INFO 2026-11-02 12:38:29,713 ttft-benchmark_LLMRouter 3jybiw18 -- Initialized DeploymentHandle idqq7lpy for Deployment(name='LLMRouter', app='ttft-benchmark').
#####
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_2o0sd86n.log 2026-11-02 14:49:01.490408851 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_dv94q8aa.log 2026-11-02 14:45:20.292688360 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_yrbkh85v.log 2026-11-02 14:36:43.550341324 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_wwhfhx0n.log 2026-11-02 12:54:26.180004985 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_nx9jlpa8.log 2026-11-02 12:50:30.228284519 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_2age6oh7.log 2026-11-02 12:38:46.851974378 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_2wxwzba3.log 2026-11-02 00:52:26.526652477 -0800
0
== replica_ttft-benchmark_LLMServer_Qwen--Qwen3_6-27B-FP8_w52qj2s6.log 2026-10-31 16:25:30.439067856 -0700
0
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
agentx-concurrency-ramp
lifecycle.json
sweep_summary.json
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/lifecycle.json
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/sweep_summary.json

/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/agentx-concurrency-ramp:
command.json
concurrency_series.json
logs
profile_export.jsonl
profile_export_aiperf.csv
profile_export_aiperf.json
profile_export_console.txt
server_metrics_export.csv
server_metrics_export.json
stderr.log
stdout.log
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/sweep_summary.json:7:        "sessions": 1,
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/sweep_summary.json:11:        "sessions": 32,
/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/benchmark/sweep_summary.json:15:        "sessions": 32,
```

</details>

<details><summary>Bash result</summary>

```text
k-5be017850000_pid-151035.0000.jsonl events 108966 queued 542 max kv usage 0.475 dropped 0 wall 1793652001..1793653026
k-5be017850000_pid-151035.0001.jsonl events 131417 queued 167 max kv usage 0.248 dropped 0 wall 1793653026..1793653912
k-5be017850000_pid-173074.0000.jsonl events 164439 queued 427 max kv usage 0.281 dropped 0 wall 1793652704..1793653922
k-5be017850000_pid-180495.0000.jsonl events 137773 queued 457 max kv usage 0.146 dropped 0 wall 1793652940..1793653880
```

</details>

**Bash** — Read the router metrics record interval and the KV event publisher setup

```bash
cd /mnt/local_storage/ray && sed -n 325,350p python/ray/serve/_private/router.py; echo '#####'; sed -n 1,80p python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/kv_events.py; grep -n "buffer_steps\|publisher\|KVEventsConfig\|enable_kv_cache_events\|topic" python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/kv_events.py
```

**Bash** — Inspect the aiperf per-request export for session and turn fields

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; head -c 1500 $R/benchmark/agentx-concurrency-ramp/profile_export.jsonl; echo; python3 -I - $R <<'EOF'
import json,sys
R=sys.argv[1]
n=0
for line in open(f'{R}/benchmark/agentx-concurrency-ramp/profile_export.jsonl'):
    d=json.loads(line); n+=1
    if n==1: print(sorted(d.keys()))
print('records',n)
EOF
```

**Bash** — Rebuild router-style and engine-style outstanding prefill from traces and compare to the scraped gauge

```bash
mkdir -p /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_decomp.py <<'EOF'
"""Decompose router gauge vs engine outstanding-prefill gap for one run."""
import json, sys, bisect, math, glob, os, statistics as st
R = sys.argv[1]
T = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
plot = json.load(open(f"{R}/analysis/plot_data.json")); t0 = plot["time_origin"]["profiling_start_ns"] / 1e9
ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json")); END = ev["periods"]["steady"]["end_s"]
manifest = json.load(open(f"{R}/manifest.json"))
import datetime
run_start = datetime.datetime.fromisoformat(manifest["started_at"]).timestamp()
run_end = datetime.datetime.fromisoformat(manifest["finished_at"]).timestamp()

# ---- gauge from scrapes
G = []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"): continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    for s in rec["samples"]:
        if s["name"] == "ray_serve_autoscaling_policy_metric" and s["labels"].get("metric") == "outstanding_effective_prefill_tokens" and "LLMServer" in s["labels"].get("deployment", ""):
            G.append((t, s["value"]))
G.sort()

# ---- traces of this run
reqs = {}   # rid -> dict
eng_deltas = []
files = [p for p in sorted(glob.glob(f"{T}/*.jsonl")) if run_start <= os.path.getmtime(p) <= run_end + 600]
for path in files:
    offset = None; chunks = {}; completed_at = {}; queued = {}; admitted = {}; first = {}; finished = {}; preempted = {}
    for line in open(path):
        try: e = json.loads(line)
        except ValueError: continue
        if offset is None and "wall_time_s" in e and "monotonic_s" in e: offset = e["wall_time_s"] - e["monotonic_s"]
        evn = e["event"]
        if evn == "batch_scheduled":
            for r in e["requests"]:
                if r["phase"] == "prefill": chunks.setdefault(r["request_id"], []).append((e["batch_index"], r["scheduled_tokens"]))
        elif evn == "batch_completed": completed_at[e["batch_index"]] = e["monotonic_s"]
        elif evn == "request_queued": queued[e["request_id"]] = e
        elif evn == "request_admitted": admitted[e["request_id"]] = e
        elif evn == "request_first_token": first[e["request_id"]] = e
        elif evn == "request_finished": finished[e["request_id"]] = e
        elif evn == "request_preempted": preempted.setdefault(e["request_id"], []).append((e["batch_index"], e["monotonic_s"]))
    if offset is None: continue
    rel = lambda mono: mono + offset - t0
    eng = os.path.basename(path).split(".")[0]
    for rid, q in queued.items():
        cs = chunks.get(rid)
        if not cs or any(b not in completed_at for b, _ in cs): continue
        starts = [(0, q["monotonic_s"])] + sorted(preempted.get(rid, []))
        for k, (batch, at) in enumerate(starts):
            until = starts[k + 1][0] if k + 1 < len(starts) else math.inf
            eng_deltas.append((rel(at), sum(t for b, t in cs if batch <= b < until)))
        eng_deltas.extend((rel(completed_at[b]), -t) for b, t in cs)
        adm = admitted.get(rid); f = first.get(rid)
        if adm is None or f is None: continue
        own = adm["prompt_tokens"] - adm["cached_prefix_tokens"]
        reqs[rid] = dict(eng=eng, q=rel(q["monotonic_s"]), adm=rel(adm["monotonic_s"]), ft=rel(f["monotonic_s"]),
                         fin=rel(finished[rid]["monotonic_s"]) if rid in finished else None,
                         own=own, isl=adm["prompt_tokens"], cached=adm["cached_prefix_tokens"],
                         pre_first_preempt=any(b <= f["batch_index"] for b, _ in preempted.get(rid, [])),
                         kind="cold" if adm["cached_prefix_tokens"] < 0.5 * adm["prompt_tokens"] else "warm")
print(f"trace files: {[os.path.basename(p) for p in files]}")
print(f"requests with queued+admitted+first_token: {len(reqs)}; in load phase: {sum(1 for r in reqs.values() if 0 <= r['q'] <= END)}")

def step(deltas):
    deltas = sorted(deltas); times = [t for t, _ in deltas]; levels = []; lv = 0
    for _, d in deltas: lv += d; levels.append(lv)
    return times, levels
def at(times, levels, t):
    i = bisect.bisect_right(times, t) - 1
    return levels[i] if i >= 0 else 0
def tmean(times, levels, a, b):
    """time-weighted mean of step fn over [a,b]"""
    i0 = bisect.bisect_right(times, a)
    pts = [(a, at(times, levels, a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, b))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(b, None)]):
        if t2 > t1: tot += v1 * (t2 - t1)
    return tot / (b - a)

def router_style(sel, a=0.0, b=0.0):
    d = []
    for r in sel:
        d.append((r["q"] - a, r["own"])); d.append((r["ft"] + b, -r["own"]))
    return step(d)

load = [r for r in reqs.values() if 0 <= r["q"] <= END]
Gt, Gv = step([(G[0][0], G[0][1])] + [(t, v - pv) for (t, v), (_, pv) in zip(G[1:], G)])
Et, El = step(eng_deltas)
Rt, Rl = router_style(reqs.values())
Rc = router_style([r for r in reqs.values() if r["kind"] == "cold"])
Rw = router_style([r for r in reqs.values() if r["kind"] == "warm"])
mG, mE, mR = tmean(Gt, Gv, 0, END), tmean(Et, El, 0, END), tmean(Rt, Rl, 0, END)
print(f"\nload-phase time means (0..{END:.0f}s): gauge {mG:.0f}  router-style rebuild {mR:.0f}  engine-style rebuild {mE:.0f}")
print(f"  gap gauge-engine {mG-mE:.0f}: whole-prompt-until-first-token (reason 1) {mR-mE:.0f} ({(mR-mE)/(mG-mE)*100:.0f}%), residual gauge-rebuild {mG-mR:.0f} ({(mG-mR)/(mG-mE)*100:.0f}%)")
print(f"  cold contribution to rebuild {tmean(*Rc, 0, END):.0f}, warm {tmean(*Rw, 0, END):.0f}")
tot_own = sum(r["own"] for r in load)
print(f"  sum own_uncached in load {tot_own:.0f}; each 100ms of extra hold adds {tot_own*0.1/END:.0f} to the mean; residual needs {(mG-mR)/tot_own*END*1000:.0f} ms of extra hold if it were lag alone")

# Does router-style match the gauge's shape? best shift
def corr(x, y):
    mx, my = st.mean(x), st.mean(y)
    sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy) if sx and sy else 0
Gs = [(t, v) for t, v in G if 0 <= t <= END]
best = None
for sh in [x/4 for x in range(-16, 17)]:
    rv = [at(Rt, Rl, t - sh) for t, _ in Gs]
    c = corr([v for _, v in Gs], rv)
    if best is None or c > best[1]: best = (sh, c)
print(f"\nbest time shift of gauge vs router-style rebuild: {best[0]:+.2f}s (corr {best[1]:.3f})")
sh = best[0]
# Regression G ~ a_c*Rc + a_w*Rw  (through origin), using scrapes
X = [(at(*Rc, t - sh), at(*Rw, t - sh)) for t, _ in Gs]; y = [v for _, v in Gs]
# normal equations
s11 = sum(a*a for a, _ in X); s22 = sum(b*b for _, b in X); s12 = sum(a*b for a, b in X); s1y = sum(a*v for (a, _), v in zip(X, y)); s2y = sum(b*v for (_, b), v in zip(X, y))
det = s11*s22 - s12*s12
ac = (s1y*s22 - s2y*s12)/det; aw = (s11*s2y - s12*s1y)/det
a1 = sum((a+b)*v for (a, b), v in zip(X, y)) / sum((a+b)**2 for a, b in X)
print(f"regression at that shift: gauge ~ {ac:.2f} x cold_rebuild + {aw:.2f} x warm_rebuild   (single slope: {a1:.2f})")
# time-weighted sanity: time-mean of gauge vs. a_c*cold + a_w*warm
print(f"  check: {ac:.2f}*{tmean(*Rc,0,END):.0f} + {aw:.2f}*{tmean(*Rw,0,END):.0f} = {ac*tmean(*Rc,0,END)+aw*tmean(*Rw,0,END):.0f} vs gauge mean {mG:.0f}")

# ---- single-request windows: scrapes where exactly one request is in router window for the preceding 3s
print("\nscrapes with exactly one request held (router-style) throughout [t-3, t]:")
byq = sorted(load, key=lambda r: r["q"])
rows = []
for t, v in Gs:
    inside = [r for r in load if r["q"] <= t - 3.0 and r["ft"] >= t]
    touching = [r for r in load if r["q"] <= t + 0.5 and r["ft"] >= t - 3.5]
    if len(inside) == 1 and len(touching) == 1:
        r = inside[0]; rows.append((t, v, r))
print(f"  n={len(rows)}")
for t, v, r in rows[:60]:
    print(f"  t={t:7.1f} gauge={v:8.0f} own_uncached={r['own']:7d} isl={r['isl']:7d} cached={r['cached']:7d} {r['kind']} ratio={v/r['own'] if r['own'] else float('nan'):.2f} engine={r['eng'][-6:]}")
exact = sum(1 for t, v, r in rows if abs(v - r["own"]) <= 64)
print(f"  gauge within one block (64) of own_uncached: {exact}/{len(rows)}; gauge within 64 of isl: {sum(1 for t,v,r in rows if abs(v-r['isl'])<=64)}/{len(rows)}")
json.dump({"reqs": {k: v for k, v in reqs.items()}, "G": G}, open(os.path.join(os.path.dirname(sys.argv[0]), "gauge_decomp_state.json"), "w"))
EOF
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_decomp.py $R
```

<details><summary>Bash result</summary>

```text
# Start the metrics pusher if autoscaling is enabled.
        autoscaling_config = self.autoscaling_config
        if autoscaling_config:
            self.metrics_pusher.start()
            # Optimization for autoscaling cold start time. If there are
            # currently 0 replicas for the deployment, and there is at
            # least one queued request on this router, then immediately
            # push handle metric to the controller.
            if self.should_send_scaled_to_zero_optimized_push(curr_num_replicas):
                self.push_autoscaling_metrics_to_controller()

            # Record number of queued + ongoing requests at regular
            # intervals into the in-memory metrics store
            record_interval_s = (
                autoscaling_config.look_back_period_s
                * RAY_SERVE_AUTOSCALING_METRIC_RECORD_INTERVAL_FACTOR
            )
            self.metrics_pusher.register_or_update_task(
                self.RECORD_METRICS_TASK_NAME,
                self._add_autoscaling_metrics_point,
                min(record_interval_s, autoscaling_config.metrics_interval_s),
            )
            # Push metrics to the controller periodically.
            self.metrics_pusher.register_or_update_task(
                self.PUSH_METRICS_TO_CONTROLLER_TASK_NAME,
#####
import logging
from typing import Any, Dict, Optional

import ray
from ray import serve
from ray.llm._internal.serve.core.configs.llm_config import LLMConfig
from ray.llm._internal.serve.routing_policies.kv_aware.constants import (
    DEFAULT_KV_EVENTS_PORT_BASE,
    DEFAULT_KV_EVENTS_REPLAY_PORT_OFFSET,
    DEFAULT_KV_TOKEN_PORT_BASE,
    KV_EVENTS_PORT_BASE_KEY,
    KV_TOKEN_METADATA_KEY,
    KV_TOKEN_PORT_BASE_KEY,
)
from ray.llm._internal.serve.routing_policies.kv_aware.kv_aware_router import (
    is_kv_aware,
)
from ray.serve._private.constants import SERVE_LOGGER_NAME

logger = logging.getLogger(SERVE_LOGGER_NAME)


def configure_kv_events_for_kv_routing(llm_config: LLMConfig) -> None:
    """
    Enable engine KV-cache events for a KV-aware-routed deployment.
    """
    engine_kwargs = llm_config.engine_kwargs
    if engine_kwargs.get("enable_prefix_caching") is False:
        logger.warning(
            "KV-aware routing is configured but enable_prefix_caching is False; "
            "the engine will not emit KV-cache events."
        )

    llm_config.update_engine_kwargs(
        kv_events_config={
            "enable_kv_cache_events": True,
            "publisher": "zmq",
            "endpoint": _default_kv_events_endpoint(llm_config),
            # Bind a replay (ROUTER) socket so the selection service can recover
            # the events it missed before its SUB connected (slow-joiner gap).
            "replay_endpoint": _default_kv_events_replay_endpoint(llm_config),
        }
    )

    _configure_runtime_env_for_kv_routing(llm_config)


def enable_native_kv_offload_events(vllm_config: Any) -> None:
    """Make vLLM's native CPU-offload events usable by the KV router."""
    cache_config = vllm_config.cache_config
    if (
        cache_config.kv_offloading_size is None
        or cache_config.kv_offloading_backend != "native"
    ):
        return

    vllm_config.kv_transfer_config.kv_connector_extra_config[
        "self_describing_kv_events"
    ] = True


def _configure_runtime_env_for_kv_routing(llm_config: LLMConfig) -> None:
    """Configure vLLM process-wide settings required by KV-aware routing.

    The KV router's global indexer chains and dedups blocks by the engines'
    block hashes, so identical content must hash identically on every
    replica. vLLM salts its block-hash chain root per process unless
    ``PYTHONHASHSEED`` is set, so pin it deployment-wide.

    Native offloading must use ``OffloadingConnector`` because it emits the
    self-describing CPU-tier events consumed by the KV router.
    """
    runtime_env = dict(llm_config.runtime_env or {})
    env_vars = dict(runtime_env.get("env_vars") or {})
    env_vars.setdefault("PYTHONHASHSEED", "0")
    env_vars["VLLM_USE_SIMPLE_KV_OFFLOAD"] = "0"
    runtime_env["env_vars"] = env_vars
    llm_config.runtime_env = runtime_env


36:            "enable_kv_cache_events": True,
37:            "publisher": "zmq",
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
{"metadata":{"session_num":0,"x_request_id":"f7fcefe8-2956-4075-8982-8768ce95888a","x_correlation_id":"1590b337-5301-4e92-8fc7-5f206ee4c7fb","conversation_id":"002001296e8a8c38ad9d7cc436d691afc602","turn_index":0,"source_trace_id":"002001296e8a8c38ad9d7cc436d691afc602","source_outer_idx":0,"source_kind":"weka_main","credit_issued_ns":1793652009128879273,"request_start_ns":1793652009144483652,"request_ack_ns":1793652009239603863,"request_end_ns":1793652009456444692,"worker_id":"worker_1291aa07","record_processor_id":"record_processor_5a4a93b6","benchmark_phase":"profiling","phase_index":0,"profiling_index":0,"phase_name":"profiling","phase_kind":"profiling","was_cancelled":false,"context_overflow_skip":false,"agent_depth":0,"root_correlation_id":"1590b337-5301-4e92-8fc7-5f206ee4c7fb"},"metrics":{"http_req_data_sent":{"value":2.283203125,"unit":"KB"},"http_req_sending":{"value":1.841087,"unit":"ms"},"output_sequence_length":{"value":20,"unit":"tokens"},"usage_prompt_cache_read_tokens":{"value":0,"unit":"tokens"},"http_req_data_received":{"value":5.5771484375,"unit":"KB"},"usage_total_tokens":{"value":685,"unit":"tokens"},"inter_chunk_latency":{"value":[13.632408999999999,3.301331,8.977698,9.009013999999999,9.086293999999999,9.188342,9.073136,9.17981,9.049711,9.149471,9.007892,9.080029999999999,9.039965,9.106308,9.017802999999999,9.020225,9.186651,9.07344,9.662339],"unit":"ms"},"http_req_blocked":{"value":0.0,"unit":"ms"},"time_to_second_token":{"value":13.632408999999999,"unit"
['metadata', 'metrics']
records 1593
```

</details>

<details><summary>Bash result</summary>

```text
trace files: ['k-5be017850000_pid-151035.0000.jsonl', 'k-5be017850000_pid-151035.0001.jsonl', 'k-5be017850000_pid-173074.0000.jsonl', 'k-5be017850000_pid-180495.0000.jsonl']
requests with queued+admitted+first_token: 1592; in load phase: 1592

load-phase time means (0..1860s): gauge 17899  router-style rebuild 13418  engine-style rebuild 8249
  gap gauge-engine 9650: whole-prompt-until-first-token (reason 1) 5169 (54%), residual gauge-rebuild 4481 (46%)
  cold contribution to rebuild 9834, warm 3584
  sum own_uncached in load 11056207; each 100ms of extra hold adds 594 to the mean; residual needs 754 ms of extra hold if it were lag alone

best time shift of gauge vs router-style rebuild: +3.50s (corr 0.586)
regression at that shift: gauge ~ 0.81 x cold_rebuild + 0.68 x warm_rebuild   (single slope: 0.79)
  check: 0.81*9834 + 0.68*3584 = 10386 vs gauge mean 17899

scrapes with exactly one request held (router-style) throughout [t-3, t]:
  n=2
  t=    3.7 gauge=       0 own_uncached=  63724 isl=  63724 cached=      0 cold ratio=0.00 engine=151035
  t=   79.7 gauge=   88347 own_uncached=  88347 isl=  88347 cached=      0 cold ratio=1.00 engine=151035
  gauge within one block (64) of own_uncached: 1/2; gauge within 64 of isl: 1/2
```

</details>

**Bash** — Test whether gauge values equal exact sums of engine uncached tokens and estimate the extra hold time

```bash
cat > /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_tests.py <<'EOF'
import json, sys, bisect, math, random, statistics as st, itertools
S = json.load(open(sys.argv[1])); reqs = list(S["reqs"].values()); G = [tuple(x) for x in S["G"]]
END = 1860.0
Gs = [(t, v) for t, v in G if 0 <= t <= END]
load = [r for r in reqs if 0 <= r["q"] <= END]
print("requests", len(load), "own_uncached==0:", sum(r["own"] == 0 for r in load), "preempted before first token:", sum(r["pre_first_preempt"] for r in load))

def step(d):
    d = sorted(d); times = [t for t, _ in d]; lv = 0; levels = []
    for _, x in d: lv += x; levels.append(lv)
    return times, levels
def tmean(times, levels, a, b, f=lambda v: v):
    i0 = bisect.bisect_right(times, a)
    at = lambda t: (levels[bisect.bisect_right(times, t) - 1] if bisect.bisect_right(times, t) else 0)
    pts = [(a, at(a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, b))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(b, None)]):
        if t2 > t1: tot += f(v1) * (t2 - t1)
    return tot / (b - a)
def rebuild(sel, lag):
    return step([(r["q"], r["own"]) for r in sel] + [(r["ft"] + lag, -r["own"]) for r in sel])

gauge_busy = sum(v > 0 for _, v in Gs) / len(Gs)
gauge_mean = st.mean(v for _, v in Gs)
print(f"\ngauge: busy fraction (sample>0) {gauge_busy:.3f}, sample mean {gauge_mean:.0f}")
print("router-style rebuild from traces, extra hold per request (ms): busy fraction, time mean, cold/warm split")
for lag in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0, 1.2, 1.5]:
    T, L = rebuild(load, lag)
    busy = tmean(T, L, 0, END, lambda v: 1.0 if v > 0 else 0.0)
    m = tmean(T, L, 0, END)
    Tc, Lc = rebuild([r for r in load if r["kind"] == "cold"], lag); Tw, Lw = rebuild([r for r in load if r["kind"] == "warm"], lag)
    print(f"  +{lag*1000:5.0f} ms: busy {busy:.3f}  mean {m:7.0f}  (cold {tmean(Tc,Lc,0,END):6.0f} warm {tmean(Tw,Lw,0,END):6.0f})")

# ---- isolated requests: no other request within 6s on either side
qs = sorted(load, key=lambda r: r["q"])
iso = []
for i, r in enumerate(qs):
    prev_ft = max((x["ft"] for x in qs[:i]), default=-1e9)
    nxt_q = qs[i + 1]["q"] if i + 1 < len(qs) else 1e9
    if r["q"] - prev_ft > 6 and nxt_q - r["ft"] > 6: iso.append(r)
print(f"\nisolated requests (gap > 6s both sides): {len(iso)}; cold {sum(r['kind']=='cold' for r in iso)} warm {sum(r['kind']=='warm' for r in iso)}")
# for each isolated request, scrapes in [q-1, ft+6] that are nonzero; value equality
nz_total = 0; hold_total = 0.0; eq = 0; neq = []; nz_by_kind = {"cold": [0, 0.0, 0], "warm": [0, 0.0, 0]}
for r in iso:
    samples = [(t, v) for t, v in Gs if r["q"] - 1 <= t <= r["ft"] + 6]
    nz = [(t, v) for t, v in samples if v > 0]
    nz_total += len(nz); hold_total += r["ft"] - r["q"]
    k = nz_by_kind[r["kind"]]; k[0] += len(nz); k[1] += r["ft"] - r["q"]; k[2] += 1
    for t, v in nz:
        if abs(v - r["own"]) <= 1: eq += 1
        else: neq.append((round(t, 1), v, r["own"], r["isl"], r["cached"], r["kind"], round(r["ft"] - r["q"], 2)))
print(f"  nonzero scrapes near them: {nz_total}; of which equal to own_uncached: {eq}; not equal: {len(neq)}")
print(f"  implied mean hold from scrapes: {nz_total*2/len(iso)*1000:.0f} ms vs engine queued->first token mean {hold_total/len(iso)*1000:.0f} ms  => extra hold ~{(nz_total*2-hold_total)/len(iso)*1000:.0f} ms")
for k, (n, h, c) in nz_by_kind.items():
    if c: print(f"    {k}: n={c} scrapes-implied hold {n*2/c*1000:.0f} ms vs engine {h/c*1000:.0f} ms => extra ~{(n*2-h)/c*1000:.0f} ms")
print("  mismatches (t, gauge, own_uncached, isl, cached, kind, engine hold s):")
for x in neq[:40]: print("   ", x)

# ---- subset-sum test on all nonzero scrapes with few candidates
def subset_match(target, vals):
    # DP over sums (values are ints up to ~200K; candidates <= 14)
    sums = {0: ()}
    for i, v in enumerate(vals):
        new = {}
        for s, sub in sums.items():
            ns = s + v
            if ns <= target + 2 and ns not in sums and ns not in new: new[ns] = sub + (i,)
        sums.update(new)
    for s in (target, target - 1, target + 1):
        if s in sums: return sums[s]
    return None
def run_subset(reqset, label):
    tested = matched = 0; mism = []
    for t, v in Gs:
        if v <= 0: continue
        cands = [r for r in reqset if r["q"] - 1.0 <= t + 0.5 and r["ft"] + 2.5 >= t - 5.0]
        if not cands or len(cands) > 14: continue
        tested += 1
        sub = subset_match(int(round(v)), [r["own"] for r in cands])
        if sub is not None: matched += 1
        else: mism.append((round(t, 1), int(v), sorted((r["own"], r["kind"]) for r in cands)))
    print(f"\n{label}: nonzero scrapes with <=14 candidate requests: {tested}, gauge value == exact subset sum of own_uncached: {matched} ({matched/max(1,tested)*100:.0f}%)")
    return mism
mism = run_subset(load, "real own_uncached")
# control: shuffle own_uncached across requests
random.seed(1); owns = [r["own"] for r in load]; random.shuffle(owns)
shuf = [dict(r, own=o) for r, o in zip(load, owns)]
run_subset(shuf, "control (own_uncached shuffled across requests)")
print("sample mismatches (t, gauge, candidate (own,kind)):")
for x in mism[:25]: print("  ", x)
EOF
python3 -I /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_tests.py /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_decomp_state.json
```

**Bash** — Check where the router tokenizes and admits requests relative to dispatch

```bash
cd /mnt/local_storage/ray && sed -n 74,145p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_aware_router.py; echo '#####'; grep -n "tokenize\|to_thread\|run_in_executor\|def _route\|async def route\|select_worker\|choose_replica" python/ray/llm/_internal/serve/core/ingress/router.py | head -30
```

**Bash** — Check the router metric record and push intervals

```bash
cd /mnt/local_storage/ray && grep -n "RAY_SERVE_AUTOSCALING_METRIC_RECORD_INTERVAL_FACTOR\|CONTROL_LOOP_INTERVAL_S\b" python/ray/serve/_private/constants.py | head; grep -n "push_autoscaling_metrics_to_controller\|PUSH_METRICS_TO_CONTROLLER_TASK_NAME" -A4 python/ray/serve/_private/router.py | sed -n 1,40p
```

<details><summary>Bash result</summary>

```text
Exit code 1
requests 1592 own_uncached==0: 0 preempted before first token: 0

gauge: busy fraction (sample>0) 0.415, sample mean 17899
router-style rebuild from traces, extra hold per request (ms): busy fraction, time mean, cold/warm split
  +    0 ms: busy 0.347  mean   13418  (cold   9834 warm   3584)
  +  100 ms: busy 0.401  mean   14012  (cold  10087 warm   3925)
  +  200 ms: busy 0.451  mean   14607  (cold  10341 warm   4266)
  +  300 ms: busy 0.496  mean   15201  (cold  10594 warm   4607)
  +  400 ms: busy 0.536  mean   15795  (cold  10847 warm   4948)
  +  500 ms: busy 0.573  mean   16390  (cold  11100 warm   5289)
  +  600 ms: busy 0.607  mean   16984  (cold  11354 warm   5630)
  +  700 ms: busy 0.639  mean   17578  (cold  11607 warm   5971)
  +  800 ms: busy 0.670  mean   18173  (cold  11860 warm   6312)
  + 1000 ms: busy 0.722  mean   19361  (cold  12367 warm   6995)
  + 1200 ms: busy 0.766  mean   20550  (cold  12873 warm   7677)
  + 1500 ms: busy 0.821  mean   22333  (cold  13633 warm   8700)

isolated requests (gap > 6s both sides): 0; cold 0 warm 0
  nonzero scrapes near them: 0; of which equal to own_uncached: 0; not equal: 0
Traceback (most recent call last):
  File "/tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/gauge_tests.py", line 53, in <module>
    print(f"  implied mean hold from scrapes: {nz_total*2/len(iso)*1000:.0f} ms vs engine queued->first token mean {hold_total/len(iso)*1000:.0f} ms  => extra hold ~{(nz_total*2-hold_total)/len(iso)*1000:.0f} ms")
                                               ~~~~~~~~~~^~~~~~~~~
ZeroDivisionError: division by zero
```

</details>

<details><summary>Bash result</summary>

```text
async def choose_replicas(
        self,
        candidate_replicas: List[RunningReplica],
        pending_request: Optional[PendingRequest] = None,
    ) -> List[List[RunningReplica]]:
        """Choose the candidate replica(s) to route ``pending_request`` to.

        Maps the candidate replicas to their Dynamo worker ids, asks the
        ``KVTokenTracker`` to rank them via ``select_worker``, and
        routes to the chosen worker's replica. With direct streaming enabled,
        HAProxy then forwards the original request to that replica.

        Requests with no prompt token ids have nothing to score on, so they
        route to a random candidate (batch prompts, truncated or unparseable
        bodies).

        A LoRA request is narrowed to the replicas already holding the adapter
        before it is scored, and carries the adapter name into scoring.

        Args:
            candidate_replicas: The replicas eligible to serve the request.
            pending_request: The request being routed.

        Returns:
            Ranked groups of replicas.
        """
        if (
            pending_request is not None
            and pending_request.metadata.multiplexed_model_id
        ):
            # Score within the multiplex candidates
            candidate_replica_ids = self.apply_multiplex_routing(
                pending_request=pending_request
            )
            candidate_replicas = [
                replica
                for replica in candidate_replicas
                if replica.replica_id in candidate_replica_ids
            ]
            if not candidate_replicas:
                # No multiplex candidates yet (e.g. the adapter's location is
                # still propagating). Return no ranks so the caller backs off.
                return []

        token_ids = (
            pending_request.kwargs.get(REQUEST_TOKEN_IDS_KWARG)
            if pending_request is not None
            else None
        )
        # No token ids to score on, or no tracker in this process (proxy fallback
        # router): load-balance.
        if not token_ids or self._kv_token_tracker is None:
            return [[random.choice(candidate_replicas)]] if candidate_replicas else []

        worker_id_to_replica = {
            get_worker_id(replica.replica_id.unique_id): replica
            for replica in candidate_replicas
        }
        selection = await self._kv_token_tracker.select_worker(
            pending_request.metadata.request_id,
            token_ids,
            list(worker_id_to_replica),
            _get_expected_output_tokens(pending_request),
            # vLLM stamps this adapter name on the request's KV-cache events and
            # the selection service salts its KV hashes with it, so each adapter
            # gets its own KV namespace. Empty means the base model.
            lora_name=pending_request.metadata.multiplexed_model_id or None,
            session_id=pending_request.metadata.session_id or None,
        )
        return [[worker_id_to_replica[selection["worker_id"]]]]


#####
96:            ``choose_replica`` positionally, exposing the request fields the way
112:        ``choose_replica`` so session-aware policies (e.g.
126:            a pre-routing ``/tokenize`` rejection is surfaced here.
147:        self._tokenizer = None
169:            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (  # noqa: E501
173:            self._tokenizer = await asyncio.to_thread(Tokenizer, llm_config)
188:    async def route(self, request: Request):
208:        if self._tokenizer is not None and routing_payload is not None:
212:            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (  # noqa: E501
219:                request_token_ids = await self._tokenizer.tokenize(
358:        ``routing_payload``, when present, is passed to ``choose_replica``
374:        choose_replica_kwargs = {"_reserve": False}
376:            choose_replica_kwargs[REQUEST_TOKEN_IDS_KWARG] = request_token_ids
377:        async with handle.choose_replica(
378:            *route_args, **choose_replica_kwargs
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
77:CONTROL_LOOP_INTERVAL_S = get_env_float_non_negative(
78:    "RAY_SERVE_CONTROL_LOOP_INTERVAL_S", 0.1
504:RAY_SERVE_AUTOSCALING_METRIC_RECORD_INTERVAL_FACTOR = get_env_float(
505:    "RAY_SERVE_AUTOSCALING_METRIC_RECORD_INTERVAL_FACTOR", 0.2
112:    PUSH_METRICS_TO_CONTROLLER_TASK_NAME = "push_metrics_to_controller"
113-    RECORD_METRICS_TASK_NAME = "record_metrics"
114-
115-    def __init__(
116-        self,
--
281:                self.push_autoscaling_metrics_to_controller()
282-
283-            yield
284-        finally:
285-            # If the request is disconnected before assignment, this coroutine
--
335:                self.push_autoscaling_metrics_to_controller()
336-
337-            # Record number of queued + ongoing requests at regular
338-            # intervals into the in-memory metrics store
339-            record_interval_s = (
--
350:                self.PUSH_METRICS_TO_CONTROLLER_TASK_NAME,
351:                self.push_autoscaling_metrics_to_controller,
352-                autoscaling_config.metrics_interval_s,
353-            )
354-
355-        else:
--
454:    def push_autoscaling_metrics_to_controller(self):
455-        """Pushes queued and running request metrics to the controller.
456-
457-        These metrics are used by the controller for autoscaling.
458-        If a previous push is already in flight, skips this push (will try again next interval).
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Run the subset-sum exactness test on gauge samples

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 - <<'EOF'
import re
p='gauge_tests.py'; s=open(p).read()
s=s.replace('if r["q"] - prev_ft > 6 and nxt_q - r["ft"] > 6: iso.append(r)','if r["q"] - prev_ft > 2 and nxt_q - r["ft"] > 2: iso.append(r)')
s=s.replace('print(f"  implied mean hold','if iso: print(f"  implied mean hold')
s=s.replace('for k, (n, h, c) in nz_by_kind.items():','for k, (n, h, c) in (nz_by_kind.items() if iso else []):')
open(p,'w').write(s)
EOF
python3 -I gauge_tests.py gauge_decomp_state.json 2>&1 | sed -n '/isolated/,$p'
```

**Bash** — Check whether any router-side KV overlap metrics were scraped

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import json,sys
R=sys.argv[1]
inv=json.load(open(f'{R}/analysis/metrics_inventory.json'))
names=inv if isinstance(inv,list) else (inv.get('metrics') or inv.get('names') or list(inv.keys()))
names=[n if isinstance(n,str) else json.dumps(n)[:120] for n in names]
print(len(names))
for n in names:
    if any(k in n.lower() for k in ('kv','overlap','hit','prefill','router','selection','dynamo','token')): print(n)
EOF
```

**Bash** — Measure inter-turn gaps within sessions from the benchmark export

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import json,sys,statistics as st
R=sys.argv[1]
recs=[]
for line in open(f'{R}/benchmark/agentx-concurrency-ramp/profile_export.jsonl'):
    d=json.loads(line); m=d['metadata']; recs.append(m|{k:v.get('value') for k,v in d['metrics'].items() if isinstance(v,dict)})
print(sorted(recs[0].keys()))
# inter-turn gap within conversation
by=dict()
for r in recs: by.setdefault(r['conversation_id'],[]).append(r)
gaps=[]
for cid,rs in by.items():
    rs.sort(key=lambda r:r['turn_index'])
    for a,b in zip(rs,rs[1:]):
        gaps.append((b['request_start_ns']-a['request_end_ns'])/1e6)
print('conversations',len(by),'turn pairs',len(gaps),'inter-turn gap ms: min %.0f p10 %.0f median %.0f p90 %.0f max %.0f'%(min(gaps),sorted(gaps)[len(gaps)//10],st.median(gaps),sorted(gaps)[int(len(gaps)*.9)],max(gaps)))
import collections
print('turns per conversation', collections.Counter(len(v) for v in by.values()).most_common(10))
json.dump(recs,open('/tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/aiperf_recs.json','w'))
EOF
```

<details><summary>Bash result</summary>

```text
isolated requests (gap > 6s both sides): 32; cold 1 warm 31
  nonzero scrapes near them: 21; of which equal to own_uncached: 4; not equal: 17
  implied mean hold from scrapes: 1312 ms vs engine queued->first token mean 478 ms  => extra hold ~835 ms
    cold: n=1 scrapes-implied hold 4000 ms vs engine 4783 ms => extra ~-783 ms
    warm: n=31 scrapes-implied hold 1226 ms vs engine 339 ms => extra ~887 ms
  mismatches (t, gauge, own_uncached, isl, cached, kind, engine hold s):
    (23.7, 67211.0, 2115, 67971, 65856, 'warm', 0.16)
    (27.7, 67211.0, 2115, 67971, 65856, 'warm', 0.16)
    (55.7, 64115.0, 3516, 66236, 62720, 'warm', 0.24)
    (83.7, 88347.0, 10689, 76545, 65856, 'warm', 0.65)
    (87.7, 88347.0, 10689, 76545, 65856, 'warm', 0.65)
    (87.7, 88347.0, 5460, 83860, 78400, 'warm', 0.36)
    (147.7, 54706.0, 20498, 54994, 34496, 'warm', 1.15)
    (203.7, 47158.0, 585, 64873, 64288, 'warm', 0.07)
    (207.7, 47158.0, 585, 64873, 64288, 'warm', 0.07)
    (207.7, 47158.0, 25083, 78395, 53312, 'warm', 1.49)
    (231.7, 64610.0, 2251, 66539, 64288, 'warm', 0.17)
    (235.7, 64610.0, 2251, 66539, 64288, 'warm', 0.17)
    (387.7, 2581.0, 1023, 51199, 50176, 'warm', 0.09)
    (387.7, 2581.0, 1092, 51268, 50176, 'warm', 0.09)
    (1315.7, 72078.0, 2063, 110255, 108192, 'warm', 0.2)
    (1359.7, 4423.0, 1175, 37239, 36064, 'warm', 0.08)
    (1387.7, 10831.0, 966, 132678, 131712, 'warm', 0.13)

real own_uncached: nonzero scrapes with <=14 candidate requests: 184, gauge value == exact subset sum of own_uncached: 122 (66%)

control (own_uncached shuffled across requests): nonzero scrapes with <=14 candidate requests: 184, gauge value == exact subset sum of own_uncached: 6 (3%)
sample mismatches (t, gauge, candidate (own,kind)):
   (27.7, 67211, [(664, 'warm'), (2115, 'warm'), (3051, 'warm')])
   (87.7, 88347, [(5460, 'warm'), (10689, 'warm')])
   (107.7, 8703, [(37765, 'cold'), (67700, 'warm')])
   (119.7, 31013, [(34891, 'cold'), (41989, 'cold')])
   (123.7, 31013, [(34891, 'cold'), (41989, 'cold')])
   (127.7, 31013, [(34891, 'cold'), (34956, 'cold'), (41989, 'cold')])
   (147.7, 54706, [(20498, 'warm'), (34689, 'cold')])
   (387.7, 2581, [(693, 'warm'), (1023, 'warm'), (1092, 'warm'), (1436, 'warm')])
   (407.7, 17554, [(507, 'warm'), (645, 'warm'), (2177, 'warm'), (2412, 'warm'), (17426, 'cold'), (88278, 'cold')])
   (411.7, 90669, [(645, 'warm'), (2177, 'warm'), (17426, 'cold'), (88278, 'cold')])
   (415.7, 90669, [(645, 'warm'), (849, 'warm'), (1339, 'warm'), (2965, 'cold'), (3035, 'cold'), (14230, 'warm'), (17426, 'cold'), (88278, 'cold')])
   (427.7, 1476, [(386, 'warm'), (522, 'warm'), (1014, 'warm'), (1996, 'warm'), (2973, 'cold'), (5330, 'warm')])
   (447.7, 52528, [(416, 'warm'), (806, 'warm'), (8055, 'warm'), (9773, 'warm'), (17108, 'warm'), (27715, 'warm')])
   (491.7, 89531, [(2452, 'warm'), (2531, 'warm'), (7130, 'warm'), (18679, 'warm'), (76845, 'cold')])
   (495.7, 89531, [(1751, 'cold'), (2452, 'warm'), (2531, 'warm'), (3363, 'warm'), (18679, 'warm'), (18778, 'cold'), (76845, 'cold')])
   (527.7, 7217, [(730, 'cold'), (1517, 'warm'), (1846, 'warm'), (1858, 'warm'), (1898, 'warm'), (2175, 'warm'), (4454, 'warm'), (46074, 'cold')])
   (587.7, 3473, [(345, 'cold'), (1127, 'warm'), (1420, 'warm'), (1721, 'warm'), (6682, 'warm'), (19625, 'warm'), (50870, 'cold')])
   (627.7, 62458, [(1397, 'warm'), (1475, 'warm'), (1604, 'warm'), (1818, 'warm'), (1965, 'warm'), (3746, 'warm'), (5497, 'warm'), (9745, 'cold')])
   (699.7, 108825, [(1447, 'warm'), (1704, 'warm'), (2206, 'warm'), (2357, 'warm'), (2566, 'warm'), (5337, 'warm'), (15052, 'warm')])
   (703.7, 108825, [(473, 'cold'), (506, 'warm'), (1447, 'warm'), (1885, 'warm'), (2096, 'warm'), (2206, 'warm'), (2970, 'cold'), (5337, 'warm'), (15052, 'warm')])
   (707.7, 108825, [(473, 'cold'), (506, 'warm'), (1885, 'warm'), (2096, 'warm'), (2206, 'warm'), (2970, 'cold'), (11702, 'warm'), (98027, 'cold')])
   (727.7, 14524, [(1496, 'warm'), (1829, 'warm'), (2122, 'warm'), (2596, 'warm'), (2948, 'warm'), (3642, 'warm'), (4375, 'warm'), (4985, 'warm'), (8718, 'warm'), (29968, 'warm'), (118171, 'cold')])
   (731.7, 126260, [(1496, 'warm'), (1829, 'warm'), (2122, 'warm'), (2948, 'warm'), (8718, 'warm'), (29968, 'warm'), (118171, 'cold')])
   (735.7, 126260, [(1829, 'warm'), (2947, 'warm'), (2948, 'warm'), (5602, 'warm'), (29968, 'warm'), (52273, 'cold'), (118171, 'cold')])
   (787.7, 108242, [(557, 'warm'), (845, 'warm'), (1562, 'warm'), (3485, 'warm'), (4975, 'warm'), (5683, 'warm'), (7190, 'warm'), (30636, 'warm'), (70118, 'cold')])
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
17
serve_request_router_queue_len
vllm_prefix_cache_hits
vllm_request_time_per_output_token_seconds
vllm_time_to_first_token_seconds
```

</details>

<details><summary>Bash result</summary>

```text
['agent_depth', 'benchmark_phase', 'context_overflow_skip', 'conversation_id', 'credit_issued_ns', 'decode_duration', 'e2e_output_token_throughput', 'http_req_blocked', 'http_req_chunks_received', 'http_req_chunks_sent', 'http_req_connecting', 'http_req_connection_overhead', 'http_req_connection_reused', 'http_req_data_received', 'http_req_data_sent', 'http_req_dns_lookup', 'http_req_duration', 'http_req_receiving', 'http_req_sending', 'http_req_total', 'http_req_waiting', 'input_sequence_length', 'inter_chunk_latency', 'inter_token_latency', 'osl_mismatch_diff_pct', 'output_sequence_length', 'output_token_count', 'output_token_throughput_per_user', 'phase_index', 'phase_kind', 'phase_name', 'prefill_throughput_per_user', 'profiling_index', 'record_processor_id', 'request_ack_ns', 'request_end_ns', 'request_latency', 'request_start_ns', 'root_correlation_id', 'session_num', 'source_kind', 'source_outer_idx', 'source_trace_id', 'time_to_first_output_token', 'time_to_first_token', 'time_to_second_token', 'turn_index', 'usage_completion_tokens', 'usage_prompt_cache_read_tokens', 'usage_prompt_tokens', 'usage_total_tokens', 'was_cancelled', 'worker_id', 'x_correlation_id', 'x_request_id']
conversations 127 turn pairs 1466 inter-turn gap ms: min 1 p10 648 median 1410 p90 15178 max 1437509
turns per conversation [(1, 37), (16, 7), (2, 7), (18, 7), (6, 5), (5, 4), (3, 4), (31, 4), (20, 4), (9, 3)]
```

</details>

**Bash** — Print raw request and gauge sequences in two early windows

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json
S=json.load(open('gauge_decomp_state.json')); reqs=sorted(S['reqs'].items(), key=lambda kv: kv[1]['q']); G=[tuple(x) for x in S['G']]
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
for lo,hi in [(0,130),(195,245)]:
    print(f"\n===== window {lo}-{hi}s: requests (q, first_token, finished, own, isl, cached, kind, conv/turn) and gauge samples")
    ev=[]
    for rid,r in reqs:
        if lo<=r['q']<=hi:
            a=recs.get(aid(rid),{})
            ev.append((r['q'],f"REQ  q={r['q']:7.2f} ft={r['ft']:7.2f} fin={r['fin'] if r['fin'] is None else round(r['fin'],2)} own={r['own']:6d} isl={r['isl']:6d} cached={r['cached']:6d} {r['kind']} conv={str(a.get('conversation_id'))[:6]} turn={a.get('turn_index')} eng={r['eng'][-6:]}"))
    for t,v in G:
        if lo<=t<=hi: ev.append((t,f"GAUGE t={t:7.2f} {v:8.0f}"))
    for _,s in sorted(ev): print(s)
EOF
```

**Bash** — Measure how often the exported gauge repeats identical nonzero values across scrapes

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json,statistics as st
S=json.load(open('gauge_decomp_state.json')); G=[tuple(x) for x in S['G']]
Gs=[(t,v) for t,v in G if 0<=t<=1860]
dt=[b[0]-a[0] for a,b in zip(Gs,Gs[1:])]
print('scrape interval: median %.2f min %.2f max %.2f'%(st.median(dt),min(dt),max(dt)))
pairs=list(zip(Gs,Gs[1:]))
nz=[(a,b) for a,b in pairs if a[1]>0 and b[1]>0]
same=[(a,b) for a,b in nz if a[1]==b[1]]
print('consecutive scrapes both nonzero:',len(nz),'identical:',len(same))
for lo,hi in [(0,300),(300,600),(600,900),(900,1200),(1200,1500),(1500,1860)]:
    n=[(a,b) for a,b in nz if lo<=a[0]<hi]; s=[(a,b) for a,b in same if lo<=a[0]<hi]
    print(f'  {lo}-{hi}: both nonzero {len(n)}, identical {len(s)}')
# runs of identical nonzero values
runs=[]; cur=None
for t,v in Gs:
    if v>0 and cur and cur[1]==v: cur[2]+=1; cur[3]=t
    else:
        if cur and cur[2]>=2: runs.append(cur)
        cur=[t,v,1,t]
print('runs of identical nonzero values length>=2:',len(runs))
for r in runs[:50]: print('   start %.1f value %.0f samples %d (%.0fs)'%(r[0],r[1],r[2],r[3]-r[0]+4))
EOF
```

**Bash** — Rerun the existing tool to see the published vs recomputed policy mean check

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment && python3 tools/policy_vs_engine_mean.py runs/20261102T203807Z-b200-workday-curve-prefill-mean --out /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/tmp_plot.png 2>&1 | tail -4
```

**Bash** — Check how the replica-side lifecycle request id is derived and the output kind used

```bash
cd /mnt/local_storage/ray && grep -n "get_serve_request_id" -A12 python/ray/llm/_internal/serve/utils/server_utils.py | head -40; echo '#####'; grep -rn "request_id" python/ray/llm/_internal/serve/core/ingress/router.py | head -20; echo '#####'; grep -n "output_kind\|RequestOutputKind" python/ray/llm/_internal/serve/engines/vllm/*.py | head
```

<details><summary>Bash result</summary>

```text
===== window 0-130s: requests (q, first_token, finished, own, isl, cached, kind, conv/turn) and gauge samples
REQ  q=   0.11 ft=   0.15 fin=0.33 own=   665 isl=   665 cached=     0 cold conv=002001 turn=0 eng=151035
REQ  q=   0.45 ft=   3.73 fin=7.46 own= 63724 isl= 63724 cached=     0 cold conv=002001 turn=1 eng=151035
GAUGE t=   3.70        0
GAUGE t=   7.70        0
REQ  q=   8.64 ft=   8.83 fin=11.34 own=  2678 isl= 65398 cached= 62720 warm conv=002001 turn=2 eng=151035
GAUGE t=  11.70        0
REQ  q=  15.07 ft=  15.11 fin=15.23 own=   474 isl=   474 cached=     0 cold conv=006c98 turn=0 eng=151035
REQ  q=  15.33 ft=  18.81 fin=19.46 own= 64624 isl= 64624 cached=     0 cold conv=006c98 turn=1 eng=151035
GAUGE t=  15.70        0
REQ  q=  16.84 ft=  18.81 fin=21.28 own=  2587 isl= 66875 cached= 64288 warm conv=002001 turn=3 eng=151035
GAUGE t=  19.70    67211
REQ  q=  22.60 ft=  22.76 fin=25.78 own=  2115 isl= 67971 cached= 65856 warm conv=002001 turn=4 eng=151035
GAUGE t=  23.70    67211
REQ  q=  27.10 ft=  27.31 fin=30.27 own=  3051 isl= 70475 cached= 67424 warm conv=002001 turn=5 eng=151035
GAUGE t=  27.70    67211
REQ  q=  29.07 ft=  29.14 fin=30.84 own=   664 isl= 64952 cached= 64288 warm conv=006c98 turn=2 eng=151035
GAUGE t=  31.70        0
REQ  q=  32.39 ft=  32.60 fin=37.58 own=  3091 isl= 72083 cached= 68992 warm conv=002001 turn=6 eng=151035
GAUGE t=  35.70        0
REQ  q=  39.32 ft=  39.84 fin=84.84 own=  8242 isl= 78802 cached= 70560 warm conv=002001 turn=7 eng=151035
GAUGE t=  39.70        0
GAUGE t=  43.70        0
REQ  q=  45.09 ft=  45.14 fin=45.37 own=   664 isl=   664 cached=     0 cold conv=00ca01 turn=0 eng=151035
REQ  q=  45.47 ft=  48.78 fin=51.31 own= 64115 isl= 64115 cached=     0 cold conv=00ca01 turn=1 eng=151035
GAUGE t=  47.70        0
GAUGE t=  51.70    64115
REQ  q=  54.44 ft=  54.68 fin=62.46 own=  3516 isl= 66236 cached= 62720 warm conv=00ca01 turn=2 eng=151035
GAUGE t=  55.70    64115
GAUGE t=  59.70        0
GAUGE t=  63.70        0
GAUGE t=  67.70        0
GAUGE t=  71.70        0
REQ  q=  75.24 ft=  80.02 fin=102.32 own= 88347 isl= 88347 cached=     0 cold conv=019608 turn=0 eng=151035
GAUGE t=  75.70        0
GAUGE t=  79.70    88347
REQ  q=  83.25 ft=  83.91 fin=96.53 own= 10689 isl= 76545 cached= 65856 warm conv=00ca01 turn=3 eng=151035
GAUGE t=  83.70    88347
REQ  q=  86.10 ft=  86.46 fin=95.0 own=  5460 isl= 83860 cached= 78400 warm conv=002001 turn=8 eng=151035
GAUGE t=  87.70    88347
GAUGE t=  91.70        0
REQ  q=  93.58 ft=  93.68 fin=96.16 own=  1120 isl= 65408 cached= 64288 warm conv=006c98 turn=3 eng=151035
GAUGE t=  95.70        0
REQ  q=  97.83 ft=  98.39 fin=153.04 own=  8703 isl= 83967 cached= 75264 warm conv=00ca01 turn=4 eng=151035
GAUGE t=  99.70     8703
GAUGE t= 103.70     8703
REQ  q= 104.50 ft= 110.51 fin=156.5 own= 67700 isl=155508 cached= 87808 warm conv=019608 turn=1 eng=151035
REQ  q= 105.55 ft= 111.09 fin=117.16 own= 37765 isl= 37765 cached=     0 cold conv=02bc0a turn=0 eng=151035
GAUGE t= 107.70     8703
GAUGE t= 111.70   105465
GAUGE t= 115.70   105465
REQ  q= 118.11 ft= 120.81 fin=129.16 own= 41989 isl= 41989 cached=     0 cold conv=02bc0a turn=1 eng=151035
REQ  q= 119.37 ft= 121.87 fin=128.33 own= 34891 isl= 34891 cached=     0 cold conv=00ca01 turn=0 eng=151035
GAUGE t= 119.70    31013
GAUGE t= 123.70    31013
REQ  q= 125.98 ft= 127.74 fin=139.33 own= 34956 isl= 34956 cached=     0 cold conv=00ca01 turn=0 eng=151035
GAUGE t= 127.70    31013
REQ  q= 129.46 ft= 131.35 fin=183.47 own= 32518 isl= 67014 cached= 34496 warm conv=00ca01 turn=1 eng=151035

===== window 195-245s: requests (q, first_token, finished, own, isl, cached, kind, conv/turn) and gauge samples
REQ  q= 195.13 ft= 197.39 fin=203.12 own= 28672 isl= 53760 cached= 25088 cold conv=03e110 turn=1 eng=151035
REQ  q= 195.13 ft= 196.06 fin=197.79 own=   409 isl=   409 cached=     0 cold conv=04dba6 turn=0 eng=151035
REQ  q= 195.13 ft= 197.47 fin=220.39 own= 12053 isl= 65365 cached= 53312 warm conv=0470d4 turn=5 eng=151035
GAUGE t= 195.70        0
REQ  q= 197.88 ft= 200.36 fin=206.24 own= 47158 isl= 47158 cached=     0 cold conv=04dba6 turn=1 eng=151035
REQ  q= 199.41 ft= 200.40 fin=202.87 own=  1887 isl= 64607 cached= 62720 warm conv=03e110 turn=4 eng=151035
GAUGE t= 199.70    47158
GAUGE t= 203.70    47158
REQ  q= 204.03 ft= 204.09 fin=289.52 own=   585 isl= 64873 cached= 64288 warm conv=03e110 turn=5 eng=151035
REQ  q= 206.39 ft= 207.88 fin=265.89 own= 25083 isl= 78395 cached= 53312 warm conv=03e110 turn=2 eng=151035
GAUGE t= 207.70    47158
GAUGE t= 211.70        0
GAUGE t= 215.70        0
GAUGE t= 219.70        0
REQ  q= 221.37 ft= 222.31 fin=235.07 own= 15422 isl= 79710 cached= 64288 warm conv=0470d4 turn=6 eng=151035
GAUGE t= 223.70        0
REQ  q= 225.05 ft= 225.10 fin=225.31 own=   474 isl=   474 cached=     0 cold conv=05c249 turn=0 eng=151035
REQ  q= 225.42 ft= 228.82 fin=230.3 own= 64610 isl= 64610 cached=     0 cold conv=05c249 turn=1 eng=151035
GAUGE t= 227.70        0
REQ  q= 231.47 ft= 231.65 fin=236.56 own=  2251 isl= 66539 cached= 64288 warm conv=05c249 turn=2 eng=151035
GAUGE t= 231.70    64610
GAUGE t= 235.70    64610
REQ  q= 236.37 ft= 236.56 fin=242.32 own=  2279 isl= 80679 cached= 78400 warm conv=0470d4 turn=7 eng=151035
REQ  q= 237.66 ft= 237.76 fin=241.65 own=  1076 isl= 66932 cached= 65856 warm conv=05c249 turn=3 eng=151035
GAUGE t= 239.70        0
REQ  q= 243.10 ft= 243.38 fin=249.92 own=  3964 isl= 69820 cached= 65856 warm conv=05c249 turn=4 eng=151035
REQ  q= 243.57 ft= 243.66 fin=247.39 own=   847 isl= 80815 cached= 79968 warm conv=0470d4 turn=8 eng=151035
GAUGE t= 243.70        0
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
scrape interval: median 4.00 min 4.00 max 4.00
consecutive scrapes both nonzero: 149 identical: 118
  0-300: both nonzero 17, identical 15
  300-600: both nonzero 27, identical 21
  600-900: both nonzero 27, identical 22
  900-1200: both nonzero 30, identical 23
  1200-1500: both nonzero 24, identical 18
  1500-1860: both nonzero 24, identical 19
runs of identical nonzero values length>=2: 75
   start 19.7 value 67211 samples 3 (12s)
   start 51.7 value 64115 samples 2 (8s)
   start 79.7 value 88347 samples 3 (12s)
   start 99.7 value 8703 samples 3 (12s)
   start 111.7 value 105465 samples 2 (8s)
   start 119.7 value 31013 samples 3 (12s)
   start 139.7 value 54706 samples 3 (12s)
   start 199.7 value 47158 samples 3 (12s)
   start 231.7 value 64610 samples 2 (8s)
   start 379.7 value 2581 samples 3 (12s)
   start 399.7 value 17554 samples 3 (12s)
   start 411.7 value 90669 samples 2 (8s)
   start 419.7 value 1476 samples 3 (12s)
   start 439.7 value 52528 samples 3 (12s)
   start 451.7 value 37488 samples 2 (8s)
   start 491.7 value 89531 samples 2 (8s)
   start 519.7 value 7217 samples 3 (12s)
   start 531.7 value 50528 samples 2 (8s)
   start 539.7 value 73065 samples 3 (12s)
   start 559.7 value 97476 samples 3 (12s)
   start 579.7 value 3473 samples 3 (12s)
   start 591.7 value 50870 samples 2 (8s)
   start 619.7 value 62458 samples 3 (12s)
   start 651.7 value 35244 samples 2 (8s)
   start 679.7 value 61111 samples 3 (12s)
   start 699.7 value 108825 samples 3 (12s)
   start 711.7 value 98027 samples 2 (8s)
   start 719.7 value 14524 samples 3 (12s)
   start 731.7 value 126260 samples 2 (8s)
   start 779.7 value 108242 samples 3 (12s)
   start 799.7 value 43848 samples 3 (12s)
   start 831.7 value 64265 samples 2 (8s)
   start 851.7 value 3172 samples 2 (8s)
   start 859.7 value 69713 samples 3 (12s)
   start 879.7 value 4842 samples 3 (12s)
   start 891.7 value 108244 samples 2 (8s)
   start 919.7 value 61098 samples 3 (12s)
   start 931.7 value 1421 samples 2 (8s)
   start 939.7 value 1830 samples 3 (12s)
   start 959.7 value 12646 samples 3 (12s)
   start 971.7 value 16552 samples 2 (8s)
   start 991.7 value 3476 samples 2 (8s)
   start 1019.7 value 64493 samples 3 (12s)
   start 1059.7 value 190165 samples 3 (12s)
   start 1079.7 value 4725 samples 3 (12s)
   start 1091.7 value 81430 samples 2 (8s)
   start 1151.7 value 3542 samples 2 (8s)
   start 1159.7 value 1215 samples 3 (12s)
   start 1179.7 value 25439 samples 3 (12s)
   start 1191.7 value 27246 samples 2 (8s)
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
wrote /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/tmp_plot.png
  policy mean / engine mean where engine mean >= 1K: median 2.04, p10 0.72, p90 4.19 (n=448)
  time-average over the load phase: policy 17.9K, engine 8.2K
  published policy mean vs recomputed from the gauge: max abs diff 35.6K
```

</details>

<details><summary>Bash result</summary>

```text
208:def get_serve_request_id() -> str:
209-    """Get request id from serve request context."""
210-    context = serve.context._serve_request_context.get()
211-    if context is not None:
212-        return context.request_id
213-    return ""
214-
215-
216-def get_model_request_id(model: str):
217:    return model + "-" + get_serve_request_id()
218-
219-
220-def replace_prefix(model: str) -> str:
221-    """Replace -- with / in model name to handle slashes within the URL path segment"""
222-    return model.replace("--", "/")
#####
#####
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Compare the policy's published 60s mean with the trace rebuild and the raw exported gauge

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > pub_check.py <<'EOF'
import json, sys, bisect, math, statistics as st
R = sys.argv[1]; END = 1860.0
plot = json.load(open(f"{R}/analysis/plot_data.json")); t0 = plot["time_origin"]["profiling_start_ns"] / 1e9
S = json.load(open("gauge_decomp_state.json")); reqs = list(S["reqs"].values())
raw, pub, req = [], [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    rec = json.loads(line)
    if rec.get("error"): continue
    t = rec["scrape_started_at_ns"] / 1e9 - t0
    d = {}
    for s in rec["samples"]:
        if "LLMServer" not in s["labels"].get("deployment", ""): continue
        if s["name"] == "ray_serve_autoscaling_policy_metric": d[s["labels"]["metric"]] = s["value"]
    if "outstanding_effective_prefill_tokens" in d: raw.append((t, d["outstanding_effective_prefill_tokens"]))
    if "prefill_required_replicas" in d: pub.append((t, d["prefill_required_replicas"] * 20000))
    if "required_replicas" in d: req.append((t, d["required_replicas"]))
def step(d):
    d = sorted(d); times = [t for t, _ in d]; lv = 0; levels = []
    for _, x in d: lv += x; levels.append(lv)
    return times, levels
def at(times, levels, t):
    i = bisect.bisect_right(times, t) - 1
    return levels[i] if i >= 0 else 0
def win_mean(times, levels, t, w=60.0):
    a = t - w; i0 = bisect.bisect_right(times, a)
    pts = [(a, at(times, levels, a))] + [(times[i], levels[i]) for i in range(i0, bisect.bisect_right(times, t))]
    tot = 0.0
    for (t1, v1), (t2, _) in zip(pts, pts[1:] + [(t, None)]):
        if t2 > t1: tot += v1 * (t2 - t1)
    return tot / w
Rt, Rl = step([(r["q"], r["own"]) for r in reqs] + [(r["ft"], -r["own"]) for r in reqs])
pubL = [(t, v) for t, v in pub if 0 <= t <= END]
rawL = [(t, v) for t, v in raw if 0 <= t <= END]
print(f"published policy mean (what the policy divided by target): time-average over load {st.mean(v for _, v in pubL):.0f}")
print(f"raw exported gauge: sample average {st.mean(v for _, v in rawL):.0f}")
print(f"router-style rebuild from traces: 60s-mean averaged at the same times {st.mean(win_mean(Rt, Rl, t) for t, _ in pubL):.0f}")
# correlation/regression of published mean vs rebuild 60s-mean at shifts
def corr(x, y):
    mx, my = st.mean(x), st.mean(y); sx = math.sqrt(sum((a-mx)**2 for a in x)); sy = math.sqrt(sum((b-my)**2 for b in y))
    return sum((a-mx)*(b-my) for a, b in zip(x, y)) / (sx*sy)
print("\nshift (s): corr(published mean, rebuild 60s-mean shifted), slope through origin")
for sh in [0, 2, 4, 6, 8, 10, 12, 16, 20]:
    x = [win_mean(Rt, Rl, t - sh) for t, _ in pubL]; y = [v for _, v in pubL]
    print(f"  {sh:3d}: corr {corr(x, y):.3f}  slope {sum(a*b for a, b in zip(x, y))/sum(a*a for a in x):.2f}")
# raw gauge vs rebuild instantaneous at shifts (wide)
print("\nraw exported gauge vs instantaneous rebuild: corr at shift")
for sh in [0, 2, 4, 6, 8, 10, 12, 14]:
    x = [at(Rt, Rl, t - sh) for t, _ in rawL]; y = [v for _, v in rawL]
    print(f"  {sh:3d}: corr {corr(x, y):.3f}")
# How does the published mean relate to the raw gauge samples? recompute 60s mean from raw samples (step) and compare pointwise
Gt, Gl = step([(raw[0][0], raw[0][1])] + [(t, v - pv) for (t, v), (_, pv) in zip(raw[1:], raw)])
diffs = [(t, v, win_mean(Gt, Gl, t)) for t, v in pubL]
print(f"\npublished mean vs 60s mean of raw scrapes: mean abs diff {st.mean(abs(v-w) for _, v, w in diffs):.0f}, mean signed (pub - raw) {st.mean(v-w for _, v, w in diffs):.0f}")
for sh in [0, 4, 8, 12]:
    x=[win_mean(Gt, Gl, t - sh) for t,_ in pubL]; y=[v for _,v in pubL]
    print(f"  shift {sh}: corr {corr(x,y):.3f} slope {sum(a*b for a,b in zip(x,y))/sum(a*a for a in x):.2f}")
EOF
python3 -I pub_check.py /home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean
```

**Bash** — Read the metrics pusher, push path, and controller ingestion of router metrics

```bash
cd /mnt/local_storage/ray && grep -n "class MetricsPusher" -A60 python/ray/serve/_private/metrics_utils.py | grep -n "thread\|Thread\|async\|loop\|def \|sleep\|interval" | head -30; echo '#####'; sed -n 454,500p python/ray/serve/_private/router.py; echo '#####'; sed -n 600,660p python/ray/serve/_private/autoscaling_state.py; echo '#####'; sed -n 1075,1115p python/ray/serve/_private/autoscaling_state.py
```

**Bash** — Find when the controller evaluates the policy and exports its metrics

```bash
cd /mnt/local_storage/ray && grep -n "def get_decision_num_replicas\|def _get_raw_router_metrics\|policy(\|_policy(\|current_time\|SERVE_AUTOSCALING_POLICY_METRICS_KEY\|policy_metric" python/ray/serve/_private/autoscaling_state.py | head -30; grep -rn "autoscaling_policy_metric\|SERVE_AUTOSCALING_POLICY_METRICS_KEY" python/ray/serve/_private/controller.py python/ray/serve/_private/deployment_state.py python/ray/serve/_private/application_state.py 2>/dev/null | head; grep -rn "def _record_autoscaling_metrics\|run_control_loop\|autoscale(" python/ray/serve/_private/controller.py | head
```

**Bash** — Scan this run's controller log for warnings or slow loops

```bash
grep -v "^$" /tmp/ray/session_latest/logs/serve/controller_149281.log | cut -c1-220 | grep -iv "long_poll\|Deploying\|Adding\|Removing\|Starting\|Replica\|Stopping\|replica" | head -40; echo '#####'; grep -c "" /tmp/ray/session_latest/logs/serve/controller_149281.log; grep -i "warn\|error\|slow\|took" /tmp/ray/session_latest/logs/serve/controller_149281.log | cut -c1-250 | head -20
```

**Bash** — Decompose each distinct exported gauge state into engine uncached counts with a wide candidate window

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > states.py <<'EOF'
import json, random, collections
S = json.load(open("gauge_decomp_state.json")); reqs = list(S["reqs"].values()); G = [tuple(x) for x in S["G"]]
Gs = [(t, v) for t, v in G if 0 <= t <= 1860]
# distinct exported states (collapse runs)
states = []
for t, v in Gs:
    if v > 0 and (not states or states[-1][1] != v or t - states[-1][2] > 4.5): states.append([t, v, t])
    elif v > 0: states[-1][2] = t
print("distinct nonzero exported states:", len(states))
def subset(target, vals, tol=1):
    sums = {0: ()}
    for i, v in enumerate(vals):
        new = {}
        for s, sub in sums.items():
            ns = s + v
            if ns <= target + tol and ns not in sums and ns not in new: new[ns] = sub + (i,)
        sums.update(new)
    for s in range(target - tol, target + tol + 1):
        if s in sums: return sums[s]
    return None
def run(rs, label, back=14.0, fwd=1.0, maxc=18):
    tested = matched = 0; mism = []
    for t, v, t2 in states:
        c = [r for r in rs if r["q"] <= t + fwd and r["ft"] >= t - back]
        if not c or len(c) > maxc: continue
        tested += 1
        sub = subset(int(round(v)), [r["own"] for r in c])
        if sub is not None: matched += 1
        else: mism.append((t, int(v), c))
    print(f"{label}: states with <= {maxc} candidates in [t-{back:.0f}, t+{fwd:.0f}]: {tested}; exact subset sum of engine own_uncached: {matched} ({matched/max(1,tested)*100:.0f}%)")
    return mism
mism = run(reqs, "real")
random.seed(3); owns = [r["own"] for r in reqs]; random.shuffle(owns); run([dict(r, own=o) for r, o in zip(reqs, owns)], "control shuffled")
# also try explaining mismatches with isl (router saw no overlap) for any candidate
print("\nunexplained states: try subset sums allowing each candidate to count as own OR isl")
expl2 = 0
for t, v, c in mism:
    vals = []
    for r in c: vals.append(r["own"])
    # alternative: each request contributes own or isl: encode as own plus optional extra (isl-own)
    extras = [r["isl"] - r["own"] for r in c if r["isl"] > r["own"]]
    sub = subset(int(round(v)), [r["own"] for r in c] + extras)
    if sub is not None: expl2 += 1
print(f"  {expl2}/{len(mism)} explained when a warm request may count as its full prompt (isl)")
print("\nfirst 30 unexplained states (t, value, candidates as (own, isl, kind, q, ft)):")
for t, v, c in mism[:30]:
    print(f"  t={t:7.1f} value={v:7d}  " + "; ".join(f"({r['own']},{r['isl']},{r['kind'][0]},q{r['q']:.1f},ft{r['ft']:.1f})" for r in sorted(c, key=lambda r: r['q'])))
EOF
python3 -I states.py
```

<details><summary>Bash result</summary>

```text
published policy mean (what the policy divided by target): time-average over load 15399
raw exported gauge: sample average 17899
router-style rebuild from traces: 60s-mean averaged at the same times 13411

shift (s): corr(published mean, rebuild 60s-mean shifted), slope through origin
    0: corr 0.843  slope 1.11
    2: corr 0.860  slope 1.12
    4: corr 0.867  slope 1.12
    6: corr 0.870  slope 1.12
    8: corr 0.865  slope 1.12
   10: corr 0.854  slope 1.12
   12: corr 0.836  slope 1.11
   16: corr 0.795  slope 1.09
   20: corr 0.748  slope 1.07

raw exported gauge vs instantaneous rebuild: corr at shift
    0: corr 0.237
    2: corr 0.504
    4: corr 0.560
    6: corr 0.538
    8: corr 0.467
   10: corr 0.210
   12: corr 0.076
   14: corr -0.086

published mean vs 60s mean of raw scrapes: mean abs diff 7594, mean signed (pub - raw) -2500
  shift 0: corr 0.701 slope 0.75
  shift 4: corr 0.693 slope 0.75
  shift 8: corr 0.667 slope 0.74
  shift 12: corr 0.633 slope 0.73
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
2:42-    """Periodically runs registered asyncio tasks."""
4:44-    def __init__(
7:47-        async_sleep: Optional[Callable[[float], Coroutine[Any, Any, None]]] = None,
9:49-        self._async_sleep = async_sleep or asyncio.sleep
11:51-        self._async_tasks: Dict[str, asyncio.Task] = dict()
14:54-        # on the main thread but its methods called on a separate asyncio loop.
15:55-        self._stop_event: Optional[asyncio.Event] = None
18:58-    def stop_event(self) -> asyncio.Event:
20:60-            self._stop_event = asyncio.Event()
24:64-    def start(self):
27:67-    async def metrics_task(self, name: str):
28:68-        """Periodically runs `task_func` every `interval_s` until `stop_event` is set.
31:71-        Supports both sync and async task functions.
34:74-        wait_for_stop_event = asyncio.create_task(self.stop_event.wait())
42:82-                if asyncio.iscoroutinefunction(task_func):
49:89-            sleep_task = asyncio.create_task(
50:90-                self._async_sleep(self._tasks[name].interval_s)
52:92-            await asyncio.wait(
53:93-                [sleep_task, wait_for_stop_event],
54:94-                return_when=asyncio.FIRST_COMPLETED,
57:97-            if not sleep_task.done():
58:98-                sleep_task.cancel()
60:100-    def register_or_update_task(
#####
    def push_autoscaling_metrics_to_controller(self):
        """Pushes queued and running request metrics to the controller.

        These metrics are used by the controller for autoscaling.
        If a previous push is already in flight, skips this push (will try again next interval).
        """
        with self._metrics_push_lock:
            if self._pending_metrics_push_ref is not None:
                if not check_obj_ref_ready_nowait(self._pending_metrics_push_ref):
                    return  # Previous push still in flight, skip and try again later
            # Handle reports carry every replica this handle routes to, so they
            # are the wide ones worth encoding columnar (see should_encode_columnar).
            # The flag is the kill switch: the controller reads either format, so
            # flipping it back needs a restart, not a redeploy.
            report = self._get_metrics_report()
            payload = (
                autoscaling_metrics_codec.encode(report)
                if RAY_SERVE_COLUMNAR_AUTOSCALING_METRICS
                else compress_metric_report(report)
            )
            self._pending_metrics_push_ref = (
                self._controller_handle.record_autoscaling_metrics_from_handle.remote(
                    payload
                )
            )

    def _add_autoscaling_metrics_point(self):
        """Adds metrics point for queued and running requests at replicas.

        Also prunes keys in the in memory metrics store with outdated datapoints.

        ┌─────────────────────────────────────────────────────────────────┐
        │  Handle-based metrics collection                
... [2534 chars trimmed] ...
 queued=MetricSamples(payload["q_ts"], payload["q_val"]),
                running=running,
                running_keys=running_keys,
                router_metrics=_router_metric_samples(payload),
            )
        )

    def drop_stale_handle_metrics(self, alive_serve_actor_ids: Set[str]) -> None:
        """Drops handle metrics that are no longer valid.

        This includes handles that live on Serve Proxy or replica actors
        that have died AND handles from which the controller hasn't
        received an update for too long.
        """

        timeout_s = max(
            2 * self._config.metrics_interval_s,
            RAY_SERVE_MIN_HANDLE_METRICS_TIMEOUT_S,
        )
        now_mono = time.monotonic()
        # Both stores hold reports with the same drop-relevant shape, so one loop. A
        # handle on a dead proxy/replica actor goes immediately; otherwise it goes when
        # the controller has not received an update for timeout_s. Age is measured from
        # controller-local receive time so producer/controller clock skew cannot drop
        # fresh reports early or retain phantoms past the timeout.
        for report in self._handle_store.all_reports():
            dead_actor = (
                report.is_serve_component_source
#####
                # Extract values from TimeStampedValue list
                raw_metrics[metric_name][replica_id] = timeseries

        return dict(raw_metrics)

    def _get_raw_router_metrics(self) -> Dict[str, Dict[str, TimeSeries]]:
        """Extract router-owned metric series, keyed by metric and handle ID."""
        raw_metrics: Dict[str, Dict[str, TimeSeries]] = defaultdict(dict)

        for handle_id, object_report in self._handle_store.objects.items():
            router_metrics = getattr(
                object_report,
                "router_metrics",
                {QUEUED_REQUESTS_KEY: object_report.queued_requests},
            )
            for metric_name, timeseries in router_metrics.items():
                raw_metrics[metric_name][handle_id] = timeseries

        for handle_id, columnar_report in self._handle_store.columnar.items():
            raw_metrics[QUEUED_REQUESTS_KEY][handle_id] = [
                TimeStampedValue(float(timestamp), float(value))
                for timestamp, value in zip(
                    columnar_report.queued.timestamps,
                    columnar_report.queued.values,
                )
            ]
            for metric_name, samples in columnar_report.router_metrics.items():
                raw_metrics[metric_name][handle_id] = [
                    TimeStampedValue(float(timestamp), float(value))
                    for timestamp, value in zip(samples.timestamps, samples.values)
                ]

        return dict(raw_metrics)


class ApplicationAutoscalingState:
    """Manages autoscaling for a single application."""

    def __init__(
        self,
        app_name: ApplicationName,
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
39:    SERVE_AUTOSCALING_POLICY_METRICS_KEY,
74:    raw = policy.get_policy()
384:        self.autoscaling_policy_metric_gauge = metrics.Gauge(
385:            "serve_autoscaling_policy_metric",
388:                f"'{SERVE_AUTOSCALING_POLICY_METRICS_KEY}' policy state key, "
677:        policy_metrics: Optional[Dict[str, Any]] = None,
693:        for name, value in (policy_metrics or {}).items():
700:                self.autoscaling_policy_metric_gauge.set(
704:    def get_decision_num_replicas(
721:        decision_num_replicas, self._policy_state = self._policy(autoscaling_context)
732:            (self._policy_state or {}).get(SERVE_AUTOSCALING_POLICY_METRICS_KEY),
763:            current_time=time.time(),
1080:    def _get_raw_router_metrics(self) -> Dict[str, Dict[str, TimeSeries]]:
1164:    def has_policy(self) -> bool:
1188:            and self.has_policy()
1231:    def get_decision_num_replicas(
1240:        if self.has_policy():
1256:            # `self._policy` is non-None here (guarded by `has_policy()` above).
1257:            decisions, returned_policy_state = self._policy(  # type: ignore[misc]
1291:                    .get(SERVE_AUTOSCALING_POLICY_METRICS_KEY),
1453:    def _application_has_policy(self, app_name: ApplicationName) -> bool:
1456:            and self._app_autoscaling_states[app_name].has_policy()
1459:    def get_decision_num_replicas(
python/ray/serve/_private/controller.py:328:        run_background_task(self.run_control_loop())
python/ray/serve/_private/controller.py:626:    async def run_control_loop(self) -> None:
python/ray/serve/_private/controller.py:636:                await self.run_control_loop_step(
python/ray/serve/_private/controller.py:668:    async def run_control_loop_step(
python/ray/serve/_private/controller.py:2037:        run_control_loop().
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
INFO 2026-11-02 12:38:18,642 controller 149281 -- HAProxy is enabled in ServeController, replacing Serve proxy with HAProxy.
INFO 2026-11-02 12:38:20,531 controller 149281 -- Raising graceful_shutdown_timeout_s for ingress deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' from 20.0s to 35.0s so the force-kill deadline covers the direct-ingress drai
INFO 2026-11-02 12:38:20,531 controller 149281 -- Application 'ttft-benchmark' has ingress request router deployment 'LLMRouter' configured.
INFO 2026-11-02 12:38:20,629 controller 149281 -- Registering autoscaling state for deployment Deployment(name='LLMServer:Qwen--Qwen3_6-27B-FP8', app='ttft-benchmark')
INFO 2026-11-02 12:38:20,630 controller 149281 -- Using custom autoscaling policy 'ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy' for deployment 'Deployment(name='LLMServer:Qwen--Qwen3_6-2
INFO 2026-11-02 12:38:20,632 controller 149281 -- Deregistering autoscaling state for deployment Deployment(name='LLMRouter', app='ttft-benchmark')
WARNING 2026-11-02 12:38:20,632 controller 149281 -- Cannot deregister autoscaling state for deployment Deployment(name='LLMRouter', app='ttft-benchmark') because it is not registered
INFO 2026-11-02 12:38:43,179 controller 149281 -- Creating node manager for node 33c8be3ae1e891d03c293636681cadf5daaff6dd64aa71a95bc02323
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
This may be caused by a slow __init__ or reconfigure method.
INFO 2026-11-02 13:16:36,904 controller 149281 -- Controller shutdown started.
INFO 2026-11-02 13:16:36,905 controller 149281 -- Deleting app 'ttft-benchmark'.
INFO 2026-11-02 13:16:36,905 controller 149281 -- Deleting Deployment(name='LLMRouter', app='ttft-benchmark')
INFO 2026-11-02 13:16:39,092 controller 149281 -- Deregistering autoscaling state for deployment Deployment(name='LLMRouter', app='ttft-benchmark')
WARNING 2026-11-02 13:16:39,092 controller 149281 -- Cannot deregister autoscaling state for deployment Deployment(name='LLMRouter', app='ttft-benchmark') because it is not registered
INFO 2026-11-02 13:16:39,195 controller 149281 -- Deleting Deployment(name='LLMServer:Qwen--Qwen3_6-27B-FP8', app='ttft-benchmark')
WARNING 2026-11-02 13:16:46,966 controller 149281 -- application not yet shutdown
WARNING 2026-11-02 13:16:46,967 controller 149281 -- deployment not yet shutdown
WARNING 2026-11-02 13:16:47,070 controller 149281 -- application not yet shutdown
WARNING 2026-11-02 13:16:47,070 controller 149281 -- deployment not yet shutdown
WARNING 2026-11-02 13:16:47,174 controller 14
... [969 chars trimmed] ...
roller 149281 -- application not yet shutdown
#####
512
WARNING 2026-11-02 12:38:20,632 controller 149281 -- Cannot deregister autoscaling state for deployment Deployment(name='LLMRouter', app='ttft-benchmark') because it is not registered
INFO 2026-11-02 12:38:43,551 controller 149281 -- Replica(id='3jybiw18', deployment='LLMRouter', app='ttft-benchmark') started successfully on node '33c8be3ae1e891d03c293636681cadf5daaff6dd64aa71a95bc02323' after 22.8s (PID: 149639). Replica construc
WARNING 2026-11-02 12:38:50,835 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
WARNING 2026-11-02 12:39:20,870 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
WARNING 2026-11-02 12:39:50,942 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
INFO 2026-11-02 12:40:03,877 controller 149281 -- Replica(id='2age6oh7', deployment='LLMServer:Qwen--Qwen3_6-27B-FP8', app='ttft-benchmark') started successfully on node '33c8be3ae1e891d03c293636681cadf5daaff6dd64aa71a95bc02323' after 103.1s (PID: 14
WARNING 2026-11-02 12:50:37,740 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
WARNING 2026-11-02 12:51:07,758 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
WARNING 2026-11-02 12:51:37,768 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
INFO 2026-11-02 12:51:46,809 controller 149281 -- Replica(id='nx9jlpa8', deployment='LLMServer:Qwen--Qwen3_6-27B-FP8', app='ttft-benchmark') started successfully on node '33c8be3ae1e891d03c293636681cadf5daaff6dd64aa71a95bc02323' after 99.1s (PID: 172
WARNING 2026-11-02 12:54:33,879 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
WARNING 2026-11-02 12:55:03,898 controller 149281 -- Deployment 'LLMServer:Qwen--Qwen3_6-27B-FP8' in application 'ttft-benchmark' has 1 replicas that have taken more than 30s to initialize.
This may be caused by a slow __init__ or reconfigure method.
```

</details>

<details><summary>Bash result</summary>

```text
distinct nonzero exported states: 75
real: states with <= 18 candidates in [t-14, t+1]: 56; exact subset sum of engine own_uncached: 46 (82%)
control shuffled: states with <= 18 candidates in [t-14, t+1]: 56; exact subset sum of engine own_uncached: 11 (20%)

unexplained states: try subset sums allowing each candidate to count as own OR isl
  4/10 explained when a warm request may count as its full prompt (isl)

first 30 unexplained states (t, value, candidates as (own, isl, kind, q, ft)):
  t=  119.7 value=  31013  (67700,155508,w,q104.5,ft110.5); (37765,37765,c,q105.5,ft111.1); (41989,41989,c,q118.1,ft120.8); (34891,34891,c,q119.4,ft121.9)
  t=  411.7 value=  90669  (17554,17554,c,q397.6,ft398.4); (507,17755,w,q402.7,ft402.8); (2412,107468,w,q403.9,ft404.2); (2177,57057,w,q404.3,ft404.5); (88278,88278,c,q405.2,ft410.8); (645,17893,w,q407.6,ft409.4); (17426,17426,c,q408.5,ft410.9)
  t=  491.7 value=  89531  (472,56920,w,q481.0,ft481.1); (2864,56176,w,q481.9,ft482.1); (3186,45522,w,q482.0,ft482.3); (1687,69111,w,q483.1,ft483.3); (7130,63578,w,q485.7,ft486.1); (76845,76845,c,q486.1,ft491.6); (2531,160899,w,q486.1,ft491.6); (18679,103351,w,q486.6,ft491.7); (2452,57332,w,q486.6,ft491.7)
  t=  699.7 value= 108825  (2485,104405,w,q686.0,ft686.3); (10711,45207,w,q686.3,ft686.9); (2430,83966,w,q688.7,ft688.9); (2366,118398,w,q688.9,ft689.7); (10614,70198,w,q688.9,ft689.8); (1084,44988,w,q689.7,ft689.8); (2566,46470,w,q694.3,ft694.4); (2357,102709,w,q695.1,ft695.4); (1704,119304,w,q695.6,ft695.8); (1447,84551,w,q696.2,ft696.4); (15052,58956,w,q697.3,ft698.1); (5337,108825,w,q698.1,ft698.5)
  t=  731.7 value= 126260  (14524,63132,w,q717.9,ft718.8); (2269,86941,w,q719.8,ft720.0); (4375,123543,w,q721.8,ft722.1); (2596,65316,w,q722.2,ft722.4); (4985,94361,w,q722.4,ft722.8); (3642,103994,w,q722.7,ft723.0); (1496,48536,w,q724.4,ft724.5); (118171,118171,c,q726.1,ft732.8); (2122,105610,w,q727.0,ft727.2); (8718,73006,w,q727.1,ft727.7); (29968,127184,w,q727.7,ft730.0); (1829,120997,w,q728.8,ft730.1); (2948,89188,w,q728.8,ft730.1)
  t= 1019.7 value=  64493  (2717,48189,w,q1006.0,ft1006.2); (6675,141523,w,q1006.3,ft1006.9); (1051,33979,w,q1006.4,ft1006.4); (7074,170146,w,q1006.8,ft1007.5); (7690,79818,w,q1008.7,ft1009.2); (2874,46778,w,q1011.5,ft1011.7); (1381,34309,w,q1012.1,ft1012.2); (3335,50375,w,q1012.4,ft1012.6); (1116,34044,w,q1012.6,ft1012.7); (280,280,c,q1013.5,ft1013.6); (4076,37004,w,q1014.9,ft1015.2); (12743,70759,w,q1015.4,ft1016.1); (2134,66422,w,q1016.1,ft1016.3); (1379,34307,w,q1017.8,ft1017.9); (3446,39510,w,q1019.1,ft1019.3); (1442,34370,w,q1019.3,ft1019.4); (2027,34955,w,q1020.4,ft1020.6)
  t= 1179.7 value=  25439  (6261,83093,w,q1166.8,ft1167.2); (2066,105554,w,q1168.5,ft1168.7); (896,176512,w,q1169.8,ft1170.0); (1591,123895,w,q1170.8,ft1171.0); (1383,97031,w,q1171.1,ft1171.2); (1903,97551,w,q1173.6,ft1173.8); (1035,176651,w,q1174.9,ft1175.0); (4849,92657,w,q1175.2,ft1175.6); (3261,108317,w,q1177.0,ft1177.3); (663,97879,w,q1178.2,ft1178.3); (351,124223,w,q1178.4,ft1178.5); (2248,53992,w,q1180.7,ft1180.8)
  t= 1191.7 value=  27246  (663,97879,w,q1178.2,ft1178.3); (351,124223,w,q1178.4,ft1178.5); (2248,53992,w,q1180.7,ft1180.8); (8884,109236,w,q1180.9,ft1181.5); (6937,88473,w,q1181.2,ft1181.6); (17294,109806,w,q1181.6,ft1182.8); (1637,125509,w,q1182.3,ft1182.5); (3151,123887,w,q1182.5,ft1182.8); (2525,58973,w,q1184.1,ft1184.3); (1607,123911,w,q1185.4,ft1185.6); (42953,42953,c,q1186.0,ft1188.2); (590,126030,w,q1188.3,ft1188.4); (1630,98846,w,q1188.9,ft1189.1); (3110,100326,w,q1189.6,ft1189.8)
  t= 1451.7 value= 116407  (1295,95375,w,q1438.3,ft1438.4); (1968,75664,w,q1440.0,ft1440.2); (2228,135508,w,q1441.9,ft1442.2); (5199,67919,w,q1442.1,ft1442.5); (908,73036,w,q1443.2,ft1443.2); (1586,148978,w,q1443.5,ft1443.7); (2261,74389,w,q1446.1,ft1446.3); (5081,139929,w,q1447.2,ft1447.7); (11599,11599,c,q1447.6,ft1448.2); (139927,139927,c,q1447.7,ft1456.0); (2788,59236,w,q1448.1,ft1448.3); (791,110551,w,q1450.8,ft1450.9); (474,474,c,q1451.9,ft1451.9); (70396,70396,c,q1452.2,ft1455.9)
  t= 1719.7 value= 150271  (5075,39571,w,q1707.0,ft1707.3); (5268,94644,w,q1707.6,ft1708.0); (706,63426,w,q1707.9,ft1708.0); (4521,101737,w,q1708.8,ft1709.1); (1341,40541,w,q1710.0,ft1710.1); (3513,97593,w,q1710.6,ft1710.8); (2736,107792,w,q1710.7,ft1711.0); (1067,33995,w,q1711.5,ft1711.6); (21438,60638,w,q1712.7,ft1713.9); (84835,84835,c,q1715.3,ft1720.3); (5334,111958,w,q1716.6,ft1717.0); (1968,34896,w,q1716.7,ft1716.8); (5852,164220,w,q1716.8,ft1720.3); (5027,64611,w,q1716.9,ft1717.2); (1880,36376,w,q1720.4,ft1720.6)
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Check whether the published mean also repeats in runs and find the Ray metrics report interval

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json,statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']))
pubL=[(t,v) for t,v in pub if 0<=t<=1860 and v>0]
pairs=list(zip(pubL,pubL[1:]))
print('published mean: consecutive nonzero scrapes',len(pairs),'identical',sum(a[1]==b[1] for a,b in pairs))
runs=[];cur=None
for t,v in pubL:
    if cur and cur[1]==v: cur[2]+=1
    else:
        if cur: runs.append(cur)
        cur=[t,v,1]
runs.append(cur)
import collections
print('run lengths of identical published values:',collections.Counter(r[2] for r in runs))
EOF
grep -rho "metrics_report_interval_ms[^,}]*" /tmp/ray/session_latest/logs/raylet.out /tmp/ray/session_latest/logs/dashboard_agent.log /tmp/ray/session_latest/logs/*.err 2>/dev/null | sort | uniq -c | head; grep -rn "metrics_report_interval_ms" /home/ray/anaconda3/lib/python3.12/site-packages/ray/_private/ray_constants.py /mnt/local_storage/ray/src/ray/common/ray_config_def.h 2>/dev/null | head -5; env | grep -i "RAY_metrics" 
```

**Bash** — Regress the policy's published mean on cold and warm trace rebuilds

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > reg.py <<'EOF'
import json, bisect, math, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']*20000))
pubL=[(t,v) for t,v in pub if 0<=t<=1860]
def step(d):
    d=sorted(d); times=[t for t,_ in d]; lv=0; levels=[]
    for _,x in d: lv+=x; levels.append(lv)
    return times,levels
def at(times,levels,t):
    i=bisect.bisect_right(times,t)-1; return levels[i] if i>=0 else 0
def wmean(times,levels,t,w=60.0):
    a=t-w; i0=bisect.bisect_right(times,a)
    pts=[(a,at(times,levels,a))]+[(times[i],levels[i]) for i in range(i0,bisect.bisect_right(times,t))]
    tot=0.0
    for (t1,v1),(t2,_) in zip(pts,pts[1:]+[(t,None)]):
        if t2>t1: tot+=v1*(t2-t1)
    return tot/w
def rb(sel,lag=0.0): return step([(r['q'],r['own']) for r in sel]+[(r['ft']+lag,-r['own']) for r in sel])
cold=[r for r in reqs if r['kind']=='cold']; warm=[r for r in reqs if r['kind']=='warm']
y=[v for _,v in pubL]
for lag in (0.0,0.13):
    C=rb(cold,lag); W=rb(warm,lag); A=rb(reqs,lag)
    for sh in (6,):
        xc=[wmean(*C,t-sh) for t,_ in pubL]; xw=[wmean(*W,t-sh) for t,_ in pubL]; xa=[wmean(*A,t-sh) for t,_ in pubL]
        s11=sum(a*a for a in xc); s22=sum(b*b for b in xw); s12=sum(a*b for a,b in zip(xc,xw)); s1y=sum(a*v for a,v in zip(xc,y)); s2y=sum(b*v for b,v in zip(xw,y))
        det=s11*s22-s12*s12; ac=(s1y*s22-s2y*s12)/det; aw=(s11*s2y-s12*s1y)/det
        resid=[v-(ac*a+aw*b) for a,b,v in zip(xc,xw,y)]
        print(f"lag {lag*1000:.0f} ms, shift {sh}s: published ~ {ac:.2f} x cold + {aw:.2f} x warm (rmse {math.sqrt(st.mean(r*r for r in resid)):.0f}); single slope {sum(a*v for a,v in zip(xa,y))/sum(a*a for a in xa):.2f}; means: pub {st.mean(y):.0f} cold {st.mean(xc):.0f} warm {st.mean(xw):.0f} all {st.mean(xa):.0f}")
# also with intercept, single slope
A=rb(reqs,0.13); xa=[wmean(*A,t-6) for t,_ in pubL]
mx,my=st.mean(xa),st.mean(y); b=sum((a-mx)*(v-my) for a,v in zip(xa,y))/sum((a-mx)**2 for a in xa); print(f"with intercept: slope {b:.2f} intercept {my-b*mx:.0f}")
EOF
python3 -I reg.py
```

**Bash** — Identify warm turns the router counted at full prompt length and their context

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
conv={}
for rid,r in reqs.items():
    a=recs.get(aid(rid)); 
    if a: conv.setdefault(a['conversation_id'],[]).append((a['turn_index'],rid,r))
# find warm requests whose isl exactly equals an exported state value, or state - subset of own
G=[tuple(x) for x in S['G']]; vals={int(v) for t,v in G if 0<=t<=1860 and v>0}
hits=[]
for rid,r in reqs.items():
    if r['kind']=='warm' and int(r['isl']) in vals:
        a=recs.get(aid(rid),{})
        prev=[x for x in conv.get(a.get('conversation_id'),[]) if x[0]==a.get('turn_index',0)-1]
        prev_eng=prev[0][2]['eng'][-6:] if prev else None; prev_fin=prev[0][2]['fin'] if prev else None
        hits.append((round(r['q'],1),r['isl'],r['own'],r['cached'],r['eng'][-6:],a.get('turn_index'),prev_eng,None if prev_fin is None else round(r['q']-prev_fin,2)))
print("warm requests whose FULL prompt length appears verbatim as an exported gauge state (q, isl, own, cached, engine, turn, prev-turn engine, gap since prev turn finished s):")
for h in sorted(hits): print("  ",h)
# how many warm requests exist in total vs. these
print("total warm", sum(r['kind']=='warm' for r in reqs.values()))
# replica (engine) first-seen times
import collections
first={}
for r in reqs.values(): first[r['eng'][-6:]]=min(first.get(r['eng'][-6:],1e9),r['q'])
print("engine first request time:",first)
EOF
```

<details><summary>Bash result</summary>

```text
published mean: consecutive nonzero scrapes 462 identical 277
run lengths of identical published values: Counter({2: 93, 3: 92, 1: 1})
/mnt/local_storage/ray/src/ray/common/ray_config_def.h:833:RAY_CONFIG(uint64_t, metrics_report_interval_ms, 10000)
ANYSCALE_RAY_METRICS_ENDPOINT=http://localhost:8085
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
lag 0 ms, shift 6s: published ~ 1.04 x cold + 1.42 x warm (rmse 5397); single slope 1.12; means: pub 15399 cold 9834 warm 3574 all 13409
lag 130 ms, shift 6s: published ~ 1.00 x cold + 1.29 x warm (rmse 5390); single slope 1.07; means: pub 15399 cold 10163 warm 4013 all 14177
with intercept: slope 1.04 intercept 660
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
warm requests whose FULL prompt length appears verbatim as an exported gauge state (q, isl, own, cached, engine, turn, prev-turn engine, gap since prev turn finished s):
   (698.1, 108825, 5337, 103488, '151035', 8, '151035', 4.62)
   (1733.9, 34187, 1259, 32928, '180495', 0, None, None)
total warm 1443
engine first request time: {'151035': 0.11082577705383301, '173074': 705.1134192943573, '180495': 961.0731387138367}
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Check whether the router's over-count is concentrated in time and list full-prompt miscounts

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > timeres.py <<'EOF'
import json, bisect, math, statistics as st, random
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values()); G=[tuple(x) for x in S['G']]
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']*20000))
def step(d):
    d=sorted(d); times=[t for t,_ in d]; lv=0; levels=[]
    for _,x in d: lv+=x; levels.append(lv)
    return times,levels
def at(times,levels,t):
    i=bisect.bisect_right(times,t)-1; return levels[i] if i>=0 else 0
def wmean(times,levels,t,w=60.0):
    a=t-w; i0=bisect.bisect_right(times,a)
    pts=[(a,at(times,levels,a))]+[(times[i],levels[i]) for i in range(i0,bisect.bisect_right(times,t))]
    tot=0.0
    for (t1,v1),(t2,_) in zip(pts,pts[1:]+[(t,None)]):
        if t2>t1: tot+=v1*(t2-t1)
    return tot/w
A=step([(r['q'],r['own']) for r in reqs]+[(r['ft']+0.13,-r['own']) for r in reqs])
print("window: mean published policy mean vs mean 60s-mean of rebuild (engine own_uncached, +130ms hold, shifted 6s), ratio")
for lo,hi in [(60,300),(300,600),(600,700),(700,800),(800,940),(940,1040),(1040,1200),(1200,1400),(1400,1600),(1600,1860)]:
    P=[v for t,v in pub if lo<=t<hi]; X=[wmean(*A,t-6) for t,v in pub if lo<=t<hi]
    print(f"  {lo:4d}-{hi:4d}: pub {st.mean(P):6.0f}  rebuild {st.mean(X):6.0f}  diff {st.mean(P)-st.mean(X):6.0f}  ratio {st.mean(P)/st.mean(X):.2f}")
# states explained only by counting a warm request at full isl
Gs=[(t,v) for t,v in G if 0<=t<=1860]
states=[]
for t,v in Gs:
    if v>0 and (not states or states[-1][1]!=v or t-states[-1][2]>4.5): states.append([t,v,t])
    elif v>0: states[-1][2]=t
def subset(target, vals, tol=1):
    sums={0:()}
    for i,v in enumerate(vals):
        new={}
        for s,sub in sums.items():
            ns=s+v
            if ns<=target+tol and ns not in sums and ns not in new: new[ns]=sub+(i,)
        sums.update(new)
    for s in range(target-tol,target+tol+1):
        if s in sums: return sums[s]
    return None
print("\nstates not explained by engine own_uncached but explained when some request counts at full prompt:")
for t,v,t2 in states:
    c=[r for r in reqs if r['q']<=t+1 and r['ft']>=t-14]
    if not c or len(c)>18: continue
    if subset(int(v),[r['own'] for r in c]) is not None: continue
    vals=[r['own'] for r in c]+[r['isl']-r['own'] for r in c]
    sub=subset(int(v),vals)
    if sub is None: continue
    n=len(c); full=[c[i-n] for i in sub if i>=n]; owns=[c[i] for i in sub if i<n]
    print(f"  t={t:.1f} value={int(v)}: full-prompt counted: {[(r['isl'],r['cached'],r['kind'],round(r['q'],1),r['eng'][-6:]) for r in full]}; plus own of {[r['own'] for r in owns]}")
EOF
python3 -I timeres.py
```

**Bash** — Read the selection service API docs for how effective prefill tokens are defined

```bash
P=/home/ray/anaconda3/lib/python3.12/site-packages/dynamo; sed -n 772,900p $P/_core.pyi | grep -v "^\s*$" | head -120; echo '#####'; ls $P/select_service; grep -rn "effective_prefill\|overlap" $P/select_service/*.py 2>/dev/null | head -20
```

**Bash** — Check worker registration in the tracker and where vLLM emits block-stored or cache-cleared events

```bash
cd /mnt/local_storage/ray && sed -n 456,555p python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py; echo '#####'; V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))"); grep -rn "AllBlocksCleared\|BlockStored(" $V/v1/core/*.py $V/v1/core/block_pool.py $V/distributed/kv_events.py 2>/dev/null | head -12; grep -n "reset_prefix_cache" $V/v1/core/sched/scheduler.py $V/v1/engine/core.py 2>/dev/null | head
```

<details><summary>Bash result</summary>

```text
window: mean published policy mean vs mean 60s-mean of rebuild (engine own_uncached, +130ms hold, shifted 6s), ratio
    60- 300: pub  11903  rebuild  12108  diff   -205  ratio 0.98
   300- 600: pub  12280  rebuild  13592  diff  -1312  ratio 0.90
   600- 700: pub  14715  rebuild  14410  diff    304  ratio 1.02
   700- 800: pub  34930  rebuild  30091  diff   4839  ratio 1.16
   800- 940: pub  31887  rebuild  26157  diff   5730  ratio 1.22
   940-1040: pub  11955  rebuild  17888  diff  -5933  ratio 0.67
  1040-1200: pub  14613  rebuild  12141  diff   2472  ratio 1.20
  1200-1400: pub  19617  rebuild  11793  diff   7824  ratio 1.66
  1400-1600: pub  10414  rebuild  12003  diff  -1589  ratio 0.87
  1600-1860: pub  10766  rebuild   9237  diff   1529  ratio 1.17

states not explained by engine own_uncached but explained when some request counts at full prompt:
  t=699.7 value=108825: full-prompt counted: [(70198, 59584, 'warm', 688.9, '151035')]; plus own of [10711, 2430, 10614, 1084, 2566, 1447, 15052, 5337]
  t=731.7 value=126260: full-prompt counted: [(63132, 48608, 'warm', 717.9, '151035'), (65316, 62720, 'warm', 722.2, '151035')]; plus own of [2596, 1496, 2122, 8718]
  t=1019.7 value=64493: full-prompt counted: [(48189, 45472, 'warm', 1006.0, '180495')]; plus own of [7074, 2874, 1381, 3335, 280, 4076]
  t=1719.7 value=150271: full-prompt counted: [(94644, 89376, 'warm', 1707.6, '180495'), (40541, 39200, 'warm', 1710.0, '180495')]; plus own of [5852, 5334, 3513, 1968, 5027]
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
class SelectionService:
    """
    In-process handle to a runtime-free Dynamo selection core.
    """
    def __init__(
        self,
        *,
        indexer_threads: int = 4,
        indexer_peers: Optional[list[str]] = None,
        replica_sync_port: Optional[int] = None,
        replica_sync_peers: Optional[list[str]] = None,
        selection_cache: Optional[SelectionCacheConfig] = None,
        session_affinity_ttl_secs: Optional[float] = None,
        session_affinity_mode: Literal["hard", "soft"] = "hard",
        session_replication_writer_id: Optional[int] = None,
    ) -> None:
        """Create a selection service. `indexer_threads` sizes the KV indexer pool.
        `session_affinity_ttl_secs` enables session affinity with an idle TTL
        between 1 and 31536000 seconds.
        `session_affinity_mode`: hard rejects dispatch away from a live binding;
        soft rebinds when the bound worker is excluded from the request's candidates.
        `session_replication_writer_id` exports affinity updates for a host transport;
        use an id unique to this replica incarnation.
        """
        ...
    def shutdown(self) -> None:
        """
        Stop the service: cancel KV-event listeners and scheduling so that
        in-flight and queued selections fail fast.
        The KV indexer thread pool is released when the handle is dropped.
        Idempotent, and also runs automatically on drop.
        """
        ...
    async def shutdown_async(self) -> None:
        """Await service shutdown without blocking the Python event loop."""
        ...
    def session_binding_updates(self) -> AsyncIterator[List[JsonLike]]:
        """Stream session bindings this service commits or refreshes, for the
        host to broadcast to its peers.
        Each item is every binding queued since the last one (at least one).
        Requires session_replication_writer_id. Ends on shutdown.
        """
        ...
    def apply_session_binding(self, update: JsonLike) -> None:
        """Merge a versioned peer binding without echoing it.
        A binding this replica cannot place yet is dropped; the owner's next
        refresh repairs it.
        """
        ...
    async def upsert_worker(self, worker: JsonLike) -> JsonLike:
        """Upsert a worker and subscribe to its live KV events; returns its catalog record."""
        ...
    async def delete_worker(self, worker_id: int) -> JsonLike:
        """Remove a worker and tear down its KV-event listener; returns its catalog record."""
        ...
    def list_workers(
        self, *, model_name: Optional[str] = None, routing_group: Optional[str] = None
    ) -> JsonLike:
        """List catalog records, optionally filtered by model and routing group."""
        ...
    def ready(self) -> JsonLike:
        """Readiness: whether at least one worker is schedulable, plus catalog state."""
        ...
    async def overlap_scores(self, request: JsonLike) -> JsonLike:
        """Per-worker KV-overlap scores for a prompt."""
        ...
    async def select(self, request: JsonLike) -> JsonLike:
        """Select the best worker by KV-overlap + load, without booking."""
        ...
    async def select_and_reserve(self, request: JsonLike) -> JsonLike:
        """Select the best worker and book its load."""
        ...
    async def create_reservation(self, request: JsonLike) -> JsonLike:
        """Book a request's load against a worker, keyed by ``selection_id``.
        Without a ``worker_id``, replays the matching ``select``'s cached
        selection (same model/routing-group), booked under ``selection_id``;
        other request fields are ignored. With a ``worker_id`` and the prompt,
        books explicitly under ``selection_id`` on that worker and discards any
        cached selection for the id. ``selection_id`` is required.
        """
        ...
    async def prefill_complete(self, selection_id: str) -> None:
        """Mark a reservation's prefill complete; its load shifts prefill -> decode."""
        ...
    def add_output_block(
        self, selection_id: str, *, decay_fraction: Optional[float] = None
    ) -> None:
        """Record one decode output block for a reservation, advancing its decode load."""
        ...
    async def free_reservation(self, selection_id: str) -> None:
        """Free a finished reservation, releasing its tracked load."""
        ...
    def loads(
        self, *, model_name: Optional[str] = None, routing_group: Optional[str] = None
    ) -> JsonLike:
        """Current per-model active load (pending counts + per-worker potential loads)."""
        ...
    async def potential_loads(self, request: JsonLike) -> JsonLike:
        """Per-worker potential loads for a prompt, without booking."""
        ...
class ModelDeploymentCard:
    """
    A model deployment card is a collection of model information
    """
#####
__init__.py
__main__.py
__pycache__
main.py
```

</details>

<details><summary>Bash result</summary>

```text
def _on_deployment_targets(self, target_info: DeploymentTargetInfo) -> None:
        """LongPoll listener: reconcile tracked workers against the running-replica
        snapshot.

        Each replica advertises its KV-events endpoint via ``record_routing_stats``
        (carried in ``RunningReplicaInfo.routing_stats``); newly advertised replicas
        are registered with the selection service and departed ones evicted.
        """
        members: Dict[int, tuple] = {}
        for replica in target_info.running_replicas:
            worker_id = get_worker_id(replica.replica_id.unique_id)
            kv_event_metadata = replica.routing_stats.get("kv_event_metadata")
            if kv_event_metadata is not None:
                members[worker_id] = (
                    replica.replica_id.to_full_id_str(),
                    kv_event_metadata,
                )

        registered = set(self._replica_id_by_worker)
        added = members.keys() - registered
        removed = registered - members.keys()

        for worker_id in removed:
            self.remove_worker(worker_id)
            self._replica_id_by_worker.pop(worker_id, None)
        for worker_id in added:
            replica_id, kv_event_metadata = members[worker_id]
            self._register_block_size(kv_event_metadata["block_size"], replica_id)
            self._replica_id_by_worker[worker_id] = replica_id
            self._schedule(
                self._upsert_worker(worker_id, replica_id, kv_event_metadata)
            )

        if added or removed:
            logger.info(
                "KV router replica membership updated: +%d -%d, tracking %d worker(s).",
                len(added),
                len(removed),
                len(self._replica_id_by_worker),
            )

    def remove_worker(self, worker_id: int) -> None:
        """Evict a departed replica's worker and its KV blocks from the
        selection service.
        """
        # Drop the departed replica's in-flight requests; their completions can
        # never arrive, so they would otherwise leak. delete_worker below frees
        # their load in the service, so no per-request free_reservation is needed.
        for request_id in self._request_ids_by_worker.pop(worker_id, set()):
            state = self._requests.pop(request_id, None)
            if state is not None:
                self._remove_autoscaling_load(state)
        self._publish_autoscaling_snapshot()
        if self._svc is None:
            return
        self._schedule(self._svc.delete_worker(worker_id))

    async def _upsert_worker(
        self, worker_id: int, replica_id: str, kv_event_metadata: Dict[str, Any]
    ) -> None:
        """Register a replica's KV-event endpoint with the selection service.

        The selection service spawns a connect-out ZMQ listener to the
        replica's ``endpoint`` and indexes its live KV events.
        """
        if self._svc is None:
            return
        dp_rank = kv_event_metadat
... [865 chars trimmed] ...
are routing for data parallel deployments.
                "data_parallel_size": 1,
                "kv_events_endpoints": {dp_rank: kv_event_metadata["endpoint"]},
                # The listener dials this on a sequence gap (slow-joiner) to replay
                # the events it missed before its SUB connected; without it those
                # events are dropped and never indexed.
                "replay_endpoint": kv_event_metadata.get("replay_endpoint"),
            }
        )
        logger.info(
            "Registered KV event worker %d for replica %s at %s.",
            worker_id,
            replica_id,
            kv_event_metadata["endpoint"],
        )

    async def select_worker(
        self,
#####
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:8:    AllBlocksCleared,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:363:        return BlockStored(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:549:                BlockStored(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:867:            self.kv_event_queue.append(AllBlocksCleared())
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/distributed/kv_events.py:56:class BlockStored(KVCacheEvent):
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/distributed/kv_events.py:137:class AllBlocksCleared(KVCacheEvent):
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/distributed/kv_events.py:142:    events: list[BlockStored | BlockRemoved | AllBlocksCleared]
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:8:    AllBlocksCleared,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:363:        return BlockStored(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:549:                BlockStored(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/block_pool.py:867:            self.kv_event_queue.append(AllBlocksCleared())
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/engine/core.py:831:    def reset_prefix_cache(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/engine/core.py:834:        return self.scheduler.reset_prefix_cache(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/engine/core.py:866:        if not self.reset_prefix_cache(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:1547:        by reset_prefix_cache, whose same-step resume would otherwise deliver
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:2774:    def reset_prefix_cache(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:2812:        reset_successful = self.kv_cache_manager.reset_prefix_cache()
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/sched/scheduler.py:2834:            # see reset_prefix_cache() flip to False purely because they
Shell cwd was reset to /home/ray/default
```

</details>

## Claude  (2026-11-03 00:45:29)

I have what I need. Let me save one non-obvious finding to memory, then write up.

**Bash** — Check existing memory index

```bash
M=/home/ray/.claude/projects/-home-ray-default/memory; ls $M 2>/dev/null; cat $M/MEMORY.md 2>/dev/null | head -30
```

<details><summary>Bash result</summary>

```text
(Bash completed with no output)
```

</details>

**Write** `/home/ray/.claude/projects/-home-ray-default/memory/serve-metrics-scrapes-are-stale.md`

```
---
name: serve-metrics-scrapes-are-stale
description: In the serve-llm-autoscaling-experiment harness, telemetry/serve_metrics.jsonl gauges are a 10s sample-and-hold of Ray metrics; average the policy's published mean, not raw gauge scrapes
metadata:
  type: project
---

Ray exports Serve controller gauges to the metrics endpoint every 10s
(`metrics_report_interval_ms` default 10000), while the harness scrapes every 4s.
So `ray_serve_autoscaling_policy_metric{metric=outstanding_effective_prefill_tokens}`
in `telemetry/serve_metrics.jsonl` repeats identical values in runs of 2-3 scrapes
(8-12s) and lags the router by 0-12s. Verified 2026-11-02 on run
`20261102T203807Z-b200-workday-curve-prefill-mean`.

**Why:** averaging those raw scrapes gave a "policy mean" of 17.9K tokens when the
policy's own published 60s mean (`metric=prefill_required_replicas` x target)
averaged 15.4K. The raw gauge is bursty (58% zeros), so ~186 independent samples
is too noisy to summarize it.

**How to apply:** for "what did the policy see" use the published mean series;
treat the raw gauge scrapes as stale snapshots good for identifying exact gauge
states (they are exact integer sums of per-request token counts), not for
time averages. `tools/policy_vs_engine_mean.py` prints the raw-scrape average in
its summary line, so read the published value instead. See [[token-policy-gauge-gap-decomposition]].

```

<details><summary>Write result</summary>

```text
File created successfully at: /home/ray/.claude/projects/-home-ray-default/memory/serve-metrics-scrapes-are-stale.md
```

</details>

**Write** `/home/ray/.claude/projects/-home-ray-default/memory/token-policy-gauge-gap-decomposition.md`

```
---
name: token-policy-gauge-gap-decomposition
description: Why the KV router's outstanding-prefill gauge runs above the engines' remaining prefill (b200 workday run, 2026-11-02) and what was ruled out
metadata:
  type: project
---

Run `20261102T203807Z-b200-workday-curve-prefill-mean` (Ray checkout
`/mnt/local_storage/ray` branch `celina/prefill-token-autoscaling-policy` at
`ed547dd6dc`). Policy 60s mean averaged 15.4K tokens over the load phase vs
8.2K rebuilt from vLLM scheduler traces (chunk burn-down). Decomposition of the
7.2K gap: ~5.2K is the gauge holding a request's whole uncached prompt until its
first token (by design), ~0.8K is the router holding each request ~130ms longer
than the engine's queued-to-first-token span, ~1.2K is the router occasionally
counting a warm session turn at its full prompt length although the engine had
most of it cached (its KV index lacked blocks; seen only after ~700s, at high
load and around new replica joins). Ruled out: router replica double counting
(one router; policy takes max), leaked requests (gauge returns to exactly 0,
never negative), dropped lifecycle events (no warnings), cache eviction (engine
KV usage peaked 47%).

**Why:** 46 of 56 distinct gauge states were exact integer sums of the engines'
own uncached counts (shuffled control: 11), so the per-request estimate is
usually exact; the published mean regresses as 1.00 x cold + 1.29 x warm rebuild.

**How to apply:** to pin down the warm over-count, log the router's
`effective_prefill_tokens` per request next to the engine's
`cached_prefix_tokens`; the gauge export cannot resolve it. Analysis scripts
were in the session scratchpad only. See [[serve-metrics-scrapes-are-stale]].

```

<details><summary>Write result</summary>

```text
File created successfully at: /home/ray/.claude/projects/-home-ray-default/memory/token-policy-gauge-gap-decomposition.md
```

</details>

**Write** `/home/ray/.claude/projects/-home-ray-default/memory/MEMORY.md`

```
- [Serve metrics scrapes are stale](serve-metrics-scrapes-are-stale.md) — raw gauge scrapes are 10s sample-and-hold; use the policy's published mean
- [Token policy gauge gap decomposition](token-policy-gauge-gap-decomposition.md) — 72% whole-prompt hold, ~10% extra hold, ~18% router over-estimates on warm turns; what was ruled out

```

<details><summary>Write result</summary>

```text
File created successfully at: /home/ray/.claude/projects/-home-ray-default/memory/MEMORY.md (file state is current in your context — no need to Read it back)
```

</details>

## Claude  (2026-11-03 00:46:14)

The gap has three causes, and the headline number in the handoff was inflated by a telemetry artifact. The policy actually saw a load-phase mean of 15.4K tokens, not 17.9K. Of the 7.2K gap to the engine's 8.2K, about 72% is the whole-prompt hold (by design), about 10% is the router holding requests ~130 ms longer than the engine, and about 18% is the router occasionally counting a warm session turn at its full prompt length.

**First, the 17.9K figure is wrong.** Ray exports controller gauges every 10 s (`metrics_report_interval_ms` defaults to 10000) while the harness scrapes every 4 s. The raw gauge value repeats identically across consecutive scrapes 79% of the time and holds for 8 to 12 s, even where the underlying state lasted 2 s. Averaging those scrapes of a bursty signal (58% zeros) is noisy. The policy's own published 60 s mean, which is what it divides by the target, averages 15.4K over the load phase. The existing tool prints the raw-scrape average in its summary line, which is where 17.9K came from.

**Decomposition of the real gap (15.4K vs 8.2K):**

| Component | Tokens | Share | Evidence |
|---|---|---|---|
| Whole uncached prompt held until first token, vs chunk burn-down | 5.2K | 72% | Rebuilding the gauge from traces with the engine's own uncached counts and queued-to-first-token windows gives 13.4K |
| Router holds each request ~130 ms longer than the engine span | 0.8K | ~10% | Gauge is nonzero 41.5% of the time; the rebuild matches that only with ~130 ms extra hold per request |
| Router counts some warm turns at full prompt length | 1.2K | ~18% | Published mean regresses as 1.00 x cold rebuild + 1.29 x warm rebuild |

**Evidence that the per-request estimate is usually exact.** The stale gauge scrapes are still real gauge states. Of 56 distinct nonzero states with at most 18 concurrent candidate requests, 46 are exact integer sums of the engines' own uncached counts, versus 11 for a control with the counts shuffled across requests. Single-request states match to the token, for example 88347 for a cold 88347-token prompt and 105465 for a 67700 + 37765 pair.

**Evidence for the warm over-count.** Four of the ten unexplained states become exact sums only if a warm turn counts at its full prompt length. Examples: at 698 s the gauge read 108825, the full prompt of a turn whose engine had 103488 tokens cached; at 1006 s a turn with 45472 cached was counted at 48189. These cluster at high load after ~700 s, around the time the second replica joined, and on the newest replica. Before 700 s the ratio of published mean to rebuild is 0.98. The mechanism is most likely the router's KV index missing blocks the engine has, from KV event loss or lag under burst, or from the slow-joiner gap on new workers. I could not verify this directly because the router's per-request estimate is not logged.

**Ruled out:**
- Double counting across router replicas. One router ran the whole time, and the policy takes a max across sources anyway.
- Leaked requests. The gauge returns to exactly 0 after load and never goes negative, and the tracking TTL of 3600 s never elapsed within the run.
- Dropped lifecycle events. No warnings in any replica log, and no trace events dropped.
- Cache eviction. Engine KV usage peaked at 47%.

**To settle the warm over-count**, add a debug line in `select_worker` in `kv_token_tracker.py` logging the request id, `effective_prefill_tokens`, and prompt length, then join against `cached_prefix_tokens` in the scheduler traces on a rerun. The analysis scripts are in `/tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad/` (gauge_decomp.py, gauge_tests.py, states.py, pub_check.py, reg.py, timeres.py). I also saved two memory notes: the stale-scrape caveat and this decomposition.

## User  (2026-11-03 00:59:55)

wdym the 17.9k numver is wrong? whats wrong with the telemetry and is the graph wrong?

## Claude  (2026-11-03 01:00:18)

Nothing in the graph is wrong, and the telemetry values are real. The problem is one number in the tool's printed summary, which was computed from the wrong series.

**What the graph shows.** The purple line is the policy's published 60 s mean, the `prefill_required_replicas` metric times the 20K target. That is literally the number the policy divided by the target each tick, and it is correct. Its load-phase average is 15.4K. The green line comes from the traces and is unaffected.

**Where 17.9K came from.** The tool also rebuilds a 60 s mean from the raw `outstanding_effective_prefill_tokens` scrapes, as a fallback for policies that do not publish a mean. The summary line "time-average over the load phase: policy 17.9K" and the ratio stats (median 2.0, p90 4.2) are computed from that rebuilt series, not from the purple line you see. So the handoff quoted a number that is not on the plot.

**Why the raw-scrape rebuild is unreliable here.** Ray's metrics agent only refreshes exported gauges every 10 s, and the harness scrapes every 4 s. So each raw gauge value is a snapshot that is up to 10 s old and gets read 2 or 3 times in a row. The values themselves are exact (they match sums of real per-request token counts), but you effectively get only about 186 independent samples of a signal that is zero 58% of the time and spikes to 200K. The time-average of that is noisy: the 2.5K difference from the published mean is about one standard error. I would call it noise rather than a systematic bias. The published mean is smooth, so the same 10 s staleness barely affects its average.

So the picture is:
- Graph: correct as drawn.
- Raw gauge scrapes: correct values, stale by up to 10 s, fine for identifying individual gauge states, poor for time averages.
- The 17.9K summary and the ratio stats: computed from the raw scrapes, so they overstate the gap. Use 15.4K.

If you want, I can change the tool's summary and ratio lines in `tools/policy_vs_engine_mean.py` to use the published series when it exists, so the printed numbers match the purple line.

## User  (2026-11-03 01:05:06)

how did you come to this conclusion that majority of the difference is from the lack of chunk burn down?

## Claude  (2026-11-03 01:05:24)

By rebuilding the gauge two ways from the same per-request trace data and changing only one thing between them.

**The three numbers.** All three are time-averages over the 1860 s load phase, in outstanding uncached prefill tokens across all replicas.

| Accounting | Mean | What it counts |
|---|---|---|
| Engine-style (green line) | 8.2K | Each request's engine-measured uncached tokens enter when the engine queues it; each prefill chunk's tokens leave when that batch completes |
| Router-style rebuild | 13.4K | Same requests, same engine-measured uncached counts, same entry time, but the whole count stays until the request's first token |
| Real router gauge (published mean) | 15.4K | What the router actually tracked |

The first two use identical inputs: the same 1592 requests, the same `cached_prefix_tokens` from the engine, the same queue timestamps. The only difference is whether tokens leave chunk by chunk or all at once at first token. So the 5.2K between them is attributable purely to that choice. The remaining 2.0K between the rebuild and the real gauge is everything the router does differently from the rebuild: holding each request a bit longer and sometimes estimating more uncached tokens than the engine found.

**Why the magnitude is plausible.** For a long prompt prefilled at a steady rate, holding the whole count traces a rectangle while burn-down traces a triangle, so the ratio is up to 2x. Queue time before the first chunk counts fully in both, which pulls the ratio below 2. The observed ratio is 13.4 / 8.2 = 1.63, and cold prompts (mean 31K tokens, 3.5 steps) account for 9.8K of the 13.4K, which is where that rectangle-versus-triangle effect lives.

**Why I trust the rebuild as a stand-in for the router.** The exported gauge values, although stale, are exact gauge states, and 46 of 56 distinct states were exact integer sums of the engine's uncached counts for requests inside the queued-to-first-token window at that time. A shuffled control matched 11. So the router's per-request number and its hold window are, most of the time, exactly what the rebuild assumes, which is what lets me pin the burn-down share at 5.2K and leave only 2.0K for the other causes.

One caveat: the split between "130 ms extra hold" and "warm over-estimates" inside that 2.0K is less certain than the 5.2K, because it relies on busy-fraction fitting and a regression rather than exact matches.

## User  (2026-11-03 01:09:05)

can you plot that ^ like the sane as our engine vs policy plot but remove the scale decision and delay stuff and add a line for the reconstructed

**Skill**

```json
{
 "skill": "dataviz"
}
```

<details><summary>Skill result</summary>

```text
Launching skill: dataviz
```

</details>


## Claude  (2026-11-03 01:09:23)

No node here, so I can't run the palette validator; I'll keep the existing purple and green, add a dashed orange line for the rebuild so the new series differs by line style as well as hue.

**Write** `/home/ray/default/serve-llm-autoscaling-experiment/tools/policy_vs_engine_rebuild.py`

```
#!/usr/bin/env python3
"""What the mean policy observed vs what the engines held, with the accounting
difference isolated.

Three lookback means of the same quantity, total outstanding uncached prefill
tokens across all replicas:
  policy:  the router gauge's lookback mean as the policy published it
           (prefill_required_replicas x target), from the run's telemetry.
  rebuild: the router's accounting rebuilt from the vLLM scheduler traces with
           the engine's own uncached counts: a request's whole uncached prompt
           enters when the engine queues it and leaves at its first token.
  engine:  the engine's accounting from the same traces: the same tokens enter
           at the same time but each prefill chunk leaves when its batch
           completes, so the total is the work still to do.
rebuild - engine is the cost of holding the whole prompt instead of burning it
down; policy - rebuild is everything else the router does differently (hold
timing, its own uncached estimate).

usage: python tools/policy_vs_engine_rebuild.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH] [--hold-ms 0]
"""
import argparse, bisect, datetime, glob, json, math, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import yaml

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
PHASE_TINT = {"ramp up": "#e8f0fb", "hold": "#f1f1ee", "ramp down": "#fdeee6"}
POLICY, REBUILD, ENGINE = "#6250d6", "#c9480f", "#1baf7a"
ARROW, DOT = "→", "·"

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--lookback", type=float, default=None, help="mean window; default: the policy's upscale_lookback_s")
ap.add_argument("--ymax", type=float, default=None)
ap.add_argument("--start", type=float, default=0.0)
ap.add_argument("--end", type=float, default=None)
ap.add_argument("--traces", default="/mnt/cluster_storage/serve-llm-autoscaling-ex
... [6330 chars trimmed] ...
ap) for v in pol_v], color=POLICY, lw=1.8)
ax.plot(ts, [min(v, ycap) for v in rebuild_mean], color=REBUILD, lw=1.8, ls=(0, (4, 2)))
ax.plot(ts, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)
ax.axhline(target, color=INK2, lw=1, ls=":")
ax.set_ylim(0, ycap * 1.1)
ax.set_xlim(a.start, end)
ax.set_xlabel("seconds since profiling start", color=INK2)
ax.set_ylabel("total outstanding uncached prefill tokens, all replicas (K)", color=INK2)

def avg(v):
    return sum(v) / len(v) if v else float("nan")
m_pol, m_reb, m_eng = avg(pol_v), avg(rebuild_mean), avg(engine_mean)
hold_note = f", +{a.hold_ms:.0f} ms hold" if a.hold_ms else ""
handles = [Line2D([], [], color=POLICY, lw=1.8, label=f"policy: {lookback:.0f}s mean of the router gauge, {pol_label} (load-phase average {m_pol:.1f}K)"),
           Line2D([], [], color=REBUILD, lw=1.8, ls=(0, (4, 2)), label=f"rebuild: router accounting from traces, whole uncached prompt held from queueing to first token{hold_note} ({m_reb:.1f}K)"),
           Line2D([], [], color=ENGINE, lw=1.8, label=f"engine: remaining uncached prefill from traces, chunks leave as batches complete ({m_eng:.1f}K)"),
           Line2D([], [], color=INK2, lw=1, ls=":", label=f"target {target:.0f}K per replica")]
fig.legend(handles=handles, loc="lower center", fontsize=8, frameon=False, ncol=1, bbox_to_anchor=(0.5, -0.17))
fig.suptitle(f"{evaluation['experiment']} {DOT} policy vs engine, with the router's whole-prompt accounting rebuilt from the engine's own numbers",
             x=0.01, ha="left", fontsize=10, color=INK)
out = a.out or f"{R}/analysis/policy_vs_engine_rebuild{int(lookback)}_{int(a.start)}_{int(end)}.png"
fig.savefig(out, facecolor=SURFACE, bbox_inches="tight")
print("wrote", out)
print(f"  load-phase averages: policy {m_pol:.1f}K  rebuild {m_reb:.1f}K  engine {m_eng:.1f}K")
print(f"  rebuild - engine (whole prompt held vs chunk burn-down): {m_reb - m_eng:.1f}K; policy - rebuild (hold timing, router estimate): {m_pol - m_reb:.1f}K")

```

<details><summary>Write result</summary>

```text
File created successfully at: /home/ray/default/serve-llm-autoscaling-experiment/tools/policy_vs_engine_rebuild.py (file state is current in your context — no need to Read it back)
```

</details>

**Bash** — Render the three-line policy vs rebuild vs engine chart

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment && python3 tools/policy_vs_engine_rebuild.py runs/20261102T203807Z-b200-workday-curve-prefill-mean
```

<details><summary>Bash result</summary>

```text
wrote runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_0_1860.png
  load-phase averages: policy 15.4K  rebuild 13.4K  engine 8.2K
  rebuild - engine (whole prompt held vs chunk burn-down): 5.2K; policy - rebuild (hold timing, router estimate): 2.0K
```

</details>

**Read** `/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_0_1860.png`

## Claude  (2026-11-03 01:10:34)

Plot written to `runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_0_1860.png`, from a new tool `tools/policy_vs_engine_rebuild.py` that takes the same arguments as the original.

What's on it:
- **Purple**: the policy's published 60 s mean, load-phase average 15.4K.
- **Orange dashed**: the router's accounting rebuilt from the traces using the engine's own uncached counts, whole prompt held from queueing to first token, 13.4K.
- **Green**: the engine's remaining prefill with chunk burn-down, 8.2K.
- Phase tints and the 20K target line are kept. Scale decisions, replica startup shading, and the upscale-delay timer strip are removed.

Two things to read off it:
- **For the first ~650 s, purple and orange sit on top of each other** while both ride well above green. In that stretch the router's number is exactly the engine's uncached count held whole, so the entire gap is the burn-down accounting.
- **After ~700 s and again around 1280 s, purple pulls above orange** (peaks of 55K vs 42K, and 48K vs 26K). Those are the episodes where the router counted warm turns at full prompt length. Purple also trails orange by about 6 to 10 s throughout, which is the 10 s metrics export staleness, not a real delay in the router.

The script prints the three averages and the two differences (5.2K and 2.0K) when run. It has a `--hold-ms` flag if you want to see the rebuild with the ~130 ms extra hold folded in.

## User  (2026-11-03 01:13:29)

how did you rebuilt this? it doesnt look like it follows the purple as well as it should

## Claude  (2026-11-03 01:13:41)

Fair challenge. Let me quantify how closely it should track before I defend it.

**Bash** — Quantify how well the rebuild tracks the published mean and how much noise the policy's own sampling adds

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > fidelity.py <<'EOF'
import json, bisect, math, statistics as st, random
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
pub=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and s['labels'].get('metric')=='prefill_required_replicas' and 'LLMServer' in s['labels'].get('deployment',''):
            pub.append((t,s['value']*20000))
pubL=[(t,v) for t,v in pub if 60<=t<=1860]
def step(d):
    d=sorted(d); times=[t for t,_ in d]; lv=0; levels=[]
    for _,x in d: lv+=x; levels.append(lv)
    return times,levels
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def wmean(T,L,t,w=60.0):
    a=t-w; i0=bisect.bisect_right(T,a)
    pts=[(a,at(T,L,a))]+[(T[i],L[i]) for i in range(i0,bisect.bisect_right(T,t))]
    tot=0.0
    for (t1,v1),(t2,_) in zip(pts,pts[1:]+[(t,None)]):
        if t2>t1: tot+=v1*(t2-t1)
    return tot/w
def sampled_mean(T,L,t,phase,w=60.0,dt=2.0):
    # policy-style: 2s samples of the gauge, time-weighted over the window with carry-in
    k0=math.floor((t-w-phase)/dt); samples=[]
    for k in range(k0,int((t-phase)/dt)+1):
        ts_=phase+k*dt
        if ts_<=t: samples.append((ts_,at(T,L,ts_)))
    samples=[s for s in samples if s[0]>=t-w-dt]
    tot=0.0; a=t-w
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(t,None)]):
        lo,hi=max(t1,a),min(t2,t)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/w
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
A=step([(r['q'],r['own']) for r in reqs]+[(r['ft']+0.13,-r['own']) for r in reqs])
y=[v for _,v in pubL]
print("published policy mean vs exact rebuild (engine counts, +130ms):")
for sh in (0,4,6,8,10):
    x=[wmean(*A,t-sh) for t,_ in pubL]; print(f"  shift {sh:2d}s: corr {corr(x,y):.3f}  rmse {rmse(x,y):.0f}")
# how much noise does the policy's own 2s sampling add? compare exact rebuild mean to 2s-sampled rebuild mean
random.seed(0)
ex=[wmean(*A,t) for t,_ in pubL]
for phase in (0.0,0.7,1.3):
    sm=[sampled_mean(*A,t,phase) for t,_ in pubL]
    print(f"  2s-sampling noise alone (phase {phase}): rmse exact-vs-sampled rebuild {rmse(ex,sm):.0f}, corr {corr(ex,sm):.3f}")
# sampled rebuild vs published at best shift
best=None
for sh in (4,6,8,10):
    for phase in (0.0,0.5,1.0,1.5):
        sm=[sampled_mean(*A,t-sh,phase) for t,_ in pubL]; c=corr(sm,y)
        if best is None or c>best[0]: best=(c,sh,phase,rmse(sm,y))
print(f"  2s-sampled rebuild vs published: best corr {best[0]:.3f} at shift {best[1]}s phase {best[2]} (rmse {best[3]:.0f})")
# time-resolved fit: by 300s block, rmse at shift 6
x=[wmean(*A,t-6) for t,_ in pubL]
for lo in range(60,1860,300):
    idx=[i for i,(t,_) in enumerate(pubL) if lo<=t<lo+300]
    print(f"  {lo:4d}-{lo+300:4d}: corr {corr([x[i] for i in idx],[y[i] for i in idx]):.3f} rmse {rmse([x[i] for i in idx],[y[i] for i in idx]):.0f}  mean pub {st.mean(y[i] for i in idx):.0f} rebuild {st.mean(x[i] for i in idx):.0f}")
EOF
python3 -I fidelity.py
```

<details><summary>Bash result</summary>

```text
published policy mean vs exact rebuild (engine counts, +130ms):
  shift  0s: corr 0.841  rmse 6125
  shift  4s: corr 0.864  rmse 5719
  shift  6s: corr 0.867  rmse 5662
  shift  8s: corr 0.861  rmse 5778
  shift 10s: corr 0.849  rmse 5982
  2s-sampling noise alone (phase 0.0): rmse exact-vs-sampled rebuild 1557, corr 0.988
  2s-sampling noise alone (phase 0.7): rmse exact-vs-sampled rebuild 1417, corr 0.990
  2s-sampling noise alone (phase 1.3): rmse exact-vs-sampled rebuild 1650, corr 0.984
  2s-sampled rebuild vs published: best corr 0.876 at shift 4s phase 0.5 (rmse 5518)
    60- 360: corr 0.956 rmse 1612  mean pub 10426 rebuild 10849
   360- 660: corr 0.878 rmse 2989  mean pub 13893 rebuild 14897
   660- 960: corr 0.779 rmse 7536  mean pub 30264 rebuild 26422
   960-1260: corr 0.751 rmse 6352  mean pub 13129 rebuild 11993
  1260-1560: corr 0.834 rmse 7793  mean pub 16851 rebuild 13980
  1560-1860: corr 0.720 rmse 4788  mean pub 9798 rebuild 8484
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Check whether pre-engine latency grows at high load

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json,csv,statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
rows=list(csv.DictReader(open(f'{R}/analysis/load_requests.csv')))
print("pre_ms (client send -> engine queued) by time window, cold / warm medians and p90; this bounds how early the router admits before the engine queues")
for lo in range(0,1860,300):
    for k in ('cold','warm'):
        v=sorted(float(r['pre_ms']) for r in rows if r['kind']==k and lo<=recs[r['request_id']]['request_start_ns']/1e9-t0<lo+300)
        if v: print(f"  {lo:4d}-{lo+300:4d} {k}: n={len(v):4d} median {v[len(v)//2]:6.0f} p90 {v[int(.9*len(v))]:6.0f} max {v[-1]:6.0f}")
EOF
```

<details><summary>Bash result</summary>

```text
pre_ms (client send -> engine queued) by time window, cold / warm medians and p90; this bounds how early the router admits before the engine queues
     0- 300 cold: n=  34 median     82 p90    247 max    567
     0- 300 warm: n=  56 median    118 p90    273 max    537
   300- 600 cold: n=  33 median     45 p90    161 max    830
   300- 600 warm: n= 186 median    120 p90    451 max   1103
   600- 900 cold: n=  36 median    107 p90    705 max    819
   600- 900 warm: n= 248 median    186 p90    508 max   1147
   900-1200 cold: n=  18 median     83 p90    413 max    785
   900-1200 warm: n= 315 median    161 p90    287 max    838
  1200-1500 cold: n=  18 median     79 p90    231 max    240
  1200-1500 warm: n= 263 median    171 p90    344 max   1519
  1500-1800 cold: n=   9 median     62 p90    166 max    166
  1500-1800 warm: n= 311 median    164 p90    327 max   1022
  1800-2100 warm: n=  57 median    178 p90    298 max    436
Shell cwd was reset to /home/ray/default
```

</details>

## Claude  (2026-11-03 01:14:49)

Here is exactly how the orange line is built, and then what "following purple" should look like given how purple itself is made.

**How the rebuild works.** For every request in this run's four scheduler trace files:
1. Its uncached token count is the sum of its prefill chunks up to its first token, from `batch_scheduled`. That equals `prompt_tokens - cached_prefix_tokens`, the engine's own cache hit.
2. That count is added at the `request_queued` timestamp and removed at the `request_first_token` timestamp, plus 130 ms.
3. The resulting step function gets the same 60 s time-weighted mean the policy computes.

Nothing from the router goes into it. It is the engine's numbers put through the router's accounting rule, so that any difference from purple is something the router does differently.

**How closely it should match.** Purple is not an exact mean of the router gauge. The router records its gauge every 2 s, and the policy integrates those samples. Re-sampling the orange step function the same way shows that sampling alone adds about 1.5K RMSE to a 60 s mean, even with perfect inputs. Purple is also delayed by the 10 s metrics export, so the best alignment is at a 4 to 6 s shift. With both accounted for, the fit over the whole load phase is correlation 0.88 and RMSE 5.5K. The fit is not uniform in time:

| Window (s) | Correlation | RMSE | Mean purple | Mean orange |
|---|---|---|---|---|
| 60–360 | 0.96 | 1.6K | 10.4K | 10.8K |
| 360–660 | 0.88 | 3.0K | 13.9K | 14.9K |
| 660–960 | 0.78 | 7.5K | 30.3K | 26.4K |
| 960–1260 | 0.75 | 6.4K | 13.1K | 12.0K |
| 1260–1560 | 0.83 | 7.8K | 16.9K | 14.0K |
| 1560–1860 | 0.72 | 4.8K | 9.8K | 8.5K |

In the first 360 s the rebuild tracks purple to within the sampling noise floor. That is the single-replica, low-load period, and it is also where the raw gauge states were exact integer sums of the engine's counts. From about 660 s on, purple runs 1 to 4K above orange with larger excursions. That is the multi-replica, high-load period, and it is where the gauge states stopped decomposing into engine counts and where the full-prompt warm turns showed up.

**What I checked that could have been a rebuild defect instead.** Client-to-engine latency, which bounds how much earlier the router admits a request than the engine queues it, stays at a median of 100 to 200 ms across the whole run and does not grow at high load. Each 100 ms is worth only about 0.6K on the mean, so a timing lead cannot produce the 4K excess after 660 s. All 1592 placed requests come from the four trace files of this run, with per-file clock offsets, and none were preempted before first token.

So the divergence you are seeing is not the rebuild failing to follow the router. It is the measurement: before 660 s the router's number is the engine's number held whole, and after that the router counts more than the engine's cache hit implies. If you want to see the rebuild line without the 6 s export delay confounding the eye, I can add a `--shift` flag that moves orange and green right by the export lag so the peaks line up with purple.

## User  (2026-11-03 01:16:32)

sure

**Bash** — Add a shift flag to the tool and render the shifted chart

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment && python3 - <<'EOF'
p='tools/policy_vs_engine_rebuild.py'; s=open(p).read()
s=s.replace('''ap.add_argument("--hold-ms", type=float, default=0.0, help="extra hold per request in the rebuild, after its first token")''',
'''ap.add_argument("--hold-ms", type=float, default=0.0, help="extra hold per request in the rebuild, after its first token")
ap.add_argument("--shift", type=float, default=0.0, help="seconds to move the trace-side lines right, to line up with the policy's export lag")''')
s=s.replace('''ax.plot(ts, [min(v, ycap) for v in rebuild_mean], color=REBUILD, lw=1.8, ls=(0, (4, 2)))
ax.plot(ts, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)''',
'''tsh = [t + a.shift for t in ts]
ax.plot(tsh, [min(v, ycap) for v in rebuild_mean], color=REBUILD, lw=1.8, ls=(0, (4, 2)))
ax.plot(tsh, [min(v, ycap) for v in engine_mean], color=ENGINE, lw=1.8)''')
s=s.replace('''hold_note = f", +{a.hold_ms:.0f} ms hold" if a.hold_ms else ""''',
'''hold_note = f", +{a.hold_ms:.0f} ms hold" if a.hold_ms else ""
shift_note = f" {DOT} trace-side lines shifted right {a.shift:g}s to match the policy metric's export lag" if a.shift else ""''')
s=s.replace('''label=f"engine: remaining uncached prefill from traces, chunks leave as batches complete ({m_eng:.1f}K)"),''',
'''label=f"engine: remaining uncached prefill from traces, chunks leave as batches complete ({m_eng:.1f}K){shift_note}"),''')
s=s.replace('''out = a.out or f"{R}/analysis/policy_vs_engine_rebuild{int(lookback)}_{int(a.start)}_{int(end)}.png"''',
'''out = a.out or f"{R}/analysis/policy_vs_engine_rebuild{int(lookback)}{f'_shift{a.shift:g}' if a.shift else ''}_{int(a.start)}_{int(end)}.png"''')
s=s.replace('''usage: python tools/policy_vs_engine_rebuild.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH] [--hold-ms 0]''',
'''usage: python tools/policy_vs_engine_rebuild.py runs/<run_dir> [--lookback 60] [--ymax K]
           [--start 0] [--end END] [--traces DIR] [--out PATH] [--hold-ms 0] [--shift 0]
--shift moves the two trace-side lines right by that many seconds: the policy
metric is exported through Ray's metrics agent every 10s, so the published
mean trails the traces by roughly 4-8s.''')
open(p,'w').write(s)
EOF
python3 tools/policy_vs_engine_rebuild.py runs/20261102T203807Z-b200-workday-curve-prefill-mean --shift 6
```

<details><summary>Bash result</summary>

```text
wrote runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_shift6_0_1860.png
  load-phase averages: policy 15.4K  rebuild 13.4K  engine 8.2K
  rebuild - engine (whole prompt held vs chunk burn-down): 5.2K; policy - rebuild (hold timing, router estimate): 2.0K
```

</details>

**Read** `/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_shift6_0_1860.png`

## Claude  (2026-11-03 01:17:04)

Shifted version written to `runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_shift6_0_1860.png`. The tool now takes `--shift <seconds>`, which moves the two trace-side lines right and notes it in the legend; the averages are unchanged by the shift.

With the 6 s export lag removed, purple and orange sit on top of each other through the whole ramp up to about 650 s, including the plateaus at 500 to 600 s and the 20K crossing around 130 s. Past 700 s the rebuild tracks the shape of purple but purple runs higher, with the two clear excursions at about 830 s (55K vs 42K) and 1280 s (48K vs 26K). Those two are the episodes where the router counted warm turns at their full prompt length; everything else in the gap is the whole-prompt hold shown by the orange-to-green distance.

## User  (2026-11-03 01:44:00)

this doesnt match exactly which is weird. can you keep looking for why the reconstruction is still off? dig deeper into the reasons for the mismatch and also be aware of telemetry errors

## Claude  (2026-11-03 01:45:44)

Good call. Before attributing anything to the router, I'll model the telemetry pipeline exactly (the router's 2 s gauge samples, the push to the controller, the 10 s metrics export, the 4 s scrape) and fit its phases, so the comparison with purple is apples to apples. Then whatever residual remains is real. First, the export tick phase and any scrape errors or gaps.

**Bash** — Check scrape errors and gaps and locate the 10 s export tick phase

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json, collections, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
errs=0; n=0; rows=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line); n+=1
    if rec.get('error'): errs+=1; print('scrape error at %.1f: %s'%(rec['scrape_started_at_ns']/1e9-t0, str(rec['error'])[:120])); continue
    t=rec['scrape_started_at_ns']/1e9-t0; d={}
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and 'LLMServer' in s['labels'].get('deployment',''): d[s['labels']['metric']]=s['value']
    rows.append((t,d.get('outstanding_effective_prefill_tokens'),d.get('prefill_required_replicas'),d.get('required_replicas'), rec.get('scrape_duration_ms') or rec.get('duration_ms')))
print('scrapes',n,'errors',errs,'missing policy metric',sum(1 for r in rows if r[1] is None))
dts=[b[0]-a[0] for a,b in zip(rows,rows[1:])]
print('scrape gaps: min %.3f max %.3f; gaps > 4.5s: %s'%(min(dts),max(dts),[(round(a[0],1),round(b[0]-a[0],1)) for a,b in zip(rows,rows[1:]) if b[0]-a[0]>4.5]))
print('scrape time fractional part (mod 4):', collections.Counter(round(r[0]%4,2) for r in rows).most_common(5))
# export tick phase: times when either exported value changes, mod 10
ch=[b[0] for a,b in zip(rows,rows[1:]) if (a[1],a[2])!=(b[1],b[2]) and 0<=b[0]<=1860]
hist=collections.Counter(round(t%10,1) for t in ch)
print('change-time mod 10 histogram:', sorted(hist.items()))
print('n changes',len(ch))
# run-length pattern
runs=[];cur=1
for a,b in zip(rows,rows[1:]):
    if (a[1],a[2])==(b[1],b[2]): cur+=1
    else: runs.append(cur); cur=1
print('run lengths (all states incl zero):', collections.Counter(runs).most_common(8))
EOF
```

<details><summary>Bash result</summary>

```text
scrapes 543 errors 0 missing policy metric 0
scrape gaps: min 4.000 max 4.000; gaps > 4.5s: []
scrape time fractional part (mod 4): [(3.7, 543)]
change-time mod 10 histogram: [(1.7, 93), (9.7, 93)]
n changes 186
run lengths (all states incl zero): [(2, 96), (3, 95), (4, 1)]
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Fit the record and export tick phases by exact-matching the exported gauge against the trace rebuild

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > pipeline_fit.py <<'EOF'
"""Fit the telemetry pipeline (router 2s record tick, controller evaluation/export tick)
so the trace rebuild can be compared to the exported gauge and published mean exactly."""
import json, bisect, math, statistics as st, itertools, sys
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
rows=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0; d={}
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and 'LLMServer' in s['labels'].get('deployment',''): d[s['labels']['metric']]=s['value']
    if 'outstanding_effective_prefill_tokens' in d: rows.append((t,d['outstanding_effective_prefill_tokens'],d['prefill_required_replicas']*20000))
rows=[r for r in rows if 0<=r[0]<=1860]
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def level_fn(lead,hold):
    return step([(r['q']-lead,r['own']) for r in reqs]+[(r['ft']+hold,-r['own']) for r in reqs])
EARLY=650
best=[]
leads=[0.0,0.05,0.1,0.2,0.3]; holds=[0.0,0.05,0.1,0.15,0.2,0.3,0.5,0.8]
fns={(l,h):level_fn(l,h) for l in leads for h in holds}
psis=[round(5.5+0.1*i,2) for i in range(43)]; phis=[round(0.05*i,2) for i in range(40)]
results=[]
for psi in psis:
    for phi in phis:
        # evaluation time per scrape, record sample time
        samp=[]
        for t,v,_ in rows:
            e=psi+10*math.floor((t-psi)/10); s=phi+2*math.floor((e-phi)/2); samp.append((s,v,t))
        for (l,h),(T,L) in fns.items():
            m_early=sum(1 for s,v,t in samp if t<=EARLY and abs(at(T,L,s)-v)<=1)
            m_all=sum(1 for s,v,t in samp if abs(at(T,L,s)-v)<=1)
            results.append((m_early,m_all,psi,phi,l,h))
results.sort(reverse=True)
n_early=sum(1 for r in rows if r[0]<=EARLY)
print(f"scrapes: early (<= {EARLY}s) {n_early}, all {len(rows)}")
print("top fits by exact matches of exported raw gauge vs rebuild sampled at the implied record tick (early matches, all matches, eval phase mod 10, record phase mod 2, lead s, hold s):")
for r in results[:12]: print("  ",r)
# baseline: how many exact matches do you get ignoring the pipeline (sample rebuild at scrape time)?
T,L=fns[(0.0,0.0)]
print("baseline, rebuild sampled at scrape time, no pipeline model: early", sum(1 for t,v,_ in rows if t<=EARLY and abs(at(T,L,t)-v)<=1), "all", sum(1 for t,v,_ in rows if abs(at(T,L,t)-v)<=1))
json.dump(results[:50],open('pipeline_fit_top.json','w'))
EOF
python3 -I pipeline_fit.py
```

**Bash** — Check whether router tokenization runs off the event loop and where the metrics sink is wired

```bash
cd /mnt/local_storage/ray && sed -n 1,80p python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/tokenizer.py | grep -n "def \|to_thread\|executor\|async\|encode" ; echo '#####'; grep -n "set_autoscaling_metrics_sink\|set_custom_router_metrics\|_run_router_in_separate_loop" -r python/ray/llm/_internal/serve python/ray/serve/_private/router.py python/ray/serve/handle.py | grep -v tests | head
```

<details><summary>Bash result</summary>

```text
scrapes: early (<= 650s) 162, all 465
top fits by exact matches of exported raw gauge vs rebuild sampled at the implied record tick (early matches, all matches, eval phase mod 10, record phase mod 2, lead s, hold s):
   (152, 374, 9.7, 0.3, 0.0, 0.1)
   (152, 374, 9.7, 0.25, 0.05, 0.05)
   (152, 374, 9.7, 0.2, 0.1, 0.0)
   (152, 374, 9.6, 0.3, 0.0, 0.1)
   (152, 374, 9.6, 0.25, 0.05, 0.05)
   (152, 374, 9.6, 0.2, 0.1, 0.0)
   (152, 374, 9.5, 0.3, 0.0, 0.1)
   (152, 374, 9.5, 0.25, 0.05, 0.05)
   (152, 374, 9.5, 0.2, 0.1, 0.0)
   (152, 374, 9.4, 0.3, 0.0, 0.1)
   (152, 374, 9.4, 0.25, 0.05, 0.05)
   (152, 374, 9.4, 0.2, 0.1, 0.0)
baseline, rebuild sampled at scrape time, no pipeline model: early 71 all 179
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
38:    def __init__(self, message: str, *, status_code: int, type: str):
45:def build_tokenize_request(
#####
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:119:            handle._init(_run_router_in_separate_loop=True)
python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:340:    def set_autoscaling_metrics_sink(
python/ray/llm/_internal/serve/utils/broadcast.py:61:        handle._init(_run_router_in_separate_loop=True)
python/ray/llm/_internal/serve/core/ingress/router.py:183:            self._kv_token_tracker.set_autoscaling_metrics_sink(
python/ray/serve/handle.py:198:        return self.init_options._run_router_in_separate_loop
python/ray/serve/_private/router.py:393:    def set_custom_router_metrics(
python/ray/serve/_private/router.py:873:        self._metrics_manager.set_custom_router_metrics(metrics_dict)
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Simulate the published mean through the fitted pipeline and classify every remaining gauge mismatch

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > pipeline_sim.py <<'EOF'
import json, bisect, math, statistics as st, collections
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
PSI,PHI,LEAD,HOLD=9.6,0.3,0.0,0.1
rows=[]
for line in open(f'{R}/telemetry/serve_metrics.jsonl'):
    rec=json.loads(line)
    if rec.get('error'): continue
    t=rec['scrape_started_at_ns']/1e9-t0; d={}
    for s in rec['samples']:
        if s['name']=='ray_serve_autoscaling_policy_metric' and 'LLMServer' in s['labels'].get('deployment',''): d[s['labels']['metric']]=s['value']
    if 'outstanding_effective_prefill_tokens' in d: rows.append((t,d['outstanding_effective_prefill_tokens'],d['prefill_required_replicas']*20000))
rows=[r for r in rows if 0<=r[0]<=1860]
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
T,L=step([(r['q']-LEAD,r['own']) for r in reqs]+[(r['ft']+HOLD,-r['own']) for r in reqs])
def policy_mean(e,w=60.0):
    # samples at record ticks <= e; time-weighted over [e-w, e] with carry-in and carry-forward (as _time_weighted_mean)
    k_last=math.floor((e-PHI)/2); ticks=[PHI+2*k for k in range(k_last-31,k_last+1)]
    samples=[(s,at(T,L,s)) for s in ticks if s<=e]
    a=e-w; tot=0.0
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
        lo,hi=max(t1,a),min(t2,e)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/w
sim=[]; raw_pred=[]
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    sim.append(policy_mean(e)); raw_pred.append(at(T,L,s))
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); return sum((a-mx)*(b-my) for a,b in zip(x,y))/math.sqrt(sum((a-mx)**2 for a in x)*sum((b-my)**2 for b in y))
pub=[p for _,_,p in rows]
print("published mean vs pipeline-simulated rebuild mean:")
for lo,hi in [(0,650),(650,1860),(0,1860)]:
    idx=[i for i,(t,_,_) in enumerate(rows) if lo<=t<hi]
    x=[sim[i] for i in idx]; y=[pub[i] for i in idx]
    print(f"  {lo:4d}-{hi:4d}: corr {corr(x,y):.3f} rmse {rmse(x,y):.0f} mean pub {st.mean(y):.0f} sim {st.mean(x):.0f} (pub-sim {st.mean(y)-st.mean(x):+.0f})")
for lo in range(0,1860,150):
    idx=[i for i,(t,_,_) in enumerate(rows) if lo<=t<lo+150]
    x=[sim[i] for i in idx]; y=[pub[i] for i in idx]
    print(f"    {lo:4d}-{lo+150:4d}: rmse {rmse(x,y):5.0f}  pub {st.mean(y):6.0f} sim {st.mean(x):6.0f} diff {st.mean(y)-st.mean(x):+6.0f}   raw mismatches {sum(1 for i in idx if abs(raw_pred[i]-rows[i][1])>1)}/{len(idx)}")
json.dump({'rows':rows,'sim':sim,'raw_pred':raw_pred,'params':[PSI,PHI,LEAD,HOLD]},open('pipeline_sim.json','w'))

# classify raw mismatches
print("\nraw gauge mismatches (export != rebuild at implied record tick), classified:")
def subset(target, vals, tol=1):
    sums={0:()}
    for i,v in enumerate(vals):
        new={}
        for s_,sub in sums.items():
            ns=s_+v
            if ns<=target+tol and ns not in sums and ns not in new: new[ns]=sub+(i,)
        sums.update(new)
    for s_ in range(target-tol,target+tol+1):
        if s_ in sums: return sums[s_]
    return None
def aid(r): return None
cls=collections.Counter(); details=[]
for i,(t,v,p) in enumerate(rows):
    pred=raw_pred[i]
    if abs(pred-v)<=1: continue
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in reqs if r['q']-LEAD<=s<r['ft']+HOLD]           # what rebuild counts at s
    diff=int(round(v-pred))
    near=[r for r in reqs if r['q']-3<=s<=r['ft']+3]                  # requests near the sample
    tag=None; info=''
    if diff>0:
        # extra: a held warm request counted at full prompt?
        ex=[r for r in held if abs((r['isl']-r['own'])-diff)<=1]
        if ex: tag='held request counted at FULL prompt'; info=f"isl {ex[0]['isl']} own {ex[0]['own']} cached {ex[0]['cached']} eng {ex[0]['eng'][-6:]}"
        if tag is None:
            sub=subset(diff,[r['isl']-r['own'] for r in held if r['isl']>r['own']])
            if sub: tag='several held requests counted at FULL prompt'
        if tag is None:
            late=[r for r in near if r['ft']+HOLD<=s and abs(r['own']-diff)<=1]
            if late: tag='decrement later than first token'; info=f"own {late[0]['own']} first token {s-late[0]['ft']:.2f}s before sample"
        if tag is None:
            early=[r for r in near if s<r['q']-LEAD and abs(r['own']-diff)<=1]
            if early: tag='admitted before engine queued'; info=f"own {early[0]['own']} queued {early[0]['q']-s:.2f}s after sample"
        if tag is None:
            sub=subset(diff,[r['own'] for r in near if not (r['q']-LEAD<=s<r['ft']+HOLD)])
            if sub: tag='timing: nearby requests held at sample (late decrement / early admit)'
    else:
        miss=[r for r in held if abs(r['own']+diff)<=1]
        if miss: tag='held request NOT counted (or decremented early)'; info=f"own {miss[0]['own']} eng {miss[0]['eng'][-6:]} q {miss[0]['q']:.1f} ft {miss[0]['ft']:.1f}"
        if tag is None:
            sub=subset(-diff,[r['own'] for r in held])
            if sub: tag='several held requests not counted'
    if tag is None:
        # maybe record tick jitter: does any s' within +-2s match exactly?
        for ds in [x/10 for x in range(-20,21)]:
            if abs(at(T,L,s+ds)-v)<=1: tag=f'record-tick jitter (matches at {ds:+.1f}s)'; break
    if tag is None: tag='unexplained'
    cls[tag]+=1; details.append((round(t,1),int(v),int(pred),diff,tag,info))
for k,n in cls.most_common(): print(f"  {n:3d}  {k}")
print("\nmismatch details (t, exported, predicted, diff, class, info):")
for d in details: print("  ",d)
EOF
python3 -I pipeline_sim.py
```

**Bash** — Check router tokenization threading and which benchmark requests are absent from the rebuild

```bash
cd /mnt/local_storage/ray && grep -n "async def tokenize\|def tokenize\|to_thread\|run_in_executor\|encode(" python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/tokenizer.py | head; echo '#####'; cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json
recs=json.load(open('aiperf_recs.json'))
print('aiperf records',len(recs),'cancelled',sum(1 for r in recs if r.get('was_cancelled')),'overflow skip',sum(1 for r in recs if r.get('context_overflow_skip')))
S=json.load(open('gauge_decomp_state.json'))
ids={k.split('chatcmpl-')[-1].rsplit('-',1)[0] for k in S['reqs']}
missing=[r for r in recs if r['x_request_id'] not in ids]
print('aiperf records not in rebuild:',[(r['x_request_id'][:8],r.get('turn_index'),r.get('input_sequence_length'),r.get('usage_prompt_cache_read_tokens'),round((r['request_end_ns']-r['request_start_ns'])/1e9,2)) for r in missing])
EOF
```

<details><summary>Bash result</summary>

```text
published mean vs pipeline-simulated rebuild mean:
     0- 650: corr 0.976 rmse 1552 mean pub 11585 sim 12297 (pub-sim -712)
   650-1860: corr 0.882 rmse 6399 mean pub 17438 sim 14959 (pub-sim +2478)
     0-1860: corr 0.889 rmse 5246 mean pub 15399 sim 14032 (pub-sim +1367)
       0- 150: rmse  1093  pub  10089 sim  10623 diff   -534   raw mismatches 3/37
     150- 300: rmse   828  pub  11161 sim  11264 diff   -103   raw mismatches 0/38
     300- 450: rmse  1563  pub   6581 sim   7536 diff   -955   raw mismatches 2/37
     450- 600: rmse  2422  pub  17830 sim  19361 diff  -1532   raw mismatches 5/38
     600- 750: rmse  3621  pub  21323 sim  19078 diff  +2245   raw mismatches 5/37
     750- 900: rmse  8260  pub  35124 sim  29002 diff  +6122   raw mismatches 0/38
     900-1050: rmse  6282  pub  14877 sim  17815 diff  -2938   raw mismatches 16/37
    1050-1200: rmse  4396  pub  14906 sim  11361 diff  +3545   raw mismatches 17/38
    1200-1350: rmse 11564  pub  23118 sim  14061 diff  +9057   raw mismatches 8/37
    1350-1500: rmse  3134  pub  11953 sim  12311 diff   -358   raw mismatches 12/38
    1500-1650: rmse  2573  pub   8070 sim   9819 diff  -1749   raw mismatches 8/37
    1650-1800: rmse  6358  pub  14334 sim  10851 diff  +3484   raw mismatches 13/38
    1800-1950: rmse  1578  pub   3230 sim   1773 diff  +1457   raw mismatches 2/15

raw gauge mismatches (export != rebuild at implied record tick), classified:
   32  unexplained
   32  admitted before engine queued
   21  held request NOT counted (or decremented early)
    3  held request counted at FULL prompt
    3  several held requests not counted

mismatch details (t, exported, predicted, diff, class, info):
   (119.7, 31013, 41989, -10976, 'unexplained', '')
   (123.7, 31013, 41989, -10976, 'unexplained', '')
   (127.7, 31013, 41989, -10976, 'unexplained', '')
   (411.7, 90669, 88923, 1746, 'unexplained', '')
   (415.7, 90669, 88923, 1746, 'unexplained', '')
   (491.7, 89531, 100507, -10976, 'unexplained', '')
   (495.7, 89531, 100507, -10976, 'unexplained', '')
   (559.7, 97476, 92565, 4911, 'admitted before engine queued', 'own 4911 queued 0.21s after sample')
   (563.7, 97476, 92565, 4911, 'admitted before engine queued', 'own 4911 queued 0.21s after sample')
   (567.7, 97476, 92565, 4911, 'admitted before engine queued', 'own 4911 queued 0.21s after sample')
   (699.7, 108825, 5337, 103488, 'held request counted at FULL prompt', 'isl 108825 own 5337 cached 103488 eng 151035')
   (703.7, 108825, 5337, 103488, 'held request counted at FULL prompt', 'isl 108825 own 5337 cached 103488 eng 151035')
   (707.7, 108825, 5337, 103488, 'held request counted at FULL prompt', 'isl 108825 own 5337 cached 103488 eng 151035')
   (731.7, 126260, 148139, -21879, 'unexplained', '')
   (735.7, 126260, 148139, -21879, 'unexplained', '')
   (919.7, 61098, 148906, -87808, 'unexplained', '')
   (923.7, 61098, 148906, -87808, 'unexplained', '')
   (927.7, 61098, 148906, -87808, 'unexplained', '')
   (931.7, 
... [3693 chars trimmed] ...
, '')
   (1375.7, 19940, 43460, -23520, 'unexplained', '')
   (1411.7, 5536, 17230, -11694, 'held request NOT counted (or decremented early)', 'own 11694 eng 180495 q 1407.6 ft 1408.3')
   (1415.7, 5536, 17230, -11694, 'held request NOT counted (or decremented early)', 'own 11694 eng 180495 q 1407.6 ft 1408.3')
   (1451.7, 116407, 142715, -26308, 'unexplained', '')
   (1455.7, 116407, 142715, -26308, 'unexplained', '')
   (1459.7, 0, 5638, -5638, 'held request NOT counted (or decremented early)', 'own 5638 eng 180495 q 1458.0 ft 1458.4')
   (1463.7, 0, 5638, -5638, 'held request NOT counted (or decremented early)', 'own 5638 eng 180495 q 1458.0 ft 1458.4')
   (1467.7, 0, 5638, -5638, 'held request NOT counted (or decremented early)', 'own 5638 eng 180495 q 1458.0 ft 1458.4')
   (1551.7, 0, 5653, -5653, 'held request NOT counted (or decremented early)', 'own 5653 eng 173074 q 1547.9 ft 1548.3')
   (1555.7, 0, 5653, -5653, 'held request NOT counted (or decremented early)', 'own 5653 eng 173074 q 1547.9 ft 1548.3')
   (1559.7, 0, 3434, -3434, 'held request NOT counted (or decremented early)', 'own 3434 eng 173074 q 1558.0 ft 1558.3')
   (1563.7, 0, 3434, -3434, 'held request NOT counted (or decremented early)', 'own 3434 eng 173074 q 1558.0 ft 1558.3')
   (1567.7, 0, 3434, -3434, 'held request NOT counted (or decremented early)', 'own 3434 eng 173074 q 1558.0 ft 1558.3')
   (1579.7, 17129, 15467, 1662, 'admitted before engine queued', 'own 1662 queued 0.16s after sample')
   (1583.7, 17129, 15467, 1662, 'admitted before engine queued', 'own 1662 queued 0.16s after sample')
   (1587.7, 17129, 15467, 1662, 'admitted before engine queued', 'own 1662 queued 0.16s after sample')
   (1691.7, 21598, 21185, 413, 'admitted before engine queued', 'own 413 queued 0.11s after sample')
   (1695.7, 21598, 21185, 413, 'admitted before engine queued', 'own 413 queued 0.11s after sample')
   (1699.7, 1488, 0, 1488, 'admitted before engine queued', 'own 1488 queued 0.15s after sample')
   (1703.7, 1488, 0, 1488, 'admitted before engine queued', 'own 1488 queued 0.15s after sample')
   (1707.7, 1488, 0, 1488, 'admitted before engine queued', 'own 1488 queued 0.15s after sample')
   (1719.7, 150271, 90687, 59584, 'unexplained', '')
   (1723.7, 150271, 90687, 59584, 'unexplained', '')
   (1727.7, 150271, 90687, 59584, 'unexplained', '')
   (1751.7, 2443, 0, 2443, 'admitted before engine queued', 'own 2443 queued 0.11s after sample')
   (1755.7, 2443, 0, 2443, 'admitted before engine queued', 'own 2443 queued 0.11s after sample')
   (1791.7, 2155, 0, 2155, 'admitted before engine queued', 'own 2155 queued 0.10s after sample')
   (1795.7, 2155, 0, 2155, 'admitted before engine queued', 'own 2155 queued 0.10s after sample')
   (1799.7, 0, 16778, -16778, 'several held requests not counted', '')
   (1803.7, 0, 16778, -16778, 'several held requests not counted', '')
   (1807.7, 0, 16778, -16778, 'several held requests not counted', '')
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
134:    async def tokenize(
#####
aiperf records 1593 cancelled 0 overflow skip 0
aiperf records not in rebuild: [('bdf2959a', 37, 102125, 0, 9.44)]
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Check whether each engine trace file's wall-to-monotonic clock offset is consistent

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json,statistics as st
T='/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces'
for f in ['k-5be017850000_pid-151035.0000.jsonl','k-5be017850000_pid-151035.0001.jsonl','k-5be017850000_pid-173074.0000.jsonl','k-5be017850000_pid-180495.0000.jsonl']:
    offs=[]; first=None
    for i,line in enumerate(open(f'{T}/{f}')):
        try: e=json.loads(line)
        except: continue
        if 'wall_time_s' in e and 'monotonic_s' in e:
            o=e['wall_time_s']-e['monotonic_s']
            if first is None: first=o
            offs.append(o)
        if i>400000: break
    print(f"{f}: first-event offset {first:.4f}; over {len(offs)} events: median {st.median(offs):.4f} min {min(offs):.4f} max {max(offs):.4f}")
EOF
```

**Bash** — Refit the published-mean simulation with series staleness and inspect the not-counted cases' timing

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > pipeline_sim2.py <<'EOF'
import json, bisect, math, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
def sim(T,L,stale,w=60.0):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10)          # evaluation/export time
        last=e-stale                              # latest sample the controller has
        k=math.floor((last-PHI)/2); ticks=[PHI+2*j for j in range(k-32,k+1)]
        samples=[(s,at(T,L,s)) for s in ticks]
        a=e-w; tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/w)
    return out
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("early RMSE / bias of published mean vs simulated, by series staleness at evaluation and by lead/hold:")
for lead,hold in [(0.0,0.1),(0.1,0.1),(0.2,0.1),(0.0,0.0),(0.2,0.0)]:
    T,L=step([(r['q']-lead,r['own']) for r in reqs]+[(r['ft']+hold,-r['own']) for r in reqs])
    line=f"  lead {lead:.1f} hold {hold:.1f}:"
    for stale in (0.0,1.0,2.0,3.0,4.0):
        s=sim(T,L,stale)
        line+=f"  stale {stale:.0f}s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} |"
    print(line)
# residual of late region at best early setting
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
for stale in (0.0,2.0):
    s=sim(T,L,stale)
    print(f"stale {stale:.0f}s, lead .1 hold .1: late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):.0f} bias {st.mean(pub[i]-s[i] for i in late):+.0f}; whole-run mean pub {st.mean(pub):.0f} sim {st.mean(s):.0f}")
json.dump(sim(T,L,2.0),open('sim_best.json','w'))
# 'not counted' cases: where is the record tick relative to the request window?
print("\n'not counted' cases: sample time relative to the request's engine window [q, ft]:")
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in reqs if r['q']<=s<r['ft']+0.1]
    if v==0 and held and all(r['eng'][-6:] in ('180495','173074') for r in held):
        for r in held: print(f"  t={t:.1f} sample s={s:.1f}: request q={r['q']:.2f} ft={r['ft']:.2f} (s-q={s-r['q']:+.2f}s, ft-s={r['ft']-s:+.2f}s) own {r['own']} eng {r['eng'][-6:]}")
EOF
python3 -I pipeline_sim2.py
```

**Bash** — Confirm the KV block size and test whether unexplained differences are whole blocks

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json, collections
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
c=[r['cached'] for r in reqs if r['cached']>0]
print('cached_prefix_tokens multiples of 784:', sum(1 for x in c if x%784==0),'/',len(c), '; of 64:', sum(1 for x in c if x%64==0))
diffs=[-10976,1746,-10976,-21879,-87808,-65264,64493,42336,24776,26656,-23520,-26308,59584]
print('unexplained diffs / 784:', [(d, round(d/784,2)) for d in diffs])
EOF
grep -o '"block_size":[^,}]*' /mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/k-5be017850000_pid-151035.0000.jsonl | head -2; grep -rn "block_size\|mamba\|align" /home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py | head -8
```

<details><summary>Bash result</summary>

```text
k-5be017850000_pid-151035.0000.jsonl: first-event offset 1792108114.5568; over 108966 events: median 1792108114.5568 min 1792108114.5568 max 1792108114.5568
k-5be017850000_pid-151035.0001.jsonl: first-event offset 1792108114.5568; over 131417 events: median 1792108114.5568 min 1792108114.5568 max 1792108114.5568
k-5be017850000_pid-173074.0000.jsonl: first-event offset 1792108114.5568; over 164439 events: median 1792108114.5568 min 1792108114.5568 max 1792108114.5568
k-5be017850000_pid-180495.0000.jsonl: first-event offset 1792108114.5568; over 137773 events: median 1792108114.5568 min 1792108114.5568 max 1792108114.5568
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
early RMSE / bias of published mean vs simulated, by series staleness at evaluation and by lead/hold:
  lead 0.0 hold 0.1:  stale 0s: rmse  1552 bias  -712 |  stale 1s: rmse  1552 bias  -712 |  stale 2s: rmse  1770 bias  -830 |  stale 3s: rmse  1770 bias  -830 |  stale 4s: rmse  1776 bias  +317 |
  lead 0.1 hold 0.1:  stale 0s: rmse  1678 bias  -957 |  stale 1s: rmse  1678 bias  -957 |  stale 2s: rmse  1916 bias -1082 |  stale 3s: rmse  1916 bias -1082 |  stale 4s: rmse  1773 bias   +84 |
  lead 0.2 hold 0.1:  stale 0s: rmse  2246 bias -1528 |  stale 1s: rmse  2246 bias -1528 |  stale 2s: rmse  2432 bias -1650 |  stale 3s: rmse  2432 bias -1650 |  stale 4s: rmse  1957 bias  -537 |
  lead 0.0 hold 0.0:  stale 0s: rmse  1618 bias  -341 |  stale 1s: rmse  1618 bias  -341 |  stale 2s: rmse  1815 bias  -475 |  stale 3s: rmse  1815 bias  -475 |  stale 4s: rmse  1998 bias  +658 |
  lead 0.2 hold 0.0:  stale 0s: rmse  2001 bias -1157 |  stale 1s: rmse  2001 bias -1157 |  stale 2s: rmse  2210 bias -1296 |  stale 3s: rmse  2210 bias -1296 |  stale 4s: rmse  1893 bias  -195 |
stale 0s, lead .1 hold .1: late rmse 6143 bias +1976; whole-run mean pub 15399 sim 14444
stale 2s, lead .1 hold .1: late rmse 6212 bias +1997; whole-run mean pub 15399 sim 14475

'not counted' cases: sample time relative to the request's engine window [q, ft]:
  t=999.7 sample s=998.3: request q=997.95 ft=998.29 (s-q=+0.35s, ft-s=-0.01s) own 6146 eng 180495
  t=1003.7 sample s=998.3: request q=997.95 ft=998.29 (s-q=+0.35s, ft-s=-0.01s) own 6146 eng 180495
  t=1007.7 sample s=998.3: request q=997.95 ft=998.29 (s-q=+0.35s, ft-s=-0.01s) own 6146 eng 180495
  t=1071.7 sample s=1068.3: request q=1067.89 ft=1068.24 (s-q=+0.41s, ft-s=-0.06s) own 1903 eng 180495
  t=1075.7 sample s=1068.3: request q=1067.89 ft=1068.24 (s-q=+0.41s, ft-s=-0.06s) own 1903 eng 180495
  t=1119.7 sample s=1118.3: request q=1118.13 ft=1118.24 (s-q=+0.17s, ft-s=-0.06s) own 1487 eng 180495
  t=1123.7 sample s=1118.3: request q=1118.13 ft=1118.24 (s-q=+0.17s, ft-s=-0.06s) own 1487 eng 180495
  t=1127.7 sample s=1118.3: request q=1118.13 ft=1118.24 (s-q=+0.17s, ft-s=-0.06s) own 1487 eng 180495
  t=1339.7 sample s=1338.3: request q=1337.90 ft=1338.30 (s-q=+0.40s, ft-s=-0.00s) own 5971 eng 180495
  t=1343.7 sample s=1338.3: request q=1337.90 ft=1338.30 (s-q=+0.40s, ft-s=-0.00s) own 5971 eng 180495
  t=1347.7 sample s=1338.3: request q=1337.90 ft=1338.30 (s-q=+0.40s, ft-s=-0.00s) own 5971 eng 180495
  t=1459.7 sample s=1458.3: request q=1458.00 ft=1458.38 (s-q=+0.30s, ft-s=+0.08s) own 5638 eng 180495
  t=1463.7 sample s=1458.3: request q=1458.00 ft=1458.38 (s-q=+0.30s, ft-s=+0.08s) own 5638 eng 180495
  t=1467.7 sample s=1458.3: request q=1458.00 ft=1458.38 (s-q=+0.30s, ft-s=+0.08s) own 5638 eng 180495
  t=1551.7 sample s=1548.3: request q=1547.88 ft=1548.29 (s-q=+0.42s, ft-s=-0.01s) own 5653 eng 173074
  t=1555.7 sample s=1548.3: request q=1547.88 ft=1548.29 (s-q=+0.42s, ft-s=-0.01s) own 5653 eng 173074
  t=1559.7 sample s=1558.3: request q=1558.01 ft=1558.26 (s-q=+0.29s, ft-s=-0.04s) own 3434 eng 173074
  t=1563.7 sample s=1558.3: request q=1558.01 ft=1558.26 (s-q=+0.29s, ft-s=-0.04s) own 3434 eng 173074
  t=1567.7 sample s=1558.3: request q=1558.01 ft=1558.26 (s-q=+0.29s, ft-s=-0.04s) own 3434 eng 173074
  t=1799.7 sample s=1798.3: request q=1797.88 ft=1798.23 (s-q=+0.42s, ft-s=-0.07s) own 3327 eng 173074
  t=1799.7 sample s=1798.3: request q=1797.47 ft=1798.33 (s-q=+0.83s, ft-s=+0.03s) own 13451 eng 180495
  t=1803.7 sample s=1798.3: request q=1797.88 ft=1798.23 (s-q=+0.42s, ft-s=-0.07s) own 3327 eng 173074
  t=1803.7 sample s=1798.3: request q=1797.47 ft=1798.33 (s-q=+0.83s, ft-s=+0.03s) own 13451 eng 180495
  t=1807.7 sample s=1798.3: request q=1797.88 ft=1798.23 (s-q=+0.42s, ft-s=-0.07s) own 3327 eng 173074
  t=1807.7 sample s=1798.3: request q=1797.47 ft=1798.33 (s-q=+0.83s, ft-s=+0.03s) own 13451 eng 180495
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
cached_prefix_tokens multiples of 784: 1451 / 1451 ; of 64: 746
unexplained diffs / 784: [(-10976, -14.0), (1746, 2.23), (-10976, -14.0), (-21879, -27.91), (-87808, -112.0), (-65264, -83.24), (64493, 82.26), (42336, 54.0), (24776, 31.6), (26656, 34.0), (-23520, -30.0), (-26308, -33.56), (59584, 76.0)]
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:189:# Steps the request needs alone: block-aligned chunks of at most the budget,
/home/ray/default/serve-llm-autoscaling-experiment/tools/load_breakdown.py:190:# then a partial-block tail in its own step (vLLM's Mamba "align" cache mode).
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Read how the router's in-memory metrics store compacts the gauge series

```bash
cd /mnt/local_storage/ray && grep -n "def prune_keys_and_compact_data" -A40 python/ray/serve/_private/metrics_utils.py | head -70; echo '#####'; grep -n "def add_metrics_point" -A14 python/ray/serve/_private/metrics_utils.py | head -30
```

**Bash** — Inspect the early-region residual pattern between the published mean and the simulation

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json, statistics as st
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json'))
print("early region: published vs simulated (lead .1 hold .1, 2s stale), every 3rd scrape; ratio pub/sim")
for i,(t,v,p) in enumerate(rows):
    if t<=650 and i%3==0: print(f"  t={t:6.1f} pub {p:7.0f} sim {sim[i]:7.0f} diff {p-sim[i]:+6.0f} ratio {p/sim[i] if sim[i] else float('nan'):.2f}")
e=[(p,sim[i]) for i,(t,v,p) in enumerate(rows) if t<=650]
print("early: sum pub/sum sim = %.3f; median ratio %.3f"%(sum(p for p,_ in e)/sum(s for _,s in e), st.median(p/s for p,s in e if s>1000)))
EOF
```

**Bash** — List the held requests behind each block-sized disagreement with session context

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json, math
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
conv={}
for rid,r in reqs.items():
    a=recs.get(aid(rid)); 
    if a: conv.setdefault(a['conversation_id'],[]).append((a['turn_index'],r))
cases=[(119.7,-10976),(411.7,1746),(491.7,-10976),(731.7,-21879),(919.7,-87808),(971.7,-65264),(1019.7,64493),(1059.7,42336),(1179.7,24776),(1191.7,26656),(1371.7,-23520),(1451.7,-26308),(1719.7,59584)]
B=784
for t,diff in cases:
    e=PSI+10*math.floor((t-PSI)/10); s=PHI+2*math.floor((e-PHI)/2)
    held=[(rid,r) for rid,r in reqs.items() if r['q']-0.5<=s<r['ft']+0.1]
    print(f"\nt={t} diff={diff:+d} ({diff/B:+.2f} blocks) sample s={s:.1f}; held requests:")
    for rid,r in held:
        a=recs.get(aid(rid),{}); ti=a.get('turn_index'); cid=a.get('conversation_id')
        prev=[x for x in conv.get(cid,[]) if x[0]==(ti or 0)-1]
        pe=prev[0][1]['eng'][-6:] if prev else '-'; gap=f"{r['q']-prev[0][1]['fin']:.1f}s" if prev and prev[0][1]['fin'] else '-'
        flag=''
        if diff>0 and abs(r['cached']-diff)<=1: flag='  <== router missed ALL of this request\'s cached blocks'
        if diff<0 and r['own']+diff>=0 and r['cached']==0: flag='  <== candidate: router found extra overlap on a cold prompt'
        print(f"   own {r['own']:6d} cached {r['cached']:6d} ({r['cached']//B:3d} blk) isl {r['isl']:6d} eng {r['eng'][-6:]} q {r['q']:.1f} ft {r['ft']:.1f} conv {str(cid)[:6]} turn {ti} prev-eng {pe} gap {gap}{flag}")
EOF
```

<details><summary>Bash result</summary>

```text
164:    def prune_keys_and_compact_data(self, start_timestamp_s: float):
165-        """Prune keys and compact data that are outdated.
166-
167-        For keys that haven't had new data recorded after the timestamp,
168-        remove them from the database.
169-        For keys that have, compact the datapoints that were recorded
170-        before the timestamp.
171-        """
172-        for key, datapoints in list(self.data.items()):
173-            if len(datapoints) == 0 or datapoints[-1].timestamp < start_timestamp_s:
174-                del self.data[key]
175-            else:
176-                self.data[key] = self._get_datapoints(key, start_timestamp_s)
177-
178-    def _get_datapoints(
179-        self, key: Hashable, window_start_timestamp_s: float
180-    ) -> TimeSeries:
181-        """Get all data points given key after window_start_timestamp_s"""
182-
183-        datapoints = self.data[key]
184-
185-        idx = bisect.bisect(
186-            a=datapoints,
187-            x=TimeStampedValue(
188-                timestamp=window_start_timestamp_s, value=0  # dummy value
189-            ),
190-        )
191-        return datapoints[idx:]
192-
193-    def get_latest(
194-        self,
195-        key: Hashable,
196-    ) -> Optional[float]:
197-        """Get the latest value for a given key."""
198-        if not self.data.get(key, None):
199-            return None
200-        return self.data[key][-1].value
201-
202-
203-def time_weighted_average(
204-    step_series: TimeSeries,
#####
149:    def add_metrics_point(self, data_points: Dict[Hashable, float], timestamp: float):
150-        """Push new data points to the store.
151-
152-        Args:
153-            data_points: dictionary containing the metrics values. The
154-              key should uniquely identify this time series
155-              and to be used to perform aggregation.
156-            timestamp: the unix epoch timestamp the metrics are
157-              collected at.
158-        """
159-
160-        for name, value in data_points.items():
161-            # Using in-sort to insert while maintaining sorted ordering.
162-            bisect.insort(a=self.data[name], x=TimeStampedValue(timestamp, value))
163-
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
early region: published vs simulated (lead .1 hold .1, 2s stale), every 3rd scrape; ratio pub/sim
  t=   3.7 pub       0 sim       0 diff     +0 ratio nan
  t=  15.7 pub    2170 sim    2124 diff    +46 ratio 1.02
  t=  27.7 pub    5147 sim    5678 diff   -532 ratio 0.91
  t=  39.7 pub    6664 sim    6622 diff    +43 ratio 1.01
  t=  51.7 pub    9553 sim   10148 diff   -595 ratio 0.94
  t=  63.7 pub   11035 sim   10896 diff   +139 ratio 1.01
  t=  75.7 pub    8869 sim    8772 diff    +97 ratio 1.01
  t=  87.7 pub    8434 sim   10021 diff  -1586 ratio 0.84
  t=  99.7 pub   10680 sim   10346 diff   +334 ratio 1.03
  t= 111.7 pub   11293 sim   12910 diff  -1618 ratio 0.87
  t= 123.7 pub   17632 sim   16908 diff   +723 ratio 1.04
  t= 135.7 pub   20665 sim   22793 diff  -2128 ratio 0.91
  t= 147.7 pub   18302 sim   21427 diff  -3125 ratio 0.85
  t= 159.7 pub   21071 sim   23660 diff  -2589 ratio 0.89
  t= 171.7 pub   16313 sim   20257 diff  -3944 ratio 0.81
  t= 183.7 pub   12894 sim   15517 diff  -2623 ratio 0.83
  t= 195.7 pub   11370 sim   11352 diff    +18 ratio 1.00
  t= 207.7 pub    8965 sim    9959 diff   -994 ratio 0.90
  t= 219.7 pub    9784 sim    9485 diff   +299 ratio 1.03
  t= 231.7 pub   10764 sim   12038 diff  -1274 ratio 0.89
  t= 243.7 pub   11049 sim   11766 diff   -717 ratio 0.94
  t= 255.7 pub   10237 sim   10915 diff   -678 ratio 0.94
  t= 267.7 pub    8202 sim   10477 diff  -2275 ratio 0.78
  t= 279.7 pub    7620 sim    7610 diff    +10 ratio 1.00
  t= 291.7 pub    3891 sim    5460 diff  -1568 ratio 0.71
  t= 303.7 pub    4917 sim    4924 diff     -7 ratio 1.00
  t= 315.7 pub    4913 sim    4866 diff    +46 ratio 1.01
  t= 327.7 pub    4786 sim    5663 diff   -877 ratio 0.85
  t= 339.7 pub    3836 sim    5110 diff  -1273 ratio 0.75
  t= 351.7 pub    4191 sim    6445 diff  -2254 ratio 0.65
  t= 363.7 pub    4122 sim    5458 diff  -1336 ratio 0.76
  t= 375.7 pub    4534 sim    5966 diff  -1432 ratio 0.76
  t= 387.7 pub    4824 sim    7310 diff  -2485 ratio 0.66
  t= 399.7 pub    5005 sim    4982 diff    +23 ratio 1.00
  t= 411.7 pub    7708 sim    8912 diff  -1204 ratio 0.86
  t= 423.7 pub   12592 sim   13268 diff   -676 ratio 0.95
  t= 435.7 pub   12229 sim   12600 diff   -371 ratio 0.97
  t= 447.7 pub   13891 sim   20074 diff  -6184 ratio 0.69
  t= 459.7 pub   17302 sim   22201 diff  -4900 ratio 0.78
  t= 471.7 pub   12899 sim   18747 diff  -5849 ratio 0.69
  t= 483.7 pub   10647 sim   14127 diff  -3480 ratio 0.75
  t= 495.7 pub   14559 sim   18502 diff  -3943 ratio 0.79
  t= 507.7 pub   15727 sim   16729 diff  -1002 ratio 0.94
  t= 519.7 pub   16146 sim   16240 diff    -94 ratio 0.99
  t= 531.7 pub   16672 sim   17359 diff   -687 ratio 0.96
  t= 543.7 pub   18409 sim   17740 diff   +669 ratio 1.04
  t= 555.7 pub   19954 sim   21076 diff  -1122 ratio 0.95
  t= 567.7 pub   20471 sim   21087 diff   -616 ratio 0.97
  t= 579.7 pub   24520 sim   23648 diff   +872 ratio 1.04
  t= 591.7 pub   22432 sim   21704 diff   +728 ratio 1.03
  t= 603.7 pub   20685 sim   21204 diff   -520 ratio 0.98
  t= 615.7 pub   15819 sim   15915 diff    -96 ratio 0.99
  t= 627.7 pub   11866 sim   12459 diff   -593 ratio 0.95
  t= 639.7 pub   10123 sim   10107 diff    +16 ratio 1.00
early: sum pub/sum sim = 0.915; median ratio 0.948
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
t=119.7 diff=-10976 (-14.00 blocks) sample s=118.3; held requests:
   own  41989 cached      0 (  0 blk) isl  41989 eng 151035 q 118.1 ft 120.8 conv 02bc0a turn 1 prev-eng 151035 gap 1.0s  <== candidate: router found extra overlap on a cold prompt

t=411.7 diff=+1746 (+2.23 blocks) sample s=408.3; held requests:
   own  88278 cached      0 (  0 blk) isl  88278 eng 151035 q 405.2 ft 410.8 conv 085411 turn 0 prev-eng - gap -
   own    645 cached  17248 ( 22 blk) isl  17893 eng 151035 q 407.6 ft 409.4 conv 02bc0a turn 2 prev-eng 151035 gap 3.4s
   own  17426 cached      0 (  0 blk) isl  17426 eng 151035 q 408.5 ft 410.9 conv 02bc0a turn 0 prev-eng - gap -

t=491.7 diff=-10976 (-14.00 blocks) sample s=488.3; held requests:
   own  76845 cached      0 (  0 blk) isl  76845 eng 151035 q 486.1 ft 491.6 conv 0a279a turn 4 prev-eng 151035 gap 1.4s  <== candidate: router found extra overlap on a cold prompt
   own   2531 cached 158368 (202 blk) isl 160899 eng 151035 q 486.1 ft 491.6 conv 019608 turn 3 prev-eng 151035 gap 215.4s
   own  18679 cached  84672 (108 blk) isl 103351 eng 151035 q 486.6 ft 491.7 conv 03e110 turn 5 prev-eng 151035 gap 3.8s
   own   2452 cached  54880 ( 70 blk) isl  57332 eng 151035 q 486.6 ft 491.7 conv 02bc0a turn 14 prev-eng 151035 gap 1.3s

t=731.7 diff=-21879 (-27.91 blocks) sample s=728.3; held requests:
   own  29968 cached  97216 (124 blk) isl 127184 eng 151035 q 727.7 ft 730.0 conv 069d7b turn 11 prev-eng 151035 gap 1.3s
   own   1829 cached 119168 (152 blk) isl 120997 eng 151035 q 728.8 ft 730.1 conv 0470d4 turn 29 prev-eng 151035 gap 2.6s
   own   2948 cached  86240 (110 blk) isl  89188 eng 151035 q 728.8 ft 730.1 conv 07dd40 turn 23 prev-eng 151035 gap 2.7s
   own 118171 cached      0 (  0 blk) isl 118171 eng 173074 q 726.1 ft 732.8 conv 0db32e turn 0 prev-eng - gap -  <== candidate: router found extra overlap on a cold prompt

t=919.7 diff=-87808 (-112.00 blocks) sample s=918.3; held requests:
   own  93793 cached      0 (  0 blk) isl  93793 eng 173074 q 915.8 ft 922.5 conv 11310d turn 2 prev-eng 173074 gap 15.1s  <== candidate: router found extra overlap on a cold prompt
   own  55113 cached      0 (  0 blk) isl  55113 eng 173074 q 916.6 ft 923.8 conv 117ebe turn 1 prev-eng 173074 gap 0.8s

t=971.7 diff=-65264 (-83.24 blocks) sample s=968.3; held requests:
   own  72743 cached      0 (  0 blk) isl  72743 eng 151035 q 965.8 ft 969.9 conv 07dd40 turn 8 prev-eng 151035 gap 332.0s  <== candidate: router found extra overlap on a cold prompt
   own   4961 cached  64288 ( 82 blk) isl  69249 eng 151035 q 967.3 ft 969.9 conv 0bcd99 turn 4 prev-eng 151035 gap 61.2s
   own   4112 cached  28224 ( 36 blk) isl  32336 eng 180495 q 968.0 ft 968.2 conv 069d7b turn 1 prev-eng 180495 gap 0.4s
   own  29771 cached      0 (  0 blk) isl  29771 eng 180495 q 968.7 ft 970.2 conv 069d7b turn 0 prev-eng - gap -

t=1019.7 diff=+64493 (+82.26 blocks) sample s=1018.3; held requests:

t=1059.7 diff=+42336 (+54.00 blocks) sample s=1058.3; held requests:
   own 122366 cached      0 (  0 blk) isl 122366 eng 151035 q 1054.9 ft 1063.5 conv 03e110 turn 6 prev-eng 151035 gap 290.4s
   own  20939 cached 144256 (184 blk) isl 165195 eng 151035 q 1057.2 ft 1063.5 conv 05c249 turn 22 prev-eng 151035 gap -
   own   4524 cached 150528 (192 blk) isl 155052 eng 173074 q 1058.1 ft 1058.5 conv 019608 turn 17 prev-eng 173074 gap 2.6s

t=1179.7 diff=+24776 (+31.60 blocks) sample s=1178.3; held requests:
   own    351 cached 123872 (158 blk) isl 124223 eng 151035 q 1178.4 ft 1178.5 conv 0470d4 turn 33 prev-eng 151035 gap 2.0s
   own    663 cached  97216 (124 blk) isl  97879 eng 173074 q 1178.2 ft 1178.3 conv 1007f0 turn 35 prev-eng 173074 gap 1.3s

t=1191.7 diff=+26656 (+34.00 blocks) sample s=1188.3; held requests:
   own    590 cached 125440 (160 blk) isl 126030 eng 151035 q 1188.3 ft 1188.4 conv 0470d4 turn 35 prev-eng 151035 gap 1.9s

t=1371.7 diff=-23520 (-30.00 blocks) sample s=1368.3; held requests:
   own  36553 cached      0 (  0 blk) isl  36553 eng 180495 q 1367.5 ft 1369.6 conv 1007f0 turn 0 prev-eng - gap -  <== candidate: router found extra overlap on a cold prompt
   own   6907 cached  36064 ( 46 blk) isl  42971 eng 180495 q 1368.3 ft 1369.8 conv 1007f0 turn 1 prev-eng 180495 gap 1.3s

t=1451.7 diff=-26308 (-33.56 blocks) sample s=1448.3; held requests:
   own 139927 cached      0 (  0 blk) isl 139927 eng 151035 q 1447.7 ft 1456.0 conv 0bc9c4 turn 5 prev-eng 151035 gap 399.1s  <== candidate: router found extra overlap on a cold prompt
   own   2788 cached  56448 ( 72 blk) isl  59236 eng 180495 q 1448.1 ft 1448.3 conv 1007f0 turn 5 prev-eng 180495 gap 0.7s

t=1719.7 diff=+59584 (+76.00 blocks) sample s=1718.3; held requests:
   own  84835 cached      0 (  0 blk) isl  84835 eng 151035 q 1715.3 ft 1720.3 conv 002001 turn 10 prev-eng 151035 gap 1437.7s
   own   5852 cached 158368 (202 blk) isl 164220 eng 151035 q 1716.8 ft 1720.3 conv 0470d4 turn 51 prev-eng 151035 gap 2.8s
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Read the metrics tick scheduling and the router's tokenize path

```bash
cd /mnt/local_storage/ray && sed -n 64,100p python/ray/serve/_private/metrics_utils.py; echo '#####'; sed -n 120,175p python/ray/llm/_internal/serve/routing_policies/kv_aware/vllm/tokenizer.py; echo '#####'; sed -n 200,235p python/ray/llm/_internal/serve/core/ingress/router.py
```

**Bash** — Simulate record ticks slipping during tokenization and see whether it closes the published-mean residual

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > slip_sim.py <<'EOF'
import json, bisect, math, statistics as st
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
S=json.load(open('gauge_decomp_state.json')); reqs=sorted(S['reqs'].values(), key=lambda r:r['q'])
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI=9.6
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
LEAD,HOLD=0.1,0.1
T,L=step([(r['q']-LEAD,r['own']) for r in reqs]+[(r['ft']+HOLD,-r['own']) for r in reqs])
def tick_schedule(phi, c, end=1900.0):
    """record ticks every 2s after the previous one completes; a tick that falls inside a
    blocked interval [q-lead-c*isl, q-lead] (tokenizing on the loop before admission) is
    delayed to the block's end."""
    blocks=sorted((r['q']-LEAD-c*r['isl'], r['q']-LEAD) for r in reqs)
    bs=[b[0] for b in blocks]
    ticks=[]; t=phi
    while t<end:
        # find a block containing t
        i=bisect.bisect_right(bs,t)-1
        if i>=0 and blocks[i][0]<=t<blocks[i][1]: t=blocks[i][1]
        ticks.append(t); t+=2.0
    return ticks
def sim(ticks, w=60.0):
    out=[]; raw=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10)
        j=bisect.bisect_right(ticks,e)
        samples=[(s,at(T,L,s)) for s in ticks[max(0,j-35):j]]
        a=e-w; tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/w); raw.append(samples[-1][1] if samples else 0)
    return out, raw
pub=[p for _,_,p in rows]; rawv=[v for _,v,_ in rows]
early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("tick slip = c x prompt tokens per admission; early/late RMSE and bias of published mean vs simulation; exact raw matches")
for c in [0, 2e-6, 4e-6, 6e-6, 8e-6, 10e-6, 12e-6, 15e-6, 20e-6]:
    best=None
    for phi in [0.1*i for i in range(20)]:
        s,raw=sim(tick_schedule(phi,c))
        r_e=rmse([s[i] for i in early],[pub[i] for i in early])
        if best is None or r_e<best[0]: best=(r_e,phi,s,raw)
    r_e,phi,s,raw=best
    print(f"  c={c*1e6:4.0f} us/token phi={phi:.1f}: early rmse {r_e:5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} | late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):5.0f} bias {st.mean(pub[i]-s[i] for i in late):+5.0f} | raw exact: early {sum(1 for i in early if abs(raw[i]-rawv[i])<=1)}/{len(early)} late {sum(1 for i in late if abs(raw[i]-rawv[i])<=1)}/{len(late)} | whole-run sim mean {st.mean(s):.0f} vs pub {st.mean(pub):.0f}")
EOF
python3 -I slip_sim.py
```

<details><summary>Bash result</summary>

```text
def start(self):
        self.stop_event.clear()

    async def metrics_task(self, name: str):
        """Periodically runs `task_func` every `interval_s` until `stop_event` is set.

        If `task_func` raises an error, an exception will be logged.
        Supports both sync and async task functions.
        """

        wait_for_stop_event = asyncio.create_task(self.stop_event.wait())
        while True:
            if wait_for_stop_event.done():
                return

            try:
                task_func = self._tasks[name].task_func
                # Check if the function is a coroutine function
                if asyncio.iscoroutinefunction(task_func):
                    await task_func()
                else:
                    task_func()
            except Exception as e:
                logger.exception(f"Failed to run metrics task '{name}': {e}")

            sleep_task = asyncio.create_task(
                self._async_sleep(self._tasks[name].interval_s)
            )
            await asyncio.wait(
                [sleep_task, wait_for_stop_event],
                return_when=asyncio.FIRST_COMPLETED,
            )

            if not sleep_task.done():
                sleep_task.cancel()

    def register_or_update_task(
#####
        )
        # Check the templates the renderer actually uses, including the model's
        # default when no chat-template override is configured.
        self._merge_inline_system = (
            AnthropicServingMessages._should_merge_inline_system(self._renderer)
        )
        self._disabled_thinking_effort = (
            frontend_args.anthropic_disabled_thinking_effort
        )
        logger.info(
            "In-process pre-routing tokenizer ready for %s",
            self._model_config.model,
        )

    async def tokenize(
        self, payload: Dict[str, Any], *, request_path: Optional[str] = None
    ) -> Optional[List[int]]:
        """Tokenize a request ``payload`` into prompt token IDs.

        Args:
            payload: The request body, already parsed into a dict by ``LLMRouter``.
            request_path: The original API path forwarded by HAProxy, if present.

        Returns:
            The prompt token IDs, or ``None`` for bodies that are not routed on.

        Raises:
            TokenizeError: vLLM's prompt renderer rejected the request.
        """
        request = build_tokenize_request(
            payload,
            merge_inline_system=self._merge_inline_system,
            request_path=request_path,
        )
        if request is None:
            return None

        try:
            if isinstance(request, ChatCompletionRequest):
                # Conversion defaults disabled thinking to "none". Resolve the
                # configured effort before rendering, since it can change tokens.
                if (
                    request_path.endswith("/v1/messages")
                    and request.reasoning_effort == "none"
                ):
                    request.reasoning_effort = (
                        await self._get_disabled_thinking_effort()
                    )
                rendered_inputs = await self._render_chat(request)
            else:
                rendered_inputs = await self._render_completion(request)
        except TokenizeError:
            raise
        except (ValueError, VLLMClientError, jinja2.TemplateError) as e:
            # vLLM maps bad inputs and chat-template errors to 400; other
            # exceptions are real bugs and should surface, not degrade routing.
#####
                "requests. For truncated bodies, raise HAProxy's routing body "
                "limit.",
                body_truncated,
            )
        # Tokenize only a parseable, routable body; a truncated or unparseable
        # body has no routing payload, so fall back to token-less routing.
        request_token_ids = None
        stage_token_ids = False
        if self._tokenizer is not None and routing_payload is not None:
            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.prompt_token_forwarding import (  # noqa: E501
                is_text_only_chat,
            )
            from ray.llm._internal.serve.routing_policies.kv_aware.vllm.tokenizer import (  # noqa: E501
                TokenizeError,
            )

            payload = vars(routing_payload)
            request_path = request.headers.get(SERVE_INGRESS_ROUTER_REQUEST_PATH_HEADER)
            try:
                request_token_ids = await self._tokenizer.tokenize(
                    payload, request_path=request_path
                )
            except TokenizeError as e:
                raise HTTPException(status_code=e.status_code, detail=e.message)
            # Media needs engine preprocessing, not just prompt token IDs.
            stage_token_ids = is_text_only_chat(payload.get("messages", []))
        # HAProxy forwards the configured session header on the same name,
        # but use the same case-insensitive, separator-tolerant matcher as
        # proxy.py / ingress.py so a `-`/`_` rewrite anywhere in the path
        # doesn't silently drop session affinity.
        session_id = next(
            (v for k, v in request.headers.items() if _matches_session_id_header(k)),
            None,
        )
        multiplexed_model_id = self._get_multiplexed_model_id(request_body)
        handle = self._get_handle(multiplexed_model_id)
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
tick slip = c x prompt tokens per admission; early/late RMSE and bias of published mean vs simulation; exact raw matches
  c=   0 us/token phi=0.3: early rmse  1678 bias  -957 | late rmse  6143 bias +1976 | raw exact: early 146/162 late 231/303 | whole-run sim mean 14444 vs pub 15399
  c=   2 us/token phi=1.0: early rmse  2239 bias -1295 | late rmse  6257 bias +1356 | raw exact: early 115/162 late 137/303 | whole-run sim mean 14966 vs pub 15399
  c=   4 us/token phi=0.0: early rmse  2413 bias -1467 | late rmse  6068 bias  +724 | raw exact: early 112/162 late 102/303 | whole-run sim mean 15438 vs pub 15399
  c=   6 us/token phi=0.1: early rmse  2801 bias -2057 | late rmse  6290 bias  -146 | raw exact: early 105/162 late 85/303 | whole-run sim mean 16210 vs pub 15399
  c=   8 us/token phi=1.9: early rmse  3512 bias -2169 | late rmse  5685 bias  -564 | raw exact: early 103/162 late 83/303 | whole-run sim mean 16522 vs pub 15399
  c=  10 us/token phi=0.1: early rmse  4134 bias -3042 | late rmse  6342 bias  -464 | raw exact: early 94/162 late 77/303 | whole-run sim mean 16760 vs pub 15399
  c=  12 us/token phi=0.1: early rmse  3841 bias -2789 | late rmse  6596 bias -1426 | raw exact: early 82/162 late 73/303 | whole-run sim mean 17299 vs pub 15399
  c=  15 us/token phi=0.1: early rmse  4490 bias -3526 | late rmse  6526 bias -2226 | raw exact: early 68/162 late 72/303 | whole-run sim mean 18077 vs pub 15399
  c=  20 us/token phi=0.3: early rmse  4413 bias -3218 | late rmse  6754 bias -1653 | raw exact: early 61/162 late 39/303 | whole-run sim mean 17597 vs pub 15399
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Read how router metric series are encoded on the push and decoded at the controller

```bash
cd /mnt/local_storage/ray && grep -n "def _router_metric_samples" -A25 python/ray/serve/_private/autoscaling_state.py | head -40; echo '#####'; grep -n "RAY_SERVE_COLUMNAR_AUTOSCALING_METRICS\b" -A3 python/ray/serve/_private/constants.py | head -8; echo '#####'; grep -n "router_metrics\|def encode\|def _encode_samples\|dedup\|compact\|max_points\|MAX_\|downsample\|float16\|quantiz" python/ray/serve/_private/autoscaling_metrics_codec.py | head -40
```

**Bash** — Read when the controller evaluates the policy and exports its metrics

```bash
cd /mnt/local_storage/ray && sed -n 670,770p python/ray/serve/_private/autoscaling_state.py
```

**Bash** — Test which effective averaging window best reproduces the published mean

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > window_fit.py <<'EOF'
import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]
def sim(w, drop_first=0, mode='full'):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        ticks=[PHI+2*j for j in range(k-40,k+1)]
        samples=[(s,at(T,L,s)) for s in ticks if s>=e-w][drop_first:]
        if not samples: out.append(0); continue
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
print("published mean vs simulated with different effective window lengths (early region):")
for w in (20,30,40,50,58,60,70,80,100):
    s=sim(w); print(f"  window {w:3d}s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
# alternative: samples carry forward only 2s each (no carry to 'now'), i.e. plain average of samples in window
def sim_avg(w):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        vals=[at(T,L,PHI+2*j) for j in range(k-40,k+1) if PHI+2*j>=e-w]
        out.append(st.mean(vals))
    return out
s=sim_avg(60); print(f"  plain average of 2s samples, 60s: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
# what if the series the controller holds is only the last N samples per push (e.g. a cap)?
for n in (5,10,15,20,25,30):
    def sim_cap(n):
        out=[]
        for t,v,p in rows:
            e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
            samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-n+1,k+1)]
            a=samples[0][0]; tot=0.0
            for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
                if t2>t1: tot+=v1*(t2-t1)
            out.append(tot/(e-a))
        return out
    s=sim_cap(n); print(f"  last {n:2d} samples only: rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f}")
EOF
python3 -I window_fit.py
```

<details><summary>Bash result</summary>

```text
def record_autoscaling_metrics(
        self,
        decision_num_replicas: int,
        total_num_requests: float,
        policy_execution_time_ms: float,
        policy_scope: str,
        policy_metrics: Optional[Dict[str, Any]] = None,
    ):
        tags = {
            "deployment": self._deployment_id.name,
            "application": self._deployment_id.app_name,
        }
        self.autoscaling_decision_gauge.set(decision_num_replicas, tags=tags)
        self.autoscaling_total_requests_gauge.set(total_num_requests, tags=tags)
        # Stash the decision's value for the scale up/down log to reuse.
        self._last_decision_total_num_requests = total_num_requests
        self.autoscaling_policy_execution_time_gauge.set(
            policy_execution_time_ms, tags={**tags, "policy_scope": policy_scope}
        )
        self.autoscaling_target_ongoing_requests_gauge.set(
            self._config.get_target_ongoing_requests(), tags=tags
        )
        for name, value in (policy_metrics or {}).items():
            # Skip malformed entries so a policy bug can't break the control loop.
            if (
                isinstance(name, str)
                and isinstance(value, (int, float))
                and math.isfinite(value)
            ):
                self.autoscaling_policy_metric_gauge.set(
                    float(value), tags={**tags, "metric": name}
                )

    def get_decision_num_replicas(
        self, curr_target_num_replicas: int, _skip_bound_check: bool = False
    ) -> int:
        """Decide the target number of replicas to autoscale to.

        The decision is based off of the number of requests received
        for this deployment. After the decision number of replicas is
        returned by the policy, it is then bounded by the bounds min
        and max adjusted by the target capacity and returned. If
        `_skip_bound_check` is True, then the bounds are not applied.
        """
        if self._policy is None:
            raise ValueError(f"Policy is not set for deployment {self._deployment_id}.")
        autoscaling_context = self.get_autoscaling_context(curr_target_num_replicas)

        # Time the policy execution
        start_time = time.time()
        decision_num_replicas, self._policy_state = self._policy(autoscaling_context)
        # The policy can return a float value.
        if isinstance(decision_num_replicas, float):
            decision_num_replicas = math.ceil(decision_num_replicas)
        policy_execution_time_ms = (time.time() - start_time) * 1000

        self.record_autoscaling_metrics(
            decision_num_replicas,
            autoscaling_context.total_num_requests,
            policy_execution_time_ms,
            "deployment",
            (self._policy_state or {}).get(SERVE_AUTOSCALING_POLICY_METRICS_KEY),
        )

        if _skip_bound_check:
            return decision_num_replicas

        return self.apply_bounds(decision_num_replicas)

    def get_autoscaling_context(
        self,
        curr_target_num_replicas,
        override_policy_state: Optional[Dict[str, Any]] = None,
    ) -> AutoscalingContext:
        # Adding this to overwrite policy state during application level autoscaling
        if override_policy_state is not None:
            current_policy_state = override_policy_state.copy()
        elif self._policy_state is not None:
            current_policy_state = self._policy_state.copy()
        else:
            current_policy_state = {}
        return AutoscalingContext(
            deployment_id=self._deployment_id,
            deployment_name=self._deployment_id.name,
            app_name=self._deployment_id.app_name,
            current_num_replicas=len(self._running_replicas),
            target_num_replicas=curr_target_num_replicas,
            running_replicas=self._running_replicas,
            total_num_requests=self.get_total_num_requests,
            capacity_adjusted_min_replicas=self.get_num_replicas_lower_bound(),
            capacity_adjusted_max_replicas=self.get_num_replicas_upper_bound(),
            policy_state=current_policy_state,
            current_time=time.time(),
            config=self._config,
            total_queued_requests=self._get_queued_requests,
            aggregated_metrics=self._get_aggregated_custom_metrics,
            raw_metrics=self._get_raw_custom_metrics,
            raw_router_metrics=self._get_raw_router_metrics,
            last_scale_up_time=self._last_scale_up_time,
            last_scale_down_time=self._last_scale_down_time,
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
266:def _router_metric_samples(payload: FlatHandleReport) -> Dict[str, MetricSamples]:
267-    """Return router-owned metrics as zero-copy sample views."""
268-    result: Dict[str, MetricSamples] = {}
269-    for metric_index, start, length in payload["router_entries"]:
270-        name = payload["router_metric_names"][int(metric_index)]
271-        a, b = int(start), int(start + length)
272-        result[name] = MetricSamples(
273-            payload["router_ts"][a:b], payload["router_val"][a:b]
274-        )
275-    return result
276-
277-
278-def _log_dropped_handle(
279-    report: HandleReport, timeout_s: float, dead_actor: bool
280-) -> None:
281-    """One copy of the operator-facing drop text, shared by both report types.
282-    total_requests gates it so handles that never took traffic stay quiet."""
283-    peak_requests = report.total_requests
284-    if peak_requests <= 0:
285-        return
286-    if dead_actor:
287-        logger.debug(
288-            f"Dropping metrics for handle '{report.handle_id}' because the Serve "
289-            f"actor it was on ({report.actor_id}) is no longer "
290-            f"alive. Its peak ongoing requests was {peak_requests}."
291-        )
#####
474:RAY_SERVE_COLUMNAR_AUTOSCALING_METRICS = get_env_bool(
475:    "RAY_SERVE_COLUMNAR_AUTOSCALING_METRICS", "1"
476-)
477-
478-# Feature flag to turn on node locality routing for proxies. On by default.
#####
87:def encode(report: HandleMetricReport) -> bytes:
129:        name for name in rep.router_metrics if name != QUEUED_REQUESTS_KEY
134:        series = rep.router_metrics[name]
179:    # the columnar float64 payload is already compact and the win is the
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
published mean vs simulated with different effective window lengths (early region):
  window  20s: rmse  7662 bias -1039
  window  30s: rmse  4962 bias -1109
  window  40s: rmse  4166 bias -1020
  window  50s: rmse  2900 bias  -970
  window  58s: rmse  1930 bias  -859
  window  60s: rmse  1622 bias  -908
  window  70s: rmse  2280 bias  -853
  window  80s: rmse  2671 bias  -780
  window 100s: rmse  2723 bias  -886
  plain average of 2s samples, 60s: rmse  1745 bias  -978
  last  5 samples only: rmse 11487 bias -1007
  last 10 samples only: rmse  7662 bias -1039
  last 15 samples only: rmse  4962 bias -1109
  last 20 samples only: rmse  4166 bias -1020
  last 25 samples only: rmse  2900 bias  -970
  last 30 samples only: rmse  1622 bias  -908
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Read the columnar metrics encoder for any subsampling of router series

```bash
cd /mnt/local_storage/ray && sed -n 87,180p python/ray/serve/_private/autoscaling_metrics_codec.py
```

**Bash** — Solve for the per-tick series staleness that would reproduce each published value

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > stale_fit.py <<'EOF'
import json, bisect, math, statistics as st, collections
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
def pm(e, stale, w=60.0):
    last=e-stale; k=math.floor((last-PHI)/2)
    samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
    if not samples: return 0.0
    a=max(e-w,samples[0][0]); tot=0.0
    for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
        lo,hi=max(t1,a),min(t2,e)
        if hi>lo: tot+=v1*(hi-lo)
    return tot/(e-a)
# one entry per export tick
ticks={}
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(v,p))
out=[]
for e,(v,p) in sorted(ticks.items()):
    if e<0: continue
    cands=[(abs(pm(e,d)-p),d) for d in [0.5*i for i in range(0,25)]]
    r0=abs(pm(e,0)-p); r,d=min(cands)
    out.append((e,p,pm(e,0),r0,d,r))
early=[o for o in out if o[0]<=650]; late=[o for o in out if o[0]>650]
for name,grp in (('early',early),('late',late)):
    print(f"{name}: ticks {len(grp)}; residual with stale=0: rmse {math.sqrt(st.mean(o[3]**2 for o in grp)):.0f}; best per-tick staleness: rmse {math.sqrt(st.mean(o[5]**2 for o in grp)):.0f}; staleness distribution: {sorted(collections.Counter(o[4] for o in grp).items())}")
print("\nearly ticks with residual > 1500 at stale=0: (e, pub, sim0, best stale, residual after)")
for o in early:
    if o[3]>1500: print(f"  e={o[0]:.1f} pub {o[1]:6.0f} sim {o[2]:6.0f} diff {o[1]-o[2]:+6.0f} -> stale {o[4]:4.1f}s residual {o[5]:5.0f}")
EOF
python3 -I stale_fit.py
```

**Bash** — Fit the record tick phase per window to detect drift

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > phase_drift.py <<'EOF'
import json, bisect, math
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI=9.6
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
T,L=step([(r['q']-0.1,r['own']) for r in reqs]+[(r['ft']+0.1,-r['own']) for r in reqs])
print("record-tick phase fitted per 300s window (exact matches of exported raw gauge, nonzero exported values only):")
for lo in range(0,1860,300):
    sub=[(t,v) for t,v,_ in rows if lo<=t<lo+300]
    best=[]
    for phi in [0.05*i for i in range(40)]:
        m=sum(1 for t,v in sub if v>0 and abs(at(T,L,phi+2*math.floor((PSI+10*math.floor((t-PSI)/10)-phi)/2))-v)<=1)
        best.append((m,phi))
    best.sort(reverse=True); nz=sum(1 for t,v in sub if v>0)
    print(f"  {lo:4d}-{lo+300:4d}: nonzero exported {nz:3d}; best phases {[(m,round(p,2)) for m,p in best[:4]]}")
EOF
python3 -I phase_drift.py
```

<details><summary>Bash result</summary>

```text
def encode(report: HandleMetricReport) -> bytes:
    return _encode_handle(report)


def should_encode_columnar(report: HandleMetricReport) -> bool:
    """Whether a producer should serialize this report columnar (vs Python objects).

    Handle reports carry every replica the handle routes to, so they are the ones worth
    encoding natively; replica reports carry one and stay on the object path. The format
    self-identifies on the wire (see is_columnar), so mixed senders decode correctly.

    Args:
        report: the metric report about to be serialized.

    Returns:
        True to encode columnar, False to use the Python-object path.
    """
    return isinstance(report, HandleMetricReport)


def _encode_handle(rep: HandleMetricReport) -> bytes:
    metric_names = list(rep.metrics.keys())
    mi = {m: i for i, m in enumerate(metric_names)}
    replica_keys: List[str] = []
    rk: Dict[str, int] = {}
    entries, series_list = [], []
    for m in metric_names:
        for key, series in rep.metrics[m].items():
            if key not in rk:
                rk[key] = len(replica_keys)
                replica_keys.append(key)
            entries.append((mi[m], rk[key], 0, len(series)))  # offset filled below
            series_list.append(series)
    ts, val = _flatten_series(series_list)
    # fill offsets
    off = 0
    for i, (a, b, _o, n) in enumerate(entries):
        entries[i] = (a, b, off, n)
        off += n
    q_ts = [p.timestamp for p in rep.queued_requests]
    q_val = [p.value for p in rep.queued_requests]
    router_metric_names = [
        name for name in rep.router_metrics if name != QUEUED_REQUESTS_KEY
    ]
    router_entries, router_series = [], []
    router_off = 0
    for i, name in enumerate(router_metric_names):
        series = rep.router_metrics[name]
        router_entries.append((i, router_off, len(series)))
        router_series.append(series)
        router_off += len(series)
    router_ts, router_val = _flatten_series(router_series)
    arrays = {
        "entries": (
            np.array(entries, dtype="<i8") if entries else np.zeros((0, 4), "<i8")
        ).reshape(-1, 4),
        "ts": np.array(ts, dtype="<f8"),
        "val": np.array(val, dtype="<f8"),
        "q_ts": np.array(q_ts, dtype="<f8"),
        "q_val": np.array(q_val, dtype="<f8"),
        "router_entries": (
            np.array(router_entries, dtype="<i8")
            if router_entries
            else np.zeros((0, 3), "<i8")
        ).reshape(-1, 3),
        "router_ts": np.array(router_ts, dtype="<f8"),
        "router_val": np.array(router_val, dtype="<f8"),
    }
    descriptors, blob = _pack(arrays)
    header = {
        "type": "handle",
        "deployment": [rep.deployment_id.name, rep.deployment_id.app_name],
        "handle_id": rep.handle_id,
        "actor_id": rep.actor_id,
        "handle_source": rep.handle_source.value,
        "timestamp": rep.timestamp,
        "metric_names": metric_names,
        "replica_keys": replica_keys,
        "router_metric_names": router_metric_names,
        "arrays": descriptors,
    }
    return _frame(header, blob)


def _frame(header: dict, blob: bytes) -> bytes:
    hb = json.dumps(header).encode()
    # Pad so the arrays land 8-byte aligned: np.frombuffer tolerates an unaligned
    # offset, but unaligned SIMD access costs more on aarch64, which premerge targets.
    # json.loads ignores the trailing spaces.
    hb += b" " * ((4 - len(hb)) % 8)
    # level=1: metric reports are serialized on the producer hot path (per handle,
    # every metrics interval), so minimize compression CPU;
    # the columnar float64 payload is already compact and the win is the
    # native decode, not the wire size.
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
early: ticks 65; residual with stale=0: rmse 1626; best per-tick staleness: rmse 916; staleness distribution: [(0.0, 29), (1.5, 11), (3.5, 11), (5.5, 4), (7.5, 4), (9.5, 4), (11.5, 2)]
late: ticks 121; residual with stale=0: rmse 6158; best per-tick staleness: rmse 4741; staleness distribution: [(0.0, 19), (1.5, 15), (3.5, 14), (5.5, 16), (7.5, 19), (9.5, 15), (11.5, 23)]

early ticks with residual > 1500 at stale=0: (e, pub, sim0, best stale, residual after)
  e=129.6 pub  20665 sim  22296 diff  -1631 -> stale  3.5s residual   452
  e=139.6 pub  18302 sim  20637 diff  -2335 -> stale  5.5s residual   534
  e=149.6 pub  21350 sim  24130 diff  -2780 -> stale  3.5s residual  2089
  e=159.6 pub  21071 sim  23837 diff  -2765 -> stale  0.0s residual  2765
  e=169.6 pub  16313 sim  18255 diff  -1942 -> stale  3.5s residual   409
  e=179.6 pub  12894 sim  14447 diff  -1553 -> stale  3.5s residual   388
  e=379.6 pub   4824 sim   6345 diff  -1520 -> stale  9.5s residual    21
  e=439.6 pub  13891 sim  18819 diff  -4929 -> stale  5.5s residual  1274
  e=449.6 pub  15516 sim  20224 diff  -4708 -> stale  1.5s residual  3886
  e=459.6 pub  17302 sim  21395 diff  -4093 -> stale 11.5s residual  1423
  e=469.6 pub  12899 sim  17019 diff  -4121 -> stale  7.5s residual  2447
  e=479.6 pub  10647 sim  14276 diff  -3629 -> stale  7.5s residual  2277
  e=489.6 pub  14559 sim  19184 diff  -4625 -> stale  5.5s residual    29
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
record-tick phase fitted per 300s window (exact matches of exported raw gauge, nonzero exported values only):
     0- 300: nonzero exported  24; best phases [(21, 1.95), (21, 1.9), (21, 1.85), (21, 1.8)]
   300- 600: nonzero exported  34; best phases [(27, 0.3), (24, 0.45), (24, 0.35), (24, 0.25)]
   600- 900: nonzero exported  36; best phases [(31, 0.3), (31, 0.25), (28, 0.4), (28, 0.35)]
   900-1200: nonzero exported  37; best phases [(18, 0.25), (16, 0.3), (13, 0.4), (13, 0.35)]
  1200-1500: nonzero exported  31; best phases [(27, 0.4), (25, 0.5), (25, 0.45), (25, 0.35)]
  1500-1800: nonzero exported  31; best phases [(28, 0.55), (28, 0.5), (28, 0.45), (28, 0.4)]
  1800-2100: nonzero exported   0; best phases [(0, 1.95), (0, 1.9), (0, 1.85), (0, 1.8)]
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Look for TTL or capacity limits on the dynamo KV indexer

```bash
P=/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.pyi; grep -n -i "ttl\|capacity\|evict\|approximate\|max_blocks\|indexer" $P | head -40; echo '#####'; grep -n "class KvRouterConfig" -A60 $P | grep -i -B1 -A3 "ttl\|approximate\|capacity" | head -60
```

**Bash** — Test whether the router's over-count tracks cached volume from long-lived sessions

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > age_corr.py <<'EOF'
import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json')); PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
t0=json.load(open('/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
conv_start={}
for rid,r in reqs.items():
    a=recs.get(aid(rid))
    if a: conv_start[a['conversation_id']]=min(conv_start.get(a['conversation_id'],1e9), a['request_start_ns']/1e9-t0)
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['age']=(r['q']-conv_start[a['conversation_id']]) if a else 0; r['turn']=a.get('turn_index') if a else None; R.append(r)
# per export tick: residual (pub - sim) vs. cached volume held in the 60s window, split by session age
ticks={}
for i,(t,v,p) in enumerate(rows):
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(p,sim[i]))
X_old=[];X_young=[];X_cold=[];Y=[];E=[]
for e,(p,s) in sorted(ticks.items()):
    if e<60: continue
    held=[r for r in R if r['ft']>e-60 and r['q']<e]
    w=lambda r: max(0,min(r['ft'],e)-max(r['q'],e-60))/60.0   # exposure fraction in window
    X_old.append(sum(r['cached']*w(r) for r in held if r['age']>600)); X_young.append(sum(r['cached']*w(r) for r in held if r['age']<=600)); X_cold.append(sum(r['own']*w(r) for r in held if r['kind']=='cold'))
    Y.append(p-s); E.append(e)
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y)); return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy) if sx and sy else 0
print("residual (published - simulated rebuild) vs exposure-weighted volume held in the window:")
print(f"  corr with cached tokens of sessions older than 600s: {corr(X_old,Y):+.3f}")
print(f"  corr with cached tokens of sessions younger than 600s: {corr(X_young,Y):+.3f}")
print(f"  corr with cold uncached tokens: {corr(X_cold,Y):+.3f}")
# regression Y ~ a*X_old + b*X_young (through origin)
s11=sum(a*a for a in X_old); s22=sum(b*b for b in X_young); s12=sum(a*b for a,b in zip(X_old,X_young)); s1y=sum(a*y for a,y in zip(X_old,Y)); s2y=sum(b*y for b,y in zip(X_young,Y))
det=s11*s22-s12*s12; a=(s1y*s22-s2y*s12)/det; b=(s11*s2y-s12*s1y)/det
print(f"  residual ~ {a:+.3f} x old-session cached + {b:+.3f} x young-session cached   (fraction of cached tokens the router appears to miss)")
# excursion at 1240-1300: held requests with ages
print("\nheld requests at export tick 1279.6 (pub peak) with session age and cached blocks:")
e=1279.6; held=sorted([r for r in R if r['ft']>e-60 and r['q']<e], key=lambda r:-r['cached'])
for r in held[:14]: print(f"  q {r['q']:.1f} ft {r['ft']:.1f} own {r['own']:6d} cached {r['cached']:6d} ({r['cached']//784:3d} blk) turn {r['turn']} session age {r['age']:5.0f}s eng {r['eng'][-6:]}")
print(f"  pub {ticks[e][0]:.0f} sim {ticks[e][1]:.0f}")
# time trend of residual vs. mean session age of held warm requests, per 300s
print("\nper 300s: mean residual, mean session age of held warm requests, old-session cached volume")
for lo in range(60,1860,300):
    idx=[i for i,e in enumerate(E) if lo<=e<lo+300]
    print(f"  {lo:4d}-{lo+300:4d}: residual {st.mean(Y[i] for i in idx):+6.0f}  old-session cached volume {st.mean(X_old[i] for i in idx):8.0f}  young {st.mean(X_young[i] for i in idx):8.0f}")
EOF
python3 -I age_corr.py
```

**Bash** — Check controller responsiveness via status poll gaps

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import json,sys,statistics as st
R=sys.argv[1]
ts=[json.loads(l)['timestamp_ns']/1e9 for l in open(f'{R}/telemetry/serve_status.jsonl')]
d=[b-a for a,b in zip(ts,ts[1:])]
print('serve_status poll gaps: median %.2f p99 %.2f max %.2f; gaps>2s: %d'%(st.median(d),sorted(d)[int(.99*len(d))],max(d),sum(1 for x in d if x>2)))
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
print('largest gaps (t, gap):',sorted([(round(a-t0),round(b-a,1)) for a,b in zip(ts,ts[1:]) if b-a>1.5], key=lambda x:-x[1])[:10])
EOF
```

<details><summary>Bash result</summary>

```text
44:def run_kv_indexer(args: List[str]) -> None:
45:    """Run the KV indexer with the given arguments."""
767:        ttl_secs: Optional[float] = None,
780:        indexer_threads: int = 4,
781:        indexer_peers: Optional[list[str]] = None,
785:        session_affinity_ttl_secs: Optional[float] = None,
789:        """Create a selection service. `indexer_threads` sizes the KV indexer pool.
791:        `session_affinity_ttl_secs` enables session affinity with an idle TTL
805:        The KV indexer thread pool is released when the handle is dropped.
947:    enable_local_indexer: bool
1100:class KvIndexer:
1102:    A KV Indexer that tracks KV Events emitted by workers. Events include add_block and remove_block.
1109:        Create a `KvIndexer` object
1134:        Return the block size of the KV Indexer.
1138:class ApproxKvIndexer:
1140:    An approximate KV Indexer that doesn't receive KV cache events from workers.
1141:    Instead, it relies on routing decisions with TTL-based expiration and pruning
1156:        router_ttl_secs: float = 120.0,
1159:        Create an `ApproxKvIndexer` object
1162:            component: The component to associate with this indexer
1164:            router_ttl_secs: TTL for blocks in seconds (default: 120.0)
1185:        Return the block size of the ApproxKvIndexer.
1196:        Notify the indexer that a token sequence has been routed to a specific worker.
1198:        This updates the indexer's internal state to track which blocks are likely
1239:        enable_local_indexer: bool = False,
1261:            enable_local_indexer: Enable worker-local KV indexer
1826:        session_affinity_ttl_secs: Optional[int] = None,
1839:            session_affinity_ttl_secs: Router-local session-affinity idle TTL in seconds.
1884:        router_ttl_secs: float = 120.0,
1885:        router_approximate_cache_policy: Literal["ttl", "lru"] = "ttl",
1889:        use_remote_indexer: bool = False,
1890:        serve_indexer: bool = False,
1894:        router_predicted_ttl_secs: Optional[float] = None,
1943:            router_ttl_secs: TTL for blocks in seconds when not using KV events (default: 120.0)
1944:            router_approximate_cache_policy: Process-local approximate-index retention policy,
1945:                "ttl" or "lru" (default: "ttl"). LRU requires use_kv_events=False.
1946:            router_queue_threshold: Optional queue threshold fraction for prefill token capacity (default: None).
1957:            router_event_threads: Number of KV indexer worker threads (default: 4).
1959:                including for approximate routing when KV events are disabled.
1964:            use_remote_indexer: Query a remote KV indexer served from the worker component (default: False).
#####
1883-        router_prefill_load_model: str = "none",
1884-        router_ttl_secs: float = 120.0,
1885-        router_approximate_cache_policy: Literal["ttl", "lru"] = "ttl",
1886-        router_queue_threshold: Optional[float] = None,
1887-        router_event_threads: int = 4,
1888-        router_queue_policy: str = "fcfs",
--
1893-        shared_cache_type: str = "none",
1894-        router_predicted_ttl_secs: Optional[float] = None,
1895-        conditional_disagg_enabled: bool = False,
1896-        conditional_disagg_policy: str = "isl_bounding",
1897-        conditional_disagg_eff_isl_threshold: int = 2048,
```

</details>

<details><summary>Bash result</summary>

```text
residual (published - simulated rebuild) vs exposure-weighted volume held in the window:
  corr with cached tokens of sessions older than 600s: +0.354
  corr with cached tokens of sessions younger than 600s: +0.164
  corr with cold uncached tokens: -0.014
  residual ~ +0.173 x old-session cached + +0.006 x young-session cached   (fraction of cached tokens the router appears to miss)

held requests at export tick 1279.6 (pub peak) with session age and cached blocks:
  q 1226.1 ft 1226.3 own   1572 cached 178752 (228 blk) turn 25 session age   461s eng 173074
  q 1220.6 ft 1224.0 own  36489 cached 169344 (216 blk) turn 7 session age  1146s eng 151035
  q 1260.3 ft 1260.5 own    809 cached 159936 (204 blk) turn 20 session age  1215s eng 151035
  q 1249.8 ft 1250.0 own   1855 cached 158368 (202 blk) turn 19 session age  1205s eng 151035
  q 1229.0 ft 1230.8 own  22709 cached 136416 (174 blk) turn 18 session age  1184s eng 151035
  q 1222.0 ft 1224.0 own   2645 cached 134848 (172 blk) turn 17 session age  1177s eng 151035
  q 1243.9 ft 1245.5 own  19460 cached 133280 (170 blk) turn 15 session age   929s eng 151035
  q 1268.7 ft 1268.8 own    775 cached 131712 (168 blk) turn 42 session age  1104s eng 151035
  q 1249.1 ft 1249.2 own   1172 cached 130144 (166 blk) turn 40 session age  1084s eng 151035
  q 1262.2 ft 1262.4 own   2077 cached 130144 (166 blk) turn 41 session age  1097s eng 151035
  q 1228.0 ft 1228.1 own    806 cached 128576 (164 blk) turn 38 session age  1063s eng 151035
  q 1238.6 ft 1238.9 own   2284 cached 128576 (164 blk) turn 39 session age  1074s eng 151035
  q 1222.0 ft 1224.0 own   1598 cached 127008 (162 blk) turn 37 session age  1057s eng 151035
  q 1262.6 ft 1262.7 own    664 cached 123872 (158 blk) turn 46 session age   738s eng 151035
  pub 47935 sim 26262

per 300s: mean residual, mean session age of held warm requests, old-session cached volume
    60- 360: residual  -1081  old-session cached volume        0  young     7505
   360- 660: residual  -1185  old-session cached volume        0  young    27026
   660- 960: residual  +1975  old-session cached volume     4521  young    32957
   960-1260: residual  +2674  old-session cached volume     8637  young    22372
  1260-1560: residual  +1937  old-session cached volume     9083  young    19427
  1560-1860: residual  +1495  old-session cached volume    18960  young    19526
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
serve_status poll gaps: median 1.00 p99 1.04 max 1.25; gaps>2s: 0
largest gaps (t, gap): []
```

</details>

**Bash** — Scan the router replica process logs for KV indexer events around replica joins

```bash
cd /tmp/ray/session_latest/logs && ls | grep -E "worker-.*-149639|python-core-worker-.*149639" | head; for f in $(ls | grep -E "149639" | head -4); do echo "== $f ($(wc -c <$f) bytes)"; grep -n -i "indexer\|capacity\|evict\|drop\|gap\|replay\|hwm\|lag\|membership\|Registered KV\|recover\|reset\|worker" $f | grep -v "^.*DEBUG" | cut -c1-230 | head -40; done
```

**Bash** — Scan engine process logs for KV event publisher warnings

```bash
cd /tmp/ray/session_latest/logs && for pid in 151035 173074 180495; do for f in $(ls | grep -E "$pid" | head -3); do echo "== $f ($(wc -c <$f) bytes)"; grep -n -i "kv.event\|kv_event\|publisher\|zmq\|dropped\|queue full\|hwm\|prefix cach\|mamba" $f | cut -c1-220 | head -12; done; done
```

**Bash** — Find vLLM's rule for usable prefix-cache hits on hybrid Mamba models

```bash
V=$(python3 -c "import vllm,os;print(os.path.dirname(vllm.__file__))"); python3 -c "import vllm;print(vllm.__version__)"; grep -rn -i "mamba" $V/v1/core/kv_cache_manager.py $V/v1/core/kv_cache_coordinator.py $V/v1/core/single_type_kv_cache_manager.py 2>/dev/null | cut -c1-200 | head -25; echo '#####'; grep -n -i "align\|last.*block\|prefix.*hit\|num_computed_tokens" $V/v1/core/single_type_kv_cache_manager.py | grep -i "mamba\|hybrid\|align\|state" | cut -c1-200 | head -20
```

**Bash** — Regress the over-count on cached blocks that predate a replica join versus after it

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > join_reg.py <<'EOF'
import json, bisect, math, statistics as st
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json')); PSI=9.6
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
JOINS=[698.0, 935.0]   # replica 2 and 3 RUNNING
conv={}
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['conv']=a['conversation_id'] if a else None; r['turn']=a.get('turn_index') if a else None; R.append(r)
    if a: conv.setdefault(a['conversation_id'],[]).append(r)
def context_before(c, t):
    """prompt length of the conversation's last turn queued before t (0 if none)"""
    return max((x['isl'] for x in conv.get(c,[]) if x['q']<t), default=0)
# per warm request: cached blocks that predate the most recent join before it
for r in R:
    joins_before=[j for j in JOINS if j<r['q']]
    if r['kind']=='warm' and joins_before:
        j=max(joins_before); r['pre']=min(r['cached'], context_before(r['conv'], j)); r['post']=r['cached']-r['pre']
    else: r['pre']=0; r['post']=r['cached'] if r['kind']=='warm' else 0
ticks={}
for i,(t,v,p) in enumerate(rows):
    e=PSI+10*math.floor((t-PSI)/10); ticks.setdefault(e,(p,sim[i]))
Xpre=[];Xpost=[];Xcold=[];Y=[];E=[]
for e,(p,s) in sorted(ticks.items()):
    if e<60: continue
    held=[r for r in R if r['ft']>e-60 and r['q']<e]
    w=lambda r: max(0,min(r['ft']+0.1,e)-max(r['q'],e-60))/60.0
    Xpre.append(sum(r['pre']*w(r) for r in held)); Xpost.append(sum(r['post']*w(r) for r in held)); Xcold.append(sum(r['own']*w(r) for r in held if r['kind']=='cold'))
    Y.append(p-s); E.append(e)
def corr(x,y):
    mx,my=st.mean(x),st.mean(y); sx=math.sqrt(sum((a-mx)**2 for a in x)); sy=math.sqrt(sum((b-my)**2 for b in y)); return sum((a-mx)*(b-my) for a,b in zip(x,y))/(sx*sy) if sx and sy else 0
print(f"residual (published - simulated) vs exposure-weighted cached tokens of held warm turns:")
print(f"  corr with cached blocks created BEFORE the latest replica join: {corr(Xpre,Y):+.3f}")
print(f"  corr with cached blocks created AFTER the latest replica join:  {corr(Xpost,Y):+.3f}")
# 2-var regression through origin
s11=sum(a*a for a in Xpre); s22=sum(b*b for b in Xpost); s12=sum(a*b for a,b in zip(Xpre,Xpost)); s1y=sum(a*y for a,y in zip(Xpre,Y)); s2y=sum(b*y for b,y in zip(Xpost,Y))
det=s11*s22-s12*s12; a=(s1y*s22-s2y*s12)/det; b=(s11*s2y-s12*s1y)/det
res=[y-(a*x1+b*x2) for x1,x2,y in zip(Xpre,Xpost,Y)]
print(f"  residual ~ {a:+.2f} x pre-join cached + {b:+.2f} x post-join cached ; rmse of fit {math.sqrt(st.mean(r*r for r in res)):.0f} vs raw residual rmse {math.sqrt(st.mean(y*y for y in Y)):.0f}")
late=[i for i,e in enumerate(E) if e>700]
print(f"  late region only: mean residual {st.mean(Y[i] for i in late):+.0f}; mean pre-join cached volume {st.mean(Xpre[i] for i in late):.0f} -> {a:.2f} x that = {a*st.mean(Xpre[i] for i in late):+.0f}")
# how much of the gauge mean would be explained: a * pre-join volume over the run
print(f"  whole-run: mean pre-join cached volume {st.mean(Xpre):.0f}, implied over-count {a*st.mean(Xpre):+.0f} tokens on the mean")
# sanity: residual by whether any pre-join sessions are held
grp=[(y,x>0) for y,x in zip(Y,Xpre)]
print(f"  mean residual when no pre-join cached volume held: {st.mean(y for y,f in grp if not f):+.0f} (n={sum(1 for y,f in grp if not f)}); when some: {st.mean(y for y,f in grp if f):+.0f} (n={sum(1 for y,f in grp if f)})")
EOF
python3 -I join_reg.py
```

<details><summary>Bash result</summary>

```text
python-core-worker-76e53d491f318d1ca3442a5e34b7fa9057f4dc83adf9ccd156551362_149639.log
== python-core-worker-76e53d491f318d1ca3442a5e34b7fa9057f4dc83adf9ccd156551362_149639.log (436586 bytes)
1:{"asctime":"2026-11-02 12:38:21,316","levelname":"I","message":"Constructing CoreWorkerProcess. pid: 149639","filename":"core_worker_process.cc","lineno":872}
2:{"asctime":"2026-11-02 12:38:21,317","levelname":"I","message":"Ray Event initialized for CORE_WORKER","filename":"event.cc","lineno":499}
5:{"asctime":"2026-11-02 12:38:21,319","levelname":"I","message":"worker server started, listening on port 14850.","filename":"grpc_server.cc","lineno":144}
6:{"asctime":"2026-11-02 12:38:21,320","levelname":"I","message":"Initializing worker at address: 10.240.13.26:14850","filename":"core_worker_process.cc","lineno":299,"worker_id":"76e53d491f318d1ca3442a5e34b7fa9057f4dc83adf9ccd156
8:{"asctime":"2026-11-02 12:38:21,321","levelname":"I","message":"Adjusted worker niceness to 15","filename":"core_worker.cc","lineno":661}
9:{"asctime":"2026-11-02 12:38:21,321","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 11 total (9 active)\nQueueing time: mean = 0.00ms, max = 0.01ms, min = 0.01ms, total = 0.02ms\nExecution time:  mean = 0.00ms, total
11:{"asctime":"2026-11-02 12:38:21,322","levelname":"I","message":"Creating actor","filename":"core_worker.cc","lineno":3258,"actor_id":"8f81cbc95354332ccc18ff6276000000"}
15:{"asctime":"2026-11-02 12:38:29,677","levelname":"I","message":"received notification on actor, state: ALIVE, ip address: 10.240.13.26, port: 16964, num_restarts: 0, death context type=CONTEXT_NOT_SET","filename":"actor_manager
18:{"asctime":"2026-11-02 12:38:43,555","levelname":"I","message":"received notification on actor, state: ALIVE, ip address: 10.240.13.26, port: 14850, num_restarts: 0, death context type=CONTEXT_NOT_SET","filename":"actor_manager
20:{"asctime":"2026-11-02 12:39:21,321","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 1063 total (13 active)\nQueueing time: mean = 0.02ms, max = 3.56ms, min = -0.00ms, total = 17.89ms\nExecution time:  mean = 88.98ms
22:{"asctime":"2026-11-02 12:40:03,884","levelname":"I","message":"received notification on actor, state: ALIVE, ip address: 10.240.13.26, port: 15965, num_restarts: 0, death context type=CONTEXT_NOT_SET","filename":"actor_manager
24:{"asctime":"2026-11-02 12:40:21,322","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 2150 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 30.01ms\nExecution time:  mean = 157.07m
25:{"asctime":"2026-11-02 12:41:21,322","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 3357 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 42.20ms\nExecution time:  mean = 163.45m
26:{"asctime":"2026-11-02 12:42:21,322","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 4548 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total =
... [3088 chars trimmed] ...
velname":"I","message":"Event stats:\n\n\nGlobal stats: 28024 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 290.68ms\nExecution time:  mean = 127.8
44:{"asctime":"2026-11-02 12:55:43,488","levelname":"I","message":"received notification on actor, state: ALIVE, ip address: 10.240.13.26, port: 14364, num_restarts: 0, death context type=CONTEXT_NOT_SET","filename":"actor_manager
46:{"asctime":"2026-11-02 12:56:21,327","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 30255 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 313.43ms\nExecution time:  mean = 122.6
47:{"asctime":"2026-11-02 12:57:21,327","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 32550 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 335.94ms\nExecution time:  mean = 141.2
48:{"asctime":"2026-11-02 12:58:21,327","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 34954 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 359.73ms\nExecution time:  mean = 135.6
49:{"asctime":"2026-11-02 12:59:21,327","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 36953 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 379.75ms\nExecution time:  mean = 132.1
50:{"asctime":"2026-11-02 13:00:21,328","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 38874 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 398.93ms\nExecution time:  mean = 131.0
51:{"asctime":"2026-11-02 13:01:21,328","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 40879 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 418.41ms\nExecution time:  mean = 129.4
52:{"asctime":"2026-11-02 13:02:21,328","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 42736 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 437.70ms\nExecution time:  mean = 127.0
53:{"asctime":"2026-11-02 13:03:21,329","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 44872 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 459.71ms\nExecution time:  mean = 125.7
54:{"asctime":"2026-11-02 13:04:21,329","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 46865 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 479.14ms\nExecution time:  mean = 123.6
55:{"asctime":"2026-11-02 13:05:21,329","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 48946 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 498.78ms\nExecution time:  mean = 122.1
56:{"asctime":"2026-11-02 13:06:21,330","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 51171 total (12 active)\nQueueing time: mean = 0.01ms, max = 3.56ms, min = -0.01ms, total = 520.69ms\nExecution time:  mean = 120.7
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
== python-core-driver-77000000ffffffffffffffffffffffffffffffffffffffffffffffff_151035.log (371951 bytes)
8:{"asctime":"2026-11-02 12:38:59,138","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 11 total (10 active)\nQueueing time: mean = 0.00ms, max = 0.01ms, min = 0.01ms, total = 0.01ms\nExecution time:  mean = 0.
16:{"asctime":"2026-11-02 12:39:59,139","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 813 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 10.20ms\nExecution time:  mean =
17:{"asctime":"2026-11-02 12:40:59,139","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 1572 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 17.20ms\nExecution time:  mean 
18:{"asctime":"2026-11-02 12:41:59,139","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 2331 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 23.91ms\nExecution time:  mean 
19:{"asctime":"2026-11-02 12:42:59,140","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 3090 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 30.72ms\nExecution time:  mean 
20:{"asctime":"2026-11-02 12:43:59,140","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 3850 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 40.25ms\nExecution time:  mean 
21:{"asctime":"2026-11-02 12:44:59,140","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 4609 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 47.99ms\nExecution time:  mean 
22:{"asctime":"2026-11-02 12:45:59,140","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 5368 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 55.82ms\nExecution time:  mean 
23:{"asctime":"2026-11-02 12:46:59,141","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 6127 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 62.01ms\nExecution time:  mean 
24:{"asctime":"2026-11-02 12:47:59,141","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 6888 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = 0.00ms, total = 68.57ms\nExecution time:  mean 
25:{"asctime":"2026-11-02 12:48:59,141","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 7648 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = -0.01ms, total = 75.17ms\nExecution time:  mean
26:{"asctime":"2026-11-02 12:49:59,142","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 8407 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.75ms, min = -0.01ms, total = 81.75ms\nExecution time:  mean
== python-core-driver-78000000ffffffffffffffffffffffffffffffffffffffffffffffff_173074.log (257639 bytes)
8:{"asctime":"2026-11-02 12:50:42,658","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 11 total (10 active)\nQueueing time: mea
... [2311 chars trimmed] ...
6-11-02 13:01:42,662","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 8407 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.64ms, min = 0.00ms, total = 78.74ms\nExecution time:  mean 
== python-core-driver-79000000ffffffffffffffffffffffffffffffffffffffffffffffff_180495.log (219546 bytes)
8:{"asctime":"2026-11-02 12:54:38,622","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 11 total (10 active)\nQueueing time: mean = 0.00ms, max = 0.01ms, min = 0.01ms, total = 0.01ms\nExecution time:  mean = 0.
16:{"asctime":"2026-11-02 12:55:38,622","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 813 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 10.38ms\nExecution time:  mean =
17:{"asctime":"2026-11-02 12:56:38,623","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 1570 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 18.71ms\nExecution time:  mean 
18:{"asctime":"2026-11-02 12:57:38,623","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 2329 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 27.19ms\nExecution time:  mean 
19:{"asctime":"2026-11-02 12:58:38,623","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 3088 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 34.33ms\nExecution time:  mean 
20:{"asctime":"2026-11-02 12:59:38,623","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 3848 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 41.47ms\nExecution time:  mean 
21:{"asctime":"2026-11-02 13:00:38,624","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 4607 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = 0.00ms, total = 49.74ms\nExecution time:  mean 
22:{"asctime":"2026-11-02 13:01:38,624","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 5366 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = -0.00ms, total = 57.83ms\nExecution time:  mean
23:{"asctime":"2026-11-02 13:02:38,624","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 6127 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = -0.00ms, total = 65.73ms\nExecution time:  mean
24:{"asctime":"2026-11-02 13:03:38,625","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 6886 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = -0.00ms, total = 73.92ms\nExecution time:  mean
25:{"asctime":"2026-11-02 13:04:38,625","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 7646 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = -0.00ms, total = 82.74ms\nExecution time:  mean
26:{"asctime":"2026-11-02 13:05:38,625","levelname":"I","message":"Event stats:\n\n\nGlobal stats: 8405 total (10 active)\nQueueing time: mean = 0.01ms, max = 0.66ms, min = -0.00ms, total = 91.51ms\nExecution time:  mean
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
0.31.0
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:17:    MambaManager,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:26:    MambaSpec,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:42:    # Retention sparsifies sliding-window and Mamba (linear-attention)
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:46:        isinstance(g.kv_cache_spec, (SlidingWindowSpec, MambaSpec))
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:53:            "no sliding-window or Mamba KV cache group, so retention has no "
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:54:            "effect. Set it to 0 (it only applies to sliding-window and Mamba "
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:150:        # Match Mamba checkpoints to Eagle's attention replay boundary.
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:153:                if isinstance(manager, MambaManager):
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:201:                retention Mamba reserves a prefill checkpoint only on the chunk
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:570:        # For models using only Mamba, block_size is set to max_model_len when
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:684:                    "full-attention, Mamba, and replicated draft groups, got: "
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:687:        # Fine-grained hash hits require Mamba "align" and compatible cache
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:688:        # managers in every group. TP needs hashing finer than the Mamba block;
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:691:        has_partial_mamba_group = any(
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:692:            isinstance(g.kv_cache_spec, MambaSpec)
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:693:            and g.kv_cache_spec.mamba_cache_mode == "align"
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:703:            allow_partial_hash_hits and has_partial_mamba_group
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:802:        ``hash_block_size`` would re-register a privatized Mamba tail.
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:909:                # mamba: its finder never drops (draft models have no mamba
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_coordinator.py:911:                if drop_eagle_block and not isinstance(spec, MambaSpec):
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py:18:from vllm.v1.core.single_type_kv_cache_manager import MambaManager
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py:24:    MambaSpec,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py:148:        enable_mamba_shared_prefix_checkpoint: bool = False,
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py:186:        self.mamba_shared_prefix_checkpoint = (
/home/ray/anaconda3/lib/python3.12/site-packages/vllm/v1/core/kv_cache_manager.py:187:            enable_mamba_shared_prefix_checkpoint
#####
107:        self.cache_hit_alignment_tokens = scheduler_block_size
141:        # aligned segment (SWA). Initialized lazily by the coordinator after
145:        # by ``KVCacheManager``; only an EAGLE Mamba "align" group ever gets it.
148:        # managers (full attention, mamba "align"); harmlessly empty elsewhere.
152:        # "align" block table is not append-only (interior states are
157:        # right hash. Populated only by mamba "align".
437:        Only mamba "align" populates this. The blocks are not kept alive by
516:            alignment_tokens=self.cache_hit_alignment_tokens,
540:        alignment_tokens: int | None,
552:        blocks that can never serve a hit at any alignment-aligned prefix length.
616:        alignment_tokens: int,
625:        last block to get the required hidden states for eagle drafting head.
643:            alignment_tokens: The returned cache hit length (in tokens) should
756:        alignment_tokens: int,
776:            alignment_tokens=alignment_tokens,
779:        # Fine-grained mode (alignment_tokens == hash_block_size <
783:            alignment_tokens < block_size and block_size % alignment_tokens == 0
789:                block_hashes, alignment_tokens, block_size
812:            scale_factor = block_size // alignment_tokens
816:                max_length // alignment_tokens,
```

</details>

<details><summary>Bash result</summary>

```text
residual (published - simulated) vs exposure-weighted cached tokens of held warm turns:
  corr with cached blocks created BEFORE the latest replica join: +0.461
  corr with cached blocks created AFTER the latest replica join:  -0.026
  residual ~ +0.17 x pre-join cached + -0.04 x post-join cached ; rmse of fit 4629 vs raw residual rmse 5262
  late region only: mean residual +2119; mean pre-join cached volume 19499 -> 0.17 x that = +3243
  whole-run: mean pre-join cached volume 12571, implied over-count +2091 tokens on the mean
  mean residual when no pre-join cached volume held: -1158 (n=63); when some: +2114 (n=117)
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Find the router replica's process logs and list KV indexer messages around the replica joins

```bash
cd /tmp/ray/session_latest/logs && grep -l "KV router replica membership\|Registered KV event worker\|Dynamo SelectionService" worker-*.out worker-*.err 2>/dev/null | head; echo '#####'; for f in $(grep -l "Registered KV event worker" worker-*.out worker-*.err 2>/dev/null | head -2); do echo "== $f"; grep -n "membership\|Registered KV\|indexer\|Indexer\|recover\|reset\|evict\|capacity\|dropped\|gap\|replay\|WARN\|ERROR\|session_affinity\|SelectionService" $f | grep "2026-11-02 1[23]:[0-5][0-9]" | cut -c1-260 | head -60; done
```

**Bash** — Test whether post-join gauge states become exact when pre-join cached blocks count as uncached

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > wipe_test.py <<'EOF'
import json, math, collections
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
JOINS=[698.0,935.0]
conv={}; R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid)); r=dict(r); r['conv']=a['conversation_id'] if a else None; r['turn']=a.get('turn_index') if a else None; R.append(r)
    if a: conv.setdefault(a['conversation_id'],[]).append(r)
def context_before(c,t): return max((x['isl'] for x in conv.get(c,[]) if x['q']<t), default=0)
for r in R:
    jb=[j for j in JOINS if j<r['q']]
    r['pre']=min(r['cached'],context_before(r['conv'],max(jb))) if (jb and r['kind']=='warm') else 0
    r['pre']=(r['pre']//784)*784
def subset(target, vals, tol=1):
    sums={0:()}
    for i,v in enumerate(vals):
        new={}
        for s_,sub in sums.items():
            ns=s_+v
            if ns<=target+tol and ns not in sums and ns not in new: new[ns]=sub+(i,)
        sums.update(new)
    for s_ in range(target-tol,target+tol+1):
        if s_ in sums: return sums[s_]
    return None
# distinct exported states at export ticks
states={}
for t,v,p in rows:
    e=PSI+10*math.floor((t-PSI)/10); states.setdefault(e,v)
tally=collections.Counter(); per_window=collections.defaultdict(collections.Counter)
for e,v in sorted(states.items()):
    if e<0 or v<=0: continue
    s=PHI+2*math.floor((e-PHI)/2)
    held=[r for r in R if r['q']-0.5<=s<r['ft']+0.1]
    if not held or len(held)>16: tally['skipped (0 or >16 held)']+=1; continue
    w=int(e//100*100)
    if subset(int(v),[r['own'] for r in held]) is not None: k='exact with engine counts'
    elif subset(int(v),[r['own'] for r in held]+[r['pre'] for r in held if r['pre']>0]) is not None: k='exact only if pre-join blocks count as uncached'
    elif subset(int(v),[r['own'] for r in held]+[r['isl']-r['own'] for r in held if r['isl']>r['own']]) is not None: k='exact only if some warm turn counts full prompt'
    else: k='unexplained'
    tally[k]+=1; per_window[w][k]+=1
print("distinct nonzero exported gauge states (one per 10s export tick):")
for k,n in tally.most_common(): print(f"  {n:3d}  {k}")
print("\nby 100s window: exact / pre-join-blocks / full-prompt / unexplained")
for w in sorted(per_window):
    c=per_window[w]; print(f"  {w:4d}: {c['exact with engine counts']:2d} / {c['exact only if pre-join blocks count as uncached']:2d} / {c['exact only if some warm turn counts full prompt']:2d} / {c['unexplained']:2d}")
# ages of correctly counted warm turns before the first join (rules out gradual aging up to that age)
ages=[]
for e,v in sorted(states.items()):
    if not (0<e<698) or v<=0: continue
    s=PHI+2*math.floor((e-PHI)/2); held=[r for r in R if r['q']-0.5<=s<r['ft']+0.1]
    if not held or len(held)>16: continue
    sub=subset(int(v),[r['own'] for r in held])
    if sub:
        for i in sub:
            r=held[i]
            if r['kind']=='warm' and r['cached']>0: ages.append((round(r['q']-min(x['q'] for x in conv[r['conv']])), r['cached']//784))
print("\nbefore the first join, warm turns counted exactly at engine counts: (session age s, cached blocks) max age:", max(ages) if ages else None, "; n=",len(ages), "; ages>400s:", sorted(a for a,_ in ages if a>400))
EOF
python3 -I wipe_test.py
```

**Bash** — Get precise replica state transition times

```bash
R=/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean; python3 -I - $R <<'EOF'
import json,sys
R=sys.argv[1]
t0=json.load(open(f'{R}/analysis/plot_data.json'))['time_origin']['profiling_start_ns']/1e9
# precise replica transitions from serve_status and deployment events
prev=None
for line in open(f'{R}/telemetry/serve_status.jsonl'):
    d=json.loads(line); st=d['deployments']['LLMServer:Qwen--Qwen3_6-27B-FP8']['replica_states']
    if st!=prev: print('%.1f'%(d['timestamp_ns']/1e9-t0), st); prev=st
import os
p=f'{R}/deployment/events.jsonl'
if os.path.exists(p):
    for line in open(p):
        d=json.loads(line); s=json.dumps(d)
        if 'LLMServer' in s and ('RUNNING' in s or 'STARTING' in s or 'replica' in s.lower()):
            print(' event', s[:200])
EOF
```

<details><summary>Bash result</summary>

```text
worker-d93dc853c08978b1d8ca90015028805c28f8053989267c22d34d0c94-7a000000-360480.err
#####
== worker-d93dc853c08978b1d8ca90015028805c28f8053989267c22d34d0c94-7a000000-360480.err
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
distinct nonzero exported gauge states (one per 10s export tick):
   61  exact with engine counts
   12  unexplained
    1  exact only if pre-join blocks count as uncached
    1  skipped (0 or >16 held)

by 100s window: exact / pre-join-blocks / full-prompt / unexplained
     0:  4 /  0 /  0 /  0
   100:  3 /  0 /  0 /  1
   200:  1 /  0 /  0 /  0
   300:  2 /  0 /  0 /  0
   400:  3 /  0 /  0 /  2
   500:  6 /  0 /  0 /  0
   600:  3 /  1 /  0 /  0
   700:  4 /  0 /  0 /  1
   800:  5 /  0 /  0 /  0
   900:  4 /  0 /  0 /  2
  1000:  2 /  0 /  0 /  1
  1100:  3 /  0 /  0 /  2
  1200:  3 /  0 /  0 /  0
  1300:  3 /  0 /  0 /  1
  1400:  4 /  0 /  0 /  1
  1500:  3 /  0 /  0 /  0
  1600:  5 /  0 /  0 /  0
  1700:  3 /  0 /  0 /  1

before the first join, warm turns counted exactly at engine counts: (session age s, cached blocks) max age: (302, 98) ; n= 15 ; ages>400s: []
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
-4.3 {'RUNNING': 1}
598.7 {'RUNNING': 1, 'STARTING': 1}
697.7 {'RUNNING': 2}
834.7 {'RUNNING': 2, 'STARTING': 1}
934.7 {'RUNNING': 3}
```

</details>

**Bash** — Check whether the dynamo worker record or indexer takes a block capacity that the Ray wrapper leaves unset

```bash
P=/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.pyi; sed -n 1100,1140p $P; echo '#####'; grep -n "total_kv_blocks\|capacity\|max_blocks\|num_blocks" $P | head -20; echo '#####'; grep -rn "total_kv_blocks\|num_gpu_blocks\|kv_event_metadata\[" /mnt/local_storage/ray/python/ray/llm/_internal/serve --include='*.py' | grep -v tests | head -12
```

**Bash** — Compute cumulative blocks stored per engine over time

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 -I - <<'EOF'
import json
S=json.load(open('gauge_decomp_state.json')); reqs=sorted(S['reqs'].values(), key=lambda r:r['q'])
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
B=784
# cumulative new blocks stored per engine: uncached prompt blocks + output blocks (approx) per request
cum={}; out=[]
for rid,r in [(k,v) for k,v in S['reqs'].items()]:
    pass
rows=[]
for rid,r in S['reqs'].items():
    a=recs.get(aid(rid),{}); osl=a.get('output_sequence_length') or 0
    rows.append((r['q'],r['eng'][-6:],r['own']//B + osl//B))
rows.sort()
for t_mark in [100,200,300,400,500,600,650,700,750,800,900,1000,1200,1400,1600,1860]:
    c={}
    for q,e,b in rows:
        if q<=t_mark: c[e]=c.get(e,0)+b
    print(f"  t<={t_mark:4d}: new blocks stored so far per engine {c}  (tokens: {{k: v*B for k,v in c.items()}})".replace("{{","{").replace("}}","}"))
EOF
```

**Bash** — Locate this run's router replica process log and scan it for indexer messages

```bash
cd /tmp/ray/session_latest/logs && ls -la --time-style=+%H:%M | grep "149639"; grep -l "3jybiw18" worker-*.out worker-*.err 2>/dev/null | head -3; for f in $(grep -l "3jybiw18" worker-*.err 2>/dev/null | head -1); do echo "== $f"; grep -n "membership\|Registered KV\|indexer\|Indexer\|recover\|evict\|capacity\|WARN\|ERROR\|dropped\|gap\|replay" $f | cut -c1-240 | head -40; done
```

<details><summary>Bash result</summary>

```text
class KvIndexer:
    """
    A KV Indexer that tracks KV Events emitted by workers. Events include add_block and remove_block.
    """

    ...

    def __init__(self, endpoint: Endpoint, block_size: int) -> None:
        """
        Create a `KvIndexer` object
        """

    def find_matches(self, sequence: List[int]) -> OverlapScores:
        """
        Find prefix matches for the given sequence of block hashes.

        Args:
            sequence: List of block hashes to find matches for

        Returns:
            OverlapScores containing worker matching scores and frequencies
        """
        ...

    def find_matches_for_request(
        self, token_ids: List[int], lora_name: Optional[str] = None, is_eagle: Optional[bool] = None
    ) -> OverlapScores:
        """
        Return the overlapping scores of workers for the given token ids.
        """
        ...

    def block_size(self) -> int:
        """
        Return the block size of the KV Indexer.
        """
        ...

class ApproxKvIndexer:
    """
    An approximate KV Indexer that doesn't receive KV cache events from workers.
#####
937:    total_kv_blocks: int | None
1946:            router_queue_threshold: Optional queue threshold fraction for prefill token capacity (default: None).
1977:            conditional_disagg_decode_busy_threshold: Decode-busy guard threshold that disables bypass when the selected decode worker's projected decode load exceeds this fraction of KV capacity (default: None).
2652:        host_num_blocks: Optional[int] = None,
2653:        device_num_blocks: Optional[int] = None,
2671:        host_num_blocks: Optional[int]
2673:        device_num_blocks: Optional[int]
3031:            A dictionary containing block_size, num_blocks, shared_cache, and
3375:            total_kv_blocks: Optional[int] = None,
3390:        def total_kv_blocks(self) -> Optional[int]: ...
#####
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:483:            self._register_block_size(kv_event_metadata["block_size"], replica_id)
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:523:        dp_rank = kv_event_metadata["dp_rank"]
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:536:                "max_num_batched_tokens": kv_event_metadata["max_num_batched_tokens"],
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:540:                "kv_events_endpoints": {dp_rank: kv_event_metadata["endpoint"]},
/mnt/local_storage/ray/python/ray/llm/_internal/serve/routing_policies/kv_aware/kv_token_tracker.py:551:            kv_event_metadata["endpoint"],
```

</details>

<details><summary>Bash result</summary>

```text
t<= 100: new blocks stored so far per engine {'151035': 425}  (tokens: {k: v*B for k,v in c.items()})
  t<= 200: new blocks stored so far per engine {'151035': 1246}  (tokens: {k: v*B for k,v in c.items()})
  t<= 300: new blocks stored so far per engine {'151035': 1715}  (tokens: {k: v*B for k,v in c.items()})
  t<= 400: new blocks stored so far per engine {'151035': 2203}  (tokens: {k: v*B for k,v in c.items()})
  t<= 500: new blocks stored so far per engine {'151035': 3149}  (tokens: {k: v*B for k,v in c.items()})
  t<= 600: new blocks stored so far per engine {'151035': 3949}  (tokens: {k: v*B for k,v in c.items()})
  t<= 650: new blocks stored so far per engine {'151035': 4328}  (tokens: {k: v*B for k,v in c.items()})
  t<= 700: new blocks stored so far per engine {'151035': 4749}  (tokens: {k: v*B for k,v in c.items()})
  t<= 750: new blocks stored so far per engine {'151035': 4960, '173074': 353}  (tokens: {k: v*B for k,v in c.items()})
  t<= 800: new blocks stored so far per engine {'151035': 5224, '173074': 832}  (tokens: {k: v*B for k,v in c.items()})
  t<= 900: new blocks stored so far per engine {'151035': 5452, '173074': 1748}  (tokens: {k: v*B for k,v in c.items()})
  t<=1000: new blocks stored so far per engine {'151035': 5672, '173074': 2288, '180495': 263}  (tokens: {k: v*B for k,v in c.items()})
  t<=1200: new blocks stored so far per engine {'151035': 6036, '173074': 2539, '180495': 942}  (tokens: {k: v*B for k,v in c.items()})
  t<=1400: new blocks stored so far per engine {'151035': 6283, '173074': 3046, '180495': 1693}  (tokens: {k: v*B for k,v in c.items()})
  t<=1600: new blocks stored so far per engine {'151035': 6692, '173074': 3691, '180495': 2184}  (tokens: {k: v*B for k,v in c.items()})
  t<=1860: new blocks stored so far per engine {'151035': 7224, '173074': 4206, '180495': 2923}  (tokens: {k: v*B for k,v in c.items()})
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
-rw-r--r-- 1 ray users   436586 13:16 python-core-worker-76e53d491f318d1ca3442a5e34b7fa9057f4dc83adf9ccd156551362_149639.log
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Find the engine's KV cache capacity and block configuration from vLLM startup logs

```bash
cd /tmp/ray/session_latest/logs && grep -l "GPU KV cache size\|Maximum concurrency for" $(ls -t worker-*.out worker-*.err 2>/dev/null | head -400) 2>/dev/null | head -5; grep -h "GPU KV cache size\|Maximum concurrency\|num_gpu_blocks\|Mamba\|mamba\|block_size" $(ls -t worker-*.out worker-*.err 2>/dev/null | head -400) 2>/dev/null | grep "11-02 12:[345]" | cut -c1-200 | sort | uniq | head -15
```

**Bash** — Read dynamo's documentation of total_kv_blocks and the selection cache config

```bash
P=/home/ray/anaconda3/lib/python3.12/site-packages/dynamo/_core.pyi; sed -n 925,945p $P; echo '#####'; sed -n 3365,3395p $P; echo '#####'; grep -n "class SelectionCacheConfig" -A25 $P | head -40
```

**Bash** — Simulate the router index as a bounded LRU and test which capacity reproduces the published mean

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > lru_sim.py <<'EOF'
"""Model the router's KV index as an LRU of C blocks per worker fed by the traces; derive each
request's router-side uncached count and push it through the telemetry pipeline."""
import json, bisect, math, statistics as st
from collections import OrderedDict
S=json.load(open('gauge_decomp_state.json')); reqs=S['reqs']
recs={r['x_request_id']:r for r in json.load(open('aiperf_recs.json'))}
P=json.load(open('pipeline_sim.json')); rows=P['rows']; PSI,PHI=9.6,0.3
def aid(rid): return rid.split('chatcmpl-')[-1].rsplit('-',1)[0]
B=784
R=[]
for rid,r in reqs.items():
    a=recs.get(aid(rid),{}); r=dict(r); r['conv']=a.get('conversation_id',rid); r['osl']=a.get('output_sequence_length') or 0; r['turn']=a.get('turn_index'); R.append(r)
R.sort(key=lambda r:r['q'])
def router_counts(C, refresh_on_hit):
    idx={}   # worker -> OrderedDict of (conv, blk) -> None  (LRU order)
    out={}
    for r in R:
        w=r['eng']; d=idx.setdefault(w,OrderedDict())
        nb_cached=r['cached']//B; nb_prompt=r['isl']//B; nb_total=(r['isl']+r['osl'])//B
        # router overlap: leading blocks present in index
        n=0
        while n<nb_prompt and (r['conv'],n) in d: n+=1
        if r['kind']=='cold': n=min(n, nb_cached)   # engine found nothing; keep router at engine for cold (hybrid cases aside)
        out[r['q']]=r['isl']-n*B if r['kind']=='warm' else r['own']
        # engine stores: blocks from cached.. total (prompt uncached + output); refresh hits optionally
        if refresh_on_hit:
            for k in range(0, nb_cached):
                if (r['conv'],k) in d: d.move_to_end((r['conv'],k))
        for k in range(nb_cached, nb_total):
            d[(r['conv'],k)]=None; d.move_to_end((r['conv'],k))
        if C is not None:
            while len(d)>C: d.popitem(last=False)
    return out
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
def sim(T,L,w=60.0):
    out=[]
    for t,v,p in rows:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
def rmse(x,y): return math.sqrt(st.mean((a-b)**2 for a,b in zip(x,y)))
pub=[p for _,_,p in rows]; early=[i for i,(t,_,_) in enumerate(rows) if t<=650]; late=[i for i,(t,_,_) in enumerate(rows) if t>650]
print("router index as LRU of C blocks per worker: published mean vs simulated rebuild")
for refresh in (False, True):
    for C in (None, 8192, 6144, 5120, 4608, 4096, 3584, 3072, 2048):
        cnt=router_counts(C, refresh)
        T,L=step([(r['q']-0.1,cnt[r['q']]) for r in R]+[(r['ft']+0.1,-cnt[r['q']]) for r in R])
        s=sim(T,L)
        print(f"  refresh_on_hit={refresh!s:5} C={str(C):5}: early rmse {rmse([s[i] for i in early],[pub[i] for i in early]):5.0f} bias {st.mean(pub[i]-s[i] for i in early):+5.0f} | late rmse {rmse([s[i] for i in late],[pub[i] for i in late]):5.0f} bias {st.mean(pub[i]-s[i] for i in late):+5.0f} | whole-run sim mean {st.mean(s):6.0f} vs pub {st.mean(pub):.0f}")
EOF
python3 -I lru_sim.py
```

<details><summary>Bash result</summary>

```text
worker-027ddf196ae3583d4c72560579c9561df9e0419d220cfbf85c890455-7a000000-386186.out
worker-f7c6d8801e1f3dd6946da5917995234d030fb5ec3dcaa67458c211cf-7a000000-378396.out
worker-d83c0da5677110bb8a94cee9e6d332c3c371c24da45f554adc14b99e-7a000000-360479.out
Shell cwd was reset to /home/ray/default
```

</details>

<details><summary>Bash result</summary>

```text
...

    def runtime_config(self) -> Any:
        """Return the runtime configuration as a dict."""
        ...

class ModelRuntimeConfig:
    """
    A model runtime configuration is a collection of runtime information
    """

    context_length: int | None
    total_kv_blocks: int | None
    max_num_seqs: int | None
    max_num_batched_tokens: int | None
    tool_call_parser: str | None
    reasoning_parser: str | None
    tokenizer_backend: str | None
    structural_tag: dict[str, Any] | None
    exclude_tools_when_tool_choice_none: bool
    data_parallel_start_rank: int
#####
        Aggregated: "backend.DisaggregationMode"
        Prefill: "backend.DisaggregationMode"
        Decode: "backend.DisaggregationMode"
        Encode: "backend.DisaggregationMode"

    class LlmRegistration:
        def __init__(
            self,
            context_length: Optional[int] = None,
            kv_cache_block_size: Optional[int] = None,
            total_kv_blocks: Optional[int] = None,
            max_num_seqs: Optional[int] = None,
            max_num_batched_tokens: Optional[int] = None,
            data_parallel_size: Optional[int] = None,
            data_parallel_start_rank: Optional[int] = None,
            bootstrap_host: Optional[str] = None,
            bootstrap_port: Optional[int] = None,
            enable_eagle: bool = False,
            max_gpu_lora_count: Optional[int] = None,
        ) -> None: ...
        @property
        def context_length(self) -> Optional[int]: ...
        @property
        def kv_cache_block_size(self) -> Optional[int]: ...
        @property
        def total_kv_blocks(self) -> Optional[int]: ...
        @property
        def max_num_seqs(self) -> Optional[int]: ...
        @property
        def max_num_batched_tokens(self) -> Optional[int]: ...
        @property
#####
758:class SelectionCacheConfig:
759-    """
760-    Bounds for the in-flight selection cache. Each field defaults to the
761-    service default when omitted.
762-    """
763-
764-    def __init__(
765-        self,
766-        *,
767-        ttl_secs: Optional[float] = None,
768-        max_entries: Optional[int] = None,
769-        max_bytes: Optional[int] = None,
770-    ) -> None: ...
771-
772-class SelectionService:
773-    """
774-    In-process handle to a runtime-free Dynamo selection core.
775-    """
776-
777-    def __init__(
778-        self,
779-        *,
780-        indexer_threads: int = 4,
781-        indexer_peers: Optional[list[str]] = None,
782-        replica_sync_port: Optional[int] = None,
783-        replica_sync_peers: Optional[list[str]] = None,
```

</details>

<details><summary>Bash result</summary>

```text
router index as LRU of C blocks per worker: published mean vs simulated rebuild
  refresh_on_hit=False C=None : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=False C=8192 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=False C=6144 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=False C=5120 : early rmse  4454 bias -2479 | late rmse 14308 bias -9744 | whole-run sim mean  22612 vs pub 15399
  refresh_on_hit=False C=4608 : early rmse  4454 bias -2479 | late rmse 15115 bias -11004 | whole-run sim mean  23433 vs pub 15399
  refresh_on_hit=False C=4096 : early rmse  4454 bias -2479 | late rmse 17599 bias -14038 | whole-run sim mean  25409 vs pub 15399
  refresh_on_hit=False C=3584 : early rmse  4454 bias -2479 | late rmse 18537 bias -15360 | whole-run sim mean  26271 vs pub 15399
  refresh_on_hit=False C=3072 : early rmse  4454 bias -2479 | late rmse 19381 bias -16225 | whole-run sim mean  26835 vs pub 15399
  refresh_on_hit=False C=2048 : early rmse  7896 bias -4620 | late rmse 24448 bias -21669 | whole-run sim mean  31128 vs pub 15399
  refresh_on_hit=True  C=None : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=8192 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=6144 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=5120 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=4608 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=4096 : early rmse  4454 bias -2479 | late rmse 13946 bias -9397 | whole-run sim mean  22385 vs pub 15399
  refresh_on_hit=True  C=3584 : early rmse  4454 bias -2479 | late rmse 13943 bias -9400 | whole-run sim mean  22387 vs pub 15399
  refresh_on_hit=True  C=3072 : early rmse  4454 bias -2479 | late rmse 15487 bias -11448 | whole-run sim mean  23722 vs pub 15399
  refresh_on_hit=True  C=2048 : early rmse  7859 bias -4509 | late rmse 21138 bias -18653 | whole-run sim mean  29124 vs pub 15399
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Rerun the bounded-index simulation with overlap anchored to the engine's cached count

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && python3 - <<'EOF'
s=open('lru_sim.py').read()
s=s.replace('''def router_counts(C, refresh_on_hit):
    idx={}   # worker -> OrderedDict of (conv, blk) -> None  (LRU order)
    out={}
    for r in R:
        w=r['eng']; d=idx.setdefault(w,OrderedDict())
        nb_cached=r['cached']//B; nb_prompt=r['isl']//B; nb_total=(r['isl']+r['osl'])//B
        # router overlap: leading blocks present in index
        n=0
        while n<nb_prompt and (r['conv'],n) in d: n+=1
        if r['kind']=='cold': n=min(n, nb_cached)   # engine found nothing; keep router at engine for cold (hybrid cases aside)
        out[r['q']]=r['isl']-n*B if r['kind']=='warm' else r['own']
        # engine stores: blocks from cached.. total (prompt uncached + output); refresh hits optionally
        if refresh_on_hit:
            for k in range(0, nb_cached):
                if (r['conv'],k) in d: d.move_to_end((r['conv'],k))
        for k in range(nb_cached, nb_total):
            d[(r['conv'],k)]=None; d.move_to_end((r['conv'],k))
        if C is not None:
            while len(d)>C: d.popitem(last=False)
    return out''','''def router_counts(C, refresh_on_hit, scope='worker'):
    idx={}; evicted={}
    out={}
    for r in R:
        key=r['eng'] if scope=='worker' else 'all'
        d=idx.setdefault(key,OrderedDict()); ev=evicted.setdefault(r['eng'],set())
        nb_cached=r['cached']//B; nb_total=(r['isl']+r['osl'])//B
        # router overlap = engine's cached prefix, cut at the first block the router index evicted
        n=nb_cached
        for k in range(nb_cached):
            if (r['eng'],r['conv'],k) in ev: n=k; break
        out[r['q']]=r['own']+(nb_cached-n)*B
        if refresh_on_hit:
            for k in range(0, nb_cached):
                if (r['eng'],r['conv'],k) in d: d.move_to_end((r['eng'],r['conv'],k))
        for k in range(nb_cached, nb_total):
            d[(r['eng'],r['conv'],k)]=None; d.move_to_end((r['eng'],r['conv'],k)); ev.discard((r['eng'],r['conv'],k))
        if C is not None:
            while len(d)>C:
                (e_,c_,k_),_=d.popitem(last=False); evicted.setdefault(e_,set()).add((e_,c_,k_))
    return out''')
s=s.replace('''for refresh in (False, True):
    for C in (None, 8192, 6144, 5120, 4608, 4096, 3584, 3072, 2048):
        cnt=router_counts(C, refresh)''','''for scope,refresh in (('worker',False),('worker',True),('all',False),('all',True)):
    for C in (None, 8192, 6144, 5120, 4608, 4096, 3584, 3072, 2560, 2048):
        cnt=router_counts(C, refresh, scope)''')
s=s.replace('print(f"  refresh_on_hit={refresh!s:5} C={str(C):5}:','print(f"  scope={scope:6} refresh_on_hit={refresh!s:5} C={str(C):5}:')
open('lru_sim.py','w').write(s)
EOF
python3 -I lru_sim.py
```

<details><summary>Bash result</summary>

```text
router index as LRU of C blocks per worker: published mean vs simulated rebuild
  scope=worker refresh_on_hit=False C=None : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=False C=8192 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=False C=6144 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=False C=5120 : early rmse  1622 bias  -908 | late rmse  6248 bias +1659 | whole-run sim mean  14634 vs pub 15399
  scope=worker refresh_on_hit=False C=4608 : early rmse  1622 bias  -908 | late rmse  6366 bias  +411 | whole-run sim mean  15447 vs pub 15399
  scope=worker refresh_on_hit=False C=4096 : early rmse  1622 bias  -908 | late rmse  7594 bias -2560 | whole-run sim mean  17383 vs pub 15399
  scope=worker refresh_on_hit=False C=3584 : early rmse  1622 bias  -908 | late rmse  7887 bias -3853 | whole-run sim mean  18226 vs pub 15399
  scope=worker refresh_on_hit=False C=3072 : early rmse  1622 bias  -908 | late rmse  8549 bias -4707 | whole-run sim mean  18782 vs pub 15399
  scope=worker refresh_on_hit=False C=2560 : early rmse  2284 bias -1408 | late rmse 11695 bias -8666 | whole-run sim mean  21536 vs pub 15399
  scope=worker refresh_on_hit=False C=2048 : early rmse  5526 bias -2979 | late rmse 13100 bias -10095 | whole-run sim mean  23015 vs pub 15399
  scope=worker refresh_on_hit=True  C=None : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=8192 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=6144 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=5120 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=4608 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=4096 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=3584 : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=worker refresh_on_hit=True  C=3072 : early rmse  1622 bias  -908 | late rmse  6244 bias   -17 | whole-run sim mean  15726 vs pub 15399
  scope=worker refresh_on_hit=True  C=2560 : early rmse  1622 bias  -908 | late rmse  6258 bias -3348 | whole-run sim mean  17897 vs pub 15399
  scope=worker refresh_on_hit=True  C=2048 : early rmse  5512 bias -2868 | late rmse 10074 bias -7115 | whole-run sim mean  21034 vs pub 15399
  scope=all    refresh_on_hit=False C=None : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=all    refresh_on_hit=False C=8192 : early rmse  1622 bias  -908 | late rmse  9784 bias -1487 | whole-run sim mean  16684 vs pub 15399
  scope=all    refresh_on_hit=False C=6144 : early rmse  1622 bias  -908 | late rmse 11336 bias -5564 | whole-run sim mean  19340 vs pub 15399
  scope=all    refresh_on_hit=False C=5120 : early rmse  1622 bias  -908 | late rmse 12066 bias -7031 | whole-run sim mean  20296 vs pub 15399
  scope=all    refresh_on_hit=False C=4608 : early rmse  1622 bias  -908 | late rmse 12801 bias -9171 | whole-run sim mean  21691 vs pub 15399
  scope=all    refresh_on_hit=False C=4096 : early rmse  1622 bias  -908 | late rmse 13883 bias -9967 | whole-run sim mean  22209 vs pub 15399
  scope=all    refresh_on_hit=False C=3584 : early rmse  1622 bias  -908 | late rmse 15935 bias -11479 | whole-run sim mean  23195 vs pub 15399
  scope=all    refresh_on_hit=False C=3072 : early rmse  1622 bias  -908 | late rmse 18941 bias -14807 | whole-run sim mean  25364 vs pub 15399
  scope=all    refresh_on_hit=False C=2560 : early rmse  2284 bias -1408 | late rmse 20042 bias -16530 | whole-run sim mean  26660 vs pub 15399
  scope=all    refresh_on_hit=False C=2048 : early rmse  5526 bias -2979 | late rmse 23807 bias -19815 | whole-run sim mean  29348 vs pub 15399
  scope=all    refresh_on_hit=True  C=None : early rmse  1622 bias  -908 | late rmse  6114 bias +2002 | whole-run sim mean  14410 vs pub 15399
  scope=all    refresh_on_hit=True  C=8192 : early rmse  1622 bias  -908 | late rmse  6148 bias +1190 | whole-run sim mean  14940 vs pub 15399
  scope=all    refresh_on_hit=True  C=6144 : early rmse  1622 bias  -908 | late rmse  6629 bias  +640 | whole-run sim mean  15298 vs pub 15399
  scope=all    refresh_on_hit=True  C=5120 : early rmse  1622 bias  -908 | late rmse  5288 bias  -687 | whole-run sim mean  16163 vs pub 15399
  scope=all    refresh_on_hit=True  C=4608 : early rmse  1622 bias  -908 | late rmse  6604 bias -2652 | whole-run sim mean  17443 vs pub 15399
  scope=all    refresh_on_hit=True  C=4096 : early rmse  1622 bias  -908 | late rmse 10644 bias -6272 | whole-run sim mean  19802 vs pub 15399
  scope=all    refresh_on_hit=True  C=3584 : early rmse  1622 bias  -908 | late rmse 10780 bias -7466 | whole-run sim mean  20580 vs pub 15399
  scope=all    refresh_on_hit=True  C=3072 : early rmse  1622 bias  -908 | late rmse 11361 bias -8251 | whole-run sim mean  21092 vs pub 15399
  scope=all    refresh_on_hit=True  C=2560 : early rmse  1622 bias  -908 | late rmse 13999 bias -10096 | whole-run sim mean  22293 vs pub 15399
  scope=all    refresh_on_hit=True  C=2048 : early rmse  5512 bias -2868 | late rmse 14980 bias -11196 | whole-run sim mean  23693 vs pub 15399
Shell cwd was reset to /home/ray/default
```

</details>

**Bash** — Plot the published mean against the rebuild pushed through the fitted telemetry pipeline

```bash
cd /tmp/claude-1000/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a/scratchpad && cat > plot_pipeline.py <<'EOF'
import json, bisect, math, statistics as st
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
R='/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean'
P=json.load(open('pipeline_sim.json')); rows=P['rows']; sim=json.load(open('sim_best.json'))
S=json.load(open('gauge_decomp_state.json')); reqs=list(S['reqs'].values())
SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
POLICY, REBUILD, ENGINE = "#6250d6", "#c9480f", "#1baf7a"
ts=[t for t,_,_ in rows]; pub=[p for _,_,p in rows]
# engine 60s mean via the same pipeline (2s samples at phase 0.3, eval at tick)
import glob, os, datetime
eng=json.load(open('eng_deltas.json')) if os.path.exists('eng_deltas.json') else None
if eng is None:
    # rebuild engine-style deltas quickly from traces
    T='/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces'
    plot=json.load(open(f'{R}/analysis/plot_data.json')); t0=plot['time_origin']['profiling_start_ns']/1e9
    m=json.load(open(f'{R}/manifest.json')); rs=datetime.datetime.fromisoformat(m['started_at']).timestamp(); re_=datetime.datetime.fromisoformat(m['finished_at']).timestamp()
    eng=[]
    for path in sorted(glob.glob(f'{T}/*.jsonl')):
        mt=os.path.getmtime(path)
        if mt<rs or mt>re_+600: continue
        chunks={}; comp={}; queued={}; pre={}; off=None
        for line in open(path):
            try: e=json.loads(line)
            except ValueError: continue
            if off is None and 'wall_time_s' in e and 'monotonic_s' in e: off=e['wall_time_s']-e['monotonic_s']
            ev=e['event']
            if ev=='batch_scheduled':
                for r in e['requests']:
                    if r['phase']=='prefill': chunks.setdefault(r['request_id'],[]).append((e['batch_index'],r['scheduled_tokens']))
            elif ev=='batch_completed': comp[e['batch_index']]=e['monotonic_s']
            elif ev=='request_queued': queued[e['request_id']]=e
            elif ev=='request_preempted': pre.setdefault(e['request_id'],[]).append((e['batch_index'],e['monotonic_s']))
        rel=lambda mono: mono+off-t0
        for rid,q in queued.items():
            cs=chunks.get(rid)
            if not cs or any(b not in comp for b,_ in cs): continue
            starts=[(0,q['monotonic_s'])]+sorted(pre.get(rid,[]))
            for k,(batch,at_) in enumerate(starts):
                until=starts[k+1][0] if k+1<len(starts) else math.inf
                eng.append((rel(at_),sum(t for b,t in cs if batch<=b<until)))
            eng.extend((rel(comp[b]),-t) for b,t in cs)
    json.dump(eng,open('eng_deltas.json','w'))
def step(d):
    d=sorted(d); T=[t for t,_ in d]; lv=0; L=[]
    for _,x in d: lv+=x; L.append(lv)
    return T,L
def at(T,L,t):
    i=bisect.bisect_right(T,t)-1; return L[i] if i>=0 else 0
PSI,PHI=9.6,0.3
def pipeline(T,L,w=60.0):
    out=[]
    for t in ts:
        e=PSI+10*math.floor((t-PSI)/10); k=math.floor((e-PHI)/2)
        samples=[(PHI+2*j,at(T,L,PHI+2*j)) for j in range(k-32,k+1) if PHI+2*j>=e-w]
        a=max(e-w,samples[0][0]); tot=0.0
        for (t1,v1),(t2,_) in zip(samples,samples[1:]+[(e,None)]):
            lo,hi=max(t1,a),min(t2,e)
            if hi>lo: tot+=v1*(hi-lo)
        out.append(tot/(e-a))
    return out
Te,Le=step(eng); eng_s=pipeline(Te,Le)
fig,ax=plt.subplots(figsize=(14,5.8),dpi=150); fig.patch.set_facecolor(SURFACE); ax.set_facecolor(SURFACE)
for s in ('top','right'): ax.spines[s].set_visible(False)
for s in ('left','bottom'): ax.spines[s].set_color(GRID)
ax.tick_params(colors=INK2,labelsize=8); ax.grid(axis='y',color=GRID,lw=0.5); ax.set_axisbelow(True)
ax.axvspan(0,930,color="#e8f0fb",zorder=0,lw=0); ax.axvspan(930,1860,color="#f1f1ee",zorder=0,lw=0)
ycap=60000
ax.text(465,ycap*1.07,"ramp up: 1→32 sessions",ha='center',va='top',fontsize=8,color=INK2); ax.text(1395,ycap*1.07,"hold: 32→32 sessions",ha='center',va='top',fontsize=8,color=INK2)
ax.axvline(697.7,color=INK2,lw=0.8,ls='--',alpha=0.6); ax.axvline(934.7,color=INK2,lw=0.8,ls='--',alpha=0.6)
ax.text(700,ycap*1.0,"replica 2 running",fontsize=7.5,color=INK2); ax.text(937,ycap*1.0,"replica 3 running",fontsize=7.5,color=INK2)
ax.step(ts,[min(v,ycap) for v in pub],where='post',color=POLICY,lw=1.8)
ax.step(ts,[min(v,ycap) for v in sim],where='post',color=REBUILD,lw=1.8,ls=(0,(4,2)))
ax.step(ts,[min(v,ycap) for v in eng_s],where='post',color=ENGINE,lw=1.8)
ax.axhline(20000,color=INK2,lw=1,ls=':')
ax.set_ylim(0,ycap*1.1); ax.set_xlim(0,1860); ax.set_xlabel("seconds since profiling start",color=INK2); ax.set_ylabel("total outstanding uncached prefill tokens, all replicas",color=INK2)
early=[i for i,t in enumerate(ts) if t<=650]; late=[i for i,t in enumerate(ts) if t>650]
rm=lambda idx,a,b: math.sqrt(st.mean((a[i]-b[i])**2 for i in idx))
handles=[Line2D([],[],color=POLICY,lw=1.8,label=f"policy: 60s mean as published (load-phase average {st.mean(pub)/1000:.1f}K)"),
 Line2D([],[],color=REBUILD,lw=1.8,ls=(0,(4,2)),label=f"rebuild: engine's own uncached counts held queued→first token, pushed through the same telemetry pipeline: 2s router samples, controller 60s mean, 10s metrics export ({st.mean(sim)/1000:.1f}K)"),
 Line2D([],[],color=ENGINE,lw=1.8,label=f"engine: chunk burn-down, same pipeline ({st.mean(eng_s)/1000:.1f}K)"),
 Line2D([],[],color=INK2,lw=1,ls=':',label="target 20K per replica"),
 Line2D([],[],color=INK2,lw=0.8,ls='--',alpha=0.6,label=f"new replica running  ·  policy vs rebuild: RMSE {rm(early,pub,sim)/1000:.1f}K before 650s, {rm(late,pub,sim)/1000:.1f}K after")]
fig.legend(handles=handles,loc='lower center',fontsize=8,frameon=False,ncol=1,bbox_to_anchor=(0.5,-0.2))
fig.suptitle("b200-workday-curve-prefill-mean · policy vs rebuild with the telemetry pipeline modeled (same sampling, same export)",x=0.01,ha='left',fontsize=10,color=INK)
out=f"{R}/analysis/policy_vs_engine_rebuild60_pipeline_0_1860.png"
fig.savefig(out,facecolor=SURFACE,bbox_inches='tight'); print("wrote",out)
EOF
python3 -I plot_pipeline.py
```

<details><summary>Bash result</summary>

```text
wrote /home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_pipeline_0_1860.png
Shell cwd was reset to /home/ray/default
```

</details>

**Read** `/home/ray/default/serve-llm-autoscaling-experiment/runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_pipeline_0_1860.png`

## Claude  (2026-11-03 02:05:05)

The reconstruction was off for three separable reasons. Two are telemetry, and once modeled they close most of the early-run gap. The third is real router behavior that appears after the first replica join and is not explained by the engine's numbers.

**Telemetry effects, now modeled exactly.** The published mean is not a clean time average of the router gauge. The router samples its gauge every 2 s (I fitted the tick phase: 0.3 s, drifting about 0.1 s per 300 s because the pusher sleeps *after* each task), the controller averages those samples over 60 s, and Ray's metrics agent exports the result every 10 s, so each scrape value is 0 to 12 s old and repeats 2 to 3 times. Pushing the engine-count rebuild through that same pipeline reproduces the exported raw gauge **to the token** at 152 of 162 early scrapes (71 without the model), and the published mean before 650 s to within RMSE 1.9K with no systematic bias worth chasing. The published value's own 2 s sampling contributes about 1.5K of that RMSE on its own. I also checked for scrape errors (none), scrape gaps (none, exactly 4.000 s), trace clock skew across engines (none), controller stalls (none), and encoder subsampling (none, float64 full series). New plot with the pipeline modeled: `runs/20261102T203807Z-b200-workday-curve-prefill-mean/analysis/policy_vs_engine_rebuild60_pipeline_0_1860.png`.

**Every remaining gauge mismatch, classified.** With the pipeline modeled, the exported gauge disagrees with the rebuild at 91 of 465 scrapes. Each disagreement is a difference of whole requests, so I could name the cause:
- Admission 0.01 to 0.45 s before the engine queues the request, and decrement within 80 ms after the engine's first token. Pure timing, both directions, tiny on the mean. This is the "not counted" set on the new replicas; the clocks were fine.
- Router sees **more** overlap than the engine uses, by exactly 14, 14, 112, and 30 KV blocks (the model's block is 784 tokens, so these are exact). All are cold prompts where the engine recomputed everything. Qwen3.6 is a hybrid Mamba model; vLLM only honors a prefix hit where a Mamba checkpoint exists, while the router's index knows the attention blocks. This pushes the router *below* the engine and is the whole reason the published mean sits about 1K under the rebuild before 650 s.
- Router sees **less** overlap than the engine, by 132, 54, 34, 76 blocks, always on warm turns of long sessions, starting right after replica 2 came up at 698 s. The over-count regresses on cached blocks created *before* the latest replica join with coefficient 0.17 and correlation 0.46, and on blocks created after it with coefficient zero. When no such sessions are held the residual is −1.2K; when they are, +2.1K. The first case is a session aged 698 s counted at its full 108825-token prompt 2 s after the join.

**What I could not pin down.** The router losing roughly a sixth of old sessions' indexed blocks has no visible cause: the router replica's log has four lines, dynamo's indexer logs nothing, and the KV event publishers report no drops. A bounded LRU index reproduces the average over-count (capacity near 4600 blocks per worker, which the first replica crossed at about 640 s) but not its timing, so I am not claiming that. Candidates are the selection service re-syncing its index when a worker is upserted, or an indexer capacity limit; we pass no `total_kv_blocks` in the worker record. Settling it needs the router to log `effective_prefill_tokens` and overlap per request, or a direct dump of the indexer after a replica join.

## User  (2026-11-03 03:42:04)

can you write this claude code trace into a file so i can commit it to github

## Claude  (2026-11-03 03:42:26)

I'll convert this session's transcript (user messages, my replies, every command I ran and a trimmed view of its output) into a markdown file in the harness repo, and copy the analysis scripts next to it so the trace is reproducible.

**Bash** — Check repo state and the session transcript file

```bash
cd /home/ray/default/serve-llm-autoscaling-experiment && git rev-parse --is-inside-work-tree 2>&1 && git status --short | head -5 && git log --oneline -1; F=/home/ray/.claude/projects/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a.jsonl; ls -la $F; python3 -I - $F <<'EOF'
import json,sys,collections
c=collections.Counter()
for line in open(sys.argv[1]):
    try: d=json.loads(line)
    except: continue
    c[d.get('type')]+=1
print(c)
EOF
```

<details><summary>Bash result</summary>

```text
true
 M README.md
 M experiments/b200_workday_curve_default.yaml
 M runs/burst-sweep-pooled/analysis/inflight_queue_by_size_4rows.png
 M runs/burst-sweep-pooled/analysis/inflight_ttft_breakdown_4rows.png
 D runs/burst-sweep-pooled/analysis/policy_queue_by_size_4rows.png
d2199b3 Trim the token-policy experiment configs
-rw------- 1 ray users 4706456 Nov  2 19:42 /home/ray/.claude/projects/-home-ray-default/0f3c0e18-6c3d-439a-b85c-5218c786df8a.jsonl
Counter({'attachment': 195, 'assistant': 174, 'user': 119, 'api-request-blob': 99, 'api-request': 56, 'mode': 48, 'permission-mode': 48, 'atis-latch': 48, 'last-prompt': 47, 'ai-title': 47, 'file-history-snapshot': 10, 'system': 8, 'tools-used': 4, 'file-history-delta': 4, 'api-request-shape': 1})
```

</details>
