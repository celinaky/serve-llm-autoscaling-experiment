# Restoring the token autoscaling workspace (2026-11-14)

Point a fresh Claude Code session at this file. It replaces
`findings-prefill-arrival-p95/RESTORE.md` (2026-11-02, B200, older branches)
for the token autoscaling work.

Source workspace: 8x H100 80GB, Python 3.12 in `/home/ray/anaconda3`, Ray
wheel 3.0.0.dev0 at `a450e8d44b`, vLLM 0.33.0, torch 2.14.0,
ai-dynamo-runtime 1.7.0 (preinstalled in the image).

Everything below is in git except caches, raw run streams and the Ray protos,
which regenerate.

## 1. Harness (this repo)

```bash
cd /home/ray/default
git clone https://github.com/celinaky/serve-llm-autoscaling-experiment.git
cd serve-llm-autoscaling-experiment          # branch main
pip install -e .                             # provides `autoscale-harness`
pip install -e serve-llm-profiler            # injected vLLM scheduler + tracing
```

`serve-llm-profiler/` is vendored here (code only; its runs and validation
data stayed in the old workspace). The harness ships
`serve-llm-profiler/serve_profiler` with each deployment when
`analysis.scheduler_trace_backend: injected`.

`workspace/` holds the files that lived loose in `/home/ray/default`: the
agent notes, the vLLM patch, the run drivers and plot scripts. They refer to
`/home/ray/default/...`, so copy them back there:

```bash
cp -a workspace/. /home/ray/default/
```

That restores `/home/ray/default/AGENTS.md` (read it first),
`agent-context/`, `vllm-0.33-device-aware-compile-cache.patch`,
`keep_alive.py` and the `run-*.py` drivers. The status/log files the notes
link to were not copied.

AIPerf is pinned by commit in each experiment's `runtime.aiperf_source`. Clone
`https://github.com/celinaky/aiperf.git` to `/home/ray/default/aiperf` only to
edit it or for the profiler's `run-configs/`.

## 2. Ray fork

The token autoscaling stack, oldest first:

| PR | Branch | Content |
|---|---|---|
| ray-project/ray#67937 (merged) | - | KV router tracks the autoscaling workload |
| ray-project/ray#67938 | `celina/prefill-token-autoscaling-2-metrics-path` | Router metrics to the controller, aggregated per handle |
| ray-project/ray#67941 | `celina/prefill-token-autoscaling-3-policy` | `LLMTokenAutoscalingPolicy` |
| ray-project/ray#68148 | `celina/prefill-token-autoscaling-4-metrics` | `serve_autoscaling_router_metric` gauge + KV Router Grafana panel |
| celinaky/ray#8 (fork, draft) | `celina/prefill-token-autoscaling-post-downscale-cooldown` | `post_downscale_cooldown_s` |

PRs 2-4 also exist upstream under the same names. Experiments use the top
of the stack:

```bash
mkdir -p /mnt/local_storage && cd /mnt/local_storage
git clone https://github.com/celinaky/ray.git && cd ray
git checkout celina/prefill-token-autoscaling-post-downscale-cooldown
```

The stack is based on upstream `235f05f06e` (2026-11-13). Only `ray.serve` and
`ray.llm` come from the checkout, so the installed Ray wheel must be a
nightly close to that base (`a450e8d44b` worked).

Wire the checkout into the installed Ray (the wheel's packages are moved
aside, not deleted), and give it generated protos, which are not in git:

```bash
SP=$(python -c "import ray,os;print(os.path.dirname(ray.__file__))")
mv $SP/serve $SP/serve.bak && ln -s /mnt/local_storage/ray/python/ray/serve $SP/serve
mv $SP/llm   $SP/llm.bak   && ln -s /mnt/local_storage/ray/python/ray/llm   $SP/llm
# Protos: the stack does not change them. Use the wheel's copy if it has one,
# else the copy generated on the old workspace (same base), else regenerate
# with `bazel run --jobs=4 //:gen_py_proto` in the checkout (bazel 7.5.0).
G=/mnt/local_storage/ray/python/ray/serve/generated
if [ -d $SP/serve.bak/generated ]; then cp -a $SP/serve.bak/generated $G
else mkdir -p $G && cp /home/ray/default/serve-llm-autoscaling-experiment/workspace/ray-serve-generated/*.py $G/; fi
python -c "import inspect, ray.serve.experimental.llm_token_autoscaling_policy as m; print(m.__file__, inspect.signature(m.LLMTokenAutoscalingPolicy))"
```

The signature should list `target_prefill_tokens_per_replica` and
`post_downscale_cooldown_s`.

## 3. vLLM

Read `/home/ray/default/agent-context/ray-vllm-runtime-patches.md`. On vLLM
0.33 with more than one replica per node, apply the device-aware compile
cache patch, or extra replicas fail at startup while the benchmark keeps
running on one. Other vLLM versions: inspect and adapt it, do not force it.
Validate two `RUNNING` model replicas before an autoscaling run.

## 4. Experiments

Configs that run on the current policy (it accepts only
`target_prefill_tokens_per_replica` and `post_downscale_cooldown_s`):

- `experiments/h100_token_metrics_injected_scheduler_local.yaml`: no cooldown.
- `experiments/h100_token_freeze_150s_injected_scheduler.yaml`,
  `h100_token_freeze_60s_injected_scheduler.yaml`,
  `h100_token_single_replica_injected_scheduler.yaml`: cooldown, with
  scaling factors 1.0 (required).
- `experiments/h100_workday_curve_prefill_mean_3_policy.yaml`.

The other token configs pass kwargs from older policy versions
(`upscale_aggregation`, `target_decode_requests_per_replica`, ...) and fail
at deploy. The averaging window is Serve's `look_back_period_s`; the policy
no longer takes `upscale_lookback_s`.

Before running on different GPUs:

- Change `deployment.accelerator_type` (the configs say `H100`).
- Recalibrate `target_prefill_tokens_per_replica`. The 20K target assumes an
  H100's prefill rate (about 7K uncached tokens/s per replica for
  Qwen/Qwen3.6-27B-FP8 with 16K-token chunks).
- `analysis.scheduler_trace_dir` points at `/mnt/cluster_storage/...`.
  Cluster storage is per Anyscale cloud; change it if the new workspace is in
  another cloud.
- The HF model cache (about 30 GB) and AIPerf dataset cache download again on
  first use. Set `HF_TOKEN`.

Launch one with `python /home/ray/default/run-local-token-metrics.py` from the
harness directory, or `autoscale-harness run experiments/<config>.yaml`.
Run `python /home/ray/default/keep_alive.py &` during long runs so the
workspace does not idle-terminate.

## 5. Watching and analyzing

- Live: in Grafana Explore (Prometheus), query
  `max by (application, deployment) (ray_serve_autoscaling_router_metric{metric="outstanding_effective_prefill_tokens"})`,
  the value the policy scales on, with `ray_serve_autoscaling_target_replicas`.
  The Serve LLM dashboard's KV Router row shows the same panel only after the
  head node regenerates its dashboards from the checkout's
  `python/ray/dashboard` files, which the symlinks above do not cover.
- After a run, the harness analysis and the policy vs engine chart
  (`analysis/policy_vs_engine_mean<window>_*.png`) read
  `serve_autoscaling_router_metric`. The raw (unaveraged) router snapshot is
  no longer exported.

## 6. Not carried over

- Raw run streams (`serve_metrics.jsonl`, profile exports, plot data,
  `mooncake_metrics.jsonl`) are gitignored; a best-effort copy is under
  `/mnt/cluster_storage/serve-llm-autoscaling-experiment/runs`.
- The profiler's `runs/` and `validation/` data (about 800 MB).
- Other projects in the old workspace (`online-prefill-harness`,
  `ray-online-prefill`, `mooncake-kv*`, `nfer`, ...).
- Claude Code transcripts and memory in `~/.claude`.
