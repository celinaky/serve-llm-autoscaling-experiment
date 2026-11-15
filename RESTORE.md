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

## 1a. Ray wheel and vLLM (when the image is older)

Check the installed versions first:

```bash
python -c "import ray, vllm, torch; print(ray.__commit__[:10], vllm.__version__, torch.__version__)"
```

The `anyscale/ray-llm:nightly-py312-cu130` image is not necessarily recent:
on 2026-11-15 a new B200 workspace on that image had Ray `0fe2388b9d`
(2026-10-07) and vLLM 0.31.0. With that wheel the linked Serve controller
fails to import (`start_metrics_wsgi_server`) and the linked `ray.llm` vLLM
engine fails (`vllm.entrypoints.common`, needs vLLM 0.33).

You don't have to change the image. Install the same workspace dependencies
as the H100 workspace. In an Anyscale workspace, `pip install` registers the
packages in the workspace's Dependencies tab for all cluster nodes. Do this
**before** linking the Ray checkout in section 2. If `serve`/`llm` are already
symlinks, undo them first (`unlink` the links, `mv serve.bak serve`,
`mv llm.bak llm`). Otherwise pip's uninstall of the old wheel follows the
links and deletes files in `/mnt/local_storage/ray`.

```bash
cd /home/ray/default
pip install pytest anyscale scikit-learn pytest-asyncio "black==22.10.0" "ruff==0.3.4" \
  "nvidia-cutlass-dsl==4.8.0" "quack-kernels==0.6.5" \
  "ai-dynamo-runtime @ https://air-example-data.s3.us-west-2.amazonaws.com/rayllm-ossci/dynamo/f9c22d7/ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl" \
  "ray[llm,serve] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/f0a9db5f0f561f01669814cdf38188529fdf85d1/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl"
# Every nightly is "3.0.0.dev0", so pip treats Ray as already installed and
# keeps the old wheel (it still upgrades vLLM to 0.33.0 / torch 2.14).
# Force the wheel itself:
pip install --force-reinstall --no-deps \
  "https://s3-us-west-2.amazonaws.com/ray-wheels/master/f0a9db5f0f561f01669814cdf38188529fdf85d1/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl"
python -c "import ray, vllm; print(ray.__commit__[:10], vllm.__version__)"   # f0a9db5f0f 0.33.0
```

`f0a9db5f0f` (2026-11-06) is the wheel the H100 workspace's dependencies
use. With it, every non-test `ray.serve`/`ray.llm` module from the stack
imports except `ray.llm._internal.common.observability.logging.setup`
(`ray._private.ray_logging.filters`), which the serving path does not import.
The running cluster accepted the new wheel without a restart: a driver and a
GPU task both reported `f0a9db5f0f` and vLLM 0.33.0.

Run Ray drivers from a small directory. Anyscale uploads the current
directory as the working dir, so a driver started from `/tmp` failed
(4.5 GB > 512 MB limit).

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
nightly close to that base (`a450e8d44b` and `f0a9db5f0f` worked; see 1a).
The wheel's `serve/generated` protos can be older than the stack's
`serve.proto` (it changed between `0fe2388b9d` and the base). If the wheel
is not near the base, use the `workspace/ray-serve-generated` copy below
instead of the wheel's. Then you don't need bazel.

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

The patch applied cleanly to the vLLM 0.33.0 installed by section 1a on the
B200 workspace. Back up `vllm/compilation/{backends,decorators}.py` first.

## 3a. B200 workspace

Set up on 2026-11-15: a single node with 8x B200 (the head node holds all the
GPUs, so the head's symlinks and patches cover every replica). Image
`anyscale/ray-llm:nightly-py312-cu130`. Follow sections 1, 1a, 2, 3 in that
order. Differences from the H100 setup:

- Section 1a is required: the image's Ray and vLLM are too old.
- Experiment configs: copy the H100 config and set
  `deployment.accelerator_type: B200`. Recalibrate
  `target_prefill_tokens_per_replica` (20000 is H100-calibrated) from a
  measured B200 prefill rate. Not done yet.
- Two `RUNNING` replicas and a full autoscaling run had not been validated on
  B200 when this was written.
- `HF_TOKEN` was not set in the new workspace.
- After a workspace restart, check `ray.__commit__` again, and check that
  `/mnt/local_storage/ray`, the `serve`/`llm` symlinks and the vLLM patch
  survived. The symlinks and patch are edits to site-packages. Local storage is
  node-local. The Dependencies tab reinstalls packages, but a same-version Ray
  wheel can be skipped again.

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
