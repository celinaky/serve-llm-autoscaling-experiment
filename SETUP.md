# Environment setup

The harness needs a specific Ray branch, vLLM 0.33 with one patch, and a GPU
cluster. These steps were last carried out on 2026-11-15 on an Anyscale
workspace: one node with 8x B200, image `anyscale/ray-llm:nightly-py312-cu130`,
Python 3.12. Follow them in order.

After any workspace restart, repeat the checks marked **Check**: the Ray
checkout lives on node-local storage and the Ray links and vLLM patch are
edits to site-packages, so all three can disappear.

## 1. Harness

```bash
cd /home/ray/default
git clone https://github.com/celinaky/serve-llm-autoscaling-experiment.git
cd serve-llm-autoscaling-experiment
pip install -e ".[dev]"        # provides `autoscale-harness`
```

Install into the Ray image's own Python, not a virtualenv: the driver must
share the replicas' environment. AIPerf needs no install; the harness runs
the pinned build through `uvx` (so `uv` must be on `PATH`).

## 2. Ray wheel and vLLM

**Check:**

```bash
python -c "import ray, vllm; print(ray.__commit__[:10], vllm.__version__)"   # f0a9db5f0f 0.33.0
```

The image's Ray and vLLM can be older than that. If so, install the versions
below **before** linking the Ray checkout in step 3. If `serve`/`llm` are
already symlinks, undo them first (`unlink` them, `mv serve.bak serve`,
`mv llm.bak llm`); otherwise pip's uninstall follows the links and deletes
files in the checkout.

```bash
pip install scikit-learn pytest-asyncio \
  "ai-dynamo-runtime @ https://air-example-data.s3.us-west-2.amazonaws.com/rayllm-ossci/dynamo/f9c22d7/ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl" \
  "ray[llm,serve] @ https://s3-us-west-2.amazonaws.com/ray-wheels/master/f0a9db5f0f561f01669814cdf38188529fdf85d1/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl"
# Every nightly is "3.0.0.dev0", so pip treats Ray as already installed and
# keeps the old wheel. Force the wheel itself:
pip install --force-reinstall --no-deps \
  "https://s3-us-west-2.amazonaws.com/ray-wheels/master/f0a9db5f0f561f01669814cdf38188529fdf85d1/ray-3.0.0.dev0-cp312-cp312-manylinux2014_x86_64.whl"
```

In an Anyscale workspace, `pip install` registers the packages for every
cluster node.

## 3. Ray branch

The harness is pinned to the top of the token-autoscaling PR stack:

| | |
|---|---|
| Repository | `https://github.com/celinaky/ray` (the same branches exist in `ray-project/ray`) |
| Branch | `celina/prefill-token-autoscaling-post-downscale-cooldown` |
| Commit | `4db7cf9007` (2026-11-20) |

It contains, oldest first:

| PR | Branch (`celina/prefill-token-autoscaling-...`) | What it adds |
|---|---|---|
| ray-project/ray#67938 | `2-metrics-path` | Replicas report custom autoscaling metrics to the controller; application policies see them |
| ray-project/ray#67941 | `3-policy` | `LLMTokenAutoscalingPolicy`, an application-level policy |
| ray-project/ray#68148 | `4-metrics` | `serve_autoscaling_replica_metric` gauge: the signal the policy scales on, which the harness records and plots |
| ray-project/ray#68414 | `post-downscale-cooldown` | `post_downscale_cooldown_s` |

What the harness relies on from this stack:

- **Policy**: `ray.llm._internal.serve.routing_policies.kv_aware.autoscaling_policy:LLMTokenAutoscalingPolicy`
  with kwargs `target_prefill_tokens_per_replica` and
  `post_downscale_cooldown_s`. It is an application-level policy, which Serve
  accepts only through an application config, so the harness deploys from a
  config rather than `serve.run()`.
- **Gauge**: `ray_serve_autoscaling_replica_metric{deployment="LLMRouter",
  metric="outstanding_effective_prefill_tokens:<server deployment>"}`, one
  series per router replica; the policy scales on the largest.

If the stack is rebased, update the commit above, re-check the two items
(`autoscale-harness run experiments/smoke.yaml` exercises both), and
regenerate the protos if `src/ray/protobuf/serve.proto` changed
(`setup/ray-serve-generated/README.md`). The harness refuses to deploy policy
kwargs the checked-out policy does not accept.

Two experiments that predate this stack are unfinished drafts against its
old design and are not supported: router prefill decay (celinaky/ray#9) and
a downscale step limit (celinaky/ray#10).

Only `ray.serve` and `ray.llm` come from the checkout; the rest of Ray is the
wheel from step 2, which must be a nightly close to the stack's base
(upstream `235f05f06e`, 2026-11-13).

```bash
mkdir -p /mnt/local_storage && cd /mnt/local_storage
git clone https://github.com/celinaky/ray.git && cd ray
git checkout celina/prefill-token-autoscaling-post-downscale-cooldown

# Link the checkout into the installed Ray (the wheel's packages are moved
# aside, not deleted).
SP=$(python -c "import ray,os;print(os.path.dirname(ray.__file__))")
mv $SP/serve $SP/serve.bak && ln -s /mnt/local_storage/ray/python/ray/serve $SP/serve
mv $SP/llm   $SP/llm.bak   && ln -s /mnt/local_storage/ray/python/ray/llm   $SP/llm

# Generated protos are not in git, and the stack changes serve.proto, so the
# wheel's copy is too old. Use the copy kept in this repo.
G=/mnt/local_storage/ray/python/ray/serve/generated
mkdir -p $G && cp /home/ray/default/serve-llm-autoscaling-experiment/setup/ray-serve-generated/*.py $G/
```

**Check:**

```bash
git -C /mnt/local_storage/ray log --oneline -1          # 4db7cf9007
python -c "import inspect, ray.llm._internal.serve.routing_policies.kv_aware.autoscaling_policy as m; print(m.__file__, inspect.signature(m.LLMTokenAutoscalingPolicy))"
```

The file must be under `/mnt/local_storage/ray` and the signature must be
`(target_prefill_tokens_per_replica: int, post_downscale_cooldown_s: float = 0)`.

## 4. vLLM compile-cache patch

vLLM 0.33 shares one compile cache between GPUs. With more than one replica
per node, the second replica loads the first one's compiled artifact, fails
at startup with `CUDA driver error: invalid argument`, and the benchmark
carries on against a single replica: the run looks fine and never scaled.
`setup/vllm-0.33-device-aware-compile-cache.patch` gives each CUDA device its
own cache directory.

```bash
VLLM=$(python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent.parent)")
patch --dry-run -p1 -d "$VLLM" < setup/vllm-0.33-device-aware-compile-cache.patch
cp $VLLM/vllm/compilation/{backends,decorators}.py /tmp/     # backup
patch -p1 -d "$VLLM" < setup/vllm-0.33-device-aware-compile-cache.patch
```

Apply it in the Python environment the Ray workers use, on every node. If
the dry run reports the patch as already applied, do not apply it again. For
another vLLM version, read the patch and adapt it; do not force it.

**Check:** `autoscale-harness warm experiments/smoke.yaml` (next section)
starts one replica per GPU and fails unless every one reaches `RUNNING`.

## 5. Warm the compile caches

A replica that lands on a GPU without a compile cache compiles from scratch:
on B200 about 112s of engine start against 32s warm, so a scale-up takes
about 170s instead of 90s. A run with only some GPUs warm mixes fast and slow
scale-ups and is not comparable to anything.

```bash
autoscale-harness warm experiments/<config>.yaml
ls ~/.cache/vllm/torch_compile_cache/torch_aot_compile/*/     # device_0 ... device_7
```

The cache key is a hash of the engine config. Warm once per engine config,
and again after changing any `deployment.engine` setting or the vLLM version.
The cache is under `~/.cache`, so it survives a restart only if the home
directory does. A cold warm-up took about 5 minutes on B200, a warm one 1.5.
`autoscale-harness run <config> --warm` does the same immediately before a
run.

## 6. Things that go wrong on a long-lived cluster

- **Metrics page grows.** Ray's per-node metrics page keeps series for every
  replica the cluster has ever run, and each deployment (and each warm-up)
  adds more. On 2026-11-20, after about 570 replicas, the page was 43 MB and
  took 1.9s to produce, so the harness sampled the policy signal every ~3.5s
  instead of every 2s. The run summary warns when scrapes fail. Restarting
  the cluster resets the page.
- **Workspace disk (10 GB).** A 64-session run writes 2-9 GB of raw
  telemetry under `runs/<run>/telemetry` and `benchmark`. Move or compress
  finished runs' raw files; a run cannot be re-analyzed without them.
- **Run drivers from this repository's directory.** Anyscale uploads the
  current directory with every Ray job, capped at 512 MiB.

## 7. Other GPUs

- Set `deployment.accelerator_type` in the config.
- Recalibrate `target_prefill_tokens_per_replica`. 20000 was chosen for an
  H100's prefill rate (about 7K uncached tokens/s per replica for
  `Qwen/Qwen3.6-27B-FP8` with 16K-token chunks) and has not been recalibrated
  for B200.
- Size `kv_offloading_size` above the GPU KV cache (see the comment in the
  experiment configs).
- With replicas on more than one node, set `analysis.scheduler_trace_dir` to
  shared storage, and apply the vLLM patch and Ray links on every node.
