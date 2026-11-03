# Restoring the serve-llm-autoscaling workspace (migrated 2026-11-02)

Point a fresh Claude Code session at this file. Everything it needs is
either in git or under `/mnt/cluster_storage/serve-llm-autoscaling-experiment/`
(cluster storage, shared across Anyscale workspaces in this cloud).

The old workspace ran 8x B200 with Ray 3.0.0.dev0 at commit
`e91b1de0e096df2a27567594b31827fe12b593b2`, vLLM 0.31.0, Python 3.12 in
`/home/ray/anaconda3`. `pip-freeze.txt` next to this file is the full list.

## 1. Repos

```bash
cd /home/ray/default
git clone https://github.com/celinaky/serve-llm-autoscaling-experiment.git
cd serve-llm-autoscaling-experiment && git checkout track-runs-dir   # 0febf88
pip install -e .                                                       # provides `autoscale-harness`

git clone https://github.com/celinaky/aiperf.git /home/ray/default/aiperf   # optional, branch workday-curve
# The experiments pin the AIPerf fork by commit in runtime.aiperf_source, so the clone is only for editing it.

mkdir -p /mnt/local_storage && cd /mnt/local_storage
git clone https://github.com/celinaky/ray.git
cd ray
git checkout celina/prefill-token-autoscaling-policy   # mean policy, e643369b6e
# or: git checkout celina/prefill-token-autoscaling-p95   # arrival policy, fbcfdbcef5
```

Wire the checkout into the installed Ray (the wheel's own `serve` and `llm`
packages are moved aside, not deleted):

```bash
SP=$(python -c "import ray,os;print(os.path.dirname(ray.__file__))")
mv $SP/serve $SP/serve.bak && ln -s /mnt/local_storage/ray/python/ray/serve $SP/serve
mv $SP/llm   $SP/llm.bak   && ln -s /mnt/local_storage/ray/python/ray/llm   $SP/llm
python -c "import ray.serve.experimental.llm_token_autoscaling_policy as m; print(m.__file__)"
```

## 2. Dynamo runtime wheel (needed for session pinning)

```bash
pip install --no-deps --force-reinstall /mnt/cluster_storage/kvfinal4/ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl
python -c "from dynamo.llm import KvRouterConfig; import inspect; print('session_affinity' in str(inspect.signature(KvRouterConfig)) or 'ok')"
```

## 3. Claude Code sessions, memory and settings

```bash
tar xzf /mnt/cluster_storage/serve-llm-autoscaling-experiment/workspace-migration-2026-11-02/claude-home.tar.gz -C /home/ray
```

This restores `~/.claude/projects/-home-ray-default/` (six session
transcripts, the `memory/` directory with `MEMORY.md`), `settings.json`,
skills, plugins and job logs. It only lines up if the project directory is
again `/home/ray/default`; otherwise rename the `-home-ray-default` folder
to match. Then `claude --resume` lists the sessions. The main one for the
autoscaling work is `c79a47ff-421c-436f-a2a9-451dc9ea4832`.

## 4. Run data

The repo tracks each run's configs, manifest, evaluation and charts. The raw
streams (profile exports, telemetry, plot data) are gitignored and live on
cluster storage; the chart tools need them.

```bash
rsync -a /mnt/cluster_storage/serve-llm-autoscaling-experiment/runs/ /home/ray/default/serve-llm-autoscaling-experiment/runs/
```

vLLM scheduler traces are already where the tools look:
`/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces/`.
Launch every run with
`RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR=/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces`
so new traces land there too.

## 5. Where the knowledge is

- `findings-prefill-arrival-p95/README.md` in the harness repo: the summary
  of 2026-10-30 to 11-02 (policies, runs, findings, tools, open threads).
  Start here.
- `findings-token-policy/` and `findings-pd-token-policy/`: the earlier
  investigations.
- `sessions/2026-10-30_to_11-02_prefill-arrival-p95-session_c79a47ff.jsonl.gz`
  next to this directory: the raw transcript of the latest session.
- `keep_alive.py` here: the script that keeps an Anyscale workspace from
  idle-terminating during long runs (`python keep_alive.py &`).

## 6. Things to check before the first run

- No stray `aiperf` processes, no `core` files in the repo (the runtime env
  uploads the working dir, 512 MiB cap), Serve fully down (`serve status`).
- The fork's `celina/prefill-token-autoscaling-p95` tip (`803834bf81`,
  "Scale token autoscaling on arrival mean") postdates the runs described in
  the notes, which used the p95 rule at `debcf256b6`.

## 7. Moving from B200 to H100: what changes

Old workspace: `celina-serve-llm-autoscaling-b200`, cloud
`cld_82np1njz31y9lwk56mc2xcc23x` (region `no-sys2-osl1`), image
`anyscale/ray-llm:nightly-py312-cu130`, Ray wheel commit `e91b1de0e0`.
The Ray branches are based on upstream `a450e8d44b`; the symlink trick in
step 1 only replaces `ray.serve` and `ray.llm`, so the new workspace's Ray
wheel must be a nightly close to that base or the checkout will not import.

- **Cluster storage is per cloud.** If the H100 workspace is in a different
  Anyscale cloud, nothing under `/mnt/cluster_storage` is visible there.
  Check first. If it is not, copy `workspace-migration-2026-11-02/` (about
  100 MB), `sessions/` (47 MB) and, if wanted, `runs/` (1.6 GB) and
  `vllm-scheduler-traces/` (9.7 GB) through a bucket you control, or
  download the small ones through the workspace file browser before
  shutting down. Everything in git needs no copying.
- **Caches that are gone and regenerate:** the HF model cache
  (`~/.cache/huggingface`, 30 GB, Qwen/Qwen3.6-27B-FP8) downloads again on
  first deploy, allow 10-20 min; the AIPerf dataset cache (`~/.cache/aiperf`,
  8.4 GB, `semianalysis_cc_traces_weka_062126_256k`) downloads again on the
  first benchmark. Set `HF_TOKEN` to avoid rate limits.
- **Every experiment file says `accelerator_type: B200`.** Change to `H100`.
- **Prefill rate.** B200 prefilled about 16K uncached tokens per second per
  replica (one 16384-token chunk per step at about one step per second).
  H100 FP8 compute is roughly 40% of B200's, so expect 6-8K tok/s. The
  calibration behind the 20K target (about 1s of engine queue per 16K
  ahead) and behind the per-replica session counts (4 sessions per replica
  under 6K p95, 5 at 15K, 6+ missing 20K) is B200-specific. Re-measure
  with the single-replica burst sweeps (`experiments/b200_burst_sweep*.yaml`,
  `b200_fast_curve_single_replica.yaml`, analysis in
  `runs/burst-sweep-pooled`) before trusting any target. A 20K target on
  H100 means roughly 2.5-3s of queue; the 1s-equivalent is about 8K.
- **Replica count.** At 270 uncached tok/s per pinned session, 32 sessions
  needed 3-4 B200 replicas. On H100 expect 6-8, which is the `max_replicas:
  8` cap. Either ramp to fewer sessions (16-20) or raise the cap, or the
  policy comparison turns into a capacity test.
- **KV cache.** H100 has 80 GB against B200's 180 GB. The 27B FP8 weights
  take about 27 GB, leaving roughly 45 GB of fp8 KV at
  `gpu_memory_utilization: 0.92` versus about 140 GB. Fewer sessions'
  contexts stay in the prefix cache, so more re-prefill and more sensitivity
  to pinning. `max_model_len: 262144` still fits a single request; check
  `max_num_seqs` and watch vLLM's preemption counters
  (`preemptions_before_first_token` in `load_requests.csv`).
- **Scheduler step time.** `max_num_batched_tokens: 16384` gives longer
  steps on H100, which raises ITL during prefill bursts. 8192 is worth
  trying if ITL p95 goes over 50ms.
- **Engine knobs that are Blackwell-only** in the logs (DeepGemm, cutlass
  DSL, `quack-kernels`, `nvidia-cutlass-dsl` in the runtime env pip list)
  are harmless on H100 but unused; `kv_cache_dtype: fp8` works on Hopper.
- **The harness's `b200_` file prefix** is a name only; copy a file to an
  `h100_` name and change `accelerator_type`, `name`, and the targets.

## 8. If cluster storage is not shared (it was not: the H100 workspace is in another cloud)

Everything that was only on the B200 cloud's storage is packed into one
folder, `/home/ray/default/migration/` on the old workspace, 1.2 GB,
with `SHA256SUMS`:

| File | Size | Restores |
|---|---|---|
| `claude-home.tar.gz` | 98 MB | step 3: Claude sessions, memory, settings |
| `2026-10-30_to_11-02_prefill-arrival-p95-session_c79a47ff.jsonl.gz` | 49 MB | the raw transcript |
| `ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl` | 78 MB | step 2; `pip install --no-deps --force-reinstall <wheel>` (not on PyPI, do not skip) |
| `runs-raw-streams.tar.gz` | 260 MB | step 4; extract at the harness repo root: `tar xzf runs-raw-streams.tar.gz -C <repo>` |
| `vllm-scheduler-traces.tar.gz` | 697 MB | extract into the new cluster storage: `mkdir -p /mnt/cluster_storage/serve-llm-autoscaling-experiment && tar xzf vllm-scheduler-traces.tar.gz -C /mnt/cluster_storage/serve-llm-autoscaling-experiment` |
| `pip-freeze.txt`, `gpus.txt`, `keep_alive.py`, `RESTORE.md` | small | reference |

Once the folder is on the new workspace (GitHub release assets, a bucket,
or a download and upload), verify with `sha256sum -c SHA256SUMS`, then
follow steps 1 to 4 using these files in place of the cluster-storage paths.
