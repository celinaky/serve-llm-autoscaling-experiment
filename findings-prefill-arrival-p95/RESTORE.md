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
