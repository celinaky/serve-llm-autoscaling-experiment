# Required Ray/vLLM runtime patches for H100 experiments

Recorded 2026-11-12. Read this before restoring the environment or starting a
Ray Serve LLM experiment. Installed-package edits can disappear after a package
reinstall or environment rebuild. Preserve the patch files in `~/default` and
check every worker node; checking the driver's environment alone is insufficient.

## Required: device-aware vLLM compilation caches

The installed vLLM 0.33.0 needs
[vllm-0.33-device-aware-compile-cache.patch](../vllm-0.33-device-aware-compile-cache.patch).
It was adapted from `vllm-device-aware-compile-cache.patch` preserved in
`restart-backup-20261109T200332Z/existing-runtime-patches.tar.gz`.

The first 150-second freeze run completed with only one running model replica.
Additional replicas repeatedly loaded the first replica's AOT artifact from the
same `torch_aot_compile/<hash>/rank_0_0/model` path and failed during startup with
`CUDA driver error: invalid argument`, including `cuPointerGetAttribute` errors.
Logs strongly implicate reuse of compiled artifacts across CUDA device indices.
A successful benchmark exit did not establish successful scale-out.

The adapted patch changes installed `vllm/compilation/backends.py` and
`vllm/compilation/decorators.py`:

- Add `_dev<CUDA ordinal>` to compilation rank directories.
- Put AOT artifacts under `torch_aot_compile/<hash>/device_<CUDA ordinal>/`.
- Isolate the shared AOT Inductor directory inside that device directory too.
  With vLLM 0.33, changing only the rank suffix leaves the parent Inductor cache
  shared across devices.

Original installed files are backed up in
`/home/ray/default/ray-runtime-compat-backup-freeze150/vllm-0.33-device-cache/`.
The fix was applied to the installed package, not committed into the Ray repo.
It does not disable compilation or change the autoscaling policy.

### Check and apply after restoring packages

First check the installed vLLM version and inspect the two files. If the
device-aware paths already exist, do not apply the patch again. For an unpatched
vLLM 0.33 installation, from `~/default`:

```bash
python -c "import importlib.metadata; print(importlib.metadata.version('vllm'))"
vllm_site_root=$(python -c "import importlib.util, pathlib; print(pathlib.Path(importlib.util.find_spec('vllm').origin).parent.parent)")
patch --dry-run -p1 -d "$vllm_site_root" < /home/ray/default/vllm-0.33-device-aware-compile-cache.patch
# Back up backends.py and decorators.py before applying.
patch -p1 -d "$vllm_site_root" < /home/ray/default/vllm-0.33-device-aware-compile-cache.patch
```

Use the Python environment that the Ray workers use. Installed-package writes
may require sandbox escalation. For a different vLLM version, inspect and adapt
the patch; do not force hunks into changed code. Restart affected model workers
after changing installed code. Clearing all caches alone is insufficient:
replicas would otherwise recreate the same shared paths.

### Validate before interpreting an experiment

Start two model replicas concurrently using the experiment's real model and
engine settings. Require two `RUNNING` model replicas, not just a target count
of two or a healthy router. Confirm device-specific paths in worker logs and
check for AOT reload, missing cubin, and CUDA startup errors. Also verify that
cached startup works after restarting replicas.

The validation-and-rerun driver is
[run-freeze150-device-cache.py](../run-freeze150-device-cache.py). It starts a
fixed two-replica validation deployment, checks both running counts, tears it
down, and only then starts the unchanged 150-second freeze experiment.
Inspect its status before launching another copy:

- [Stage/status](../freeze150-device-cache-status.json)
- [Process and log location](../freeze150-device-cache-run.json)
- [Validation artifacts](../freeze150-device-cache-validation/)

Live validation passed on 2026-11-12: Serve reported two `RUNNING` model
replicas simultaneously, and worker logs recorded separate `device_0` and
`device_1` AOT paths. The readiness evidence is in
`freeze150-device-cache-validation/ready_status.json`. The supervisor then
tears down validation and starts the experiment, reusing those device-specific
caches. Cached startup also passed in the rerun
`20261112T193950Z-h100-token-freeze-150s-injected-scheduler`: its first worker
loaded `device_0/rank_0_0_dev0/model`, the application became ready, and AIPerf
started profiling. Consult the supervisor status and run artifacts for the
subsequent scale-out and experiment outcome.

## Ray wheel and linked Serve compatibility

The observed wheel is Ray `3.0.0.dev0`, commit
`a450e8d44b938a7523f8d647517e3485b116b233` (October 28). LLM and Serve are
symlinked into `/mnt/local_storage/ray`; the freeze experiment checkout is
`celina/prefill-token-metrics-local-experiment` at `9b66a28e6541`.
The same development version string does not imply matching source commits.

The linked Serve imports `start_metrics_wsgi_server`, added October 30 in
`62a5cc1a07`, which the installed wheel lacked. Two definitions were backported
from the checkout into installed
`ray/_private/authentication/http_token_authentication.py`:
`_ReusePortThreadingWSGIServer` and `start_metrics_wsgi_server`. They reuse the
existing authentication wrapper; authentication was not disabled.

- Original: `ray-runtime-compat-backup-freeze150/http_token_authentication.py`.
- Exact patch and provenance: the first freeze run's
  `deployment/runtime-compat.patch` and `deployment/runtime-compat.json` under
  `serve-llm-autoscaling-experiment/runs/20261112T180926Z-h100-token-freeze-150s-injected-scheduler/`.

If the controller import fails on this helper after a reinstall, restore this
backport or align the wheel with the linked checkout. Do not replace the entire
core package just to satisfy that import. The existing
`serve_profiler.compat.setup` hook also handles the vLLM 0.33 `FrontendArgs`
module rename and is used by the experiment driver and model workers.
