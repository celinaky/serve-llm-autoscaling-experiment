"""Process-local compatibility between the Ray branch and vLLM 0.33.

Runs in the driver (``cli``) and, as the ``worker_process_setup_hook``, in
every replica worker.

vLLM 0.33 moved FrontendArgs from entrypoints.launchers.cli_args to
entrypoints.common.cli_args. No installed package or API implementation changes.
"""
import importlib
import importlib.util
import os
import sys


def setup() -> None:
    # Nested vLLM Ray drivers use the explicitly uploaded runtime environment.
    # The workspace hook would instead try to upload the entire checkout.
    os.environ.pop("RAY_RUNTIME_ENV_HOOK", None)
    old = "vllm.entrypoints.launchers.cli_args"
    if importlib.util.find_spec(old) is None:
        new = importlib.import_module("vllm.entrypoints.common.cli_args")
        if not hasattr(new, "FrontendArgs"):
            raise RuntimeError("Unsupported vLLM: FrontendArgs is unavailable")
        sys.modules[old] = new
