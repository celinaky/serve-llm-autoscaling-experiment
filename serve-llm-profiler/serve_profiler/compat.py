"""Process-local compatibility for the workspace's Ray/vLLM import rename.

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
