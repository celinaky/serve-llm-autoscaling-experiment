"""Run the H100 freeze experiment using the preserved vLLM compatibility hook."""
from serve_profiler.compat import setup

setup()

from serve_llm_autoscaling.cli import main

raise SystemExit(main([
    "run", "experiments/h100_token_freeze_150s_injected_scheduler.yaml"
]))
