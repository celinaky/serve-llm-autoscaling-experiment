from serve_profiler.compat import setup
setup()
from serve_llm_autoscaling.cli import main
raise SystemExit(main(["run", "experiments/h100_token_metrics_injected_scheduler_local.yaml"]))
