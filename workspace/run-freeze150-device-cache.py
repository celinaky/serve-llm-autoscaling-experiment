"""Gate the freeze rerun on two healthy replicas using device-aware caches."""
import json
import os
import shutil
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from serve_profiler.compat import setup

setup()

from serve_llm_autoscaling.backend import (
    RayServeBackend,
    check_autoscaling_support,
    check_routing_support,
    verify_topology,
)
from serve_llm_autoscaling.config import load_config, write_config
from serve_llm_autoscaling.runner import run_experiment

ROOT = Path('/home/ray/default')
CONFIG = ROOT / 'serve-llm-autoscaling-experiment/experiments/h100_token_freeze_150s_injected_scheduler.yaml'
CHECK = ROOT / 'freeze150-device-cache-validation'
CHECK.mkdir(exist_ok=True)
STATE = ROOT / 'freeze150-device-cache-status.json'


def record(stage, **details):
    data = {'stage': stage, 'timestamp': datetime.now(timezone.utc).isoformat(), **details}
    STATE.write_text(json.dumps(data, indent=2) + '\n')
    print(json.dumps(data), flush=True)


def wait_for_validation_shutdown():
    from ray import serve

    record('validation_teardown_wait')
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        names = set(serve.status().applications)
        if not names:
            return
        if names != {'freeze150-cache-validation'}:
            raise RuntimeError(f'Unexpected active applications: {sorted(names)}')
        time.sleep(2)
    raise TimeoutError('Validation application did not finish shutting down')


def main():
    config = load_config(CONFIG)
    smoke = config.model_copy(deep=True)
    smoke.name = 'h100-freeze150-device-cache-validation'
    smoke.deployment.application_name = 'freeze150-cache-validation'
    auto = smoke.deployment.autoscaling
    auto.min_replicas = auto.initial_replicas = auto.max_replicas = 2
    smoke.analysis.scheduler_trace_dir = Path(
        '/mnt/cluster_storage/serve-llm-autoscaling-experiment/paired-scheduler-traces/freeze150-cache-validation'
    ) / datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    write_config(smoke, CHECK / 'resolved.yaml')
    record('two_replica_startup')
    backend = RayServeBackend(smoke)
    check_routing_support(smoke)
    check_autoscaling_support(smoke)
    backend.connect()
    resume = '--resume-after-validation' in sys.argv
    if resume:
        ready = json.loads((CHECK / 'ready_status.json').read_text())
    else:
        backend.deploy()
        ready = backend.wait_healthy()
    try:
        deployments = ready['serve_status']['applications'][smoke.deployment.application_name]['deployments']
        counts = {
            name: info['replica_states'].get('RUNNING', 0)
            for name, info in deployments.items() if name.startswith('LLMServer')
        }
        assert counts and all(count == 2 for count in counts.values()), counts
        verify_topology(smoke, ready['serve_status'])
        (CHECK / 'ready_status.json').write_text(json.dumps(ready, indent=2, default=str) + '\n')
        if not resume:
            (CHECK / 'serve_details.json').write_text(json.dumps(backend.serve_details(), indent=2, default=str) + '\n')
        record('two_replicas_healthy', running_replicas=counts)
    finally:
        if not resume:
            backend.teardown()
    wait_for_validation_shutdown()
    # Reuse the device-specific caches in the unchanged autoscaling experiment.
    record('experiment_running', config=str(CONFIG))
    run = run_experiment(config, CONFIG)
    shutil.copy2(ROOT / 'vllm-0.33-device-aware-compile-cache.patch', run / 'deployment')
    shutil.copy2(CHECK / 'ready_status.json', run / 'deployment/two_replica_validation.json')
    record('finished', run_dir=str(run))


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        record('failed', error=f'{type(error).__name__}: {error}')
        raise
