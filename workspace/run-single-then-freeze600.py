"""Run the approved single-replica baseline, then the freeze/downscale-600 run."""
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time
from datetime import datetime, timezone

ROOT = Path('/home/ray/default')
HARNESS = ROOT / 'serve-llm-autoscaling-experiment'
CONFIGS = [
    HARNESS / 'experiments/h100_token_single_replica_injected_scheduler.yaml',
    HARNESS / 'experiments/h100_token_freeze_150s_injected_scheduler.yaml',
]


def write_json(path, data):
    temp = path.with_suffix('.tmp')
    temp.write_text(json.dumps(data, indent=2, default=str) + '\n')
    temp.replace(path)


def run_one(config_path, result_path):
    from serve_profiler.compat import setup
    setup()
    from serve_llm_autoscaling.config import load_config
    from serve_llm_autoscaling.runner import run_experiment

    config = load_config(config_path)
    run = run_experiment(config, config_path).resolve()
    manifest = json.loads((run / 'manifest.json').read_text())
    assert manifest['status'] == 'succeeded', manifest
    assert not manifest.get('analysis', {}).get('plot_error'), manifest.get('analysis')
    for name in ['arrival_stats_stack180_0_1050.png', 'policy_vs_engine_mean60_0_1050.png']:
        assert (run / 'analysis' / name).is_file(), name
    if config.deployment.autoscaling.max_replicas == 1:
        plot_data = json.loads((run / 'analysis/plot_data.json').read_text())
        samples = [s for s in plot_data['serve_status_samples']
                   if not s.get('error') and 0 <= s['relative_time_s'] <= 1050]
        assert samples and all(s['running_replicas'] == 1 for s in samples)
    write_json(result_path, {'run_dir': str(run), 'status': 'succeeded'})


def wait_until_idle():
    # Use a fresh Ray/Serve client after the previous driver shuts down.
    code = '''
import json, os, ray
os.environ.pop('RAY_RUNTIME_ENV_HOOK', None)
ray.init(address='auto', ignore_reinit_error=True, log_to_driver=False)
from ray import serve
print(json.dumps({'applications': list(serve.status().applications),
                  'available_gpus': ray.available_resources().get('GPU', 0)}))
ray.shutdown()
'''
    deadline = time.monotonic() + 300
    while time.monotonic() < deadline:
        out = subprocess.run([sys.executable, '-c', code], capture_output=True,
                             text=True, check=True, timeout=45)
        state = json.loads(out.stdout.strip().splitlines()[-1])
        if not state['applications'] and state['available_gpus'] >= 8:
            return
        if set(state['applications']) - {'ttft-benchmark'}:
            raise RuntimeError(f'Unexpected active application: {state}')
        time.sleep(5)
    raise TimeoutError('Previous deployment did not fully release the cluster')


def main():
    import yaml

    lock = (ROOT / 'single-then-freeze600.lock').open('a')
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    batch = ROOT / f'single-then-freeze600-{stamp}'
    batch.mkdir()
    state_path = ROOT / 'single-then-freeze600-status.json'
    state = {'batch_dir': str(batch), 'pid': os.getpid(), 'stage': 'preflight',
             'started_at': datetime.now(timezone.utc).isoformat(), 'completed': []}

    def record(**updates):
        state.update(updates)
        state['updated_at'] = datetime.now(timezone.utc).isoformat()
        write_json(state_path, state)

    try:
        snapshots = []
        for config in CONFIGS:
            snapshot = batch / config.name
            shutil.copy2(config, snapshot)
            snapshots.append(snapshot)
        first, second = [yaml.safe_load(p.read_text()) for p in snapshots]
        assert first['benchmark'] == second['benchmark']
        assert first['deployment']['autoscaling']['min_replicas'] == 1
        assert first['deployment']['autoscaling']['max_replicas'] == 1
        assert second['deployment']['autoscaling']['downscale_delay_s'] == 600
        assert second['deployment']['autoscaling']['upscale_delay_s'] == 40
        assert second['deployment']['autoscaling']['policy']['kwargs']['post_downscale_cooldown_s'] == 150
        record(configs=[{'source': str(p), 'snapshot': str(s),
                         'sha256': hashlib.sha256(s.read_bytes()).hexdigest()}
                        for p, s in zip(CONFIGS, snapshots)])
        for index, snapshot in enumerate(snapshots, 1):
            config = yaml.safe_load(snapshot.read_text())
            record(stage='waiting_for_idle', active_index=index,
                   experiment=config['name'], active_run_dir=None)
            wait_until_idle()
            log = batch / f'{index}-{config["name"]}.log'
            result = batch / f'{index}-result.json'
            existing = set((HARNESS / 'runs').glob(f'*-{config["name"]}'))
            with log.open('w') as output:
                child = subprocess.Popen([sys.executable, __file__, '--one',
                                          str(snapshot), str(result)],
                                         cwd=HARNESS, stdout=output,
                                         stderr=subprocess.STDOUT)
                record(stage='running', child_pid=child.pid, log=str(log))
                offset = 0
                while child.poll() is None:
                    candidates = set((HARNESS / 'runs').glob(f'*-{config["name"]}')) - existing
                    if len(candidates) == 1:
                        run = next(iter(candidates))
                        if state.get('active_run_dir') != str(run):
                            record(active_run_dir=str(run))
                            shutil.copy2(ROOT / 'vllm-0.33-device-aware-compile-cache.patch',
                                         run / 'deployment/vllm-device-cache.patch')
                            shutil.copy2(ROOT / 'freeze150-device-cache-validation/ready_status.json',
                                         run / 'deployment/two_replica_validation.json')
                    with log.open() as reader:
                        reader.seek(offset)
                        chunk = reader.read()
                        offset = reader.tell()
                    if 'CUDA driver error: invalid argument' in chunk:
                        record(stage='stopping_after_cuda_error')
                        child.send_signal(signal.SIGINT)
                        child.wait(timeout=180)
                        raise RuntimeError(f'CUDA error during run {index}; see {log}')
                    time.sleep(5)
                if child.returncode:
                    raise RuntimeError(f'Run {index} exited {child.returncode}; see {log}')
            outcome = json.loads(result.read_text())
            state['completed'].append({'index': index, **outcome, 'log': str(log)})
            record(stage='run_completed')
        record(stage='finished', finished_at=datetime.now(timezone.utc).isoformat())
    except Exception as error:
        record(stage='failed', error=f'{type(error).__name__}: {error}')
        raise


if __name__ == '__main__':
    if len(sys.argv) > 1 and sys.argv[1] == '--one':
        run_one(Path(sys.argv[2]), Path(sys.argv[3]))
    else:
        main()
