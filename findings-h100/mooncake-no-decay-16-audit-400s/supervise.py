"""Run the requested unchanged 16-session curve with decay=0 and audited KV reuse."""
from pathlib import Path
from datetime import datetime, timezone
from collections import Counter
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time

import requests

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
CONFIG = REPO/'experiments/h100_workday_curve_mooncake_no_decay_audit_400s.yaml'
NAME = 'h100-16-sessions-mooncake-no-decay-audit-scale-down-400s'
SERVICE_ROOT = Path('/home/ray/default/mooncake-kv')


def now():
    return datetime.now(timezone.utc).isoformat()


def save(name, value):
    path = OUT/name
    tmp = path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,indent=2)+'\n')
    tmp.replace(path)


def pool_metrics():
    response = requests.get('http://10.1.154.0:9003/metrics',timeout=5)
    response.raise_for_status()
    return {line.split()[0]:float(line.split()[1]) for line in response.text.splitlines()
            if line.startswith('master_') and '{' not in line}


def rows(paths):
    for path in paths:
        for line in path.open():
            try:
                yield json.loads(line)
            except ValueError:
                pass


def main():
    stage = 'preflight'
    driver = None
    launched = time.time()
    save('supervisor.json',dict(pid=os.getpid(),started_at=now(),status='running'))
    try:
        response = requests.get('http://127.0.0.1:8265/api/serve/applications/',timeout=5)
        response.raise_for_status()
        assert not response.json().get('applications'), 'Refusing to replace another Serve application'
        gpu = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip()
        assert not gpu, f'GPU compute workloads are present: {gpu}'
        spec = importlib.util.spec_from_file_location('owned_mooncake_services',SERVICE_ROOT/'services.py')
        services = importlib.util.module_from_spec(spec);spec.loader.exec_module(services)
        if not (OUT/'previous_services.json').exists():
            save('previous_services.json',services.status())
        if not (OUT/'previous_pool_metrics.json').exists():
            save('previous_pool_metrics.json',pool_metrics())
        stage = 'cold_pool_reset'
        save('status.json',dict(stage=stage,updated_at=now()))
        services.stop()
        # Native RPC listeners can briefly outlive the owning process's exit.
        time.sleep(2)
        for name in ('mooncake_master.log','mooncake_client.log'):
            path = SERVICE_ROOT/'state'/name
            if path.exists():shutil.move(path,OUT/f'previous_{name}')
        deadline = time.monotonic()+30
        while True:
            try:
                running_services = services.start(256,'120s')
                break
            except RuntimeError as exc:
                if 'already occupied' not in str(exc) or time.monotonic() >= deadline:
                    raise
                time.sleep(1)
        save('services.json',running_services)
        metrics = pool_metrics()
        assert metrics.get('master_allocated_bytes') == 0, metrics
        save('cold_pool_metrics.json',metrics)
        stage = 'benchmark'
        command = [sys.executable,'-u','-m','serve_llm_autoscaling.cli','run',str(CONFIG)]
        with (OUT/'run.log').open('w') as log:
            driver = subprocess.Popen(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
        launch = dict(started_at=now(),supervisor_pid=os.getpid(),driver_pid=driver.pid,
            command=command,experiment=str(CONFIG),driver_log=str(OUT/'run.log'),
            decay_tokens_per_second=0,baseline_run='20261104T162835Z-h100-16-sessions-mooncake-decay-10k-scale-down-400s',
            status='running',cold_pool_gib=256)
        save('launch.json',launch)
        root = None
        verified = False
        while driver.poll() is None:
            candidates = [p for p in (REPO/'runs').glob(f'*-{NAME}') if p.stat().st_mtime >= launched-2]
            if candidates:
                root = max(candidates,key=lambda p:p.stat().st_mtime)
                launch['run_dir'] = str(root)
                save('launch.json',launch)
                state = dict(stage=stage,updated_at=now(),run_dir=str(root),driver_pid=driver.pid)
                manifest = json.loads((root/'manifest.json').read_text())
                state['manifest_status'] = manifest.get('status')
                resolved = root/'resolved.yaml'
                if resolved.exists():
                    import yaml
                    config = yaml.safe_load(resolved.read_text())
                    trace = Path(config['analysis']['scheduler_trace_dir'])
                    if not verified:
                        selections = [r for r in rows((trace/'router-request-audit').glob('*.jsonl')) if r.get('event')=='router_selection']
                        admissions = []
                        for path in trace.glob('*.jsonl'):
                            # Initial admissions are near the start; avoid rescanning a growing run.
                            with path.open() as f:
                                for i,line in enumerate(f):
                                    if i > 3000:break
                                    try:r=json.loads(line)
                                    except ValueError:continue
                                    if r.get('event')=='request_admitted':admissions.append(r)
                        gpu_file = root/'telemetry/gpu_kv_events.jsonl'
                        gpu_rows = list(rows([gpu_file])) if gpu_file.exists() else []
                        matched = bool(selections and any(
                            r.get('local_cached_prefix_tokens') is not None
                            and r.get('external_cached_prefix_tokens') is not None
                            and r.get('initial_gpu_prefill_tokens') is not None for r in admissions))
                        gpu_sources = [r for r in gpu_rows if r.get('record_type')=='source_added']
                        if matched and gpu_sources:
                            verified = True
                            audit = dict(verified_at=now(),run_dir=str(root),trace_dir=str(trace),
                                first_router_selection=selections[0],first_engine_admission=admissions[0],
                                gpu_event_sources=gpu_sources,
                                first_gpu_event_batch=next((r for r in gpu_rows if r.get('record_type')=='kv_event_batch'),None))
                            save('live_capture_verified.json',audit)
                            shutil.copyfile(OUT/'preparation.json',root/'deployment/comparison_settings.json')
                    state['request_and_gpu_event_capture_verified'] = verified
                save('status.json',state)
            time.sleep(5)
        launch.update(driver_exit_code=driver.returncode,driver_finished_at=now(),run_dir=str(root) if root else None)
        save('launch.json',launch)
        if root is None:
            raise RuntimeError('Driver exited before creating a run directory')
        if not verified:
            raise RuntimeError('Run ended without verifying the requested live instrumentation')
        stage = 'cache_credit_analysis'
        save('status.json',dict(stage=stage,updated_at=now(),run_dir=str(root)))
        command = [sys.executable,'-m','serve_llm_autoscaling.diagnostics.router_cache_credit',str(root)]
        text = re.sub(r'\x1b\[[0-9;]*m','',(OUT/'run.log').read_text())
        weights = re.findall(r'host_cache_hit_weight=([\d.]+)',text)
        if weights and len(set(weights)) == 1:
            command += ['--host-credit',weights[0]]
        with (OUT/'cache_credit_analysis.log').open('w') as log:
            result = subprocess.run(command,cwd=REPO,stdout=log,stderr=subprocess.STDOUT,timeout=300)
        assert result.returncode == 0, 'Cache-credit analysis failed'
        report = json.loads((root/'analysis/router_cache_credit/audit.json').read_text())
        counts = Counter()
        capture_errors = []
        for r in rows([root/'telemetry/gpu_kv_events.jsonl']):
            if r.get('error'):capture_errors.append(r)
            for event in r.get('events',[]):
                counts[(r.get('replica_id'),event.get('type'),event.get('medium'),event.get('group_idx'))] += event.get('block_count',1)
        capture = dict(counts=[dict(replica_id=k[0],event=k[1],medium=k[2],group_idx=k[3],count=v)
            for k,v in counts.items()],errors=capture_errors)
        (root/'analysis/gpu_kv_event_audit.json').write_text(json.dumps(capture,indent=2)+'\n')
        completion = dict(completed_at=now(),run_dir=str(root),driver_exit_code=driver.returncode,
            metrics_capture_verified=verified,cache_credit_status=report['status'],
            cache_credit_valid_rows=report['valid_rows'],cache_credit_snapshot_sum_errors=len(report['snapshot_sum_errors']),
            dropped_request_trace_events=report['dropped_trace_events'],gpu_event_capture_errors=len(capture_errors),
            cache_credit_audit=str(root/'analysis/router_cache_credit/audit.json'),
            plots=str(root/'analysis/index.html'),status='complete' if driver.returncode==0 and report['status']=='available' else 'failed')
        save('completion.json',completion)
        launch.update(status=completion['status'],completed_at=now());save('launch.json',launch)
        save('status.json',dict(stage='finished',**completion))
    except Exception as exc:
        save('failure.json',dict(at=now(),stage=stage,error=f'{type(exc).__name__}: {exc}',
            driver_pid=driver.pid if driver else None,driver_still_running=driver.poll() is None if driver else False))
        raise


if __name__ == '__main__':
    main()
