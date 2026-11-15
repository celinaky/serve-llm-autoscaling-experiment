"""Durable supervisor for the local cohort-8 30-minute experiment."""
import datetime as dt
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
STAMP = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
LOG = ROOT / 'runs' / f'cohort8-30min-{STAMP}.log'
STATUS = ROOT / 'runs' / 'cohort8-30min-status.json'

def write(data):
    temporary=STATUS.with_suffix('.tmp')
    temporary.write_text(json.dumps(data,indent=2)+'\n'); temporary.replace(STATUS)

def snapshot(run):
    target=run/'source-provenance';target.mkdir(exist_ok=True)
    shutil.copytree(ROOT/'serve_profiler',target/'serve_profiler',ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(__file__,target/'run_cohort8_30min.py')
    fork=ROOT.parent/'aiperf'
    patch=subprocess.check_output(['git','diff','HEAD'],cwd=fork)
    (target/'aiperf-working-tree.patch').write_bytes(patch)
    shutil.copy2(fork/'src/aiperf/timing/cohort.py',target/'aiperf-cohort.py')
    manifest={'workload':{'session_trees':8,'cohort_sizes':[8],'phase_seconds':1800,'seed':42,'grace_seconds':300},'aiperf_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=fork,text=True).strip(),'aiperf_patch_sha256':hashlib.sha256(patch).hexdigest(),'aiperf_cohort_sha256':hashlib.sha256((target/'aiperf-cohort.py').read_bytes()).hexdigest(),'python':sys.version,'note':'Local experiment only. nfer PR and architecture site remain unchanged; CI design revisions deferred.'}
    (target/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')

def main():
    env=dict(os.environ);env.pop('RAY_RUNTIME_ENV_HOOK',None)
    env['PYTHONUNBUFFERED']='1';env['MPLCONFIGDIR']=str(ROOT/'runs'/'.matplotlib')
    command=[sys.executable,'-u','-m','serve_profiler','run',str(ROOT/'run-configs/h100-cohort8-30min.yaml')]
    with LOG.open('w') as log:
        process=subprocess.Popen(command,cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
    status={'status':'running','launched_utc':STAMP,'supervisor_pid':os.getpid(),'driver_pid':process.pid,'log':str(LOG),'command':command,'stage':'starting'};write(status)
    run=None
    while True:
        if run is None:
            with LOG.open() as log:
                for line in log:
                    if line.startswith('Run artifacts: '):
                        run=Path(line.partition(': ')[2].strip());snapshot(run);status['run_dir']=str(run);break
        if run is not None and (run/'run.json').exists():
            try: status['run']=json.loads((run/'run.json').read_text());status['stage']=status['run']['stage']
            except (OSError,ValueError):pass
            if (run/'workload.json').exists():
                try:status['workload']=json.loads((run/'workload.json').read_text())
                except (OSError,ValueError):pass
        code=process.poll()
        status['updated_utc']=dt.datetime.now(dt.timezone.utc).isoformat()
        if code is not None:
            status.update(status='completed' if code==0 else 'failed',returncode=code);write(status);return code
        write(status);time.sleep(15)

if __name__=='__main__':raise SystemExit(main())
