import json,os,re,time,shutil,collections
from pathlib import Path
from datetime import datetime,timezone
import requests
RUN=Path(__file__).resolve().parents[2]
OUT=RUN/'analysis/monitoring'; PROXY=OUT/'haproxy';PROXY.mkdir(exist_ok=True)
LOG=Path('/home/ray/default/serve-llm-autoscaling-experiment/findings-h100/mooncake-decay-16-400s/run.log')
ROOT=Path('/tmp/haproxy-serve/c7e54ddf3e8987340282a5b7bdf244d27c8912f06b3540e3efa17c32')
PID=4105947
previous=None
start=time.time()
def now():return datetime.now(timezone.utc).isoformat()
def append(path,row):
 with path.open('a') as f:f.write(json.dumps(row)+'\n')
def save(path,row):path.write_text(json.dumps(row,indent=2)+'\n')
save(OUT/'background_monitor_pid.json',{'pid':os.getpid(),'driver_pid':PID,'started_at':now()})
while True:
 stamp=now();state={'timestamp':stamp}
 try:
  detail=requests.get('http://127.0.0.1:8265/api/serve/applications/',timeout=8).json()
  append(OUT/'serve_details.jsonl',{'timestamp':stamp,'details':detail})
  app=detail.get('applications',{}).get('ttft-benchmark',{})
  dep=next((v for k,v in app.get('deployments',{}).items() if k.startswith('LLMServer')), {})
  state.update(app_status=app.get('status'),target=dep.get('target_num_replicas'),states=dict(collections.Counter(x['state'] for x in dep.get('replicas',[]))),replicas=[{'id':r['replica_id'],'state':r['state'],'pid':r.get('pid')} for r in dep.get('replicas',[])],app_absent=not bool(app))
  cfg=dep.get('deployment_config',{})
  if cfg and not (OUT/'runtime_decay_config.json').exists():save(OUT/'runtime_decay_config.json',{'observed_at':stamp,'request_router_config':cfg.get('request_router_config'),'autoscaling_config':cfg.get('autoscaling_config'),'graceful_shutdown_timeout_s':cfg.get('graceful_shutdown_timeout_s')})
 except Exception as e:state['serve_error']=repr(e)
 try:
  txt=requests.get('http://10.1.154.0:9003/metrics',timeout=5).text
  d={}
  for l in txt.splitlines():
   if l.startswith('master_') and not l.startswith('#'):
    k,v=l.rsplit(' ',1)
    if '{' not in k:d[k]=float(v)
  append(OUT/'pool_snapshots.jsonl',{'timestamp':stamp,**d});state['pool']={k:v for k,v in d.items() if any(s in k for s in ['allocated_bytes','capacity_bytes','evicted','evictions'])}
 except Exception as e:state['pool_error']=repr(e)
 for p in ROOT.glob('admin.sock.4107065.*.stderr.log'):
  try:shutil.copyfile(p,PROXY/p.name)
  except FileNotFoundError:pass
 if (ROOT/'haproxy.cfg').exists():shutil.copyfile(ROOT/'haproxy.cfg',PROXY/'haproxy.cfg')
 txt=re.sub(r'\x1b\[[0-9;]*m','',LOG.read_text(errors='replace')) if LOG.exists() else ''
 state['log_counts']={k:txt.count(k) for k in ['ClientPayloadError','ServerDisconnectedError','Address already in use','lease expired','failed keys','free_reservation']}
 state['driver_alive']=Path(f'/proc/{PID}').exists()
 profiles=list((RUN/'benchmark').rglob('profile_export.jsonl'))
 if profiles:
  try:
   rows=[json.loads(l) for l in profiles[0].read_text().splitlines() if l.strip()]
   state['profile_count']=len(rows)
   state['profile_path']=str(profiles[0])
  except Exception as e:state['profile_read_error']=repr(e)
 manifest=json.loads((RUN/'manifest.json').read_text());state['manifest_status']=manifest.get('status');state['teardown_completed']=manifest.get('teardown_completed')
 # Teardown recorded in deployment/events rather than assuming process exit implies cleanup.
 events=(RUN/'deployment/events.jsonl').read_text() if (RUN/'deployment/events.jsonl').exists() else ''
 state['teardown_event_present']='teardown_completed' in events
 save(OUT/'monitor_status.json',state)
 change={k:state.get(k) for k in ['app_status','target','states','log_counts','manifest_status','teardown_event_present']}
 if change!=previous:append(OUT/'monitor_events.jsonl',state);print(json.dumps(state),flush=True);previous=change
 if manifest.get('status') in ['succeeded','failed'] and state.get('teardown_event_present') and state.get('app_absent') and not state['driver_alive']:
  state['monitor_finished']=True;save(OUT/'monitor_status.json',state);print('MONITOR_FINISHED '+json.dumps(state),flush=True);break
 if time.time()-start>14400:
  print('Observation limit reached; not declaring run failed.',flush=True);break
 time.sleep(30)
