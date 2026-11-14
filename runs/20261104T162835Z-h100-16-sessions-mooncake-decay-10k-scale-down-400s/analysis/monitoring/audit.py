"""Read-only post-run audits. Outputs stay inside this run."""
import json,re,collections,hashlib
from pathlib import Path
from datetime import datetime,timezone
RUN=Path(__file__).resolve().parents[2];OUT=RUN/'analysis/monitoring';REPO=RUN.parents[1]
LOG=REPO/'findings-h100/mooncake-decay-16-400s/run.log'
def read(p):return json.loads(p.read_text())
def save(p,x):p.write_text(json.dumps(x,indent=2)+'\n')
def stamp():return datetime.now(timezone.utc).isoformat()
def utc(ns):return datetime.fromtimestamp(ns/1e9,timezone.utc).isoformat()
def metric(r,k):return r.get('metrics',{}).get(k,{}).get('value')
log=re.sub(r'\x1b\[[0-9;]*m','',LOG.read_text(errors='replace'))
# Runtime configs and topology from live dashboard snapshots, including draining replicas.
actors={};topology=[];last_cfg=None
for l in (OUT/'serve_details.jsonl').open():
 x=json.loads(l)
 for app in x['details'].get('applications',{}).values():
  for name,dep in app.get('deployments',{}).items():
   if not name.startswith('LLMServer'):continue
   last_cfg=dep.get('deployment_config',last_cfg)
   for r in dep.get('replicas',[])+dep.get('recent_dead_replicas',[]):
    if r.get('pid'):actors[str(r['pid'])]=r.get('replica_id')
    topology.append({'timestamp':x['timestamp'],'replica_id':r.get('replica_id'),'node_id':r.get('node_id'),'node_ip':r.get('node_ip'),'state':r.get('state')})
save(RUN/'deployment/node_topology_audit.json',{'captured_at':stamp(),'observations':len(topology),'node_ids':dict(collections.Counter(x['node_id'] for x in topology)),'node_ips':dict(collections.Counter(x['node_ip'] for x in topology)),'replicas':sorted(set(x['replica_id'] for x in topology if x['replica_id']))})
engines={};compat=[]
for line in log.splitlines():
 m=re.search(r'LLMServer:.*? pid=(\d+)\).*?EngineCore pid=(\d+).*Initializing a V1',line)
 if m:
  actor,core=m.groups();engines[core]=actors.get(actor)
  vals={k:(re.search(r'\b'+k+r'=([^,]+)',line).group(1) if re.search(r'\b'+k+r'=([^,]+)',line) else None) for k in ['revision','tokenizer_revision','tensor_parallel_size','pipeline_parallel_size','kv_cache_dtype','enable_prefix_caching','max_seq_len']}
  compat.append({'actor_pid':int(actor),'engine_core_pid':int(core),'replica_id':actors.get(actor),**vals})
save(RUN/'deployment/engine_compatibility_audit.json',{'captured_at':stamp(),'engine_initializations':compat})
# Preserve queue/admission/prefill per request, including failed HTTP records.
profiles=list((RUN/'benchmark').rglob('profile_export.jsonl'))
records=[json.loads(l) for p in profiles for l in p.open() if l.strip()]
records=[r for r in records if r.get('metadata',{}).get('benchmark_phase')=='profiling']
record_by_id={r['metadata']['x_request_id']:r for r in records}
traces=Path(read(RUN/'manifest.json')['scheduler_traces']['directory'])
reqs=collections.defaultdict(lambda:collections.defaultdict(lambda:{'queued':[],'admissions':[],'prefill_chunks':[],'first_token':None}))
UUID=re.compile(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}')
for p in sorted(traces.glob('*.jsonl')):
 engine=p.name.split('.')[0]
 for line in p.open():
  try:x=json.loads(line)
  except json.JSONDecodeError:continue
  event=x.get('event')
  if event in ['request_queued','request_admitted','request_first_token']:
   rid=UUID.search(x.get('external_request_id',x.get('request_id','')))
   if rid and rid.group() in record_by_id:
    d=reqs[rid.group()][engine]
    if event=='request_queued':d['queued'].append(x)
    elif event=='request_admitted':d['admissions'].append(x)
    else:d['first_token']=x
  elif event=='batch_scheduled':
   for q in x.get('requests',[]):
    if q.get('phase')!='prefill':continue
    rid=UUID.search(q.get('external_request_id',q.get('request_id','')))
    if rid and rid.group() in record_by_id:reqs[rid.group()][engine]['prefill_chunks'].append({'wall_time_s':x['wall_time_s'],'batch_index':x.get('batch_index'),**q})
engine_times={e:min(x['wall_time_s'] for d in reqs.values() for k,v in d.items() if k==e for x in v['queued']) for e in {k for d in reqs.values() for k,v in d.items() if v['queued']}}
order=sorted(engine_times,key=engine_times.get)
# Stop timestamps use controller logs in local UTC-8 timezone, confirmed against dashboard timestamps.
stopped={}
for line in log.splitlines():
 m=re.search(r'INFO (2026-11-04 \d\d:\d\d:\d\d,\d+) .*Replica\(id=\x27([^\x27]+)\x27.*is stopped',line)
 if m:
  t=datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc).timestamp()+8*3600;stopped[m.group(2)]=t
rows=[]
for r in records:
 m=r['metadata'];rid=m['x_request_id'];ds=reqs.get(rid,{})
 es=sorted([e for e,v in ds.items() if v['queued']],key=lambda e:ds[e]['queued'][0]['wall_time_s'])
 e=es[0] if es else None;d=ds.get(e,{})
 core=e.rsplit('pid-',1)[-1] if e else None
 chunks=d.get('prefill_chunks',[]);first=d.get('first_token');ad=d.get('admissions',[])
 before=[c for c in chunks if first is None or c['wall_time_s']<=first['wall_time_s']]
 rows.append({'request_id':rid,'trace_id':m.get('source_trace_id','')[-6:],'source_trace_id':m.get('source_trace_id'),'conversation_id':m.get('conversation_id'),'root_correlation_id':m.get('root_correlation_id'),'agent_depth':m.get('agent_depth',0),'turn_index':m.get('turn_index'),'start_ns':m.get('request_start_ns'),'end_ns':m.get('request_end_ns'),'engine':e,'replica_id':engines.get(core),'replica_ordinal':order.index(e)+1 if e in order else None,'all_queued_engines':es,'routing_ambiguous':len(es)>1,'input_tokens':metric(r,'input_sequence_length'),'client_ttft_ms':metric(r,'time_to_first_token'),'cache_read_tokens':metric(r,'usage_prompt_cache_read_tokens'),'profile_error':r.get('error'),'was_cancelled':m.get('was_cancelled',False),'context_overflow_skip':m.get('context_overflow_skip',False),'admissions':ad,'actual_computed_prefill_tokens_through_first_token':sum(c.get('scheduled_tokens',0) for c in before),'prefill_chunks_through_first_token':before,'first_token':first})
def changes(items):
 groups=collections.defaultdict(list)
 for r in items:groups[(r['root_correlation_id'],r['conversation_id'])].append(r)
 out=[]
 for group in groups.values():
  group.sort(key=lambda r:r['start_ns'])
  for a,b in zip(group,group[1:]):
   if a['engine'] and b['engine'] and a['engine']!=b['engine']:
    stop=stopped.get(a['replica_id'])
    out.append({'previous_request_id':a['request_id'],'previous_turn_index':a['turn_index'],'previous_profile_error':a['profile_error'],'from_engine':a['engine'],'from_replica_id':a['replica_id'],'to_engine':b['engine'],'to_replica_id':b['replica_id'],'producer_stopped_at_utc':utc(stop*1e9) if stop else None,'started_after_producer_stopped':bool(stop and b['start_ns']/1e9>stop),**b})
 return sorted(out,key=lambda r:r['start_ns'])
roots=[r for r in rows if r['agent_depth']==0];trans=changes(roots);alltrans=changes(rows)
save(OUT/'root_request_routing.json',{'captured_at':stamp(),'note':'All profiling root turns including failures. Engine ordinals follow first queued request and are process instances, not physical GPUs. Last HTTP end does not establish session completion.','root_request_count':len(roots),'root_sessions':len(set(r['root_correlation_id'] for r in roots)),'unmatched_root_requests':sum(not r['engine'] for r in roots),'root_requests_with_multiple_engine_ids':sum(r['routing_ambiguous'] for r in roots),'engine_order':order,'engine_to_replica_id':engines,'root_requests':roots,'all_root_replica_transitions':trans})
save(OUT/'all_conversation_routing.json',{'captured_at':stamp(),'profile_request_count':len(rows),'matched_request_count':sum(bool(r['engine']) for r in rows),'ambiguous_request_count':sum(r['routing_ambiguous'] for r in rows),'all_conversation_replica_transitions':alltrans,'completed_receiving_transition_count':sum(not r['profile_error'] for r in alltrans),'failed_receiving_transition_count':sum(bool(r['profile_error']) for r in alltrans)})
with (OUT/'request_errors.jsonl').open('w') as f:
 for r in rows:
  if r['profile_error']:f.write(json.dumps(r)+'\n')
# Source hashes remain the launched baseline; never modify sources.
launch=read(REPO/'findings-h100/mooncake-decay-16-400s/launch.json');hashes=[]
for path,expected in launch['router_source_sha256'].items():
 got=hashlib.sha256(Path(path).read_bytes()).hexdigest();hashes.append({'path':path,'expected':expected,'actual':got,'match':got==expected})
save(OUT/'router_source_hash_audit.json',{'captured_at':stamp(),'files':hashes,'all_match':all(x['match'] for x in hashes)})
print(json.dumps({'records':len(rows),'root_trees':len(set(r['root_correlation_id'] for r in roots)),'root_transitions':len(trans),'all_transitions':len(alltrans),'errors':sum(bool(r['profile_error']) for r in rows),'engines':len(order),'replica_mapping':engines},indent=2))
# Audit actual interpolation updates without importing 32-session-specific assumptions.
stdout_path=RUN/'benchmark/agentx-concurrency-ramp/stdout.log'
stdout=stdout_path.read_text(errors='replace')
def sec(s):
 h,m,v=s.split(':');return int(h)*3600+int(m)*60+float(v)
updates=[]
for line in stdout.splitlines():
 m=re.search(r'^(\d\d:\d\d:\d\d\.\d+) .*Concurrency series: session limit set to (\d+) \(live sessions (\d+), debt (\d+), idle sessions retired (\d+)\)',line)
 if m:
  tm,limit,live,debt,ret=m.groups();updates.append({'local_time':tm,'local_seconds':sec(tm),'limit':int(limit),'live_sessions':int(live),'debt':int(debt),'idle_sessions_retired':int(ret)})
if updates:
 origin=updates[0]['local_seconds']
 for x in updates:x['seconds_after_series_initialization']=x['local_seconds']-origin
stop_lines=[l for l in stdout.splitlines() if 'Phase profiling (profiling)' in l and 'sending complete' in l]
stop_time=re.match(r'^(\d\d:\d\d:\d\d\.\d+)',stop_lines[-1]).group(1) if stop_lines else None
save(OUT/'concurrency_limit_events.json',{'captured_at':stamp(),'schedule':'Original linear interpolation: [0,1], [450,16], [1050,16]. No down-ramp configured. Rounded interpolation reaches later integer limits at half-step crossings.','expected_limits':list(range(1,17)),'observed_limits':[x['limit'] for x in updates],'all_expected_limits_observed':[x['limit'] for x in updates]==list(range(1,17)),'observed_update_count':len(updates),'expected_crossings_s_after_series_initialization':{str(i):(i-1.5)*30 for i in range(2,17)},'max_crossing_timing_deviation_s':max((abs(x['seconds_after_series_initialization']-(x['limit']-1.5)*30) for x in updates if x['limit']>1),default=0),'events':updates,'sending_complete_local_time':stop_time,'sending_complete_line':stop_lines[-1] if stop_lines else None,'note':'Series initialization and records-manager profiling-credit clock differ. Relative local-clock elapsed times avoid assuming a timezone. No endpoint=1 update or down-ramp is applicable.'})
(OUT/'concurrency_limit_events.txt').write_text('\n'.join(l for l in stdout.splitlines() if any(s in l for s in ['Concurrency series:','Phase agentic.profiling','retiring idle','Retiring idle','Credit phase start: profiling']))+'\n')
# More roots than initial16 demonstrates slot reuse; last HTTP end is only an observed span.
trees=collections.defaultdict(list)
for r in rows:trees[r['root_correlation_id']].append(r)
first_http=min((r['start_ns'] for r in rows),default=0)
root_trees=[]
for root,rs in trees.items():
 rootrs=sorted([r for r in rs if r['agent_depth']==0],key=lambda r:r['start_ns'])
 if not rootrs:continue
 a=rootrs[0];r0=record_by_id[a['request_id']]['metadata']
 root_trees.append({'root_correlation_id':root,'source_trace_id':a['source_trace_id'],'trace_id':a['trace_id'],'first_root_request_ns':a['start_ns'],'first_credit_issued_ns':r0.get('credit_issued_ns'),'last_observed_request_end_ns':max(r['end_ns'] for r in rs),'root_request_count':len(rootrs),'descendant_request_count':len(rs)-len(rootrs),'request_errors':dict(collections.Counter(r['profile_error'].get('type','unknown') for r in rs if r['profile_error'])),'first_request_s_from_first_http':(a['start_ns']-first_http)/1e9})
root_trees.sort(key=lambda r:r['first_root_request_ns'])
source_counts=collections.Counter(r['source_trace_id'] for r in root_trees)
retirements=[l for l in stdout.splitlines() if 'retir' in l.lower() and ('session' in l.lower() or 'root' in l.lower()) and 'Concurrency series:' not in l]
save(OUT/'session_tree_audit.json',{'captured_at':stamp(),'profile_records':len(rows),'root_tree_count':len(root_trees),'initial_source_lane_count':16,'fresh_root_starts_beyond_initial':max(0,len(root_trees)-16),'minimum_slot_reuse_events':max(0,len(root_trees)-16),'source_trace_count':len(source_counts),'repeated_source_traces':{k:v for k,v in source_counts.items() if v>1},'traffic_origin_ns_first_actual_http':first_http,'phase_boundaries_s':[0,450,1050],'root_trees':root_trees,'explicit_retirement_log_lines':retirements,'context_overflow_records_by_depth':dict(collections.Counter(r['agent_depth'] for r in rows if r['context_overflow_skip'])),'request_error_types':dict(collections.Counter(r['profile_error'].get('type','unknown') for r in rows if r['profile_error'])),'was_cancelled_record_count':sum(r['was_cancelled'] for r in rows),'limitations':['Last observed HTTP end does not establish natural session completion.','Recycled lane ownership is not exported; graph rows are inferred observed request spans.','This run has an original linear ramp and hold, with no concurrency down-ramp.']})
# Preserve engine-init failures once per core PID, avoiding repeated propagated tracebacks.
failed={}
lines=log.splitlines()
for i,line in enumerate(lines):
 if 'Address already in use' not in line:continue
 core=re.search(r'EngineCore pid=(\d+)',line)
 if core:
  port=re.search(r'tcp://[^\s\x27\x22]+',line)
  pid=core.group(1);failed[pid]={'engine_core_pid':int(pid),'replica_id':engines.get(pid),'endpoint':port.group() if port else None,'line':line}
save(OUT/'engine_startup_failure_summary.json',{'captured_at':stamp(),'distinct_address_in_use_engine_failures':len(failed),'failures':list(failed.values()),'note':'Only actual EngineCore address-in-use failures are counted; duplicated parent traceback lines are excluded.'})
(OUT/'runtime_warning_evidence.txt').write_text('\n'.join(l for l in log.splitlines() if any(s in l for s in ['free_reservation','lease expired','Address already in use','hard-stop','Hard stop','graceful shutdown timeout']))+'\n')
# Exact transport failures: per-request timing plus retained proxy stderr; never call cache failures.
reloads=[]
for line in lines:
 if 'Successfully performed graceful HAProxy reload' not in line:continue
 m=re.search(r'2026-11-04 \d\d:\d\d:\d\d,\d+',line)
 if m:
  at=datetime.strptime(m.group(),'%Y-%m-%d %H:%M:%S,%f').replace(tzinfo=timezone.utc).timestamp()+8*3600;reloads.append({'at_s':at,'utc':utc(at*1e9),'line':line})
proxyfiles=[]
for p in (OUT/'haproxy').glob('*stderr.log'):
 text=p.read_text()
 if 'performing a hard-stop' in text:proxyfiles.append({'file':str(p),'hard_stop_lines':[l for l in text.splitlines() if 'hard-stop' in l or 'hard-stopped' in l]})
correlated=[]
for r in rows:
 if not r['profile_error']:continue
 near=min(reloads,key=lambda e:abs(r['end_ns']/1e9-e['at_s']-120)) if reloads else None
 delta=r['end_ns']/1e9-near['at_s'] if near else None
 correlated.append({'request_id':r['request_id'],'error_type':r['profile_error'].get('type'),'replica_id':r['replica_id'],'end_at_utc':utc(r['end_ns']),'nearest_reload_at_utc':near['utc'] if near else None,'seconds_after_reload':delta,'timing_consistent_with_reload_plus120s':delta is not None and abs(delta-120)<3})
save(OUT/'proxy_error_correlation.json',{'captured_at':stamp(),'effective_hard_stop_after_s':120,'error_count':len(correlated),'timing_consistent_count':sum(r['timing_consistent_with_reload_plus120s'] for r in correlated),'unattributed_count':sum(not r['timing_consistent_with_reload_plus120s'] for r in correlated),'reload_events':reloads,'retained_direct_hardstop_stderr':proxyfiles,'failed_requests':correlated,'limitations':['Reload completion logging differs by about1s from actual old-worker soft-stop.','Timing plus retained hard-stop stderr is strong evidence but no per-socket tracing was collected.','No exact HTTP stream-to-proxy-worker association is claimed.']})
