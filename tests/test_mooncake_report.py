"""Guard report attribution against Ray's retained dead-replica counters."""
import importlib.util
import json
from pathlib import Path


SPEC = importlib.util.spec_from_file_location('mooncake_report', Path(__file__).parents[1]/'tools/report_mooncake.py')
REPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(REPORT)


def test_report_uses_current_replicas_and_counter_increments(tmp_path):
    path=tmp_path/'telemetry';path.mkdir()
    rows=[]
    for at,value in [(0,5),(10,12),(20,3)]:
        rows.append({'timestamp_ns':1_000_000_000+at*1_000_000_000,'samples':[
            {'name':'ray_vllm_mooncake_store_operation_bytes_total',
             'labels':{'ReplicaId':'current1','WorkerId':'w','operation':'load_get','status':'ok'},'value':value},
            {'name':'ray_vllm_mooncake_store_operation_bytes_total',
             'labels':{'ReplicaId':'oldsmoke','WorkerId':'old','operation':'load_get','status':'ok'},'value':999999},
        ]})
    (path/'serve_metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    totals,buckets,replicas=REPORT.operation_data(tmp_path,{'current1'},1_000_000_000)
    assert totals['load_get_bytes_ok']==15
    assert set(replicas)=={'current1'}
    assert [buckets[t]['load_get_bytes_ok'] for t in [0,10,20]]==[5,7,3]


def test_report_accepts_prometheus_counter_suffix_and_pool_start_occupancy(tmp_path):
    path=tmp_path/'telemetry';path.mkdir()
    rows=[]
    for at,allocated,evicted in [(0,100,7),(10,400,9),(20,300,10)]:
        rows.append({'timestamp_ns':1_000_000_000+at*1_000_000_000,'component':'master','samples':[
            {'name':'master_allocated_bytes','labels':{},'value':allocated},
            {'name':'master_total_capacity_bytes','labels':{},'value':1000},
            {'name':'master_evicted_key_count_total','labels':{},'value':evicted},
        ]})
    (path/'mooncake_metrics.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in rows))
    summary,_=REPORT.pool_data(tmp_path,1_000_000_000)
    assert summary['master_allocated_bytes']=={'initial':100,'peak':400,'last':300}
    assert summary['master_evicted_key_count']==3
