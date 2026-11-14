"""CPU-only synthetic test of the installed connector/router event contract.

This does not replay the run's unsaved event stream. It compares Mooncake's
current CPU store-event shape against a self-describing variant on identical
synthetic tokens/hashes. No Ray application or model is deployed.
"""
import asyncio
import json
import time
from pathlib import Path

from dynamo.llm import SelectionService
from vllm.distributed.kv_events import (
    BlockRemoved, BlockStored, KVEventBatch, ZmqEventPublisher,
)

OUT = Path(__file__).resolve().parent


async def scenario(name, cpu_medium, cpu_kind):
    publisher = ZmqEventPublisher(0, endpoint='tcp://*:0', replay_endpoint='tcp://*:0')
    cfg = publisher.get_publisher_config()
    service = SelectionService(indexer_threads=1)
    request = dict(model_name='cpu-event-probe', routing_group='default',
        token_ids=list(range(64)), allowed_worker_ids=[1])
    try:
        await service.upsert_worker(dict(worker_id=1, model_name=request['model_name'],
            routing_group='default', endpoint='ray://synthetic-cpu-only', block_size=16,
            max_num_batched_tokens=1024, data_parallel_start_rank=0, data_parallel_size=1,
            kv_events_endpoints={0:cfg.endpoint}, replay_endpoint=cfg.replay_endpoint))
        for _ in range(50):
            if service.ready()['ready']:
                break
            await asyncio.sleep(.02)
        assert service.ready()['ready']
        await asyncio.sleep(.5)  # Let the ZMQ SUB handshake finish as well.
        def store(medium,kind):
            return BlockStored(block_hashes=[1001,1002,1003,1004], parent_block_hash=None,
                token_ids=list(range(64)), block_size=16, lora_id=None, medium=medium,
                lora_name=None, group_idx=0, kv_cache_spec_kind=kind)
        async def emit(event):
            publisher.publish(KVEventBatch(ts=time.time(),events=[event]))
            await asyncio.sleep(.3)
        async def snapshot(tag):
            score = await service.overlap_scores(request)
            selection = await service.select({**request,'selection_id':f'{name}-{tag}'})
            return dict(overlap_scores=score, effective_prefill_tokens=selection['effective_prefill_tokens'],
                overlap=selection['overlap'])
        await emit(store('GPU','full_attention'))
        gpu = await snapshot('gpu')
        assert gpu['overlap_scores']['workers'][0]['device_blocks'] == 4, gpu
        await emit(store(cpu_medium,cpu_kind))
        both = await snapshot('both')
        await emit(BlockRemoved(block_hashes=[1001,1002,1003,1004],medium='GPU',group_idx=0))
        cpu = await snapshot('after-gpu-removal')
        return dict(name=name,cpu_medium=cpu_medium,cpu_cache_spec_kind=cpu_kind,
            gpu_only=gpu, gpu_plus_cpu=both, after_gpu_removal=cpu)
    finally:
        service.shutdown()
        publisher.shutdown()


async def main():
    rows = []
    for args in [('mooncake-current','cpu',None),
                 ('uppercase-only','CPU',None),
                 ('self-describing','CPU','full_attention')]:
        row = await scenario(*args)
        rows.append(row)
        print(json.dumps(row),flush=True)
    (OUT/'cpu_event_compatibility_probe.json').write_text(json.dumps(dict(
        synthetic_cpu_only=True,historical_event_replay=False,scenarios=rows),indent=2)+'\n')


if __name__ == '__main__':
    asyncio.run(main())
