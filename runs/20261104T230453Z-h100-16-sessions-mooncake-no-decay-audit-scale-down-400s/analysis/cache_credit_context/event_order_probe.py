"""CPU-only check: does lowercase cpu reinsert blocks into the GPU index?"""
import asyncio
import json
import time
from pathlib import Path

from dynamo.llm import SelectionService
from vllm.distributed.kv_events import BlockStored, BlockRemoved, KVEventBatch, ZmqEventPublisher

OUT = Path(__file__).resolve().parent


async def scenario(medium):
    publisher = ZmqEventPublisher(0,endpoint='tcp://*:0',replay_endpoint='tcp://*:0')
    cfg = publisher.get_publisher_config()
    service = SelectionService(indexer_threads=1)
    query = dict(model_name='cpu-event-order',routing_group='default',token_ids=list(range(64)),allowed_worker_ids=[1])
    def store(tier):
        return BlockStored(block_hashes=[1001,1002,1003,1004],parent_block_hash=None,
            token_ids=list(range(64)),block_size=16,lora_id=None,medium=tier,
            lora_name=None,group_idx=0,kv_cache_spec_kind='full_attention' if tier=='GPU' else None)
    async def emit(event):
        publisher.publish(KVEventBatch(ts=time.time(),events=[event]))
        await asyncio.sleep(.3)
    async def score(stage):
        scores = await service.overlap_scores(query)
        selection = await service.select({**query,'selection_id':f'{medium}-{stage}'})
        return dict(stage=stage,scores=scores,effective_prefill_tokens=selection['effective_prefill_tokens'],overlap=selection['overlap'])
    try:
        await service.upsert_worker(dict(worker_id=1,model_name=query['model_name'],routing_group='default',
            endpoint='ray://synthetic-cpu-only',block_size=16,max_num_batched_tokens=1024,
            data_parallel_start_rank=0,data_parallel_size=1,kv_events_endpoints={0:cfg.endpoint},
            replay_endpoint=cfg.replay_endpoint))
        await asyncio.sleep(.7)
        await emit(store('GPU'))
        initial = await score('gpu-stored')
        assert initial['scores']['workers'][0]['device_blocks']==4
        await emit(BlockRemoved(block_hashes=[1001,1002,1003,1004],medium='GPU',group_idx=0))
        removed = await score('gpu-removed')
        assert removed['scores']['workers'][0]['device_blocks']==0
        await emit(store(medium))
        after = await score('cpu-store-after-gpu-removal')
        await emit(BlockRemoved(block_hashes=[1001,1002,1003,1004],medium='GPU',group_idx=0))
        after_second_removal = await score('another-gpu-removal')
        return dict(medium=medium,initial=initial,removed=removed,
            cpu_after_gpu_removal=after,after_second_gpu_removal=after_second_removal)
    finally:
        service.shutdown();publisher.shutdown()


async def main():
    rows = [await scenario(tier) for tier in ['cpu','CPU']]
    (OUT/'event_order_probe.json').write_text(json.dumps(dict(synthetic_cpu_only=True,scenarios=rows),indent=2)+'\n')
    for r in rows:print(json.dumps(r),flush=True)


if __name__=='__main__':asyncio.run(main())
