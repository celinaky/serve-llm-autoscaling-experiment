#!/usr/bin/env python3
"""Step-by-step eviction: after each big filler, compare the index's view of A with the engine's.

Fresh prompts every run (seeded from the clock). After A is cached, send
fillers of --filler-tokens one at a time; after each, read the index's cached
block count for A, then re-send A and read the engine's cached_tokens (which
also refreshes A in the engine's LRU, so A is evicted only once the fillers
since its last touch exceed the cache). Prints one row per step.

usage: python tools/kv_index_partial_eviction_probe.py [--filler-tokens 250000] [--fillers 7] [--prompt-tokens 100000]
"""
import argparse, asyncio, json, random, time
import requests
ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100); ap.add_argument("--events", default="tcp://127.0.0.1:5557"); ap.add_argument("--replay", default="tcp://127.0.0.1:5558")
ap.add_argument("--prompt-tokens", type=int, default=100000); ap.add_argument("--filler-tokens", type=int, default=250000); ap.add_argument("--fillers", type=int, default=7)
ap.add_argument("--block-size", type=int, default=1568); ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
a = ap.parse_args(); URL = f"http://127.0.0.1:{a.port}/v1/completions"; SEED = int(time.time()); T0 = time.time()
def prompt(n, seed): return random.Random(seed).choices(range(2000, 140000), k=n)
def send(ids):
    r = requests.post(URL, json={"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=900); r.raise_for_status()
    u = r.json()["usage"]; return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
async def main():
    from dynamo.llm import SelectionService
    svc = SelectionService(indexer_threads=4, session_affinity_ttl_secs=1800.0, session_affinity_mode="soft", session_replication_writer_id=None)
    await svc.upsert_worker({"worker_id": 1, "model_name": "default", "routing_group": "default", "endpoint": "ray://probe", "block_size": a.block_size,
                             "max_num_batched_tokens": 16384, "data_parallel_start_rank": 0, "data_parallel_size": 1, "kv_events_endpoints": {0: a.events}, "replay_endpoint": a.replay})
    await asyncio.sleep(2)
    async def idx(ids):
        o = await svc.overlap_scores({"model_name": "default", "routing_group": "default", "token_ids": ids}); return o["workers"][0]["device_blocks"]
    async def eff(ids, tag):
        s = await svc.select_and_reserve({"model_name": "default", "routing_group": "default", "selection_id": tag, "token_ids": ids, "allowed_worker_ids": [1], "expected_output_tokens": 1, "lora_name": None, "session_id": "sA"})
        await svc.free_reservation(tag); return s["effective_prefill_tokens"]
    A = prompt(a.prompt_tokens, SEED)
    c = send(A); await asyncio.sleep(2)
    print(f"{'step':28s} {'index blocks of A':>18s} {'router effective':>17s} {'engine cached':>14s}")
    print(f"{'A first send':28s} {await idx(A):18d} {await eff(A, 'p0'):17d} {c:14d}", flush=True)
    for i in range(a.fillers):
        send(prompt(a.filler_tokens, SEED + 10 + i)); await asyncio.sleep(2)
        b = await idx(A); e = await eff(A, f"p{i+1}"); c = send(A); await asyncio.sleep(1)
        print(f"{'after filler %d (%dK)' % (i + 1, a.filler_tokens // 1000):28s} {b:18d} {e:17d} {c:14d}   {'<-- index says cached, engine says evicted' if b > 0 and c == 0 else ''}", flush=True)
    svc.shutdown(); await asyncio.sleep(0.5)
asyncio.run(main())
