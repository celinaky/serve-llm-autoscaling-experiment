#!/usr/bin/env python3
"""How fast does the router's KV index reflect an eviction under concurrent load?

Like kv_index_eviction_probe.py, but the filler prompts are sent --concurrency at
a time (so the engine chunk-prefills several at once and emits dense event
batches), and the index is polled every second for how many of prompt A's
blocks it still believes are cached. Logs the moment the engine's view (A
re-sent: cached_tokens) and the index's view diverge or agree.

usage: python tools/kv_index_lag_probe.py [--concurrency 8] [--fillers 16] [--prompt-tokens 100000]
"""
import argparse, asyncio, json, random, threading, time
from concurrent.futures import ThreadPoolExecutor
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100); ap.add_argument("--events", default="tcp://127.0.0.1:5557"); ap.add_argument("--replay", default="tcp://127.0.0.1:5558")
ap.add_argument("--prompt-tokens", type=int, default=100000); ap.add_argument("--fillers", type=int, default=16); ap.add_argument("--concurrency", type=int, default=8)
ap.add_argument("--block-size", type=int, default=1568); ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8"); ap.add_argument("--out")
a = ap.parse_args(); URL = f"http://127.0.0.1:{a.port}/v1/completions"; T0 = time.time(); log = []
def note(**kw):
    kw["t"] = round(time.time() - T0, 1); log.append(kw); print(json.dumps(kw), flush=True)
def prompt(n, seed): return random.Random(seed).choices(range(2000, 140000), k=n)
def send(ids, tag):
    t = time.time(); r = requests.post(URL, json={"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=900); r.raise_for_status()
    u = r.json()["usage"]; cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    note(step="engine", tag=tag, engine_cached=cached, seconds=round(time.time() - t, 1)); return cached
async def main():
    from dynamo.llm import SelectionService
    svc = SelectionService(indexer_threads=4, session_affinity_ttl_secs=1800.0, session_affinity_mode="soft", session_replication_writer_id=None)
    await svc.upsert_worker({"worker_id": 1, "model_name": "default", "routing_group": "default", "endpoint": "ray://probe", "block_size": a.block_size,
                             "max_num_batched_tokens": 16384, "data_parallel_start_rank": 0, "data_parallel_size": 1, "kv_events_endpoints": {0: a.events}, "replay_endpoint": a.replay})
    await asyncio.sleep(2)
    async def index_blocks(ids):
        o = await svc.overlap_scores({"model_name": "default", "routing_group": "default", "token_ids": ids}); return o["workers"][0]["device_blocks"], o["num_blocks"]
    A = prompt(a.prompt_tokens, 1)
    await asyncio.get_running_loop().run_in_executor(None, send, A, "A first send")
    await asyncio.sleep(2); note(step="index", tag="A after first send", blocks=await index_blocks(A))
    pool = ThreadPoolExecutor(max_workers=a.concurrency); loop = asyncio.get_running_loop()
    futs = [loop.run_in_executor(pool, send, prompt(a.prompt_tokens, 100 + i), f"filler {i+1}") for i in range(a.fillers)]
    note(step="flood_start", fillers=a.fillers, concurrency=a.concurrency, tokens=a.fillers * a.prompt_tokens)
    done = asyncio.gather(*futs); first_zero = None
    while not done.done():
        b, n = await index_blocks(A)
        if b == 0 and first_zero is None: first_zero = round(time.time() - T0, 1); note(step="index", tag="A gone from index", blocks=b, of=n)
        await asyncio.sleep(1)
    await done; note(step="flood_done")
    for k in range(10):
        b, n = await index_blocks(A); note(step="index", tag=f"A after flood, poll {k}", blocks=b, of=n)
        if b == 0 and first_zero is None: first_zero = round(time.time() - T0, 1)
        if b == 0: break
        await asyncio.sleep(2)
    cached = await loop.run_in_executor(None, send, A, "A re-send after flood")
    b, n = await index_blocks(A); note(step="verdict", engine_cached_on_resend=cached, index_blocks_before_resend=b, index_zero_at=first_zero)
    svc.shutdown(); await asyncio.sleep(0.5)
    out = a.out or f"runs/kv_index_lag_probe_c{a.concurrency}_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    with open(out, "w") as f:
        for row in log: f.write(json.dumps(row) + "\n")
    print("wrote", out)
asyncio.run(main())
