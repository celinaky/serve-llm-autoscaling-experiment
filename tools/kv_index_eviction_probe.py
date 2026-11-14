#!/usr/bin/env python3
"""Does the router's KV index learn about engine evictions? A direct probe.

Runs a Dynamo SelectionService (the router's index, exactly as the Ray branch
creates it) subscribed to a live vLLM engine's KV-cache events (the standalone
engine started with --kv-events), then:
  1. sends prompt A (random tokens) so the engine caches it; asks the index
     how much of A it believes is cached (overlap) -> should be ~all of A;
  2. sends enough distinct filler prompts to exceed the KV cache, so the
     engine evicts A;
  3. asks the index again about A, then sends A again and reads how many
     tokens the engine actually found cached.
If the index still reports A as cached after the engine evicted it, the
router under-counts A's re-prefill, which is the dip seen in the runs.
Also records select_and_reserve's effective_prefill_tokens for A under a
session id, the number the router adds to its gauge.

usage: python tools/kv_index_eviction_probe.py [--port 8100] [--events tcp://127.0.0.1:5557]
           [--prompt-tokens 100000] [--fillers 13] [--block-size 1568]
"""
import argparse, asyncio, json, random, time
import requests

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100); ap.add_argument("--events", default="tcp://127.0.0.1:5557"); ap.add_argument("--replay", default="tcp://127.0.0.1:5558")
ap.add_argument("--prompt-tokens", type=int, default=100000); ap.add_argument("--fillers", type=int, default=13); ap.add_argument("--block-size", type=int, default=1568)
ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8"); ap.add_argument("--out")
a = ap.parse_args()
URL = f"http://127.0.0.1:{a.port}/v1/completions"
log = []
def note(**kw):
    kw["t"] = round(time.time(), 1); log.append(kw); print(json.dumps(kw), flush=True)

def prompt(n, seed): return random.Random(seed).choices(range(2000, 140000), k=n)

def send(ids, tag):
    t = time.time()
    r = requests.post(URL, json={"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=600)
    r.raise_for_status(); u = r.json()["usage"]
    cached = (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
    note(step="engine", tag=tag, prompt_tokens=u["prompt_tokens"], engine_cached=cached, seconds=round(time.time() - t, 1))
    return cached

async def main():
    from dynamo.llm import SelectionService
    svc = SelectionService(indexer_threads=2, session_affinity_ttl_secs=1800.0, session_affinity_mode="soft", session_replication_writer_id=None)
    await svc.upsert_worker({"worker_id": 1, "model_name": "default", "routing_group": "default", "endpoint": "ray://probe",
                             "block_size": a.block_size, "max_num_batched_tokens": 16384, "data_parallel_start_rank": 0, "data_parallel_size": 1,
                             "kv_events_endpoints": {0: a.events}, "replay_endpoint": a.replay})
    await asyncio.sleep(2)
    note(step="ready", ready=svc.ready())
    async def router_view(ids, tag, session):
        res = {}
        try:
            o = await svc.overlap_scores({"model_name": "default", "routing_group": "default", "token_ids": ids})
            res["overlap_scores"] = o
        except Exception as e:
            res["overlap_scores_error"] = str(e)[:300]
        try:
            sel = await svc.select_and_reserve({"model_name": "default", "routing_group": "default", "selection_id": tag, "token_ids": ids,
                                                "allowed_worker_ids": [1], "expected_output_tokens": 1, "lora_name": None, "session_id": session})
            res["longest_matched"] = sel["overlap"]["longest_matched"]; res["effective_prefill_tokens"] = sel["effective_prefill_tokens"]; res["isl_tokens"] = sel.get("isl_tokens")
            await svc.free_reservation(tag)
        except Exception as e:
            res["select_error"] = str(e)[:300]
        try:
            snap = svc.indexer_snapshot(); s = json.dumps(snap); res["index_snapshot_bytes"] = len(s); res["index_snapshot_head"] = s[:200]
        except Exception as e:
            res["snapshot_error"] = str(e)[:200]
        note(step="router", tag=tag, **res)
        return res
    A = prompt(a.prompt_tokens, 1)
    send(A, "A first send (cold)"); await asyncio.sleep(3)
    await router_view(A, "A-after-first", "session-A")
    send(A, "A second send (should be cached)"); await asyncio.sleep(3)
    await router_view(A, "A-after-second", "session-A")
    total = 0
    for i in range(a.fillers):
        F = prompt(a.prompt_tokens, 100 + i); send(F, f"filler {i+1}/{a.fillers}"); total += len(F)
    note(step="fillers_done", filler_tokens=total)
    await asyncio.sleep(5)
    await router_view(A, "A-after-eviction-index-view", "session-A")
    cached = send(A, "A third send (after fillers)"); await asyncio.sleep(3)
    await router_view(A, "A-after-third", "session-A")
    note(step="verdict", engine_cached_on_resend=cached)
    svc.shutdown()
    out = a.out or f"runs/kv_index_eviction_probe_{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}.jsonl"
    with open(out, "w") as f:
        for row in log: f.write(json.dumps(row) + "\n")
    print("wrote", out)
asyncio.run(main())
