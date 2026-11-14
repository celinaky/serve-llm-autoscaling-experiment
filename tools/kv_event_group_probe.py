#!/usr/bin/env python3
"""Which KV cache groups' blocks does the engine evict, and does the router's index follow?

Subscribes to the standalone engine's vLLM KV-event stream (ZMQ PUB) and decodes
every BlockStored / BlockRemoved by KV cache group, alongside a Dynamo
SelectionService (the router's index) on the same stream. Sends a fresh prompt
A alone (so every stored block in that window is A's), then distinct fillers,
and after each filler reports, for A's block hashes: how many are still stored
per group (stored minus removed, from the events), how many blocks the
router's index says it holds, and finally what the engine reports cached when
A is re-sent once at the end.

usage: python tools/kv_event_group_probe.py [--fillers 12] [--filler-tokens 100000] [--prompt-tokens 100000]
"""
import argparse, asyncio, json, random, threading, time
from collections import Counter, defaultdict
import msgspec, requests, zmq
from vllm.distributed.kv_events import KVEventBatch, BlockStored, BlockRemoved, AllBlocksCleared

ap = argparse.ArgumentParser()
ap.add_argument("--port", type=int, default=8100); ap.add_argument("--events", default="tcp://127.0.0.1:5557"); ap.add_argument("--replay", default="tcp://127.0.0.1:5558")
ap.add_argument("--prompt-tokens", type=int, default=100000); ap.add_argument("--filler-tokens", type=int, default=100000); ap.add_argument("--fillers", type=int, default=12)
ap.add_argument("--block-size", type=int, default=1568); ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
a = ap.parse_args(); URL = f"http://127.0.0.1:{a.port}/v1/completions"; SEED = int(time.time())
lock = threading.Lock(); stored = defaultdict(set); removed = defaultdict(set); kinds = {}; capture = {"on": False, "A": defaultdict(set)}; counts = Counter()
def listen():
    ctx = zmq.Context.instance(); s = ctx.socket(zmq.SUB); s.connect(a.events); s.setsockopt(zmq.SUBSCRIBE, b"")
    dec = msgspec.msgpack.Decoder(type=KVEventBatch)
    while True:
        parts = s.recv_multipart(); batch = dec.decode(parts[-1])
        with lock:
            for ev in batch.events:
                if isinstance(ev, BlockStored):
                    g = ev.group_idx; kinds[g] = (ev.kv_cache_spec_kind, ev.kv_cache_spec_sliding_window); counts[("stored", g)] += len(ev.block_hashes)
                    for h in ev.block_hashes:
                        stored[g].add(h)
                        if capture["on"]: capture["A"][g].add(h)
                elif isinstance(ev, BlockRemoved):
                    g = ev.group_idx; counts[("removed", g)] += len(ev.block_hashes)
                    for h in ev.block_hashes: removed[g].add(h); stored[g].discard(h)
                elif isinstance(ev, AllBlocksCleared):
                    counts[("cleared", None)] += 1
threading.Thread(target=listen, daemon=True).start()
def prompt(n, seed): return random.Random(seed).choices(range(2000, 140000), k=n)
def send(ids):
    r = requests.post(URL, json={"model": a.model, "prompt": ids, "max_tokens": 1, "temperature": 0}, timeout=900); r.raise_for_status()
    u = r.json()["usage"]; return (u.get("prompt_tokens_details") or {}).get("cached_tokens", 0)
async def main():
    from dynamo.llm import SelectionService
    svc = SelectionService(indexer_threads=4, session_affinity_ttl_secs=1800.0, session_affinity_mode="soft", session_replication_writer_id=None)
    await svc.upsert_worker({"worker_id": 1, "model_name": "default", "routing_group": "default", "endpoint": "ray://probe", "block_size": a.block_size,
                             "max_num_batched_tokens": 16384, "data_parallel_start_rank": 0, "data_parallel_size": 1, "kv_events_endpoints": {0: a.events}, "replay_endpoint": a.replay})
    await asyncio.sleep(3)
    async def idx(ids):
        o = await svc.overlap_scores({"model_name": "default", "routing_group": "default", "token_ids": ids}); return o["workers"][0]["device_blocks"]
    A = prompt(a.prompt_tokens, SEED)
    with lock: capture["on"] = True
    c0 = send(A); await asyncio.sleep(2)
    with lock: capture["on"] = False; Ag = {g: set(h) for g, h in capture["A"].items()}
    groups = sorted(Ag)
    print("KV cache groups (spec kind, sliding window):", {g: kinds.get(g) for g in groups})
    print("A's blocks stored per group:", {g: len(Ag[g]) for g in groups}, "| engine cached on first send:", c0)
    hdr = "  ".join(f"g{g} left" for g in groups)
    print(f"{'after':14s} {hdr}   index blocks of A")
    def left():
        with lock: return [len(Ag[g] & stored[g]) for g in groups]
    print(f"{'A only':14s} " + "  ".join(f"{v:7d}" for v in left()) + f"   {await idx(A):8d}", flush=True)
    for i in range(a.fillers):
        send(prompt(a.filler_tokens, SEED + 100 + i)); await asyncio.sleep(1.5)
        print(f"{'filler %d' % (i + 1):14s} " + "  ".join(f"{v:7d}" for v in left()) + f"   {await idx(A):8d}", flush=True)
    b = await idx(A); c = send(A)
    print(f"\nfinal: index says {b} of A's blocks cached -> router would count {max(0, a.prompt_tokens - b * a.block_size)} uncached;"
          f" engine found {c} cached -> engine prefilled {a.prompt_tokens - c}")
    with lock: print("event totals:", dict(counts))
    svc.shutdown(); await asyncio.sleep(0.5)
asyncio.run(main())
