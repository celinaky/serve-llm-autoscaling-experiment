"""Check KV-aware session affinity through Ray Serve.

Sends S sessions of T sequential chat turns. Each turn re-sends the whole
history plus a new user message (append-only, like an agent loop), so turn
k+1 should reuse turn k's KV on whichever replica served it. Each request
carries an X-Request-ID; the Serve LLMServer access logs then show which
replica served it. Prints, per session, the replica sequence and TTFTs.

usage: affinity_probe.py --sessions 4 --turns 6 --ctx-words 40000 \
         --logs "/tmp/ray/session_latest/logs/serve/replica_*LLMServer*.log"
"""
import argparse
import asyncio
import glob
import json
import random
import re
import time
import uuid

import aiohttp

WORDS = ("alpha beta gamma delta epsilon zeta theta iota kappa lambda mu nu xi "
         "omicron pi rho sigma tau upsilon phi chi psi omega river stone cloud "
         "branch signal vector buffer kernel thread socket module").split()


def text(n, rng):
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def turn(session, base, model, messages, rid):
    body = {"model": model, "messages": messages, "max_tokens": 8, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0}
    t0 = time.perf_counter()
    ttft, usage = None, None
    async with session.post(f"{base}/v1/chat/completions", json=body,
                            headers={"X-Request-ID": rid}) as r:
        r.raise_for_status()
        async for raw in r.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            msg = json.loads(line[5:])
            if ttft is None and msg.get("choices"):
                ttft = time.perf_counter() - t0
            if msg.get("usage"):
                usage = msg["usage"]
    return ttft, usage


async def run_session(s, args, sid, rng, results):
    messages = [{"role": "system", "content": f"[session {sid}] " + text(args.ctx_words, rng)}]
    for k in range(args.turns):
        messages.append({"role": "user", "content": text(args.delta_words, rng)})
        rid = str(uuid.uuid4())
        ttft, usage = await turn(s, args.base, args.model, messages, rid)
        results.append({"session": sid, "turn": k, "rid": rid, "ttft_s": round(ttft, 3),
                        "prompt_tokens": (usage or {}).get("prompt_tokens")})
        messages.append({"role": "assistant", "content": text(20, rng)})
        await asyncio.sleep(args.think_s)


async def main(args):
    rng = random.Random(args.seed)
    results = []
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=3600)) as s:
        await asyncio.gather(*(run_session(s, args, f"{args.tag}-{i}", random.Random(rng.random()),
                                           results) for i in range(args.sessions)))
    await asyncio.sleep(3)  # access log flush
    rid2rep = {}
    pat = re.compile(r"LLMServer:\S+ (\w+) ([0-9a-f-]{36}) -- POST")
    for path in glob.glob(args.logs):
        for line in open(path, errors="replace"):
            m = pat.search(line)
            if m:
                rid2rep[m.group(2)] = m.group(1)
    by_sess = {}
    for r in sorted(results, key=lambda r: (r["session"], r["turn"])):
        r["replica"] = rid2rep.get(r["rid"], "?")
        by_sess.setdefault(r["session"], []).append(r)
    moves = follow = 0
    for sid, rs in by_sess.items():
        reps = [r["replica"] for r in rs]
        follow += len(reps) - 1
        moves += sum(1 for a, b in zip(reps, reps[1:]) if a != b)
        print(sid, "replicas:", reps, "ttft:", [r["ttft_s"] for r in rs],
              "prompt:", rs[-1]["prompt_tokens"])
    print(f"follow-up turns that moved replica: {moves}/{follow}")
    if args.out:
        with open(args.out, "w") as f:
            for r in results:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8000")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--sessions", type=int, default=4)
    p.add_argument("--turns", type=int, default=6)
    p.add_argument("--ctx-words", type=int, default=40000)
    p.add_argument("--delta-words", type=int, default=800)
    p.add_argument("--think-s", type=float, default=2.0)
    p.add_argument("--tag", default="s")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logs", default="/tmp/ray/session_latest/logs/serve/replica_*LLMServer*.log")
    p.add_argument("--out")
    asyncio.run(main(p.parse_args()))
