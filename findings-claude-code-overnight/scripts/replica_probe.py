"""Single-replica probes against a vLLM OpenAI server (no Ray Serve in the path).

Prompts are random token ids sent to /v1/completions, so lengths are exact and
nothing is shared unless a probe reuses a prefix on purpose.

  cold      TTFT and prefill rate for unique prompts of several lengths
  warm      prompt P, then P + delta: TTFT and cached tokens of the follow-up
  hol       a short request issued while a long cold prefill is running
  conc      N concurrent cold prompts of one length: per-request TTFT spread
"""
import argparse
import asyncio
import json
import random
import time

import aiohttp

VOCAB = (1000, 140000)


def rand_ids(n, rng):
    return [rng.randrange(*VOCAB) for _ in range(n)]


async def one(session, url, model, ids, max_tokens=1):
    body = {"model": model, "prompt": ids, "max_tokens": max_tokens, "stream": True,
            "stream_options": {"include_usage": True}, "temperature": 0}
    t0 = time.perf_counter()
    ttft = None
    usage = None
    async with session.post(url, json=body) as resp:
        resp.raise_for_status()
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line.endswith("[DONE]"):
                continue
            msg = json.loads(line[5:])
            if ttft is None and msg.get("choices") and msg["choices"][0].get("text") is not None:
                ttft = time.perf_counter() - t0
            if msg.get("usage"):
                usage = msg["usage"]
    cached = ((usage or {}).get("prompt_tokens_details") or {}).get("cached_tokens")
    return {"len": len(ids), "ttft_s": round(ttft, 3), "e2e_s": round(time.perf_counter() - t0, 3),
            "cached": cached}


async def main(a):
    rng = random.Random(a.seed)
    url = f"{a.base}/v1/completions"
    out = []
    timeout = aiohttp.ClientTimeout(total=3600)
    async with aiohttp.ClientSession(timeout=timeout) as s:
        await one(s, url, a.model, rand_ids(512, rng))  # warm up the server path
        if "cold" in a.probes:
            for n in a.cold_lens:
                for rep in range(a.reps):
                    r = await one(s, url, a.model, rand_ids(n, rng))
                    r.update(probe="cold", rep=rep, prefill_tok_s=round(n / r["ttft_s"]))
                    out.append(r); print(json.dumps(r), flush=True)
        if "warm" in a.probes:
            for n in a.warm_lens:
                base = rand_ids(n, rng)
                r0 = await one(s, url, a.model, base)
                r0.update(probe="warm_base"); out.append(r0); print(json.dumps(r0), flush=True)
                for delta in a.deltas:
                    ids = base + rand_ids(delta, rng)
                    r = await one(s, url, a.model, ids)
                    r.update(probe="warm_followup", base_len=n, delta=delta)
                    out.append(r); print(json.dumps(r), flush=True)
                    base = ids  # next turn extends this one, like an agent session
        if "hol" in a.probes:
            long_ids = rand_ids(a.hol_long, rng)
            long_task = asyncio.create_task(one(s, url, a.model, long_ids))
            await asyncio.sleep(a.hol_delay)
            short = await one(s, url, a.model, rand_ids(a.hol_short, rng))
            lr = await long_task
            r = {"probe": "hol", "long_len": a.hol_long, "long_ttft_s": lr["ttft_s"],
                 "short_len": a.hol_short, "short_ttft_s": short["ttft_s"],
                 "short_issued_after_s": a.hol_delay}
            out.append(r); print(json.dumps(r), flush=True)
        if "conc" in a.probes:
            for c in a.conc_levels:
                t0 = time.perf_counter()
                rs = await asyncio.gather(*(one(s, url, a.model, rand_ids(a.conc_len, rng))
                                            for _ in range(c)))
                wall = time.perf_counter() - t0
                tt = sorted(r["ttft_s"] for r in rs)
                r = {"probe": "conc", "concurrency": c, "len": a.conc_len,
                     "ttft_min": tt[0], "ttft_p50": tt[len(tt) // 2], "ttft_max": tt[-1],
                     "agg_prefill_tok_s": round(c * a.conc_len / wall)}
                out.append(r); print(json.dumps(r), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", default="http://127.0.0.1:8100")
    p.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--probes", nargs="+", default=["cold", "warm", "hol", "conc"])
    p.add_argument("--cold-lens", type=int, nargs="+", default=[2048, 8192, 32768, 65536, 131072])
    p.add_argument("--warm-lens", type=int, nargs="+", default=[65536])
    p.add_argument("--deltas", type=int, nargs="+", default=[512, 2048, 8192])
    p.add_argument("--hol-long", type=int, default=131072)
    p.add_argument("--hol-short", type=int, default=2048)
    p.add_argument("--hol-delay", type=float, default=1.0)
    p.add_argument("--conc-levels", type=int, nargs="+", default=[1, 4, 8])
    p.add_argument("--conc-len", type=int, default=32768)
    p.add_argument("--reps", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out")
    main_args = p.parse_args()
    asyncio.run(main(main_args))
