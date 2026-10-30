#!/usr/bin/env python3
"""Open-loop side probes of a fixed new-token size, sent alongside a running benchmark.

The closed-loop AgentX replay rarely produces mid-size (4-16K uncached) turns
while the engine has a prefill backlog: running sessions stall until it
drains. This sends Poisson-timed chat requests whose prompt is a shared
prefix (cached after the first probe, like a warm turn's history) plus a
unique random suffix of the target uncached size, with few output tokens so
they add almost no decode load. Each request carries its own X-Request-ID,
and the results are written in AIPerf's profile_export.jsonl record shape,
so tools/load_breakdown.py joins them to the engine traces like any AIPerf
request when the file is placed in runs/<run>/benchmark/side-probe/.

usage: python tools/side_probe.py --out <dir> --until-ns <epoch ns> \
           [--rate 0.2] [--min-new 4000] [--max-new 16000] [--prefix 40000]
"""
import argparse, json, os, random, threading, time, uuid

import requests
from transformers import AutoTokenizer

ap = argparse.ArgumentParser()
ap.add_argument("--out", required=True)
ap.add_argument("--until-ns", type=int, required=True, help="stop sending at this wall time")
ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
ap.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
ap.add_argument("--rate", type=float, default=0.2, help="mean probes per second (Poisson)")
ap.add_argument("--min-new", type=int, default=4000)
ap.add_argument("--max-new", type=int, default=16000)
ap.add_argument("--prefix", type=int, default=40000, help="shared prefix tokens")
ap.add_argument("--max-tokens", type=int, default=8)
ap.add_argument("--seed", type=int, default=7)
a = ap.parse_args()

rng = random.Random(a.seed)
tok = AutoTokenizer.from_pretrained(a.model)
vocab = [t for t in range(1000, 100000)]


def text_of(n, r):
    # Random token ids decode to text that re-tokenizes to roughly n tokens.
    return tok.decode([r.choice(vocab) for _ in range(n)])


prefix = text_of(a.prefix, random.Random(0))
os.makedirs(a.out, exist_ok=True)
out = open(os.path.join(a.out, "profile_export.jsonl"), "a")
lock = threading.Lock()


def probe(new_tokens, seed):
    rid = str(uuid.uuid4())
    body = {"model": a.model, "stream": True, "max_tokens": a.max_tokens,
            "stream_options": {"include_usage": True},
            "messages": [{"role": "user", "content": prefix + "\n" + text_of(new_tokens, random.Random(seed))}]}
    start = time.time_ns()
    first = end = None
    usage = {}
    error = None
    try:
        with requests.post(a.url, json=body, headers={"X-Request-ID": rid}, stream=True, timeout=900) as resp:
            resp.raise_for_status()
            for line in resp.iter_lines():
                if not line.startswith(b"data: ") or line == b"data: [DONE]":
                    continue
                chunk = json.loads(line[6:])
                if chunk.get("usage"):
                    usage = chunk["usage"]
                for c in chunk.get("choices") or []:
                    d = c.get("delta") or {}
                    if first is None and (d.get("content") or d.get("reasoning_content") or d.get("reasoning")):
                        first = time.time_ns()
        end = time.time_ns()
    except Exception as e:  # recorded, never fatal to the prober
        error = repr(e)
    rec = {"metadata": {"x_request_id": rid, "request_start_ns": start, "request_end_ns": end,
                        "benchmark_phase": "profiling", "source_kind": "side_probe",
                        "target_new_tokens": new_tokens},
           "metrics": {}, "error": error}
    if first is not None and usage:
        cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens") or 0
        rec["metrics"] = {"time_to_first_token": {"value": (first - start) / 1e6, "unit": "ms"},
                          "input_sequence_length": {"value": usage["prompt_tokens"], "unit": "tokens"},
                          "usage_prompt_cache_read_tokens": {"value": cached, "unit": "tokens"}}
    with lock:
        out.write(json.dumps(rec) + "\n")
        out.flush()


threads = []
n = 0
while time.time_ns() < a.until_ns:
    t = threading.Thread(target=probe, args=(rng.randint(a.min_new, a.max_new), rng.random()), daemon=True)
    t.start()
    threads.append(t)
    n += 1
    time.sleep(rng.expovariate(a.rate))
print(f"sent {n} probes; waiting for them to finish", flush=True)
for t in threads:
    t.join()
print("done", flush=True)
