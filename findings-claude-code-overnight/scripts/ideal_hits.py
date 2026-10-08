"""Upper bounds on prefix-cache hits for the AgentX/Weka trace dataset.

Each request lists hash_ids (one per 64-token block). A request can reuse the
KV of the longest leading run of hash_ids it shares with an earlier request.
Scopes:
  stream  - only earlier requests of the same stream (root agent or one
            subagent). AIPerf's cache-bust marker is per trajectory and
            SPAWN children are busted separately, so this is what replay
            can actually hit.
  trace   - any earlier request in the same trace (no busting).
Also reports the vLLM align-mode ceiling: hits floored to the 1568-token
attention block of this model.
"""
import glob
import json
import sys
from collections import Counter

import pyarrow as pa

BLOCK = 64
VLLM_BLOCK = int(sys.argv[1]) if len(sys.argv) > 1 else 1568


def lcp(a, b):
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


def load():
    files = sorted(glob.glob(
        "/home/ray/.cache/huggingface/datasets/semianalysisai___cc-traces-weka-062126-256k/"
        "default/0.0.0/*/*.arrow"))
    table = pa.concat_tables([pa.ipc.open_stream(open(f, "rb")).read_all() for f in files])
    for row in table.to_pylist():
        reqs = []
        for r in row["requests"]:
            r = json.loads(r) if isinstance(r, str) else r
            if r.get("type") == "subagent":
                for s in r["requests"]:
                    s = json.loads(s) if isinstance(s, str) else s
                    if "hash_ids" in s:
                        reqs.append((s["t"], f"sa:{r['agent_id']}", s))
            elif "hash_ids" in r:
                reqs.append((r["t"], "root", r))
        reqs.sort(key=lambda x: x[0])
        yield row["id"], reqs


def main():
    tot_in = 0
    hit = Counter()
    n_req = 0
    breaks = 0  # requests sharing < 50% of the previous same-stream prompt
    first_in = 0
    per_req_uncached = []
    for _, reqs in load():
        seen_stream = {}
        seen_trace = []
        for _, stream, r in reqs:
            ids, n_in = r["hash_ids"], r["in"]
            tot_in += n_in
            n_req += 1
            prev = seen_stream.get(stream, [])
            best_stream = max((lcp(ids, p) for p in prev), default=0)
            best_trace = max((lcp(ids, p) for p in seen_trace), default=0)
            last = prev[-1] if prev else None
            last_lcp = lcp(ids, last) if last else 0
            if last and last_lcp < 0.5 * len(last):
                breaks += 1
            if not prev:
                first_in += n_in
            for name, blocks in (("stream", best_stream), ("trace", best_trace),
                                 ("stream_last_only", last_lcp)):
                toks = min(blocks * BLOCK, n_in)
                hit[name] += toks
                if name == "stream":
                    hit["stream_vllm_align"] += toks // VLLM_BLOCK * VLLM_BLOCK
                    per_req_uncached.append(n_in - toks // VLLM_BLOCK * VLLM_BLOCK)
            seen_stream.setdefault(stream, []).append(ids)
            seen_trace.append(ids)
    print(f"requests={n_req} input_tokens={tot_in/1e6:.1f}M "
          f"first-request-of-stream tokens={first_in/tot_in:.1%}")
    for k in ("trace", "stream", "stream_last_only", "stream_vllm_align"):
        print(f"  ideal hit rate [{k:18s}] = {hit[k]/tot_in:.1%}")
    print(f"  requests whose prompt keeps <50% of previous same-stream prompt: {breaks} "
          f"({breaks/n_req:.1%})")
    u = sorted(per_req_uncached)
    q = lambda p: u[min(len(u) - 1, int(p * len(u)))]
    print(f"  uncached tokens/request at the vLLM-align ceiling: p50={q(.5)} p90={q(.9)} "
          f"p99={q(.99)} mean={sum(u)/len(u):.0f}")


if __name__ == "__main__":
    main()
