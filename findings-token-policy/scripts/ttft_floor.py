"""The workload's queue-free TTFT floor, from completed runs' request records.

For each request, the fastest TTFT seen in its uncached-token bucket (5th
percentile of a factor-of-2 bucket) is taken as the cost of the prefill
itself, free of queueing; the percentiles of that over the request mix are
the best any autoscaler could reach.

usage: ttft_floor.py RUN_DIR [RUN_DIR ...]
"""
import collections
import glob
import json
import math
import sys


def main(run_dirs):
    rows = []
    for d in run_dirs:
        for path in glob.glob(f"{d}/benchmark/agentx-*/profile_export.jsonl"):
            for line in open(path):
                x = json.loads(line)
                m = x.get("metrics") or {}
                if x["metadata"]["benchmark_phase"] != "profiling" or x.get("error"):
                    continue
                g = lambda k: (m.get(k) or {}).get("value")  # noqa: E731
                isl, cached, ttft = (g("input_sequence_length"), g("usage_prompt_cache_read_tokens"),
                                     g("time_to_first_token"))
                if None in (isl, cached, ttft):
                    continue
                rows.append((isl - cached, ttft))

    def bucket(uncached):
        return 0 if uncached <= 0 else int(math.log2(uncached))

    by_bucket = collections.defaultdict(list)
    for uncached, ttft in rows:
        by_bucket[bucket(uncached)].append(ttft)
    floor = {b: sorted(v)[int(0.05 * len(v))] for b, v in by_bucket.items()}
    inherent = sorted(floor[bucket(u)] for u, _ in rows)
    actual = sorted(t for _, t in rows)

    def pct(values, q):
        return values[min(len(values) - 1, int(q * len(values)))]

    print(f"{len(rows)} requests from {len(run_dirs)} runs")
    print("uncached tokens      n   floor ms    p50 ms")
    for b in sorted(by_bucket):
        v = sorted(by_bucket[b])
        print(f"  >= {2 ** b:7d}  {len(v):6d}  {floor[b]:8.0f}  {pct(v, .5):8.0f}")
    for label, values in (("actual", actual), ("floor", inherent)):
        print(f"TTFT {label:6s} p90/p95/p99: " + " / ".join(f"{pct(values, q):.0f}ms" for q in (.9, .95, .99)))


if __name__ == "__main__":
    main(sys.argv[1:])
