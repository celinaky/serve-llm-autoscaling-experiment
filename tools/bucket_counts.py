"""Count requests per (own uncached band, 20K engine-load bin) for runs' load_requests.csv.

usage: python tools/bucket_counts.py runs/A [runs/B ...]  (after tools/load_breakdown.py)
"""
import csv, sys

S = [(0, 4e3, "<4K"), (4e3, 16e3, "4-16K"), (16e3, 48e3, "16-48K"), (48e3, 1e12, ">=48K")]
B = [(0, 0, "0")] + [(i * 20e3, (i + 1) * 20e3, f"{i*20}-{(i+1)*20}K") for i in range(10)] + [(200e3, 1e12, ">200K")]
rows = []
for R in sys.argv[1:]:
    rs = list(csv.DictReader(open(f"{R.rstrip('/')}/analysis/load_requests.csv")))
    rows += [r for r in rs if int(float(r["preemptions_before_first_token"])) == 0]
print(f"{len(rows)} requests (no preemption before first token)")
print(f"{'band':7}" + "".join(f"{b[2]:>9}" for b in B))
short = 0
for lo, hi, name in S:
    cells = []
    for i, (a, b, _) in enumerate(B):
        n = sum(lo <= float(r["own_uncached"]) < hi and
                (float(r["engine_load_tokens"]) == 0 if i == 0 else a < float(r["engine_load_tokens"]) <= b)
                for r in rows)
        cells.append(n)
        short += n < 10
    print(f"{name:7}" + "".join(f"{n:>9}" for n in cells))
print(f"cells under 10: {short}")
