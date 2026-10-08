"""Summarize vLLM periodic stats lines per replica log, split by time phase."""
import re
import sys
from datetime import datetime

PAT = re.compile(
    r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, Running: (\d+) reqs, "
    r"Waiting: (\d+) reqs, GPU KV cache usage: ([\d.]+)%, Prefix cache hit rate: ([\d.]+)%"
)
INIT = re.compile(r"INFO (\d\d-\d\d \d\d:\d\d:\d\d).*init engine.*took ([\d.]+) s \(compilation: ([\d.]+) s\)")


def ts(s):
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S")


def main(t0_str, split_str, paths):
    t0, split = ts(t0_str), ts(split_str)
    for path in paths:
        rows, init = [], None
        for line in open(path, errors="replace"):
            m = PAT.search(line)
            if m:
                rows.append((ts(m[1]), *map(float, m.groups()[1:])))
            elif init is None and (mi := INIT.search(line)):
                init = (ts(mi[1]), float(mi[2]), float(mi[3]))
        rows = [r for r in rows if r[3] > 0 or r[4] > 0]
        if not rows:
            continue
        print(f"== {path.split('-')[1][:8]}  engine ready t={(init[0]-t0).total_seconds():.0f}s "
              f"(init {init[1]:.0f}s, compile {init[2]:.0f}s)" if init else f"== {path}")
        for name, sel in (("before all 4 up", [r for r in rows if r[0] < split]),
                          ("all 4 up", [r for r in rows if r[0] >= split])):
            if not sel:
                continue
            n = len(sel)
            print(f"   {name:16s} windows={n:3d} prompt_tps avg={sum(r[1] for r in sel)/n:6.0f} "
                  f"max={max(r[1] for r in sel):6.0f} | running avg={sum(r[3] for r in sel)/n:4.1f} "
                  f"| waiting>0 in {sum(r[4] > 0 for r in sel):3d}/{n} max={max(r[4] for r in sel):.0f} "
                  f"| kv max={max(r[5] for r in sel):4.1f}% | hit(cum) end={sel[-1][6]:.1f}%")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], sys.argv[3:])
