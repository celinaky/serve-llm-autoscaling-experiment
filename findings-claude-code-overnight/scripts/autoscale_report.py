"""One-page analysis of an AgentX autoscaling run.

- Scale timeline: when desired replicas rose (Serve autoscaling metric) vs
  when the extra replicas reached RUNNING (serve_status telemetry).
- TTFT/ITL in windows (whole run, and by replica count at request start).
- Session affinity: follow-up turns that stayed on / moved off the replica
  that served the previous turn (Serve access logs), with TTFT for each.
- Ideal hit rate of the requests sent, and errors.

usage: autoscale_report.py <run_dir> [serve_log_glob]
"""
import bisect
import glob
import json
import re
import sys
from collections import defaultdict

LOG = re.compile(r"LLMServer:\S+ (\w+) ([0-9a-f-]{36}) -- POST /v1/chat/completions")


def val(m, k):
    v = m.get(k)
    return v.get("value") if isinstance(v, dict) else v


def q(xs, p):
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p * len(xs)))] if xs else float("nan")


def fmt(xs):
    return f"n={len(xs):4d} p50={q(xs,.5):6.2f} p90={q(xs,.9):6.2f} p99={q(xs,.99):6.2f}"


def main(run_dir, log_glob):
    bench = glob.glob(f"{run_dir}/benchmark/agentx-*/profile_export.jsonl")[0]
    recs, errors = [], 0
    for line in open(bench):
        r = json.loads(line)
        md, m = r["metadata"], r["metrics"]
        if r.get("error"):
            errors += 1
            continue
        t = val(m, "time_to_first_token")
        if t is None:
            continue
        recs.append(dict(ts=md["request_start_ns"] / 1e9, conv=md["conversation_id"],
                         rid=md["x_request_id"], ttft=t / 1000, itl=val(m, "inter_token_latency"),
                         isl=val(m, "input_sequence_length") or 0))
    recs.sort(key=lambda r: r["ts"])
    t0 = recs[0]["ts"]

    # running LLMServer replicas over time
    status = []
    for line in open(f"{run_dir}/telemetry/serve_status.jsonl"):
        s = json.loads(line)
        dep = next((v for k, v in (s.get("deployments") or {}).items() if k.startswith("LLMServer")), None)
        if dep:
            st = dep.get("replica_states") or {}
            status.append((s["timestamp_ns"] / 1e9, st.get("RUNNING", 0),
                           sum(v for k, v in st.items() if k != "RUNNING")))
    status.sort()
    st_ts = [s[0] for s in status]

    def running_at(t):
        i = bisect.bisect_right(st_ts, t) - 1
        return status[i][1] if i >= 0 else None

    # desired replicas from Serve metrics
    desired = []
    for line in open(f"{run_dir}/telemetry/serve_metrics.jsonl"):
        r = json.loads(line)
        for smp in r.get("samples", []):
            if smp["normalized_name"] == "serve_autoscaling_desired_replicas" and \
                    smp["labels"].get("deployment", "").startswith("LLMServer"):
                desired.append((r["scrape_started_at_ns"] / 1e9, smp["value"]))
    desired.sort()

    print(f"== {run_dir.rstrip('/').split('/')[-1]}")
    print(f"requests ok={len(recs)} errors={errors}")
    # scale timeline
    print("scale timeline (t since first request):")
    last_d = None
    for t, d in desired:
        if d != last_d:
            print(f"   t={t - t0:7.1f}s desired -> {d:.0f}")
            last_d = d
    last_r = None
    for t, r, other in status:
        if r != last_r:
            print(f"   t={t - t0:7.1f}s running -> {r} (other states {other})")
            last_r = r

    print("TTFT (s) by running replicas at request start:")
    by_rep = defaultdict(list)
    for r in recs:
        by_rep[running_at(r["ts"])].append(r["ttft"])
    for k in sorted(by_rep, key=lambda x: (x is None, x)):
        print(f"   replicas={k}: {fmt(by_rep[k])}")
    print("TTFT (s) by 120s window:")
    win = defaultdict(list)
    for r in recs:
        win[int((r["ts"] - t0) // 120) * 120].append(r["ttft"])
    for k in sorted(win):
        print(f"   t={k:4d}-{k+120:4d}: {fmt(win[k])}")
    itl = [r["itl"] for r in recs if r["itl"] is not None]
    print(f"ITL (ms): p50={q(itl,.5):.0f} p90={q(itl,.9):.0f}  TTFT all: {fmt([r['ttft'] for r in recs])}")

    # affinity
    rid2rep = {}
    for path in glob.glob(log_glob):
        for line in open(path, errors="replace"):
            m = LOG.search(line)
            if m:
                rid2rep[m.group(2)] = m.group(1)
    last_rep, last_isl = {}, {}
    kinds = defaultdict(list)
    lost = 0
    unc = tot = 0
    for r in recs:
        rep = rid2rep.get(r["rid"])
        prev_isl = last_isl.get(r["conv"])
        unc += r["isl"] if prev_isl is None else max(r["isl"] - prev_isl, 0)
        tot += r["isl"]
        last_isl[r["conv"]] = r["isl"]
        if rep is None:
            continue
        prev = last_rep.get(r["conv"])
        kind = "first" if prev is None else ("stayed" if prev == rep else "moved")
        if kind == "moved" and prev_isl:
            lost += min(prev_isl, r["isl"])
        kinds[kind].append(r["ttft"])
        last_rep[r["conv"]] = rep
    mapped = sum(len(v) for v in kinds.values())
    print(f"affinity (mapped {mapped}/{len(recs)} requests to replicas):")
    for k in ("first", "stayed", "moved"):
        print(f"   {k:6s}: {fmt(kinds[k])}")
    f = len(kinds["stayed"]) + len(kinds["moved"])
    if f:
        print(f"   follow-ups moved: {len(kinds['moved'])}/{f} = {len(kinds['moved'])/f:.1%}; "
              f"reusable prefix tokens re-prefilled: {lost/1e6:.2f}M")
    print(f"ideal hit rate of requests sent: {1 - unc/tot:.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else
         "/tmp/ray/session_latest/logs/serve/replica_ttft-benchmark_LLMServer_*.log")
