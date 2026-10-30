#!/usr/bin/env python3
"""Per-run diagnostic table for the workday-curve policy tuning loop.

usage: python tools/rundiag.py runs/<run_dir> [--step 30]
"""
import json, sys, glob, argparse, statistics as st
from collections import defaultdict

ap = argparse.ArgumentParser()
ap.add_argument("run")
ap.add_argument("--step", type=float, default=30)
ap.add_argument("--quiet", action="store_true", help="summary + events only")
a = ap.parse_args()
R = a.run.rstrip("/")

ev = json.load(open(f"{R}/analysis/autoscaling_evaluation.json"))
pd = json.load(open(f"{R}/analysis/plot_data.json"))
rq = ev["slo_attainment"]["whole_run"]["requests"]
win = ev["slo_attainment"]["whole_run"]["windows"]
cost = ev["cost"]["whole_run"]
dec = ev.get("ttft_decomposition", {})
print(f"== {R}")
print(f"policy {ev['autoscaling_policy']} kwargs {json.dumps(ev.get('autoscaling_policy_kwargs', {}))}")
t = rq["ttft"]; i = rq["itl"]
print(f"TTFT ms  p50 {t['p50_ms']:.0f}  p90 {t['p90_ms']:.0f}  p95 {t['p95_ms']:.0f}  p99 {t['p99_ms']:.0f}   attain {t['slo_attainment']:.3f}")
print(f"ITL  ms  p50 {i['p50_ms']:.1f}  p90 {i['p90_ms']:.1f}  p95 {i['p95_ms']:.1f}  p99 {i['p99_ms']:.1f}   attain {i['slo_attainment']:.3f}")
bc = rq.get("ttft_by_cache", {})
if bc:
    print(f"TTFT p95 cold {bc['cold']['p95_ms']:.0f} (n={bc['cold']['sample_count']})  warm {bc['warm']['p95_ms']:.0f} (n={bc['warm']['sample_count']})")
print(f"requests {rq['requests']} failed {rq['failed_requests']}  windows both-ok {win['both_within_slo']:.3f} ttft-ok {win['ttft_within_slo']:.3f} itl-ok {win['itl_within_slo']:.3f}")
print(f"cost gpu-h {cost['gpu_hours']:.2f}  mean reps {cost['mean_running_replicas']:.2f}  max {cost['max_running_replicas']}  starting-s {cost['starting_replica_seconds']:.0f}")
if dec:
    print(f"ttft decomposition p95: actual {dec['actual_ms']:.0f} instant-startup {dec['instant_startup_ms']:.0f} own-prefill-floor {dec['own_prefill_floor_ms']:.0f} waiting {dec['waiting_ms']:.0f}")
sc = ev.get("scaling", {})
print(f"scaling: {sc}")
evs = ev.get("scale_events", [])
print("scale events: " + "  ".join(f"{e['time_s']:.0f}s {e['from_replicas']:.0f}->{e['to_replicas']:.0f}" + (f"(lag {e['lag_s']:.0f})" if e.get('lag_s') else "") for e in evs))
# flap detection: direction reversal within 180s
flaps = []
for p, q in zip(evs, evs[1:]):
    if p["direction"] != q["direction"] and q["time_s"] - p["time_s"] < 180:
        flaps.append(f"{p['time_s']:.0f}s {p['direction']} then {q['time_s']:.0f}s {q['direction']}")
print("flaps (<180s reversal): " + ("; ".join(flaps) if flaps else "none"))
if a.quiet:
    sys.exit()

# ---- per-step table
ss = pd["serve_status_samples"]
sms = pd["serve_metric_series"]
rw = pd["request_windows"]
def series(name):
    s = sms.get(name)
    return [(p["relative_time_s"], p["value"]) for p in s["points"]] if s else []
def at(pts, t0, t1, fn=max):
    v = [v for ts, v in pts if t0 <= ts < t1]
    return fn(v) if v else None
def rep(t0, t1):
    v = [s["running_replicas"] for s in ss if t0 <= s["relative_time_s"] < t1]
    return (min(v), max(v)) if v else (None, None)
# policy gauges straight from serve_metrics.jsonl (window_capacity is not in the harness map)
t_origin = pd["time_origin"]["profiling_start_ns"]
gauges = defaultdict(list)
try:
    for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
        rec = json.loads(line)
        ts = (rec["scrape_started_at_ns"] - t_origin) / 1e9
        for smp in rec["samples"]:
            n = smp.get("normalized_name", "")
            if n.startswith("serve_token_autoscaling_") or n == "serve_autoscaling_desired_replicas":
                gauges[n].append((ts, smp["value"]))
except FileNotFoundError:
    pass
pre = gauges.get("serve_token_autoscaling_outstanding_prefill_tokens", [])
cap = gauges.get("serve_token_autoscaling_window_capacity", [])
dcap = gauges.get("serve_token_autoscaling_downscale_capacity", [])
pol = gauges.get("serve_token_autoscaling_desired_replicas", [])
dq = gauges.get("serve_token_autoscaling_active_decode_requests", [])
sdes = gauges.get("serve_autoscaling_desired_replicas", [])
vw = series("vllm_waiting_requests"); vr = series("vllm_running_requests"); hit = series("vllm_prefix_cache_hit_rate")
end = max(w["window_end_s"] for w in rw)
print()
print(f"{'t':>5} {'live':>4} {'reps':>5} {'sdes':>4} {'pdes':>4} {'cap':>5} {'dcap':>5} {'prefillK':>8} {'dec':>4} {'offer':>5} {'p50ttft':>7} {'p90ttft':>7} {'p99ttft':>7} {'p90itl':>6} {'vrun':>4} {'vwait':>5} {'hit':>4}")
step = a.step
tt = 0.0
while tt < end:
    ws = [w for w in rw if tt <= w["window_start_s"] < tt + step]
    if not ws:
        tt += step; continue
    live = ws[-1]["live_session_trees_at_end"]
    offered = sum(w["offered_requests"] for w in ws)
    tsamp = [w for w in ws if w["ttft_sample_count"]]
    def wp(k):
        v = [w[k] for w in tsamp if w.get(k) is not None]
        return f"{max(v):7.0f}" if v else f"{'-':>7}"
    def wi(k):
        v = [w[k] for w in ws if w.get("itl_sample_count") and w.get(k) is not None]
        return f"{max(v):6.0f}" if v else f"{'-':>6}"
    lo, hi = rep(tt, tt + step)
    reps = f"{lo}-{hi}" if lo != hi else f"{lo}"
    f = lambda x, w=5, d=1: f"{x:{w}.{d}f}" if x is not None else f"{'-':>{w}}"
    print(f"{tt:5.0f} {live:4d} {reps:>5} {f(at(sdes,tt,tt+step),4,0)} {f(at(pol,tt,tt+step),4,0)} {f(at(cap,tt,tt+step,fn=lambda v: sum(v)/len(v)),5,2)} {f(at(dcap,tt,tt+step,fn=lambda v: sum(v)/len(v)),5,2)} {f((at(pre,tt,tt+step,fn=lambda v: sum(v)/len(v)) or 0)/1000 if pre else None,8,1)} {f(at(dq,tt,tt+step,fn=lambda v: sum(v)/len(v)),4,0)} {offered:5d} {wp('p50_ttft_ms')} {wp('p90_ttft_ms')} {wp('p99_ttft_ms')} {wi('p90_itl_ms')} {f(at(vr,tt,tt+step),4,0)} {f(at(vw,tt,tt+step),5,0)} {f(at(hit,tt,tt+step,fn=lambda v: sum(v)/len(v)),4,2)}")
    tt += step
