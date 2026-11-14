#!/usr/bin/env python3
"""Zoom: the router's gauge against the engine's actual uncached prefill, request by request.

Top: two instantaneous signals over a short window. Grey step: the engine's
uncached prefill tokens outstanding (each request's prompt minus what the
engine found cached, counted from queued to first token; scheduler traces).
Purple step: the router's gauge as scraped (its estimate of the same thing;
scrapes are 2s and the value holds for up to 10s). Each prompt of --min-tokens
or more is marked at its queue time: green label = a new session's first
turn, red label = a returning session whose context the engine had evicted
(turn >= 1, engine cached 0). When the router has the context in its index it
adds only the small delta, so purple barely moves while grey jumps.
Bottom: the 60s means: the router's published mean and the engine's mean
under both definitions (whole prompt held to first token; chunk burn-down as
in policy_vs_engine_mean.py).

usage: python tools/router_undercount_zoom.py runs/<run_dir> --start 330 --end 460 [--min-tokens 30000] [--out PATH]
"""
import argparse, bisect, csv, glob, json, os
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt

SURFACE, INK, INK2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
TR = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces"
ap = argparse.ArgumentParser(); ap.add_argument("run"); ap.add_argument("--start", type=float, required=True); ap.add_argument("--end", type=float, required=True)
ap.add_argument("--min-tokens", type=int, default=30000); ap.add_argument("--lookback", type=float, default=60); ap.add_argument("--target", type=float, default=20000); ap.add_argument("--traces", default=TR); ap.add_argument("--out")
a = ap.parse_args(); R = a.run.rstrip("/")
t0 = json.load(open(f"{R}/analysis/plot_data.json"))["time_origin"]["profiling_start_ns"] / 1e9
meta = {m["x_request_id"]: m for p in glob.glob(f"{R}/benchmark/*/profile_export.jsonl") for m in (json.loads(l)["metadata"] for l in open(p)) if m.get("benchmark_phase") == "profiling"}
engines = []
for r in csv.DictReader(open(f"{R}/analysis/load_requests.csv")):
    e = r["engine"].split("pid-")[-1].split(".")[0]
    if e not in engines: engines.append(e)
whole, burn, marks = [], [], []
for e in engines:
    p = glob.glob(f"{a.traces}/*pid-{e}*.jsonl")
    if not p: continue
    off = None; q, ad, ft, chunks, done = {}, {}, {}, {}, {}
    for line in open(p[0]):
        d = json.loads(line); k = d["event"]
        if off is None and "wall_time_s" in d: off = d["wall_time_s"] - d["monotonic_s"]
        if k == "request_queued": q[d["request_id"]] = d
        elif k == "request_admitted": ad[d["request_id"]] = d
        elif k == "request_first_token": ft[d["request_id"]] = d
        elif k == "batch_scheduled":
            for r in d["requests"]:
                if r["phase"] == "prefill": chunks.setdefault(r["request_id"], []).append((d["batch_index"], r["scheduled_tokens"]))
        elif k == "batch_completed": done[d["batch_index"]] = d["monotonic_s"] + off - t0
    for rid, qq in q.items():
        if rid not in ad or rid not in ft: continue
        tq = qq["arrival_time_s"] - t0; u = ad[rid]["prompt_tokens"] - ad[rid]["cached_prefix_tokens"]
        whole.append((tq, u)); whole.append((ft[rid]["monotonic_s"] + off - t0, -u))
        cs = chunks.get(rid)
        if cs and all(b in done for b, _ in cs):
            burn.append((tq, sum(t for _, t in cs))); burn.extend((done[b], -t) for b, t in cs)
        if u >= a.min_tokens and a.start <= tq <= a.end:
            m = meta.get(rid.split("chatcmpl-", 1)[-1].rsplit("-", 1)[0], {})
            cid = m.get("conversation_id", "?"); turn = int(m.get("turn_index", 0))
            marks.append((tq, u, cid.split("::")[0][-6:] + ("/sa" if "::" in cid else ""), turn, ad[rid]["cached_prefix_tokens"], engines.index(e) + 1))
def step(deltas):
    deltas = sorted(deltas); ts = [t for t, _ in deltas]; lv = []; l = 0
    for _, dd in deltas: l += dd; lv.append(l)
    return ts, lv
wts, wlv = step(whole); bts, blv = step(burn)
def level(ts, lv, t):
    i = bisect.bisect_right(ts, t) - 1; return lv[i] if i >= 0 else 0
def mean(ts, lv, t):
    lo = t - a.lookback; i = bisect.bisect_right(ts, lo) - 1; pt = lo; pl = lv[i] if i >= 0 else 0; j = i + 1; acc = 0
    while j < len(ts) and ts[j] < t: acc += pl * (ts[j] - pt); pt, pl = ts[j], lv[j]; j += 1
    acc += pl * (t - pt); return acc / a.lookback
raw, pub = [], []
for line in open(f"{R}/telemetry/serve_metrics.jsonl"):
    d = json.loads(line); t = d["timestamp_ns"] / 1e9 - t0
    if not a.start - 15 <= t <= a.end + 5: continue
    for s in d.get("samples") or []:
        if s["name"] == "ray_serve_autoscaling_policy_metric":
            if s["labels"].get("metric") == "outstanding_effective_prefill_tokens": raw.append((t, s["value"]))
            if s["labels"].get("metric") == "prefill_required_replicas": pub.append((t, s["value"] * a.target))
grid = [a.start + i * 0.5 for i in range(int((a.end - a.start) * 2) + 1)]
fig, (ax, ax2) = plt.subplots(2, 1, figsize=(14, 10.5), dpi=150, sharex=True, gridspec_kw={"height_ratios": [1.9, 1], "hspace": 0.1}); fig.patch.set_facecolor(SURFACE)
ax.step(grid, [level(wts, wlv, t) / 1000 for t in grid], where="post", color="#9b9a95", lw=2, label="engine: uncached prompt tokens actually outstanding (prompt − what the engine found cached), queued to first token")
ax.plot(grid, [mean(wts, wlv, t) / 1000 for t in grid], color=INK, lw=1.8, label="engine: 60s mean of the grey line")
ax.plot([t for t, _ in pub], [v / 1000 for _, v in pub], color="#6250d6", lw=2, ls="--", label="router: the 60s mean it published of its own gauge (prompt − what its index says is cached). If it had counted the grey jumps, this would track the black line")
top = max(level(wts, wlv, t) for t in grid) / 1000 * 1.9
for tq, u, name, turn, cached, rep in marks:
    new = turn == 0
    col = "#1baf7a" if new else "#e34948"
    ax.axvline(tq, color=col, lw=0.8, ls=":", alpha=0.8)
    ax.text(tq + 0.6, top * (0.98 - 0.045 * (marks.index((tq, u, name, turn, cached, rep)) % 4)), f"{name} t{turn} {u//1000}K r{rep}", fontsize=6.8, color=col, rotation=0, va="top")
ax.plot([], [], color="#1baf7a", ls=":", label="new session's first turn: the router cannot think it is cached")
ax.plot([], [], color="#e34948", ls=":", label="returning session, context evicted by the engine (cached 0): the router's index may still list it")
ax.set_ylim(0, top); ax.set_ylabel("K tokens, all replicas"); ax.legend(fontsize=7.5, loc="upper left", bbox_to_anchor=(0.0, 0.80), frameon=True, facecolor=SURFACE, edgecolor=GRID); ax.grid(color=GRID, lw=0.6); ax.set_facecolor(SURFACE)
ax.set_title("what the engine really had to prefill (grey, instantaneous), its 60s mean (black), and the router's 60s mean (purple) · labels: session, turn, engine-uncached tokens, replica", fontsize=9, loc="left", color=INK2)
for sp in ax.spines.values(): sp.set_visible(False)
ax2.plot(grid, [mean(wts, wlv, t) / 1000 for t in grid], color=INK, lw=1.8, label="engine, 60s mean, whole prompt held to first token (same as the black line above)")
ax2.plot(grid, [mean(bts, blv, t) / 1000 for t in grid], color="#1baf7a", lw=1.8, label="engine, 60s mean, chunk burn-down (the policy_vs_engine_mean chart's green line)")
ax2.plot([t for t, _ in pub], [v / 1000 for _, v in pub], color="#6250d6", lw=1.8, ls="--", label="router, the 60s mean the policy published (its purple line)")
ax2.axhline(a.target / 1000, color=INK, ls=":", lw=0.8); ax2.axhline(2 * a.target / 1000, color=INK, ls=":", lw=0.8)
ax2.set_ylabel("K tokens, 60s mean"); ax2.set_xlabel("seconds since profiling start"); ax2.legend(fontsize=7.5, loc="upper left", frameon=True, facecolor=SURFACE, edgecolor=GRID); ax2.grid(color=GRID, lw=0.6); ax2.set_facecolor(SURFACE)
ax2.set_title("the same 60s means next to the harness chart's chunk burn-down line: even against burn-down, the router is under the engine here", fontsize=9, loc="left", color=INK2)
for sp in ax2.spines.values(): sp.set_visible(False)
ax.set_xlim(a.start, a.end)
fig.suptitle(f"{os.path.basename(R).split('-', 1)[-1]} · {a.start:.0f}-{a.end:.0f}s · why the router's gauge fell below the engine", fontsize=10, x=0.01, ha="left")
out = a.out or f"{R}/analysis/router_undercount_{int(a.start)}_{int(a.end)}.png"
fig.savefig(out, bbox_inches="tight", facecolor=SURFACE); print("wrote", out)
for tq, u, name, turn, cached, rep in marks:
    after = [v for t, v in raw if tq < t <= tq + 12]; before = [v for t, v in raw if tq - 12 <= t <= tq]
    print(f"  {tq:6.1f}s {name:9s} t{turn:<2d} engine uncached {u:7d}  router gauge before {max(before) if before else 0:7.0f} after {max(after) if after else 0:7.0f}  {'NEW' if turn == 0 else 'RETURNING (evicted)'}")
