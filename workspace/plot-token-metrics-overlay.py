"""Plot the completed run's raw router gauge and exported policy mean."""
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import requests

OUTPUT = Path("/home/ray/default/serve-llm-autoscaling-experiment/runs/20261111T002035Z-h100-token-metrics-injected-scheduler-local/analysis/token-metrics-overlay")
OUTPUT.mkdir(parents=True, exist_ok=True)
FILTER = 'application="ttft-benchmark",deployment="LLMServer:Qwen--Qwen3_6-27B-FP8",ClusterId="ses_97mprzyvn9vu78btggazn3w5me"'
QUERIES = {
    "router": "ray_serve_router_uncached_prefill_tokens{" + FILTER + "}",
    "policy_mean": 'ray_serve_autoscaling_policy_metric{' + FILTER + ',metric="prefill_tokens_lookback_mean"}',
}
series = {}
for key, query in QUERIES.items():
    response = requests.get("http://localhost:9481/api/v1/query_range", params={
        "query": query, "start": "2026-11-11T00:24:00Z",
        "end": "2026-11-11T00:47:00Z", "step": 10,
    }, timeout=30)
    response.raise_for_status()
    payload = response.json()
    if payload.get("status") != "success":
        raise RuntimeError(payload)
    results = payload["data"]["result"]
    if len(results) != 1:
        raise RuntimeError(f"Expected one {key} series, found {len(results)}")
    (OUTPUT / f"{key}-prometheus.json").write_text(json.dumps(payload, indent=2) + "\n")
    series[key] = results[0]

plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})
fig, ax = plt.subplots(figsize=(14, 5.5))
for key, label, color, width in [
    ("router", "KV tracker: current uncached prefill tokens", "#3298bd", 1.5),
    ("policy_mean", "Policy: exported 60-second lookback mean", "#8b3dbb", 2.4),
]:
    points = series[key]["values"]
    times = [datetime.fromtimestamp(float(t), timezone.utc) for t, _ in points]
    values = [float(v) for _, v in points]
    ax.plot(times, values, color=color, linewidth=width, label=label)
ax.set(title="Router token metric vs. the mean used by the autoscaling policy",
       xlabel="November 11, 2026 · UTC", ylabel="Uncached prefill tokens", ylim=(0, None))
ax.xaxis.set_major_formatter(mdates.DateFormatter("%H:%M", tz=timezone.utc))
ax.yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
ax.grid(alpha=0.18)
ax.legend(loc="upper right", frameon=False)
fig.text(0.08, 0.02, "Both lines are exported gauges queried from Prometheus at 10-second resolution. Purple is the policy's published mean; no mean was recomputed from blue.", fontsize=9, color="#555555")
fig.tight_layout(rect=(0, 0.06, 1, 1))
for extension in ("png", "svg"):
    fig.savefig(OUTPUT / f"router-vs-policy-mean.{extension}", dpi=180)
print(OUTPUT / "router-vs-policy-mean.png")
for key, result in series.items():
    values = [float(v) for _, v in result["values"]]
    print(key, "points", len(values), "max", max(values))
