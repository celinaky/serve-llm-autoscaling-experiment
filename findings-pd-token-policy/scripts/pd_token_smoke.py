"""GPU smoke test for the P/D token autoscaling prototype.

Deploys a KV-aware prefill/decode app (NIXL, direct streaming, HAProxy) where
the prefill deployment autoscales on the router's outstanding uncached prefill
tokens and the decode deployment on active decode requests, drives a burst of
long cold prompts, and records every 2s:

- each LLMRouter replica's per-deployment autoscaling snapshot,
- Serve's replica counts per deployment,

into ``<out_dir>/timeline.jsonl``, then prints a summary. Run on a node with
at least 4 GPUs; needs ai-dynamo, nixl and a cached model.

    python pd_token_smoke.py --model Qwen/Qwen3.6-27B-FP8 --out-dir runs/pd-smoke

The three direct-streaming environment variables are set here before Ray Serve
is imported, and passed to the controller, which does not inherit them.
"""

import argparse
import asyncio
import json
import os
import random
import sys
import threading
import time
from pathlib import Path

ENV = {
    "RAY_SERVE_ENABLE_HA_PROXY": "1",
    "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
    "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
    # Long prompts: let HAProxy forward the whole body to the router.
    "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE": str(8 * 1024 * 1024),
}
os.environ.update(ENV)

APP_NAME = "pd-token-smoke"
MODEL_ID = "pd-smoke"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.6-27B-FP8")
    p.add_argument("--out-dir", default="runs/pd-token-smoke")
    p.add_argument("--prompt-tokens", type=int, default=16000)
    p.add_argument("--output-tokens", type=int, default=400)
    p.add_argument("--concurrency", type=int, default=12)
    p.add_argument("--load-s", type=float, default=240)
    p.add_argument("--idle-s", type=float, default=150)
    p.add_argument("--prefill-target-tokens", type=int, default=60000)
    p.add_argument("--decode-target-requests", type=int, default=4)
    p.add_argument("--max-replicas", type=int, default=2)
    p.add_argument("--readiness-timeout-s", type=float, default=1800)
    p.add_argument("--keep", action="store_true", help="leave the app running")
    return p.parse_args()


def build_app(args):
    from ray.serve.llm import LLMConfig, build_pd_openai_app
    from ray.serve.llm.request_router import KVAwareRouter

    def llm_config(policy_kwargs):
        return LLMConfig(
            model_loading_config={"model_id": MODEL_ID, "model_source": args.model},
            engine_kwargs={
                "kv_transfer_config": {
                    "kv_connector": "NixlConnector",
                    "kv_role": "kv_both",
                },
                "max_model_len": 32768,
                "gpu_memory_utilization": 0.85,
                "enable_prefix_caching": True,
                "max_num_seqs": 32,
                # Chunk prefill so the prefill backlog is observable.
                "max_num_batched_tokens": 8192,
                "kv_cache_dtype": "fp8",
                "language_model_only": True,
            },
            # Hybrid (Mamba) models need the DS conv-state layout for NIXL
            # transfer of their SSM state; harmless for pure attention models.
            runtime_env={"env_vars": {"VLLM_SSM_CONV_STATE_LAYOUT": "DS"}},
            deployment_config={
                "autoscaling_config": {
                    "min_replicas": 1,
                    "initial_replicas": 1,
                    "max_replicas": args.max_replicas,
                    "target_ongoing_requests": 8,
                    "upscale_delay_s": 0,
                    "downscale_delay_s": 60,
                    "metrics_interval_s": 2,
                    "look_back_period_s": 10,
                    "policy": {
                        "policy_function": (
                            "ray.serve.experimental.llm_token_autoscaling_policy:"
                            "LLMTokenAutoscalingPolicy"
                        ),
                        "policy_kwargs": policy_kwargs,
                    },
                },
                "max_ongoing_requests": 128,
                "request_router_config": {"request_router_class": KVAwareRouter},
            },
        )

    return build_pd_openai_app(
        {
            "prefill_config": llm_config(
                {"target_prefill_tokens_per_replica": args.prefill_target_tokens},
            ),
            "decode_config": llm_config(
                {"target_decode_requests_per_replica": args.decode_target_requests},
            ),
        }
    )


def deploy(args):
    import ray
    from ray import serve

    ray.init(address="auto", ignore_reinit_error=True)
    status = serve.status()
    if status.applications:
        sys.exit(f"Serve already runs {sorted(status.applications)}; aborting.")
    # A controller started without the direct-streaming env must be replaced.
    serve.shutdown()
    serve.run(
        build_app(args),
        name=APP_NAME,
        route_prefix="/",
        controller_options={"runtime_env": {"env_vars": ENV}},
    )


def wait_ready(args, base_url):
    import requests

    deadline = time.monotonic() + args.readiness_timeout_s
    while time.monotonic() < deadline:
        try:
            r = requests.get(f"{base_url}/v1/models", timeout=5)
            if r.status_code == 200 and any(
                m.get("id") == MODEL_ID for m in r.json().get("data", [])
            ):
                return
        except Exception:
            pass
        time.sleep(5)
    sys.exit("deployment did not become ready")


def deployment_ids():
    from ray import serve
    from ray.serve._private.common import DeploymentID

    status = serve.status().applications[APP_NAME]
    prefill = next(n for n in status.deployments if n.startswith("Prefill:"))
    decode = next(n for n in status.deployments if n.startswith("Decode:"))
    return (
        DeploymentID(name=prefill, app_name=APP_NAME),
        DeploymentID(name=decode, app_name=APP_NAME),
    )


class Recorder(threading.Thread):
    """Samples router snapshots and Serve replica counts every ``period_s``."""

    def __init__(self, out_path, period_s=2.0):
        super().__init__(daemon=True)
        self.out_path = out_path
        self.period_s = period_s
        self.stop_event = threading.Event()
        self.rows = []
        self.t0 = time.time()
        self.marks = {}

    def mark(self, name):
        self.marks[name] = time.time() - self.t0

    def run(self):
        from ray import serve

        prefill_id, decode_id = deployment_ids()
        router = serve.get_deployment_handle("LLMRouter", app_name=APP_NAME)
        with open(self.out_path, "w") as f:
            while not self.stop_event.is_set():
                row = {"t": round(time.time() - self.t0, 1)}
                try:
                    snaps = {}
                    for label, dep in (("prefill", prefill_id), ("decode", decode_id)):
                        results = router.broadcast(
                            "get_autoscaling_snapshot", deployment_id=dep
                        ).results(timeout_s=5, return_exceptions=True)
                        snaps[label] = [r for r in results if isinstance(r, dict)]
                    row["snapshots"] = snaps
                except Exception as e:  # noqa: BLE001
                    row["snapshot_error"] = repr(e)
                try:
                    app = serve.status().applications[APP_NAME]
                    row["replicas"] = {
                        name: dict(d.replica_states)
                        for name, d in app.deployments.items()
                    }
                except Exception as e:  # noqa: BLE001
                    row["status_error"] = repr(e)
                self.rows.append(row)
                f.write(json.dumps(row) + "\n")
                f.flush()
                self.stop_event.wait(self.period_s)


WORDS = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu copper silver carbon helium neon argon krypton xenon radon"
).split()


def cold_prompt(rng, approx_tokens):
    # ~1.3 tokens per short word; random order defeats prefix caching.
    n = int(approx_tokens / 1.3)
    return " ".join(rng.choice(WORDS) for _ in range(n))


async def load(args, base_url, recorder):
    import httpx

    rng = random.Random(7)
    results = {"ok": 0, "failed": 0, "ttft_s": []}
    deadline = time.monotonic() + args.load_s

    async def worker(i):
        async with httpx.AsyncClient(timeout=httpx.Timeout(600, connect=30)) as c:
            while time.monotonic() < deadline:
                body = {
                    "model": MODEL_ID,
                    "prompt": cold_prompt(rng, args.prompt_tokens),
                    "max_tokens": args.output_tokens,
                    "stream": True,
                    "ignore_eos": True,
                    "temperature": 0.0,
                }
                t_start = time.monotonic()
                first = None
                try:
                    async with c.stream(
                        "POST", f"{base_url}/v1/completions", json=body
                    ) as r:
                        if r.status_code != 200:
                            results["failed"] += 1
                            await r.aread()
                            print(f"[load] {r.status_code}: {r.text[:200]}")
                            await asyncio.sleep(2)
                            continue
                        async for line in r.aiter_lines():
                            if first is None and line.startswith("data:"):
                                first = time.monotonic() - t_start
                    results["ok"] += 1
                    if first is not None:
                        results["ttft_s"].append(first)
                except Exception as e:  # noqa: BLE001
                    results["failed"] += 1
                    print(f"[load] error: {e!r}")
                    await asyncio.sleep(2)

    recorder.mark("load_start")
    await asyncio.gather(*(worker(i) for i in range(args.concurrency)))
    recorder.mark("load_end")
    return results


def summarize(recorder, results, out_dir):
    def running(row, prefix):
        for name, states in row.get("replicas", {}).items():
            if name.startswith(prefix):
                return states.get("RUNNING", 0)
        return None

    def snap_max(row, label, key):
        vals = [s.get(key) for s in row.get("snapshots", {}).get(label, [])]
        vals = [v for v in vals if isinstance(v, (int, float))]
        return max(vals) if vals else None

    lines = []
    lines.append(f"requests ok={results['ok']} failed={results['failed']}")
    if results["ttft_s"]:
        s = sorted(results["ttft_s"])
        lines.append(
            f"TTFT p50={s[len(s)//2]:.2f}s p90={s[int(len(s)*0.9)]:.2f}s "
            f"max={s[-1]:.2f}s"
        )
    lines.append(f"marks: {recorder.marks}")
    prefill_tokens = [
        (r["t"], snap_max(r, "prefill", "outstanding_effective_prefill_tokens"))
        for r in recorder.rows
    ]
    decode_active = [
        (r["t"], snap_max(r, "decode", "active_decode_requests")) for r in recorder.rows
    ]
    decode_pending = [
        (r["t"], snap_max(r, "decode", "pending_prefill_requests"))
        for r in recorder.rows
    ]
    prefill_running = [(r["t"], running(r, "Prefill:")) for r in recorder.rows]
    decode_running = [(r["t"], running(r, "Decode:")) for r in recorder.rows]

    def peak(series):
        vals = [v for _, v in series if v is not None]
        return max(vals) if vals else None

    lines.append(
        f"peak prefill outstanding tokens={peak(prefill_tokens)} "
        f"peak active decodes={peak(decode_active)} "
        f"peak pending(decode)={peak(decode_pending)}"
    )
    lines.append(
        f"prefill RUNNING max={peak(prefill_running)} decode RUNNING max={peak(decode_running)}"
    )

    def transitions(series, label):
        last = None
        out = []
        for t, v in series:
            if v is not None and v != last:
                out.append(f"t={t:.0f}s {label}={v}")
                last = v
        return out

    lines.append("prefill replica transitions: " + "; ".join(transitions(prefill_running, "P")))
    lines.append("decode replica transitions: " + "; ".join(transitions(decode_running, "D")))
    # Final snapshot should be fully drained.
    last = recorder.rows[-1] if recorder.rows else {}
    lines.append(f"final snapshots: {json.dumps(last.get('snapshots'))}")
    text = "\n".join(lines)
    print(text)
    Path(out_dir, "summary.txt").write_text(text + "\n")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_url = "http://127.0.0.1:8000"

    t0 = time.time()
    deploy(args)
    print(f"[smoke] deployed in {time.time() - t0:.0f}s; waiting for readiness")
    wait_ready(args, base_url)
    print(f"[smoke] ready after {time.time() - t0:.0f}s")

    recorder = Recorder(out_dir / "timeline.jsonl")
    recorder.start()
    time.sleep(6)
    results = asyncio.run(load(args, base_url, recorder))
    print(f"[smoke] load done: {results['ok']} ok, {results['failed']} failed; idling")
    time.sleep(args.idle_s)
    recorder.mark("idle_end")
    recorder.stop_event.set()
    recorder.join(timeout=30)
    summarize(recorder, results, out_dir)
    if not args.keep:
        from ray import serve

        serve.shutdown()


if __name__ == "__main__":
    main()
