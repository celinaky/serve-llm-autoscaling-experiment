#!/usr/bin/env python3
"""One vLLM engine outside Ray Serve, on a GPU that Ray keeps reserved.

For single-replica profiling while the other GPUs run harness experiments.
The harness owns Serve (deploy and teardown call serve.shutdown()), so a
profiling replica inside Serve would be torn down by the next experiment.
This instead:

  1. starts a detached Ray actor that holds num_gpus=1 and does nothing, so
     Ray never places a Serve replica on that GPU (experiments see 7 GPUs);
  2. runs `vllm serve` on that physical GPU with the same engine settings as
     experiments/h100_*.yaml and the Ray fork's tracing scheduler, writing
     traces to their own directory (pass it to load_breakdown.py --traces).

Engine request IDs come from the X-Request-Id header AIPerf and
side_probe.py send (chatcmpl-<id>-<8 hex>), as under Serve, so
tools/load_breakdown.py joins them the same way.

usage: python tools/standalone_engine.py start [--port 8100] [--trace-dir DIR]
       python tools/standalone_engine.py status
       python tools/standalone_engine.py stop
"""
import argparse, json, os, signal, subprocess, sys, time

import requests

ACTOR = "standalone-engine-gpu"
NAMESPACE = "standalone-engine"
STATE = os.path.expanduser("~/.standalone_engine.json")
TRACE_ROOT = "/mnt/cluster_storage/serve-llm-autoscaling-experiment/standalone-engine-traces"
MODEL = "Qwen/Qwen3.6-27B-FP8"
# Same engine as experiments/h100_workday_curve_prefill_mean.yaml.
ENGINE_ARGS = [
    "--tensor-parallel-size", "1",
    "--max-model-len", "262144",
    "--gpu-memory-utilization", "0.92",
    "--max-num-seqs", "512",
    "--max-num-batched-tokens", "16384",
    "--enable-prefix-caching",
    "--kv-cache-dtype", "fp8",
    "--language-model-only",
    # The harness sets this too: AIPerf records cached tokens from it.
    "--enable-prompt-tokens-details",
    "--async-scheduling",
    "--scheduler-cls", "ray.llm._internal.serve.engines.vllm.scheduler_trace.TracingAsyncScheduler",
]


def ray_init():
    import ray
    ray.init(address="auto", namespace=NAMESPACE, logging_level="ERROR")
    return ray


def reserve_gpu():
    ray = ray_init()

    @ray.remote(num_gpus=1, accelerator_type="H100")
    class GpuHold:
        def gpu(self):
            return [int(g) for g in ray.get_gpu_ids()]

    try:
        actor = ray.get_actor(ACTOR)
    except ValueError:
        actor = GpuHold.options(name=ACTOR, lifetime="detached", max_restarts=-1,
                                num_cpus=0).remote()
    return ray.get(actor.gpu.remote(), timeout=120)[0]


def release_gpu():
    ray = ray_init()
    try:
        ray.kill(ray.get_actor(ACTOR), no_restart=True)
        print(f"released the GPU held by {ACTOR}")
    except ValueError:
        print(f"no {ACTOR} actor")


def start(a):
    if os.path.exists(STATE):
        sys.exit(f"{STATE} exists; run `stop` first")
    gpu = reserve_gpu()
    trace_dir = a.trace_dir or f"{TRACE_ROOT}/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}"
    os.makedirs(trace_dir, exist_ok=True)
    log = f"{trace_dir}/vllm_serve.log"
    env = {k: v for k, v in os.environ.items() if not k.startswith("RAY_ADDRESS")}
    env.update(CUDA_VISIBLE_DEVICES=str(gpu), RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR=trace_dir)
    cmd = ["vllm", "serve", MODEL, "--port", str(a.port), *ENGINE_ARGS]
    if a.kv_events:
        # Same publisher the Ray branch configures for KV-aware routing, so a
        # Dynamo SelectionService can subscribe to this engine's cache events.
        cmd += ["--kv-events-config", json.dumps({"enable_kv_cache_events": True, "publisher": "zmq",
                                                  "endpoint": f"tcp://*:{a.kv_events_port}",
                                                  "replay_endpoint": f"tcp://*:{a.kv_events_port + 1}"})]
    proc = subprocess.Popen(cmd, env=env, stdout=open(log, "a"), stderr=subprocess.STDOUT,
                            stdin=subprocess.DEVNULL, start_new_session=True, cwd="/tmp")
    state = {"pid": proc.pid, "gpu": gpu, "port": a.port, "trace_dir": trace_dir, "log": log,
             "cmd": cmd, "started": time.time()}
    json.dump(state, open(STATE, "w"), indent=2)
    print(json.dumps(state, indent=2))
    url = f"http://127.0.0.1:{a.port}/v1/models"
    deadline = time.time() + a.timeout
    while time.time() < deadline:
        if proc.poll() is not None:
            sys.exit(f"vllm serve exited with {proc.returncode}; see {log}")
        try:
            if requests.get(url, timeout=5).ok:
                print(f"ready on port {a.port}, GPU {gpu}")
                return
        except requests.RequestException:
            pass
        time.sleep(5)
    sys.exit(f"not ready after {a.timeout}s; see {log}")


def status(_):
    if not os.path.exists(STATE):
        print("not running")
        return
    s = json.load(open(STATE))
    try:
        os.kill(s["pid"], 0)
        alive = True
    except OSError:
        alive = False
    print(json.dumps({**s, "alive": alive}, indent=2))


def stop(_):
    if os.path.exists(STATE):
        s = json.load(open(STATE))
        try:
            os.killpg(s["pid"], signal.SIGTERM)
            for _ in range(60):
                os.killpg(s["pid"], 0)
                time.sleep(1)
            os.killpg(s["pid"], signal.SIGKILL)
        except OSError:
            pass
        os.remove(STATE)
        print(f"stopped vllm serve (pid {s['pid']})")
    release_gpu()


ap = argparse.ArgumentParser()
sub = ap.add_subparsers(dest="cmd", required=True)
p = sub.add_parser("start")
p.add_argument("--port", type=int, default=8100)
p.add_argument("--kv-events", action="store_true", help="publish vLLM KV-cache events over ZMQ (PUB on --kv-events-port, replay on +1)")
p.add_argument("--kv-events-port", type=int, default=5557)
p.add_argument("--trace-dir")
p.add_argument("--timeout", type=int, default=1200)
sub.add_parser("status")
sub.add_parser("stop")
a = ap.parse_args()
{"start": start, "status": status, "stop": stop}[a.cmd](a)
