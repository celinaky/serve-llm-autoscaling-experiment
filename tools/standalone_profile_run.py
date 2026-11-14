#!/usr/bin/env python3
"""Drive AIPerf (and side probes) at the standalone engine, then plot queue time.

Runs the same AIPerf fork, dataset and scenario as the harness's
agentx_concurrency_ramp mode against tools/standalone_engine.py, with the
concurrency series from a JSON spec (experiments/h100_queue_profile.json), and
tools/side_probe.py alongside. Lays the run out like a harness run
(benchmark/agentx-concurrency-ramp, benchmark/side-probe) so the analysis
tools work on it, then runs tools/load_breakdown.py with the engine's own
trace directory and tools/load_ttft_breakdown.py --bands split, whose
<out>/load_queue_by_size.png is queue time against uncached prefill load per
request size, with requests of <=704 uncached tokens (16384 mod the 1568-token
block) as their own group.

usage: python tools/standalone_profile_run.py experiments/h100_queue_profile.json [--name NAME]
"""
import argparse, json, os, shutil, subprocess, sys, time

AIPERF = "git+https://github.com/celinaky/aiperf@1c61dfbb7644b26c70ce21163f0b4cda893438b0"
MODEL = "Qwen/Qwen3.6-27B-FP8"
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

ap = argparse.ArgumentParser()
ap.add_argument("spec")
ap.add_argument("--name")
a = ap.parse_args()
os.chdir(REPO)
spec = json.load(open(a.spec))
engine = json.load(open(os.path.expanduser("~/.standalone_engine.json")))
url = f"http://127.0.0.1:{engine['port']}"
name = a.name or os.path.splitext(os.path.basename(a.spec))[0].replace("_", "-")
run = f"runs/{time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())}-{name}"
bench = f"{run}/benchmark/agentx-concurrency-ramp"
probe_dir = f"{run}/benchmark/side-probe"
os.makedirs(bench)
os.makedirs(probe_dir)
os.makedirs(f"{run}/analysis")
shutil.copy(a.spec, f"{run}/spec.json")
json.dump(engine, open(f"{run}/engine.json", "w"), indent=2)
series = os.path.abspath(f"{bench}/concurrency_series.json")
json.dump({"points": spec["points"]}, open(series, "w"), indent=2)

argv = ["uvx", "--python", "3.11", "--from", AIPERF, "aiperf", "profile",
        "--model", MODEL, "--tokenizer", MODEL, "--url", url, "--endpoint-type", "chat",
        "--benchmark-duration", str(float(spec["duration_s"])),
        "--benchmark-grace-period", str(float(spec["grace_period_s"])),
        "--artifact-dir", bench, "--export-level", "records", "--ui", "none", "--streaming",
        "--use-server-token-count", "--session-header", "X-Session-ID",
        "--scenario", "inferencex-agentx-mvp",
        "--public-dataset", "semianalysis_cc_traces_weka_062126_256k",
        "--concurrency", str(spec["target_concurrency"]), "--concurrency-series", series,
        "--trajectory-start-min-ratio", "0.0", "--trajectory-start-max-ratio", "0.0",
        "--random-seed", str(spec["seed"])]
json.dump({"argv": argv}, open(f"{bench}/command.json", "w"), indent=2)
print(f"run dir {run}", flush=True)

aiperf = subprocess.Popen(argv, stdout=open(f"{bench}/stdout.log", "w"),
                          stderr=open(f"{bench}/stderr.log", "w"))
# AIPerf spends its first seconds configuring; wait for the first request so the
# probes cover the same window.
deadline = time.time() + 600
prof = f"{bench}/profile_export.jsonl"
start = None
while time.time() < deadline and aiperf.poll() is None:
    out = open(f"{bench}/stdout.log").read() + open(f"{bench}/stderr.log").read()
    if "Profiling" in out or "profiling" in out or os.path.exists(prof):
        start = time.time()
        break
    time.sleep(2)
if start is None:
    start = time.time()
# "probe" (one prober, in benchmark/side-probe) or "probes" (several, in side-probe-<i>).
probes = [spec["probe"]] if spec.get("probe") else spec.get("probes", [])
until_ns = int((start + spec["duration_s"]) * 1e9)
probers = []
for i, probe in enumerate(probes):
    d = probe_dir if i == 0 else f"{probe_dir}-{i}"
    os.makedirs(d, exist_ok=True)
    probers.append(subprocess.Popen(
        [sys.executable, "tools/side_probe.py", "--out", d, "--until-ns", str(until_ns),
         "--url", f"{url}/v1/chat/completions", "--rate", str(probe["rate"]),
         "--min-new", str(probe["min_new"]), "--max-new", str(probe["max_new"]),
         "--prefix", str(probe["prefix"]), "--seed", str(probe["seed"])],
        stdout=open(f"{d}/stdout.log", "w"), stderr=subprocess.STDOUT))
rc = aiperf.wait()
for p in probers:
    p.wait()
print(f"aiperf exited {rc}", flush=True)

subprocess.run([sys.executable, "tools/load_breakdown.py", run, "--traces", engine["trace_dir"]], check=True)
subprocess.run([sys.executable, "tools/load_ttft_breakdown.py", "--out", f"{run}/analysis", run,
                "--bands", "split"], check=True)
print(f"done: {run}/analysis/load_queue_by_size.png")
