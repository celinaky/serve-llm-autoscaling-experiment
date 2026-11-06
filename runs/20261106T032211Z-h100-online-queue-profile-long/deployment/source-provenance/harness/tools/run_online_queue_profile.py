#!/usr/bin/env python3
"""Run the native-metrics harness with finite side probes and saved provenance.

Launch with the online Ray source overlay on PYTHONPATH. This wrapper changes
no installed packages or original checkouts. It requires an idle Serve cluster;
the harness's normal existing-application check remains in force.
"""

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def snapshot(root, config_path, probes_path, overlay):
    destination = root / "deployment/source-provenance"
    destination.mkdir(parents=True, exist_ok=True)
    for label, repo in [
        ("ray", Path("/mnt/local_storage/ray-online-prefill")),
        ("harness", Path(__file__).resolve().parents[1]),
    ]:
        (destination / f"{label}.patch").write_bytes(
            subprocess.check_output(["git", "diff", "HEAD"], cwd=repo)
        )
        untracked = subprocess.check_output(
            ["git", "ls-files", "--others", "--exclude-standard"], cwd=repo, text=True
        ).splitlines()
        for relative in untracked:
            if relative.startswith("runs/"):
                continue
            source = repo / relative
            if source.is_file():
                target = destination / label / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, target)
    shutil.copy2(config_path, destination / "experiment.yaml")
    shutil.copy2(probes_path, destination / "probes.json")
    shutil.copy2(overlay / "sitecustomize.py", destination / "sitecustomize.py")


def supervise(config, config_path, probes_path, overlay, started, stop, state):
    processes = []
    try:
        root = None
        while not stop.wait(1):
            candidates = [
                p
                for p in config.runtime.results_dir.glob(f"*-{config.name}")
                if (p / "manifest.json").exists() and p.stat().st_mtime >= started - 1
            ]
            if candidates:
                root = sorted(candidates)[-1]
                state["run_directory"] = str(root.resolve())
                snapshot(root, config_path, probes_path, overlay)
                write_json(root / "deployment/profile_supervisor.json", state)
                print(f"PROFILE_RUN_DIRECTORY={root.resolve()}", flush=True)
                break
        if root is None:
            return
        records = root / "benchmark/agentx-concurrency-ramp/profile_export.jsonl"
        first_start = None
        while not stop.wait(1):
            if not records.exists():
                continue
            with records.open() as stream:
                for line in stream:
                    try:
                        first_start = json.loads(line)["metadata"]["request_start_ns"]
                    except (ValueError, KeyError):
                        continue
                    break
            if first_start is not None:
                break
        if first_start is None:
            return
        until = first_start + round(config.benchmark.duration_s * 1e9)
        for probe in json.loads(probes_path.read_text())["probes"]:
            output = root / "benchmark" / f"side-probe-{probe['name']}"
            output.mkdir(parents=True, exist_ok=True)
            argv = [
                sys.executable,
                str(Path(__file__).with_name("side_probe.py")),
                "--out",
                str(output),
                "--until-ns",
                str(until),
                "--url",
                config.runtime.endpoint_url.rstrip("/") + "/v1/chat/completions",
                "--model",
                config.deployment.model_id,
                "--max-tokens",
                "8",
            ]
            for field in ["rate", "min_new", "max_new", "prefix", "seed"]:
                argv += ["--" + field.replace("_", "-"), str(probe[field])]
            write_json(output / "command.json", {"argv": argv})
            with (output / "stdout.log").open("w") as log:
                process = subprocess.Popen(
                    argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
                )
            processes.append((probe["name"], process))
        state.update(
            probe_stop_s=until / 1e9,
            probe_processes={name: process.pid for name, process in processes},
        )
        write_json(root / "deployment/profile_supervisor.json", state)
        print(
            "PROFILE_PROBES_STARTED=" + json.dumps(state["probe_processes"]), flush=True
        )
        while not stop.wait(5):
            for name, process in processes:
                if process.poll() not in (None, 0):
                    print(
                        f"Side probe {name} exited with {process.returncode}",
                        flush=True,
                    )
            if all(process.poll() is not None for _, process in processes):
                break
    except Exception as exc:
        state["supervisor_error"] = repr(exc)
        print("PROFILE_SUPERVISOR_ERROR=" + repr(exc), flush=True)
    finally:
        for name, process in processes:
            if process.poll() is None:
                try:
                    process.wait(timeout=30)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGTERM)
                    process.wait(timeout=10)
            state.setdefault("probe_exit_codes", {})[name] = process.returncode
        if state.get("run_directory"):
            write_json(
                Path(state["run_directory"]) / "deployment/profile_supervisor.json",
                state,
            )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("config", type=Path)
    parser.add_argument("--probes", type=Path, required=True)
    parser.add_argument(
        "--overlay", type=Path, default=Path("/tmp/online-prefill-overlay")
    )
    args = parser.parse_args()
    environment = {
        "RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR": "",
        "RAY_SERVE_ENABLE_HA_PROXY": "1",
        "RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING": "1",
        "RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY": "1",
        "RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE": "8388608",
    }
    os.environ.update(environment)
    import ray
    from ray import serve
    from serve_llm_autoscaling.config import load_config
    from serve_llm_autoscaling.runner import run_experiment

    if "/ray-online-prefill/" not in str(serve.__file__):
        raise RuntimeError("Launch with the online source overlay on PYTHONPATH")
    config = load_config(args.config)
    ray.init(
        address="auto",
        ignore_reinit_error=True,
        runtime_env={
            "env_vars": {
                **environment,
                "PYTHONPATH": os.environ["PYTHONPATH"],
                "ONLINE_CALIBRATION_OVERLAY_REV": "long-profile-1",
            }
        },
    )
    state = dict(
        driver_pid=os.getpid(),
        started_at_s=time.time(),
        config=str(args.config.resolve()),
    )
    stop = threading.Event()
    watcher = threading.Thread(
        target=supervise,
        args=(
            config,
            args.config,
            args.probes,
            args.overlay,
            state["started_at_s"],
            stop,
            state,
        ),
    )
    watcher.start()
    try:
        root = run_experiment(config, args.config)
    finally:
        stop.set()
        watcher.join()
    # Regenerate verification after side probes have flushed their records.
    from serve_llm_autoscaling.calibration import render

    render(root, config.deployment.autoscaling.policy.kwargs["target_queue_time_ms"])
    write_json(root / "deployment/profile_supervisor.json", state)
    print("PROFILE_COMPLETED=" + str(root.resolve()), flush=True)


if __name__ == "__main__":
    main()
