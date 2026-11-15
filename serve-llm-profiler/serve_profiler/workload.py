"""Invoke AIPerf; never reimplement AgentX conversation replay."""

from __future__ import annotations

import os
import signal
import subprocess
import time
from pathlib import Path

from .analysis import write_json

SESSION_POOL_SIZE = 8
COHORT_SIZES = (8,)
PHASE_SECONDS = 1800
GRACE_SECONDS = 300
SEED = 42
DATASET = "semianalysis_cc_traces_weka_062126_256k"
# Use the cohort-enabled fork beside this profiler until it has a published pin.
AIPERF_SOURCE = str(Path(__file__).resolve().parents[2] / "aiperf")


def prefix(config: dict) -> list[str]:
    source = config.get("runtime", {}).get("aiperf_source", AIPERF_SOURCE)
    # AIPerf gets its own Python environment; Ray stays on the image's Python.
    argv = ["uvx", "--python", "3.11", "--from", source]
    if Path(source).is_dir():
        # Local source edits must not reuse an older uvx build of the fork.
        argv.extend(["--refresh-package", "aiperf"])
    return [*argv, "aiperf"]


def preflight(config: dict, root: Path) -> None:
    required = (
        "--ignore-trace-delays",
        "--agentx-cohort-size",
        "--unsafe-override",
        "--scenario",
        "--trajectory-start-min-ratio",
        "--session-header",
    )
    with (root / "aiperf-help.log").open("w") as log:
        subprocess.run(
            [*prefix(config), "profile", "--help"],
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=True,
        )
    help_text = (root / "aiperf-help.log").read_text()
    missing = [flag for flag in required if flag not in help_text]
    if missing:
        raise RuntimeError(f"Selected AIPerf does not support required flags: {missing}")
    version = subprocess.check_output(
        [*prefix(config), "--version"], text=True, timeout=60
    )
    write_json(
        root / "aiperf.json",
        {
            "source": prefix(config)[4],
            "version": version.strip(),
            "delay_mode": "ignore_trace_delays",
            "scenario_override": True,
        },
    )


def command(
    config: dict, url: str, out: Path, cohort_size: int, phase: int
) -> list[str]:
    deployment = config["deployment"]
    model = deployment["model_id"]
    tokenizer = deployment.get("tokenizer") or deployment.get("model_source") or model
    return [
        *prefix(config),
        "profile",
        "--model",
        model,
        "--tokenizer",
        tokenizer,
        "--url",
        url,
        "--endpoint-type",
        "chat",
        "--streaming",
        "--scenario",
        "inferencex-agentx-mvp",
        "--unsafe-override",
        "--ignore-trace-delays",
        "--public-dataset",
        DATASET,
        "--concurrency",
        str(SESSION_POOL_SIZE),
        "--agentx-cohort-size",
        str(cohort_size),
        "--benchmark-duration",
        str(PHASE_SECONDS),
        "--benchmark-grace-period",
        str(GRACE_SECONDS),
        "--trajectory-start-min-ratio",
        "0",
        "--trajectory-start-max-ratio",
        "0",
        "--session-header",
        "X-Session-ID",
        "--use-server-token-count",
        "--cache-bust",
        "first_turn_prefix",
        "--extra-inputs",
        "ignore_eos:true",
        "--random-seed",
        str(SEED + phase),
        "--artifact-dir",
        str(out.resolve()),
        "--export-level",
        "records",
        "--ui",
        "none",
    ]


def run_phases(config: dict, url: str, root: Path) -> None:
    manifest = {
        "style": "AgentX replay without recorded think times; drain-to-drain cohorts",
        "dataset": DATASET,
        "phase_duration_s": PHASE_SECONDS,
        "session_tree_pool_size": SESSION_POOL_SIZE,
        "cold_session_policy": "replace only after the previous tree drains; no cold quota",
        "phases": [],
    }
    write_json(root / "workload.json", manifest)
    for phase, cohort_size in enumerate(COHORT_SIZES):
        out = root / "workload" / f"cohort-{cohort_size}"
        out.mkdir(parents=True)
        argv = command(config, url, out, cohort_size, phase)
        write_json(out / "command.json", {"argv": argv})
        record = {
            "concurrency": SESSION_POOL_SIZE,
            "cohort_size": cohort_size,
            "seed": SEED + phase,
            "launched_at_s": time.time(),
            "status": "running",
        }
        manifest["phases"].append(record)
        write_json(root / "workload.json", manifest)
        print(
            f"AIPerf: pool {SESSION_POOL_SIZE}, cohort {cohort_size}, {PHASE_SECONDS}s issuance",
            flush=True,
        )
        # A process group allows cancellation to include uvx and AIPerf workers.
        with (out / "console.log").open("w") as log:
            process = subprocess.Popen(
                argv, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
            )
            try:
                # Includes dataset loading/warmup; workload duration remains PHASE_SECONDS.
                code = process.wait(timeout=PHASE_SECONDS + GRACE_SECONDS + 1800)
            except BaseException:
                os.killpg(process.pid, signal.SIGTERM)
                try:
                    process.wait(timeout=20)
                except subprocess.TimeoutExpired:
                    os.killpg(process.pid, signal.SIGKILL)
                    process.wait()
                record.update(status="interrupted", exited_at_s=time.time())
                write_json(root / "workload.json", manifest)
                raise
        record.update(
            returncode=code,
            exited_at_s=time.time(),
            status="completed" if code == 0 else "failed",
        )
        write_json(root / "workload.json", manifest)
        if code:
            raise RuntimeError(
                f"AIPerf failed at cohort {cohort_size}; see {out / 'console.log'}"
            )
        files = list(out.glob("profile_export.jsonl"))
        if not files or not files[0].stat().st_size:
            raise RuntimeError(f"AIPerf produced no client records at {out}")
