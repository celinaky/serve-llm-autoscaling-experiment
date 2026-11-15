"""Run a fixed 30-minute AgentX profile, or reanalyze saved scheduler traces."""
from __future__ import annotations

import argparse
import datetime as dt
import json
import shutil
import signal
import sys
import uuid
from pathlib import Path

import yaml

from .analysis import analyze, write_json


def run(config_path: Path) -> Path:
    from .deployment import Deployment, load_config, resolved_config
    from .workload import preflight, run_phases

    config = load_config(config_path)
    root = Path("runs") / (
        dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        + "-"
        + uuid.uuid4().hex[:8]
    )
    root = root.resolve()
    (root / "traces").mkdir(parents=True)
    shutil.copy2(config_path, root / "input_config.yaml")
    config, name, route = resolved_config(config, root)
    (root / "resolved_config.yaml").write_text(yaml.safe_dump(config, sort_keys=False))
    print(f"Run artifacts: {root}", flush=True)
    manifest = {"status": "running", "stage": "connect", "application": name}
    write_json(root / "run.json", manifest)
    deployment = None
    error = None

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Signal {signum}")

    previous = signal.signal(signal.SIGTERM, interrupted)
    try:
        deployment = Deployment(config, root)
        deployment.connect()
        manifest["stage"] = "aiperf_preflight"
        write_json(root / "run.json", manifest)
        preflight(config, root)
        manifest["stage"] = "deploy"
        write_json(root / "run.json", manifest)
        deployment.start()
        manifest["stage"] = "workload"
        write_json(root / "run.json", manifest)
        run_phases(config, deployment.url, root)
    except BaseException as exc:
        error = exc
        manifest.update(
            status="interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
            error=f"{type(exc).__name__}: {exc}",
        )
    finally:
        # Stop the deployment first: shutdown flushes the scheduler trace writer.
        if deployment is not None:
            try:
                deployment.close()
            except Exception as exc:
                manifest["cleanup_error"] = f"{type(exc).__name__}: {exc}"
                if error is None:
                    error = exc
                    manifest["status"] = "failed"
        signal.signal(signal.SIGTERM, previous)
        write_json(root / "run.json", manifest)
    if list((root / "traces").glob("*.jsonl")):
        try:
            result = analyze(root)
            manifest["analysis"] = {
                "samples": result["fit"]["samples"],
                "fit_status": result["fit"]["status"],
            }
        except Exception as exc:
            manifest["analysis_error"] = f"{type(exc).__name__}: {exc}"
            if error is None:
                error = exc
    if error is None:
        manifest.update(status="completed", stage="done")
    elif manifest["status"] == "running":
        manifest["status"] = "failed"
    write_json(root / "run.json", manifest)
    if error is not None:
        raise error
    return root


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    profile = commands.add_parser(
        "run", help="Deploy one replica and run 8-request cohorts with eight session trees for 30 minutes"
    )
    profile.add_argument("config", type=Path)
    offline = commands.add_parser(
        "analyze", help="Reconstruct points from saved logs without Ray or a GPU"
    )
    offline.add_argument("run_dir", type=Path)
    offline.add_argument(
        "--traces", type=Path, help="External scheduler trace directory"
    )
    offline.add_argument(
        "--trace-only",
        action="store_true",
        help="Explicitly include all traced requests without client records",
    )
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            print(f"Completed: {run(args.config)}")
        else:
            result = analyze(args.run_dir, args.traces, trace_only=args.trace_only)
            print(
                json.dumps({"output": result["output"], "fit": result["fit"]}, indent=2)
            )
        return 0
    except KeyboardInterrupt:
        print("Interrupted; partial artifacts preserved", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
