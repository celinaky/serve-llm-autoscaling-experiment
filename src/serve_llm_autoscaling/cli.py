from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from .artifacts import format_evaluation
from .backend import (
    RayServeBackend,
    check_autoscaling_support,
    check_routing_support,
    verify_autoscaling_policy,
    verify_topology,
)
from .config import load_config
from .runner import environment_check, run_experiment, warm_compile_caches


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="autoscale-harness")
    subparsers = parser.add_subparsers(dest="command", required=True)

    subparsers.add_parser("check", help="Validate the local and Ray environments")
    for name, text in (
        ("validate", "Print the resolved experiment config"),
        ("status", "Print the Serve status of the cluster"),
        ("teardown", "Shut down the Serve application"),
    ):
        command = subparsers.add_parser(name, help=text)
        command.add_argument("config", type=Path)
    deploy = subparsers.add_parser(
        "deploy", help="Deploy the application without running a workload")
    deploy.add_argument("config", type=Path)
    deploy.add_argument("--replace", action="store_true")
    warm = subparsers.add_parser(
        "warm", help="Warm every GPU's vLLM compile cache for this config's engine")
    warm.add_argument("config", type=Path)
    warm.add_argument("--replicas", type=int,
                      help="replicas to start (default: one per GPU in the cluster)")
    warm.add_argument("--replace", action="store_true")
    run = subparsers.add_parser(
        "run", help="Deploy, replay the workload, analyze and plot, tear down")
    run.add_argument("config", type=Path)
    run.add_argument("--replace", action="store_true",
                     help="replace Serve applications that are already running")
    run.add_argument("--keep-deployment", action="store_true", default=None)
    run.add_argument("--warm", action="store_true",
                     help="warm every GPU's compile cache before the run")
    analyze = subparsers.add_parser(
        "analyze", help="Regenerate a run's analysis and plots from its saved artifacts"
    )
    analyze.add_argument("run_dir", type=Path)
    analyze.add_argument("--window-s", type=float)
    analyze.add_argument("--tail-window-s", type=float)
    analyze.add_argument("--no-plot", action="store_true",
                         help="recompute the scorecard only, without the trace join and plots")
    analyze.add_argument("--traces-dir", type=Path,
                         help="Scheduler trace directory for diagnostic plots")
    compare = subparsers.add_parser(
        "compare",
        help="Tabulate several analyzed runs and plot them on one grid",
        description="Print one scorecard row per run, then draw their per-window "
                    "load at arrival, engine queue time and client TTFT on one grid.",
    )
    compare.add_argument("run_dirs", type=Path, nargs="+")
    compare.add_argument("--no-plot", action="store_true", help="print the table only")
    compare.add_argument("--labels", help="legend labels, one per run, separated by ';'")
    compare.add_argument("--window", type=float, help="window length in seconds (default 120)")
    compare.add_argument("--cols", help="subset of mean,p90,p95 to plot (default all)")
    compare.add_argument("--min-samples", type=int, help="fewest requests for a window's point")
    compare.add_argument("--start", type=float, help="first second to plot")
    compare.add_argument("--end", type=float, help="last second to plot")
    compare.add_argument("--out", type=Path, help="output PNG (default: beside the first run)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "check":
            print(json.dumps(environment_check(connect=True), indent=2, default=str))
            return 0
        if args.command == "analyze":
            from .analysis import analyze_run

            result = analyze_run(
                args.run_dir,
                window_s=args.window_s,
                tail_window_s=args.tail_window_s,
                plot=False if args.no_plot else None,
                traces_dir=args.traces_dir,
            )
            for warning in result["warnings"]:
                print(f"WARNING: {warning}", file=sys.stderr)
            print(f"Analysis artifacts: {args.run_dir / 'analysis'}")
            if evaluation := format_evaluation(args.run_dir):
                print(evaluation)
            diagnostics = result.get("diagnostic_plots") or {}
            if diagnostics:
                print(f"Diagnostic plots: {diagnostics['status']}")
                print(f"Plot gallery: {args.run_dir / 'analysis/index.html'}")
                if diagnostics["status"] in ("failed", "partial"):
                    return 1
            return 0
        if args.command == "compare":
            from .compare import format_comparison, plot_comparison

            print(format_comparison(args.run_dirs))
            if args.no_plot:
                return 0
            options = []
            for flag in ("labels", "window", "cols", "min_samples", "start", "end", "out"):
                value = getattr(args, flag)
                if value is not None:
                    options += [f"--{flag.replace('_', '-')}", str(value)]
            return plot_comparison(args.run_dirs, options)

        config = load_config(args.config)
        if args.command in ("deploy", "warm", "run"):
            # Before the first Ray import: vLLM import alias and no
            # workspace-wide runtime_env upload.
            from .vllm_compat import setup

            setup()
        if args.command == "validate":
            print(yaml.safe_dump(config.resolved_dict(), sort_keys=False))
        elif args.command == "status":
            backend = RayServeBackend(config)
            backend.connect()
            print(json.dumps(backend.status(), indent=2, default=str))
        elif args.command == "deploy":
            backend = RayServeBackend(config, replace=args.replace)
            print(json.dumps(backend.connect(), indent=2, default=str))
            print(json.dumps(check_routing_support(config), indent=2, default=str))
            print(json.dumps(check_autoscaling_support(config), indent=2, default=str))
            print(json.dumps(backend.deploy(), indent=2, default=str))
            ready = backend.wait_healthy()
            print(json.dumps(ready, indent=2, default=str))
            topology = verify_topology(config, ready["serve_status"])
            print(json.dumps(topology, indent=2, default=str))
            policy = verify_autoscaling_policy(config, backend.serve_details())
            print(json.dumps(policy, indent=2, default=str))
        elif args.command == "teardown":
            backend = RayServeBackend(config)
            backend.connect()
            backend.teardown()
        elif args.command == "warm":
            result = warm_compile_caches(config, replicas=args.replicas, replace=args.replace)
            print(json.dumps(result, indent=2, default=str))
        elif args.command == "run":
            root = run_experiment(
                config,
                args.config,
                replace=args.replace,
                keep_deployment=args.keep_deployment,
                warm=args.warm,
            )
            print(f"Run artifacts: {root}")
        return 0
    except Exception as exc:
        print(f"ERROR: {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

