"""Automatic, offline AgentX diagnostics from saved requests and engine traces."""
from __future__ import annotations

import datetime as dt
import html
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

from ..artifacts import write_json
from ..config import AgentXWorkloadConfig, CONTINUOUS_MODES, ExperimentConfig

TRACE_ENV = "RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR"
_ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")


def prepare_tracing(config: ExperimentConfig, root: Path) -> ExperimentConfig:
    """Give each new AgentX run its own trace directory and replica config."""
    if not (config.analysis.generate_plots and config.analysis.diagnostic_plots
            and config.benchmark.mode in CONTINUOUS_MODES
            and isinstance(config.benchmark.workload, AgentXWorkloadConfig)):
        return config
    config = config.model_copy(deep=True)
    base = config.analysis.scheduler_trace_dir or os.environ.get(TRACE_ENV)
    directory = Path(base) / root.name if base else root / "telemetry/scheduler-traces"
    directory = directory.resolve()
    directory.mkdir(parents=True, exist_ok=True)
    config.analysis.scheduler_trace_dir = directory
    return config


def _request_ids(root: Path) -> set[str]:
    ids = set()
    for path in root.glob("benchmark/*/profile_export.jsonl"):
        for line in path.open():
            try:
                meta = json.loads(line)["metadata"]
                if meta.get("benchmark_phase") == "profiling" and meta.get("x_request_id"):
                    ids.add(meta["x_request_id"])
            except (ValueError, KeyError, TypeError):
                continue
    return ids


def select_traces(root: Path, config: ExperimentConfig,
                  traces_dir: Path | None = None) -> list[Path]:
    """Select only engines that actually saw this run's request IDs.

    Shared legacy trace directories can contain other simultaneous runs. A
    timestamp alone cannot safely distinguish their engine load.
    """
    saved = root / "analysis/scheduler_trace_inputs.json"
    directory = traces_dir or getattr(config.analysis, "scheduler_trace_dir", None) or os.environ.get(TRACE_ENV)
    saved_paths = json.loads(saved.read_text()) if saved.exists() else []
    if traces_dir is None and saved_paths:
        candidates = [Path(p) for p in saved_paths]
    elif directory:
        candidates = list(Path(directory).glob("*.jsonl"))
    else:
        candidates = list((root / "telemetry/scheduler-traces").glob("*.jsonl"))
    manifest = root / "manifest.json"
    start = None
    if manifest.exists():
        raw = json.loads(manifest.read_text()).get("started_at")
        if raw:
            start = dt.datetime.fromisoformat(raw).timestamp()
    ids = _request_ids(root)
    if not ids:
        return []
    selected = []
    for path in sorted({p.resolve() for p in candidates if p.is_file()}):
        if start is not None and path.stat().st_mtime < start:
            continue
        with path.open() as stream:
            for line in stream:
                try:
                    event = json.loads(line)
                except ValueError:
                    continue
                if event.get("event") != "request_queued":
                    continue
                rid = event.get("request_id", "")
                match = _ENGINE_ID.match(rid)
                if (match.group(1) if match else rid) in ids:
                    selected.append(path)
                    break
    return selected


def _run_plot(root: Path, name: str, args: list[str], output: Path) -> dict[str, Any]:
    log = root / "analysis" / f"{name}_generation.log"
    output.unlink(missing_ok=True)  # Never present an old chart as a new success.
    command = [sys.executable, "-m", f"serve_llm_autoscaling.diagnostics.{name}",
               str(root), *args]
    try:
        with log.open("w") as stream:
            result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT,
                                    timeout=600, check=False)
        if result.returncode or not output.is_file() or output.stat().st_size == 0:
            return {"status": "failed", "log": log.name,
                    "reason": f"plot command exited {result.returncode}; see {log.name}"}
        return {"status": "generated", "file": output.name, "log": log.name}
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"status": "failed", "log": log.name,
                "reason": f"{type(exc).__name__}: {exc}"}


def _gallery(root: Path, report: dict[str, Any], title: str) -> None:
    out = root / "analysis"
    sections = []
    if (out / "autoscaling_timeline.png").exists():
        sections.append('<section><h2>Scaling timeline</h2><a href="autoscaling_timeline.png"><img src="autoscaling_timeline.png" alt="Scaling timeline"></a></section>')
    for name, step in report["plots"].items():
        caption = html.escape(step["title"])
        if step["status"] == "generated":
            filename = html.escape(step["file"])
            sections.append(f'<section><h2>{caption}</h2><a href="{filename}"><img src="{filename}" alt="{caption}" loading="lazy"></a><p>{html.escape(step["description"])}</p></section>')
        else:
            sections.append(f'<section><h2>{caption}</h2><p>{html.escape(step["status"] + ": " + step["reason"])}</p></section>')
    warnings = ''.join(f'<li>{html.escape(w)}</li>' for w in report["warnings"])
    session_link = '<a href="session_lanes.csv">Session assignments CSV</a>' if (out / 'session_lanes.csv').exists() else ''
    request_link = '<a href="load_requests.csv">Per-request engine data</a>' if (out / 'load_requests.csv').exists() else ''
    body = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><style>body{{font:16px/1.5 system-ui,sans-serif;margin:32px auto;max-width:1200px;padding:0 20px;color:#18222e;background:#fafaf8}}h1{{font-size:30px}}section{{margin:36px 0;padding:20px;background:white;border:1px solid #ddd;border-radius:10px}}img{{width:100%;height:auto}}code{{overflow-wrap:anywhere}}a{{color:#205bb0}}</style></head><body><h1>{html.escape(title)}</h1><p><code>{html.escape(root.name)}</code></p><p>Saved post-run report. Click any chart to view its full size. Latency plots use successful requests matched to scheduler traces; inspect the timeline/evaluation for failures.</p><ul>{warnings}</ul>{''.join(sections)}<p>Session bars end at the last observed request, not verified session completion. Colors identify the first root replica; the session CSV lists every observed root replica. {session_link}</p><p><a href="diagnostic_plots.json">Plot generation status and trace inputs</a> · <a href="autoscaling_evaluation.json">Autoscaling evaluation</a> {request_link}</p></body></html>'''
    (out / "index.html").write_text(body)


def generate_diagnostics(root: Path, config: ExperimentConfig, *,
                         traces_dir: Path | None = None) -> dict[str, Any]:
    root = root.resolve()
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {"status": "generated", "warnings": [], "plots": {},
                              "trace_files": [], "gallery": "index.html"}
    end = config.benchmark.duration_s
    suffix = f"0_{int(end)}"
    lookback = config.deployment.autoscaling.policy.kwargs.get("upscale_lookback_s") or 60
    specifications = [
        ("arrival_stats_stack", f"arrival_stats_stack180_{suffix}.png",
         ["--window", "180", "--end", str(end)], "Arrival statistics · 180s windows",
         "Successful requests matched to traces: p95, p90 and mean load at arrival, engine queue time and client TTFT; request counts under each point."),
        ("replica_percentile_lines", f"replica_percentiles180_{suffix}.png",
         ["--window", "180", "--min-samples", "10", "--end", str(end)], "Replica percentiles · 180s windows",
         "Matched successful requests per replica: p95, p90 and mean arrival load; windows with fewer than 10 requests are blank."),
        ("policy_vs_engine_mean", f"policy_vs_engine_mean{int(lookback)}_{suffix}.png",
         ["--lookback", str(lookback), "--end", str(end)], "Policy vs engine mean",
         f"Router policy signal and actual remaining engine prefill, averaged over {lookback:g} seconds."),
        ("session_lanes", f"session_lanes_{suffix}.png", [], "Session lanes and replica placement",
         "Observed first-to-last request spans with recorded lane IDs when available; inferred rows are explicitly labeled."),
    ]
    # A reanalysis must never leave an old chart/CSV looking current when its
    # input traces disappear or the join fails.
    for filename in ["load_requests.csv", "load_breakdown.csv", "load_breakdown_steps.csv",
                     "load_breakdown.png", "session_lanes.csv",
                     *(spec[1] for spec in specifications)]:
        (out / filename).unlink(missing_ok=True)
    try:
        traces = select_traces(root, config, traces_dir)
        report["trace_files"] = [str(p) for p in traces]
        # Record the inputs even when none were found; the CLI override can
        # explicitly replace this selection on a later reanalysis.
        write_json(out / "scheduler_trace_inputs.json", report["trace_files"])
        linked = out / "diagnostic-traces"
        linked.mkdir(exist_ok=True)
        for path in linked.glob("*.jsonl"):
            path.unlink()
        for path in traces:
            target = linked / path.name
            if target.exists():
                raise ValueError(f"Duplicate trace basename {path.name}; engine identities must remain distinct")
            target.symlink_to(path)
        if not traces:
            prerequisite = {"status": "skipped", "reason": "No scheduler traces matched this run's request IDs. Configure analysis.scheduler_trace_dir on shared storage before running, or reanalyze with --traces-dir DIR."}
        else:
            prerequisite = _run_plot(root, "load_breakdown", ["--traces", str(linked)], out / "load_requests.csv")
        report["load_breakdown"] = prerequisite
        for name, filename, args, title, description in specifications:
            output = out / filename
            if prerequisite["status"] != "generated" and not (name == "policy_vs_engine_mean" and traces):
                output.unlink(missing_ok=True)
                step = {"status": "skipped", "reason": "Scheduler request join unavailable: " + prerequisite.get("reason", "")}
            else:
                if name == "policy_vs_engine_mean":
                    args = [*args, "--traces", str(linked)]
                step = _run_plot(root, name, [*args, "--out", str(output)], output)
            report["plots"][name] = {**step, "title": title, "description": description}
            if step["status"] != "generated":
                report["warnings"].append(f"{title}: {step['reason']}")
        statuses = {p["status"] for p in report["plots"].values()}
        report["status"] = ("generated" if statuses == {"generated"}
                            else "skipped" if statuses == {"skipped"}
                            else "failed" if statuses == {"failed"} else "partial")
    except Exception as exc:
        report["status"] = "failed"
        report["warnings"].append(f"Diagnostic plots failed: {type(exc).__name__}: {exc}")
    write_json(out / "diagnostic_plots.json", report)
    _gallery(root, report, config.name)
    return report
