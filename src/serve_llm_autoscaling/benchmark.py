from __future__ import annotations

import json
import shlex
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import (
    AIPERF_SOURCE_DEFAULT,
    SESSION_HEADER,
    WORKLOAD_DIR,
    ExperimentConfig,
    WorkloadConfig,
)
from .windows import profiling_qps_summary

AGENTX_SCENARIO = "inferencex-agentx-mvp"
CONCURRENCY_SERIES_FILE = "concurrency_series.json"


def aiperf_command(source: str = AIPERF_SOURCE_DEFAULT) -> list[str]:
    # AIPerf needs Python 3.11+, which the Ray image's driver may not have,
    # so it runs in its own uvx-managed environment.
    if shutil.which("uvx") is None:
        raise RuntimeError("uvx not on PATH; install uv to run AIPerf")
    return ["uvx", "--python", "3.11", "--from", source, "aiperf"]


def _metric_value(data: dict[str, Any], key: str, stat: str = "avg") -> Any:
    value = data.get(key)
    if isinstance(value, dict):
        return value.get(stat)
    return value


def normalize_aiperf(data: dict[str, Any]) -> dict[str, Any]:
    errors = data.get("error_summary") or []
    failed_requests = _metric_value(data, "error_request_count")
    if failed_requests is None:
        failed_requests = sum(
            int(item.get("count", 0)) for item in errors if isinstance(item, dict)
        )
    return {
        "request_count": _metric_value(data, "request_count"),
        "failed_requests": failed_requests,
        "request_throughput": _metric_value(data, "request_throughput"),
        "output_token_throughput": _metric_value(data, "output_token_throughput"),
        "total_token_throughput": _metric_value(data, "total_token_throughput"),
        "mean_ttft_ms": _metric_value(data, "time_to_first_token", "avg"),
        "p50_ttft_ms": _metric_value(data, "time_to_first_token", "p50"),
        "p90_ttft_ms": _metric_value(data, "time_to_first_token", "p90"),
        "p99_ttft_ms": _metric_value(data, "time_to_first_token", "p99"),
        "mean_tpot_ms": _metric_value(data, "inter_token_latency", "avg"),
        "p50_tpot_ms": _metric_value(data, "inter_token_latency", "p50"),
        "p90_tpot_ms": _metric_value(data, "inter_token_latency", "p90"),
        "p99_tpot_ms": _metric_value(data, "inter_token_latency", "p99"),
        "mean_e2el_ms": _metric_value(data, "request_latency", "avg"),
        "p99_e2el_ms": _metric_value(data, "request_latency", "p99"),
        # Present only when the server reports prompt_tokens_details.
        "total_prompt_cache_read_tokens": _metric_value(
            data, "total_usage_prompt_cache_read_tokens"
        ),
        "prompt_cache_read_pct": _metric_value(
            data, "overall_usage_prompt_cache_read_pct"
        ),
    }


def workload_row_fields(workload: WorkloadConfig) -> dict[str, Any]:
    """Summary-row fields describing the configured session curve."""
    return {
        "target_concurrency": workload.target_concurrency,
        "concurrency_series": [p.model_dump() for p in workload.concurrency_series],
    }


@dataclass
class AIPerfRunner:
    """Replays the AgentX session curve against the deployment with AIPerf."""

    config: ExperimentConfig
    benchmark_root: Path

    def build_command(self, artifact_dir: Path) -> list[str]:
        """AgentX replay following the session-concurrency curve.

        The scenario owns ignore_eos, cache busting and warmup.
        """
        benchmark = self.config.benchmark
        deployment = self.config.deployment
        workload = benchmark.workload
        cmd = [
            *aiperf_command(self.config.runtime.aiperf_source),
            "profile",
            "--model", deployment.model_id,
            "--tokenizer", str(deployment.tokenizer),
            "--url", self.config.runtime.endpoint_url,
            "--endpoint-type", "chat",
            "--benchmark-duration", str(benchmark.duration_s),
            "--benchmark-grace-period", str(benchmark.grace_period_s),
            "--artifact-dir", str(artifact_dir),
            "--export-level", "records",
            "--ui", "none",
            "--streaming",
            "--use-server-token-count",
        ]
        if deployment.routing.policy == "kv_aware":
            # AIPerf sends one ID per session. The router scores the first
            # turn on prompt overlap and load, then pins the session's later
            # turns to that replica.
            cmd.extend(["--session-header", SESSION_HEADER])
        cmd.extend([
            "--scenario", AGENTX_SCENARIO,
            "--public-dataset", workload.public_dataset,
            "--concurrency", str(workload.target_concurrency),
            # AIPerf refuses series paths with symlinked components; resolve them.
            "--concurrency-series",
            str((artifact_dir / CONCURRENCY_SERIES_FILE).resolve()),
            "--trajectory-start-min-ratio", str(workload.trajectory_start_min_ratio),
            "--trajectory-start-max-ratio", str(workload.trajectory_start_max_ratio),
            "--random-seed", str(workload.seed),
        ])
        if workload.unsafe_override:
            cmd.append("--unsafe-override")
        cmd.extend(benchmark.extra_args)
        return cmd

    def write_concurrency_series(self, path: Path) -> None:
        """The session curve in the JSON shape AIPerf's flag reads."""
        points = [
            {"time_s": p.time_s, "concurrency": p.sessions}
            for p in self.config.benchmark.workload.concurrency_series
        ]
        with path.open("w") as fh:
            json.dump({"points": points}, fh, indent=2)
            fh.write("\n")

    def run(self) -> dict[str, Any]:
        benchmark = self.config.benchmark
        workload = benchmark.workload
        run_dir = self.benchmark_root / WORKLOAD_DIR
        run_dir.mkdir(parents=True, exist_ok=False)
        self.write_concurrency_series(run_dir / CONCURRENCY_SERIES_FILE)
        row = self._execute(self.build_command(run_dir), run_dir)
        row.update(workload_row_fields(workload))
        # Kept distinct from the harness-derived rates below.
        row["aiperf_request_throughput"] = row["request_throughput"]
        try:
            qps = profiling_qps_summary(run_dir, benchmark.duration_s)
        except OSError as exc:
            row["qps_summary_error"] = f"{type(exc).__name__}: {exc}"
        else:
            row.update({k: qps[k] for k in (
                "profiling_interval_s", "mean_offered_qps", "mean_started_qps",
                "mean_successful_qps", "mean_failed_qps",
            )})
        return row

    def _execute(self, cmd: list[str], point_dir: Path) -> dict[str, Any]:
        with (point_dir / "command.json").open("w") as fh:
            json.dump({"argv": cmd, "display": shlex.join(cmd)}, fh, indent=2)
            fh.write("\n")

        started = time.monotonic()
        with (point_dir / "stdout.log").open("w") as stdout, (
            point_dir / "stderr.log"
        ).open("w") as stderr:
            completed = subprocess.run(cmd, stdout=stdout, stderr=stderr, check=False)
        elapsed = time.monotonic() - started
        if completed.returncode != 0:
            raise RuntimeError(
                f"AIPerf exited {completed.returncode}; see {point_dir / 'stderr.log'}"
            )

        result_path = point_dir / "profile_export_aiperf.json"
        if not result_path.exists():
            candidates = list(point_dir.rglob("profile_export_aiperf.json"))
            if len(candidates) != 1:
                raise RuntimeError(
                    f"expected one profile_export_aiperf.json below {point_dir}, "
                    f"found {len(candidates)}"
                )
            result_path = candidates[0]
        with result_path.open() as fh:
            raw = json.load(fh)
        normalized = normalize_aiperf(raw)
        normalized.update(
            {
                # Kept for readers of pre-cleanup summaries.
                "mode": "agentx_concurrency_ramp",
                "level": self.config.benchmark.workload.target_concurrency,
                "elapsed_s": elapsed,
                "artifact_dir": str(point_dir),
                "error": None,
            }
        )
        return normalized
