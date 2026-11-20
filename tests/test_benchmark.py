import json
from pathlib import Path

import pytest

from conftest import KV_AWARE, series, short_config
from serve_llm_autoscaling import benchmark
from serve_llm_autoscaling.benchmark import (
    AIPerfRunner,
    aiperf_command,
    normalize_aiperf,
    workload_row_fields,
)
from serve_llm_autoscaling.config import AIPERF_SOURCE_DEFAULT

pytestmark = pytest.mark.usefixtures("uvx")


def _arg(command, flag):
    return command[command.index(flag) + 1]


# --- Command ----------------------------------------------------------------


def test_build_command(agentx_config, tmp_path: Path):
    command = AIPerfRunner(agentx_config(), tmp_path).build_command(tmp_path / "ramp")
    assert command[:7] == [
        "uvx", "--python", "3.11", "--from", AIPERF_SOURCE_DEFAULT, "aiperf", "profile"
    ]
    assert _arg(command, "--model") == _arg(command, "--tokenizer") == "long-context-model"
    assert _arg(command, "--url") == "http://127.0.0.1:8000"
    assert _arg(command, "--scenario") == "inferencex-agentx-mvp"
    assert _arg(command, "--public-dataset") == "semianalysis_cc_traces_weka_062126_256k"
    # The peak of the curve; the curve itself is passed as a file.
    assert _arg(command, "--concurrency") == "16"
    assert _arg(command, "--concurrency-series") == str(
        (tmp_path / "ramp" / "concurrency_series.json").resolve()
    )
    assert _arg(command, "--trajectory-start-min-ratio") == "0.0"
    assert _arg(command, "--trajectory-start-max-ratio") == "0.0"
    assert _arg(command, "--benchmark-duration") == "1800.0"
    assert _arg(command, "--benchmark-grace-period") == "300.0"
    assert _arg(command, "--random-seed") == "42"
    assert _arg(command, "--endpoint-type") == "chat"
    assert _arg(command, "--export-level") == "records"
    assert _arg(command, "--ui") == "none"
    assert _arg(command, "--artifact-dir") == str(tmp_path / "ramp")
    assert "--streaming" in command and "--use-server-token-count" in command
    assert "--unsafe-override" not in command
    # Without the KV-aware router nothing pins a session to a replica.
    assert "--session-header" not in command


def test_command_follows_the_config(agentx_config, tmp_path: Path):
    config = agentx_config(
        {"concurrency_series": series((0, 2), (60, 8), (120, 3)), "unsafe_override": True,
         "trajectory_start_min_ratio": 0.25, "trajectory_start_max_ratio": 1, "seed": 7},
        deployment={**KV_AWARE, "tokenizer": "org/tokenizer"},
        runtime={"aiperf_source": "aiperf==9.9", "endpoint_url": "http://head:8000"},
        duration_s=120, extra_args=["--request-timeout-seconds", "600"],
    )
    command = AIPerfRunner(config, tmp_path).build_command(tmp_path)
    assert _arg(command, "--from") == "aiperf==9.9"
    assert _arg(command, "--tokenizer") == "org/tokenizer"
    assert _arg(command, "--url") == "http://head:8000"
    assert _arg(command, "--concurrency") == "8"
    assert _arg(command, "--trajectory-start-min-ratio") == "0.25"
    assert _arg(command, "--trajectory-start-max-ratio") == "1.0"
    assert _arg(command, "--random-seed") == "7"
    assert command.count("--unsafe-override") == 1
    # The KV-aware router pins each session's later turns by this header.
    assert _arg(command, "--session-header") == "X-Session-ID"
    assert command[-2:] == ["--request-timeout-seconds", "600"]


def test_command_resolves_symlinked_series_path(agentx_config, tmp_path: Path):
    real = tmp_path / "real"
    real.mkdir()
    (tmp_path / "link").symlink_to(real, target_is_directory=True)
    command = AIPerfRunner(agentx_config(), tmp_path).build_command(tmp_path / "link")
    assert _arg(command, "--concurrency-series") == str(real / "concurrency_series.json")


def test_aiperf_command_requires_uvx(monkeypatch):
    monkeypatch.setattr(benchmark.shutil, "which", lambda name: None)
    with pytest.raises(RuntimeError, match="uvx not on PATH"):
        aiperf_command()


def test_write_concurrency_series(tmp_path: Path):
    config = short_config({"concurrency_series": series((0, 1), (60.5, 4), (120, 2))})
    path = tmp_path / "curve.json"
    AIPerfRunner(config, tmp_path).write_concurrency_series(path)
    # AIPerf's flag reads "concurrency", not the config's "sessions".
    assert json.loads(path.read_text()) == {"points": [
        {"time_s": 0.0, "concurrency": 1}, {"time_s": 60.5, "concurrency": 4},
        {"time_s": 120.0, "concurrency": 2},
    ]}


def test_workload_row_fields():
    config = short_config()
    assert workload_row_fields(config.benchmark.workload) == {
        "target_concurrency": 4,
        "concurrency_series": [{"time_s": 0.0, "sessions": 1}, {"time_s": 120.0, "sessions": 4}],
    }


# --- Results ----------------------------------------------------------------


def test_normalize_aiperf():
    raw = {
        "request_count": {"avg": 10},
        "request_throughput": {"avg": 2.5},
        "time_to_first_token": {"avg": 11, "p50": 9, "p90": 15, "p99": 20},
        "inter_token_latency": {"avg": 3},
        "request_latency": {"avg": 100, "p99": 150},
        "error_summary": [{"count": 2}],
    }
    result = normalize_aiperf(raw)
    assert result["request_count"] == 10
    assert result["request_throughput"] == 2.5
    assert result["p99_ttft_ms"] == 20
    assert result["prompt_cache_read_pct"] is None
    assert result["failed_requests"] == 2
    # AIPerf's own count wins over the summed error summary.
    assert normalize_aiperf({**raw, "error_request_count": {"avg": 5}})["failed_requests"] == 5


T0 = 1_000 * 10**9


def _record(credit_s, end_s, error=None):
    record = {
        "metadata": {"credit_issued_ns": T0 + int(credit_s * 1e9),
                     "request_start_ns": T0 + int(credit_s * 1e9),
                     "request_end_ns": T0 + int(end_s * 1e9),
                     "benchmark_phase": "profiling", "agent_depth": 0},
        "metrics": {},
    }
    if error:
        record["error"] = {"message": error}
    return record


def _fake_aiperf(monkeypatch, calls, *, returncode=0, export="profile_export_aiperf.json",
                 requests=True):
    """Stand in for the AIPerf process, writing what a 10s run would."""

    def fake_run(cmd, **kwargs):
        calls.append(cmd)
        artifact_dir = Path(_arg(cmd, "--artifact-dir"))
        if returncode == 0 and export:
            target = artifact_dir / export
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_text(json.dumps({
                "request_count": {"avg": 4}, "request_throughput": {"avg": 0.35},
            }))
        if returncode == 0 and requests:
            (artifact_dir / "phase_manifest.json").write_text(json.dumps({"phases": [
                {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + 12 * 10**9},
            ]}))
            records = [_record(1, 2), _record(3, 4), _record(5, 6, error="boom"),
                       # Completes in the grace period: outside the profiling interval.
                       _record(9, 11)]
            (artifact_dir / "profile_export.jsonl").write_text(
                "".join(json.dumps(r) + "\n" for r in records)
            )
        return benchmark.subprocess.CompletedProcess(cmd, returncode)

    monkeypatch.setattr(benchmark.subprocess, "run", fake_run)


def _ten_second_config():
    return short_config({"concurrency_series": series((0, 1), (5, 16))}, duration_s=10)


def test_run(monkeypatch, tmp_path: Path):
    calls = []
    _fake_aiperf(monkeypatch, calls)
    row = AIPerfRunner(_ten_second_config(), tmp_path).run()
    run_dir = tmp_path / "agentx-concurrency-ramp"
    assert len(calls) == 1
    assert row["artifact_dir"] == str(run_dir)
    assert {k: row[k] for k in ("mode", "level", "target_concurrency", "error")} == {
        "mode": "agentx_concurrency_ramp", "level": 16, "target_concurrency": 16, "error": None,
    }
    assert row["concurrency_series"] == [{"time_s": 0.0, "sessions": 1},
                                         {"time_s": 5.0, "sessions": 16}]
    assert row["request_count"] == 4 and row["elapsed_s"] >= 0
    # AIPerf's number and the harness-derived rates are distinct fields.
    assert row["aiperf_request_throughput"] == 0.35
    assert row["profiling_interval_s"] == 10
    assert row["mean_offered_qps"] == pytest.approx(0.4)
    assert row["mean_started_qps"] == pytest.approx(0.4)
    assert row["mean_successful_qps"] == pytest.approx(0.2)
    assert row["mean_failed_qps"] == pytest.approx(0.1)
    for name in ["command.json", "stdout.log", "stderr.log", "concurrency_series.json",
                 "profile_export.jsonl", "profile_export_aiperf.json", "phase_manifest.json"]:
        assert (run_dir / name).exists()
    # The curve was on disk before AIPerf was started with its path.
    command = json.loads((run_dir / "command.json").read_text())
    assert command["argv"] == calls[0]
    assert _arg(calls[0], "--concurrency-series") == str(run_dir / "concurrency_series.json")
    assert "--unsafe-override" in command["display"]


def test_run_finds_a_nested_export_and_tolerates_missing_records(monkeypatch, tmp_path: Path):
    _fake_aiperf(monkeypatch, [], export="nested/profile_export_aiperf.json", requests=False)
    row = AIPerfRunner(_ten_second_config(), tmp_path).run()
    assert row["request_count"] == 4
    assert row["qps_summary_error"].startswith("FileNotFoundError")
    assert "mean_offered_qps" not in row


def test_run_failure_points_to_stderr(monkeypatch, tmp_path: Path):
    _fake_aiperf(monkeypatch, [], returncode=2)
    with pytest.raises(RuntimeError, match="AIPerf exited 2; see .*agentx-concurrency-ramp/stderr.log"):
        AIPerfRunner(_ten_second_config(), tmp_path).run()


def test_run_requires_one_export(monkeypatch, tmp_path: Path):
    _fake_aiperf(monkeypatch, [], export=None)
    with pytest.raises(RuntimeError, match="expected one profile_export_aiperf.json .* found 0"):
        AIPerfRunner(_ten_second_config(), tmp_path).run()


def test_run_refuses_to_overwrite_an_earlier_run(monkeypatch, tmp_path: Path):
    calls = []
    _fake_aiperf(monkeypatch, calls)
    (tmp_path / "agentx-concurrency-ramp").mkdir()
    with pytest.raises(FileExistsError):
        AIPerfRunner(_ten_second_config(), tmp_path).run()
    assert calls == []
