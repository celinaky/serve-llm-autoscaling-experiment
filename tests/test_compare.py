from pathlib import Path

from serve_llm_autoscaling.artifacts import write_json
from serve_llm_autoscaling.cli import main
from serve_llm_autoscaling.compare import format_comparison


def _evaluation(policy, attainment=0.9, replica_seconds=1800.0):
    return {
        "autoscaling_policy": policy,
        "slo": {"ttft_ms": 3000, "itl_ms": 50, "percentile": 0.95},
        "slo_attainment": {"steady": {
            "requests": {"failed_requests": 0, "slo_met": attainment >= 0.95,
                         "ttft": {"mean_ms": 1500.0, "slo_attainment": attainment,
                                  "slo_percentile_ms": 2800.0},
                         "itl": {"mean_ms": 30.0, "slo_attainment": 0.99,
                                 "slo_percentile_ms": 45.0}},
            "windows": {"ttft_within_slo": 0.95, "itl_within_slo": 1.0},
        }},
        "cost": {"load": {"running_replica_seconds": replica_seconds, "gpu_hours": replica_seconds / 3600,
                          "mean_running_replicas": 2.0, "max_running_replicas": 3}},
        "scaling": {"upscales": 2, "downscales": 0, "first_upscale_at_s": 65.0,
                    "mean_upscale_lag_s": 90.0},
    }


def test_comparison_table(tmp_path: Path):
    runs = [tmp_path / "a-default", tmp_path / "b-token", tmp_path / "c-unanalyzed"]
    write_json(runs[0] / "analysis" / "autoscaling_evaluation.json", _evaluation("default", attainment=0.96))
    write_json(runs[1] / "analysis" / "autoscaling_evaluation.json",
               _evaluation("token", attainment=0.8, replica_seconds=1200.0))
    runs[2].mkdir()
    lines = format_comparison(runs).splitlines()
    assert lines[0].split()[:23] == [
        "run", "policy", "SLO", "TTFT", "pN", "ITL", "pN", "TTFT", "req", "ITL", "req",
        "TTFT", "mean", "ITL", "mean", "failed", "GPU-h", "mean", "max", "up/down",
        "1st", "up", "lag",
    ]
    assert lines[1].split()[:16] == [
        "a-default", "default", "PASS", "2800ms", "45ms", "96%", "99%", "1500ms", "30ms", "0",
        "0.50", "2.0", "3", "2/0", "65s", "90s",
    ]
    assert lines[2].split()[:6] == ["b-token", "token", "FAIL", "2800ms", "45ms", "80%"]
    assert lines[2].split()[10] == "0.33"
    assert "c-unanalyzed  (no autoscaling_evaluation.json" in lines[3]
    assert lines[4].startswith("SLO (p95 of requests): TTFT <= 3000ms, ITL <= 50ms")


def test_comparison_warns_about_different_slos(tmp_path: Path):
    runs = [tmp_path / "a", tmp_path / "b"]
    write_json(runs[0] / "analysis" / "autoscaling_evaluation.json", _evaluation("default"))
    other = _evaluation("default")
    other["slo"]["ttft_ms"] = 6000
    write_json(runs[1] / "analysis" / "autoscaling_evaluation.json", other)
    assert format_comparison(runs).splitlines()[-1] == "WARNING: runs were scored against different SLOs"


def test_compare_command(tmp_path: Path, capsys, monkeypatch):
    run = tmp_path / "run"
    write_json(run / "analysis" / "autoscaling_evaluation.json", _evaluation("default"))
    assert main(["compare", str(run), "--no-plot"]) == 0
    assert "default" in capsys.readouterr().out

    # Without the per-request table the plot is skipped with a pointer, not a crash.
    assert main(["compare", str(run)]) == 1
    assert "run `autoscale-harness analyze`" in capsys.readouterr().err

    # With it, the plot module gets the runs and the options that were set.
    (run / "analysis" / "load_requests.csv").write_text("request_id\n")
    calls = []

    def fake_run(argv, check):
        calls.append(argv)
        return type("Done", (), {"returncode": 0})()

    monkeypatch.setattr("subprocess.run", fake_run)
    assert main(["compare", str(run), "--cols", "p95", "--window", "180"]) == 0
    assert calls[0][1:3] == ["-m", "serve_llm_autoscaling.diagnostics.arrival_stats_compare"]
    assert calls[0][3:] == [str(run), "--window", "180.0", "--cols", "p95"]
