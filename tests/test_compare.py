from pathlib import Path

from serve_llm_autoscaling.artifacts import write_json
from serve_llm_autoscaling.cli import main
from serve_llm_autoscaling.compare import format_comparison


def _evaluation(policy, advisory=False, attainment=0.9, replica_seconds=1800.0, model_ready=None):
    return {
        "autoscaling_policy": policy,
        "autoscaling_policy_kwargs": {"advisory": advisory} if policy == "sla" else {},
        "slo": {"ttft_ms": 3000, "itl_ms": 50},
        "slo_attainment": {"steady": {
            "requests": {"failed_requests": 0,
                         "ttft": {"mean_ms": 1500.0, "slo_attainment": attainment},
                         "itl": {"mean_ms": 30.0, "slo_attainment": 0.99}},
            "windows": {"ttft_within_slo": 0.95, "itl_within_slo": 1.0},
        }},
        "cost": {"load": {"running_replica_seconds": replica_seconds,
                          "mean_running_replicas": 2.0, "max_running_replicas": 3}},
        "scaling": {"upscales": 2, "downscales": 0, "first_upscale_at_s": 65.0,
                    "mean_upscale_lag_s": 90.0},
        "sla_policy": {"model_ready_at_s": model_ready} if policy == "sla" else None,
    }


def test_comparison_table(tmp_path: Path):
    runs = [tmp_path / "a-default", tmp_path / "b-sla", tmp_path / "c-unanalyzed"]
    write_json(runs[0] / "analysis" / "autoscaling_evaluation.json", _evaluation("default"))
    write_json(runs[1] / "analysis" / "autoscaling_evaluation.json",
               _evaluation("sla", advisory=True, attainment=0.8, replica_seconds=1200.0,
                           model_ready=910.0))
    runs[2].mkdir()
    lines = format_comparison(runs).splitlines()
    assert lines[0].split() == [
        "run", "policy", "TTFT", "req", "TTFT", "time", "ITL", "req", "ITL", "time",
        "TTFT", "mean", "ITL", "mean", "failed", "replica-s", "mean", "max", "up/down",
        "1st", "up", "lag", "model", "ready",
    ]
    assert lines[1].split() == [
        "a-default", "default", "90%", "95%", "99%", "100%", "1500ms", "30ms", "0",
        "1800", "2.0", "3", "2/0", "65s", "90s", "-",
    ]
    assert lines[2].split()[:3] == ["b-sla", "sla", "(advisory)"]
    assert lines[2].split()[-1] == "910s"
    assert "c-unanalyzed  (no autoscaling_evaluation.json" in lines[3]
    assert lines[4] == "SLO: mean TTFT <= 3000ms, mean ITL <= 50ms"


def test_comparison_warns_about_different_slos(tmp_path: Path):
    runs = [tmp_path / "a", tmp_path / "b"]
    write_json(runs[0] / "analysis" / "autoscaling_evaluation.json", _evaluation("default"))
    other = _evaluation("default")
    other["slo"]["ttft_ms"] = 6000
    write_json(runs[1] / "analysis" / "autoscaling_evaluation.json", other)
    assert format_comparison(runs).splitlines()[-1] == "WARNING: runs were scored against different SLOs"


def test_compare_command(tmp_path: Path, capsys):
    run = tmp_path / "run"
    write_json(run / "analysis" / "autoscaling_evaluation.json", _evaluation("default"))
    assert main(["compare", str(run)]) == 0
    assert "run  " in capsys.readouterr().out
