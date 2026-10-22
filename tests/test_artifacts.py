import json
from pathlib import Path

from serve_llm_autoscaling.artifacts import format_run_summary, write_json


def _make_run(root: Path, status: str, points: list, **manifest) -> Path:
    write_json(
        root / "manifest.json",
        {
            "name": "demo",
            "status": status,
            "started_at": "2026-09-23T16:37:36+00:00",
            "finished_at": "2026-09-23T16:39:51+00:00",
            **manifest,
        },
    )
    write_json(root / "benchmark" / "sweep_summary.json", points)
    (root / "deployment").mkdir()
    (root / "deployment" / "events.jsonl").write_text(
        json.dumps({"event": "teardown_completed"}) + "\n"
    )
    return root


def _point(level, **fields):
    return {"mode": "concurrency", "level": level, "error": None,
            "request_count": 280.0, "failed_requests": 0, "p50_ttft_ms": 20.5,
            **fields}


def test_summary_clean_success(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [_point(1)], readiness_s=85.25)
    text = format_run_summary(root)
    assert "SUCCEEDED (no errors)" in text
    assert "2m 15s (deployment ready after 85.2s)" in text
    assert "Teardown:  completed" in text
    assert "280" in text and "20.5ms" in text


def test_summary_flags_partial_failures(tmp_path: Path):
    points = [_point(1, failed_requests=3), {"mode": "concurrency", "level": 2,
                                             "error": "RuntimeError: boom"}]
    text = format_run_summary(_make_run(tmp_path, "succeeded", points))
    assert "SUCCEEDED WITH ERRORS (1 of 2 points failed, 3 failed requests)" in text
    assert "ERROR: RuntimeError: boom" in text


def test_summary_failed_run(tmp_path: Path):
    root = _make_run(tmp_path, "failed", [], failure_stage="deploy",
                     error="RuntimeError: Deploying application failed")
    text = format_run_summary(root)
    assert "FAILED at stage 'deploy'" in text
    assert "Error:     RuntimeError: Deploying application failed" in text


def test_summary_prints_recommended_ttft_slo(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [
        {"mode": "agentx_fixed_concurrency", "level": 8, "request_count": 100,
         "failed_requests": 0, "error": None},
    ])
    write_json(root / "analysis" / "slo_calibration.json", {
        "recommended_ttft_slo_ms": 2412.3, "ttft_sample_count": 100,
        "failed_request_count": 0,
    })
    text = format_run_summary(root)
    assert "recommended 2412.3ms" in text
    assert "analysis.ttft_slo_ms: 2412" in text
    assert "agentx_fixed_concurrency" in text


def test_summary_omits_slo_without_calibration(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [])
    assert "TTFT SLO" not in format_run_summary(root)


def test_summary_prints_autoscaling_evaluation(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [])
    write_json(root / "analysis" / "autoscaling_evaluation.json", {
        "autoscaling_policy": "sla",
        "slo": {"ttft_ms": 3000, "itl_ms": 50, "percentile": 0.95},
        "periods": {"steady": {"start_s": 150, "end_s": 700}},
        "slo_attainment": {"steady": {
            "requests": {"requests": 400, "failed_requests": 2, "slo_met": False,
                         "ttft": {"mean_ms": 1820.4, "slo_percentile_ms": 4100.0,
                                  "slo_attainment": 0.84, "slo_met": False},
                         "itl": {"mean_ms": 31.2, "slo_percentile_ms": 48.0,
                                 "slo_attainment": 0.96, "slo_met": True}},
            "windows": {"ttft_within_slo": 0.93, "ttft_violation_s": 40.0,
                        "itl_within_slo": 1.0, "itl_violation_s": 0.0},
        }},
        "cost": {"load": {"running_replica_seconds": 1840.0, "mean_running_replicas": 2.63,
                          "max_running_replicas": 4, "starting_replica_seconds": 210.0,
                          "gpu_hours": 0.57}},
        "scaling": {"upscales": 3, "downscales": 1, "first_upscale_at_s": 67.0,
                    "mean_upscale_lag_s": 98.0, "max_upscale_lag_s": 120.0},
        "sla_policy": {"model_ready_at_s": 912.0, "first_capacity_decision_at_s": 940.0,
                       "prediction": {"ttft": {
                           "predicted_vs_server": {"bias_ms": 210.0, "samples": 40},
                           "client_vs_server": {"bias_ms": 340.0}}}},
    })
    text = format_run_summary(root)
    assert "policy sla, steady state 150-700s, 400 requests (2 failed): SLO FAIL" in text
    assert "TTFT: p95 4100ms vs SLO 3000ms: FAIL · 84% of requests within · mean 1820ms" in text
    assert "ITL : p95 48ms vs SLO 50ms: PASS · 96% of requests within" in text
    assert "within SLO 93% of the time (40s over)" in text
    assert ("GPU time: 0.57 GPU-h during load · 1840 replica-s running (mean 2.6, max 4), "
            "210 replica-s starting") in text
    assert "Scaling: 3 up / 1 down · first up at 67s · upscale lag mean 98s, max 120s" in text
    assert "model ready at 912s · first capacity decision at 940s" in text
    assert "TTFT predicted-server bias +210ms (n=40) · client-server +340ms" in text


def test_summary_omits_evaluation_without_artifact(tmp_path: Path):
    root = _make_run(tmp_path, "succeeded", [])
    assert "Autoscaling evaluation" not in format_run_summary(root)
