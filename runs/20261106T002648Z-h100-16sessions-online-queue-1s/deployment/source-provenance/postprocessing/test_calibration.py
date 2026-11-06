import json

from serve_llm_autoscaling.calibration import collect_and_render, render
from serve_llm_autoscaling.config import AutoscalingPolicyConfig


def test_reproduces_window_and_flags_dropped_audit(tmp_path):
    source = tmp_path / "shared"
    source.mkdir()
    rows = [
        dict(
            sample_id=i + 1,
            request_id=str(i),
            workload_tokens=i * 10,
            queue_time_ms=5 + 2 * i * 10,
            worker_id=1,
            prompt_tokens=100,
            routed_at_s=1,
            observed_at_s=2,
        )
        for i in range(100)
    ]
    model = dict(intercept_ms=5, slope_ms_per_token=2, sample_count=100, updated_at_s=3)
    events = [dict(event="sample_accepted", router_id="router", sample=r) for r in rows]
    events += [
        dict(
            event="fit_snapshot",
            router_id="router",
            attempt_id=1,
            samples=rows,
            candidate=model,
            published_model={},
            reason="published",
            wall_time_s=3,
        ),
        dict(event="audit_closed", dropped=0),
    ]
    path = source / "router.jsonl"
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    report = collect_and_render(tmp_path / "run", source, 2000)
    assert report["status"] == "verified"
    assert report["samples"] == 100
    fits = json.loads(
        (tmp_path / "run/analysis/prefill-calibration/fits.json").read_text()
    )
    assert fits[0]["verified"]
    assert fits[0]["derived_budget"] == 997.5
    assert (tmp_path / "run/analysis/prefill-calibration/router-00001.png").exists()
    saved = tmp_path / "run/telemetry/prefill-calibration/router.jsonl"
    saved.write_text(path.read_text().replace('"dropped": 0', '"dropped": 1'))
    assert render(tmp_path / "run", 2000)["status"] == "incomplete"


def test_queue_policy_configuration():
    policy = AutoscalingPolicyConfig(
        name="token", kwargs={"target_queue_time_ms": 2000}
    )
    assert policy.kwargs["target_queue_time_ms"] == 2000


def test_model_timeline_excludes_ingress_replica_targets(tmp_path):
    path = tmp_path / "telemetry/serve_metrics.jsonl"
    path.parent.mkdir()

    def sample(name, deployment, value, **labels):
        return dict(
            normalized_name=name,
            labels=dict(application="app", deployment=deployment, **labels),
            value=value,
        )

    path.write_text(
        json.dumps(
            dict(
                timestamp_ns=10**9,
                samples=[
                    sample("serve_autoscaling_target_replicas", "router", 1),
                    sample("serve_autoscaling_target_replicas", "engine", 2),
                    sample(
                        "serve_autoscaling_policy_metric",
                        "engine",
                        20000,
                        metric="learned_prefill_tokens_per_replica",
                    ),
                    sample("serve_autoscaling_target_replicas", "other_engine", 5),
                ],
            )
        )
        + "\n"
    )
    render(tmp_path, 1000)
    timeline = json.loads(
        (tmp_path / "analysis/prefill-calibration/controller_timeline.json").read_text()
    )
    assert timeline["replica_target_points"] == [[1, 2]]
    assert timeline["budget_points"] == [[1, 20000]]


def test_backend_enables_native_statistics_and_learning():
    from serve_llm_autoscaling.backend import RayServeBackend
    from serve_llm_autoscaling.config import ExperimentConfig

    config = ExperimentConfig.model_validate(
        {
            "name": "learned",
            "deployment": {
                "model_id": "m",
                "routing": {"policy": "kv_aware"},
                "autoscaling": {
                    "policy": {
                        "name": "token",
                        "kwargs": {"target_queue_time_ms": 1000},
                    }
                },
            },
        }
    )
    spec = RayServeBackend(config).deployment_spec()
    assert spec["engine_kwargs"]["disable_log_stats"] is False
    assert (
        spec["deployment_config"]["request_router_config"]["request_router_kwargs"][
            "enable_prefill_queue_learning"
        ]
        is True
    )
    assert config.deployment.routing.request_router_kwargs == {}
