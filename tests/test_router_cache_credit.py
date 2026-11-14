import json
from pathlib import Path

from serve_llm_autoscaling.diagnostics.router_cache_credit import audit
from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import load_config


def write(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


def test_same_engine_cache_hit_distinguishes_missing_and_present_router_credit(
    tmp_path,
):
    """An engine cache hit alone is not proof of a router overestimate."""
    traces = tmp_path / "traces"
    bookings, engines = [], []
    for rid, estimate in [("missing-credit", 100000), ("credited", 4000)]:
        bookings.append(
            dict(
                event="router_selection",
                request_id=rid,
                wall_time_s=10,
                prompt_tokens=100000,
                overlap={"gpu": 0, "cpu": 0},
                effective_prefill_tokens=estimate,
            )
        )
        engines.extend(
            [
                dict(
                    event="request_admitted",
                    request_id=f"chatcmpl-{rid}-abcdef12",
                    external_request_id=f"chatcmpl-{rid}-abcdef12",
                    wall_time_s=11,
                    prompt_tokens=100000,
                    local_cached_prefix_tokens=0,
                    external_cached_prefix_tokens=96000,
                    initial_gpu_prefill_tokens=4000,
                ),
                dict(
                    event="batch_scheduled",
                    batch_index=1,
                    requests=[
                        dict(
                            request_id=f"chatcmpl-{rid}-abcdef12",
                            phase="prefill",
                            scheduled_tokens=4000,
                        )
                    ],
                ),
                dict(
                    event="request_first_token",
                    request_id=f"chatcmpl-{rid}-abcdef12",
                    wall_time_s=12,
                    batch_index=1,
                ),
            ]
        )
    write(traces / "router-request-audit/router.jsonl", bookings)
    write(traces / "engine.0000.jsonl", engines)
    report, rows = audit(tmp_path, traces, host_credit=0.75)
    assert report["valid_rows"] == 2
    assert report["external_reuse_with_initial_router_excess"] == 1
    by_id = {r["request_id"]: r for r in rows}
    assert by_id["missing-credit"]["router_initial_excess_vs_gpu_tokens"] == 96000
    assert by_id["credited"]["router_initial_excess_vs_gpu_tokens"] == 0
    assert (
        by_id["missing-credit"]["router_initial_excess_after_transfer_allowance"]
        == 72000
    )


def test_historical_combined_cache_count_cannot_establish_router_credit(tmp_path):
    traces = tmp_path / "traces"
    write(
        traces / "engine.jsonl",
        [
            dict(
                event="request_admitted",
                request_id="r",
                wall_time_s=10,
                prompt_tokens=100000,
                cached_prefix_tokens=96000,
            )
        ],
    )
    report, rows = audit(tmp_path, traces)
    assert report["status"] == "unavailable"
    assert report["valid_rows"] == 0 and rows == []


def test_request_audit_configuration_reaches_router_runtime(tmp_path):
    config = load_config("experiments/h100_workday_curve_mooncake_decay_400s.yaml")
    config.analysis.scheduler_trace_dir = tmp_path
    env = RayServeBackend(config).deployment_spec()["runtime_env"]["env_vars"]
    key = "RAY_SERVE_LLM_ROUTER_REQUEST_TRACE_DIR"
    assert env[key] == str(tmp_path / "router-request-audit")
    config.analysis.record_router_estimates = False
    assert (
        key not in RayServeBackend(config).deployment_spec()["runtime_env"]["env_vars"]
    )
