import csv
import json

import pytest

from serve_profiler.analysis import (
    analyze,
    coverage,
    linear_fit,
    read_engines,
    reconstruct_engine,
)


def event(kind, at, **fields):
    return dict(event=kind, monotonic_s=at, wall_time_s=1000 + at, **fields)


def batch(index, at, requests):
    return event(
        "batch_scheduled",
        at,
        batch_index=index,
        scheduled_at_monotonic_s=at - 0.01,
        requests=[
            dict(request_id=rid, phase="prefill", scheduled_tokens=tokens)
            for rid, tokens in requests
        ],
    )


def done(index, at):
    # Explicit callback time must win over the logger's emission time.
    return event(
        "batch_completed", at + 0.1, batch_index=index, completed_at_monotonic_s=at
    )


def fixture_events():
    return [
        event("request_queued", 1.1, request_id="a", queued_at_monotonic_s=1),
        batch(1, 1.2, [("a", 40)]),
        event(
            "request_admitted",
            1.2,
            request_id="a",
            batch_index=1,
            prompt_tokens=100,
            cached_prefix_tokens=0,
        ),
        event("request_queued", 2, request_id="b"),
        batch(2, 2.9, [("a", 60)]),
        done(1, 3),
        event("request_queued", 4, request_id="c"),
        batch(3, 4.9, [("b", 20)]),
        done(2, 5),
        event("request_first_token", 5, request_id="a", batch_index=2),
        event(
            "request_admitted",
            4.9,
            request_id="b",
            batch_index=3,
            prompt_tokens=80,
            cached_prefix_tokens=60,
        ),
        done(3, 7),
        event("request_first_token", 7, request_id="b", batch_index=3),
        batch(4, 7.5, [("c", 10)]),
        event(
            "request_admitted",
            7.5,
            request_id="c",
            batch_index=4,
            prompt_tokens=10,
            cached_prefix_tokens=0,
        ),
        done(4, 8),
        event("request_first_token", 8, request_id="c", batch_index=4),
    ]


def test_load_cache_chunks_and_first_execution():
    points, quality = reconstruct_engine("engine", fixture_events(), None)
    assert quality["excluded"] == {"no_previous_batch": 1}
    b, c = points
    assert b["request_id"] == "b"
    assert b["load_tokens"] == 100  # own increment excluded, A not completed
    assert b["own_uncached_tokens"] == 20
    assert b["queue_ms"] == pytest.approx(3000)  # previous callback at 5, arrival 2
    assert c["load_tokens"] == 80  # A has 60 left, B has 20
    assert c["queue_ms"] == pytest.approx(3490)  # scheduled after previous callback


def test_all_requests_contribute_even_without_client_record():
    points, _ = reconstruct_engine("engine", fixture_events(), {"b": {"phase": "c4"}})
    assert len(points) == 1
    assert points[0]["load_tokens"] == 100


def test_unresolved_request_invalidates_later_loads():
    events = fixture_events() + [event("request_queued", 1.5, request_id="unresolved")]
    points, quality = reconstruct_engine("engine", events, None)
    assert points == []
    assert quality["incomplete_requests"] == 1
    assert quality["excluded"]["incomplete_engine_history"] == 3


def test_decode_only_previous_batch_is_used():
    events = fixture_events()
    events += [
        batch(5, 8.1, []),
        done(5, 9),
        event("request_queued", 8.5, request_id="d"),
        batch(6, 8.8, [("d", 4)]),
        event(
            "request_admitted",
            8.8,
            request_id="d",
            batch_index=6,
            prompt_tokens=4,
            cached_prefix_tokens=0,
        ),
        done(6, 10),
        event("request_first_token", 10, request_id="d", batch_index=6),
    ]
    points, _ = reconstruct_engine("engine", events, None)
    assert points[-1]["queue_ms"] == pytest.approx(500)


def test_post_decode_preemption_work_added_at_preemption():
    events = fixture_events() + [
        event("request_preempted", 8.1, request_id="a", batch_index=5),
        batch(5, 8.2, [("a", 100)]),
        event("request_queued", 8.3, request_id="d"),
        done(5, 9),
        batch(6, 9.1, [("d", 5)]),
        event(
            "request_admitted",
            9.1,
            request_id="d",
            batch_index=6,
            prompt_tokens=5,
            cached_prefix_tokens=0,
        ),
        done(6, 10),
        event("request_first_token", 10, request_id="d", batch_index=6),
    ]
    points, _ = reconstruct_engine("engine", events, None)
    assert points[0]["load_tokens"] == 100  # no future recompute at original arrival
    assert points[-1]["load_tokens"] == 100


def test_rotated_files_are_one_engine(tmp_path):
    events = fixture_events()
    for i, subset in enumerate((events[:8], events[8:])):
        (tmp_path / f"host_pid-1.{i:04}.jsonl").write_text(
            "".join(json.dumps(e) + "\n" for e in subset)
        )
    engines, quality = read_engines(sorted(tmp_path.glob("*.jsonl")))
    assert list(engines) == ["host_pid-1"]
    assert quality["malformed_lines"] == 0
    assert len(reconstruct_engine("host_pid-1", engines["host_pid-1"], None)[0]) == 2


def test_new_trace_sequence_detects_loss_without_drop_marker(tmp_path):
    path = tmp_path / "engine.0000.jsonl"
    events = [
        dict(event("trace_started", 1), event_index=1),
        dict(event("scheduler_idle_step", 2), event_index=3),
    ]
    path.write_text("".join(json.dumps(e) + "\n" for e in events))
    _, quality = read_engines([path])
    assert quality["event_index_gaps"] == 1


def test_fit_known_coefficients_and_degenerate_data():
    points = [dict(load_tokens=x, queue_ms=5 + 0.07 * x) for x in (0, 10_000, 40_000)]
    fit = linear_fit(points)
    assert fit["slope_ms_per_token"] == pytest.approx(0.07)
    assert fit["intercept_ms"] == pytest.approx(5)
    assert fit["r_squared"] == pytest.approx(1)
    assert linear_fit([])["status"] == "unavailable"
    assert linear_fit([points[0], points[0]])["status"] == "unavailable"
    assert (
        linear_fit([dict(load_tokens=x, queue_ms=5 - x) for x in (0, 1)])["status"]
        == "nonpositive_slope"
    )


def test_coverage_zero_boundaries_and_empty_bins():
    points = [dict(load_tokens=x, queue_ms=1) for x in (0, 1, 20_000, 40_001)]
    assert [r["requests"] for r in coverage(points)] == [1, 2, 0, 1]


def test_analyze_rejects_loss_and_does_not_require_ray(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    events = fixture_events() + [event("trace_events_dropped", 9, count=1)]
    (traces / "engine.0000.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    result = analyze(tmp_path, trace_only=True)
    assert result["fit"]["status"] == "unavailable"
    assert result["quality"]["dropped_events"] == 1
    assert list(csv.DictReader((tmp_path / "analysis/points.csv").open())) == []
    assert (tmp_path / "analysis/queue_vs_load.png").exists()


def test_client_selection_explicit(tmp_path):
    traces = tmp_path / "traces"
    traces.mkdir()
    (traces / "engine.0000.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in fixture_events())
    )
    with pytest.raises(ValueError, match="No profiling client"):
        analyze(tmp_path)
