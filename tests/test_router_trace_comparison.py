import importlib.util
import json
from pathlib import Path


def test_chunk_replay_tracks_progress_and_recomputation(tmp_path):
    path = Path(__file__).parents[1] / "tools/compare_router_scheduler_trace.py"
    spec = importlib.util.spec_from_file_location("router_trace_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    events = [
        dict(event="request_queued", request_id="a", wall_time_s=1),
        dict(event="request_queued", request_id="b", wall_time_s=3.2),
        dict(event="request_preempted", request_id="a", batch_index=3, wall_time_s=6),
    ]
    for batch, rid, tokens, done in [
        (1, "a", 5, 3),
        (2, "a", 5, 5),
        (3, "a", 8, 7),
        (4, "b", 4, 8),
    ]:
        events += [
            dict(
                event="batch_scheduled",
                batch_index=batch,
                requests=[
                    dict(request_id=rid, phase="prefill", scheduled_tokens=tokens)
                ],
            ),
            dict(event="batch_completed", batch_index=batch, wall_time_s=done),
        ]
    (tmp_path / "engine.0000.jsonl").write_text(
        "".join(json.dumps(e) + "\n" for e in events)
    )
    replay, _, _, quality = module.read_traces(tmp_path)
    assert module.load_before(replay, "engine", 1) == 0
    assert module.load_before(replay, "engine", 2.5) == 10
    assert module.load_before(replay, "engine", 3.2) == 5
    assert module.load_before(replay, "engine", 3.5) == 9
    assert module.load_before(replay, "engine", 5.5) == 4
    assert module.load_before(replay, "engine", 6.5) == 12
    assert module.load_before(replay, "engine", 8.1) == 0
    assert quality["dropped_trace_events"] == 0
    assert quality["incomplete_trace_requests"] == 0
