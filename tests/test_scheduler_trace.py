import json
from types import SimpleNamespace

import pytest

pytest.importorskip("vllm")

from serve_llm_autoscaling import scheduler_trace  # noqa: E402
from serve_llm_autoscaling.backend import TRACE_DIR_ENV, TRACING_SCHEDULER  # noqa: E402


def _events(directory):
    return [json.loads(line) for path in sorted(directory.glob("*.jsonl"))
            for line in path.read_text().splitlines()]


def test_backend_names_match_the_module():
    module, _, name = TRACING_SCHEDULER.rpartition(".")
    assert module == scheduler_trace.__name__
    assert getattr(scheduler_trace, name) is scheduler_trace.AutoTracingScheduler
    assert TRACE_DIR_ENV == scheduler_trace.SCHEDULER_TRACE_DIR_ENV


def test_writer_writes_jsonl_and_rotates(tmp_path):
    writer = scheduler_trace._JsonlTraceWriter(
        str(tmp_path / "traces"), file_prefix="replica/0", max_file_bytes=64)
    for index in range(4):
        writer.emit({"event": "batch", "index": index, "padding": "x" * 30})
    writer.close()
    writer.emit({"event": "after_close"})  # ignored
    files = sorted((tmp_path / "traces").glob("*.jsonl"))
    assert len(files) == 4  # one event fits in each 64-byte file
    assert all(path.name.startswith("replica-0_") for path in files)
    assert [event["index"] for event in _events(tmp_path / "traces")] == [0, 1, 2, 3]


def test_writer_counts_events_it_had_to_drop(tmp_path, monkeypatch):
    # No writer thread, so the one-slot queue stays full.
    monkeypatch.setattr(scheduler_trace.threading.Thread, "start", lambda self: None)
    writer = scheduler_trace._JsonlTraceWriter(str(tmp_path), max_queue_size=1)
    for index in range(3):
        writer.emit({"event": "batch", "index": index})
    assert writer._dropped_events == 2
    assert writer._queue.get_nowait()["index"] == 0
    # The next event that fits is preceded by the count of those lost.
    writer.emit({"event": "batch", "index": 3})
    dropped = writer._queue.get_nowait()
    assert (dropped["event"], dropped["count"]) == ("trace_events_dropped", 2)
    assert writer._dropped_events == 1  # index 3 itself did not fit behind it


@pytest.mark.parametrize("async_scheduling, expected", [
    (False, "TracingScheduler"), (True, "TracingAsyncScheduler"),
])
def test_auto_scheduler_follows_the_resolved_mode(monkeypatch, async_scheduling, expected):
    for name in ("TracingScheduler", "TracingAsyncScheduler"):
        monkeypatch.setattr(scheduler_trace, name,
                            lambda *args, _name=name, **kwargs: (_name, args, kwargs))
    config = SimpleNamespace(scheduler_config=SimpleNamespace(async_scheduling=async_scheduling))
    assert scheduler_trace.AutoTracingScheduler(config, "x")[:2] == (expected, (config, "x"))
    assert scheduler_trace.AutoTracingScheduler(vllm_config=config)[0] == expected


def test_tracing_schedulers_subclass_vllms_own():
    from vllm.v1.core.sched.async_scheduler import AsyncScheduler
    from vllm.v1.core.sched.scheduler import Scheduler

    assert issubclass(scheduler_trace.TracingScheduler, Scheduler)
    assert issubclass(scheduler_trace.TracingAsyncScheduler, AsyncScheduler)
    # The mixin comes first so its hooks wrap the scheduler's.
    assert scheduler_trace.TracingScheduler.__mro__[1] is scheduler_trace.SchedulerTraceMixin
