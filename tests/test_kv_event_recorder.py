"""Exercise live/replay recording with real local ZMQ sockets, without Ray/GPU jobs."""
import json
import queue
import threading
import time

import pytest

msgspec = pytest.importorskip("msgspec")
zmq = pytest.importorskip("zmq")

from serve_llm_autoscaling.kv_event_recorder import GPUKVEventRecorder, event_record


class Publisher:
    """The vLLM PUB/ROUTER multipart protocol, with controllable dropped messages."""

    def __init__(self, replica):
        self.replica = replica
        self.commands = queue.Queue()
        self.stop = threading.Event()
        self.ready = threading.Event()
        self.thread = threading.Thread(target=self.run, daemon=True)
        self.thread.start()
        assert self.ready.wait(3)

    def run(self):
        ctx = zmq.Context()
        pub, replay = ctx.socket(zmq.PUB), ctx.socket(zmq.ROUTER)
        p1 = pub.bind_to_random_port("tcp://127.0.0.1")
        p2 = replay.bind_to_random_port("tcp://127.0.0.1")
        self.source = {
            "replica_id": self.replica,
            "actor_id": self.replica,
            "application": "app",
            "deployment": "LLMServer:model",
            "dp_rank": 0,
            "endpoint": f"tcp://127.0.0.1:{p1}",
            "replay_endpoint": f"tcp://127.0.0.1:{p2}",
        }
        buffered = {}
        self.ready.set()
        try:
            while not self.stop.is_set():
                try:
                    sequence, event, live, done = self.commands.get_nowait()
                except queue.Empty:
                    pass
                else:
                    payload = msgspec.msgpack.encode([time.time(), [event], 0])
                    buffered[sequence] = payload
                    if live:
                        pub.send_multipart([b"", sequence.to_bytes(8, "big"), payload])
                    done.set()
                if replay.poll(10):
                    identity, empty, start = replay.recv_multipart()
                    for sequence, payload in sorted(buffered.items()):
                        if sequence >= int.from_bytes(start, "big"):
                            replay.send_multipart(
                                [
                                    identity,
                                    empty,
                                    b"",
                                    sequence.to_bytes(8, "big"),
                                    payload,
                                ]
                            )
                    replay.send_multipart(
                        [
                            identity,
                            empty,
                            b"",
                            (-1).to_bytes(8, "big", signed=True),
                            b"",
                        ]
                    )
        finally:
            pub.close(0)
            replay.close(0)
            ctx.term()

    def publish(self, sequence, event, live=True):
        done = threading.Event()
        self.commands.put((sequence, event, live, done))
        assert done.wait(3)

    def close(self):
        self.stop.set()
        self.thread.join(3)
        assert not self.thread.is_alive()


def records(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line]


def wait_for(path, predicate):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        rows = records(path)
        if predicate(rows):
            return rows
        threading.Event().wait(0.01)
    pytest.fail(f"Recorder did not produce expected rows: {records(path)}")


def removed(group=0, medium="GPU"):
    return {
        "type": "BlockRemoved",
        "block_hashes": [123, 456],
        "medium": medium,
        "group_idx": group,
    }


def test_replay_recovers_join_and_live_loss_and_tracks_new_replica(tmp_path):
    first, second = Publisher("original"), Publisher("scaled-up")
    active = [first.source]
    path = tmp_path / "gpu_kv_events.jsonl"
    recorder = GPUKVEventRecorder(
        path, "app", discover=lambda: list(active), interval_s=0.05
    )
    try:
        # Published before discovery: must be recovered by the initial replay.
        first.publish(
            0,
            {
                "type": "BlockStored",
                "block_hashes": [123],
                "parent_block_hash": None,
                "token_ids": [999, 998],
                "block_size": 16,
                "medium": "GPU",
                "group_idx": 0,
            },
        )
        recorder.start()
        wait_for(path, lambda rows: any(r.get("sequence") == 0 for r in rows))
        # Lost PUB batch followed by a live one. Replay deduplicates overlaps.
        first.publish(1, removed(), live=False)
        first.publish(2, removed(1, "CPU"))
        second.publish(0, removed(3), live=False)
        active.append(second.source)
        rows = wait_for(
            path,
            lambda rows: sum(r["record_type"] == "kv_event_batch" for r in rows) == 4,
        )
    finally:
        recorder.stop(5)
        first.close()
        second.close()
    rows = records(path)
    batches = [r for r in rows if r["record_type"] == "kv_event_batch"]
    assert len(batches) == 4
    assert {(r["replica_id"], r["sequence"]) for r in batches} == {
        ("original", 0),
        ("original", 1),
        ("original", 2),
        ("scaled-up", 0),
    }
    missed = next(
        r for r in batches if r["replica_id"] == "original" and r["sequence"] == 1
    )
    assert missed["delivery"] == "replay"
    assert missed["events"][0]["block_count"] == 2
    assert missed["events"][0]["block_hashes"] == ["123", "456"]
    assert all(
        r["timestamp_ns"] > 0 and r["received_timestamp_ns"] > 0 for r in batches
    )
    assert "token_ids" not in path.read_text()
    assert recorder.error_records == recorder.sequence_gaps == 0
    assert not recorder._thread.is_alive() and recorder._fh.closed
    assert rows[-1]["record_type"] == "recorder_stopped"


def test_missing_replay_history_is_explicit_gap_not_zero_evictions(tmp_path):
    publisher = Publisher("replica")
    path = tmp_path / "events.jsonl"
    recorder = GPUKVEventRecorder(
        path, "app", discover=lambda: [publisher.source], interval_s=0.05
    )
    try:
        # Publisher has already discarded sequences 0 and 1.
        publisher.publish(2, removed(), live=False)
        recorder.start()
        wait_for(
            path, lambda rows: any(r["record_type"] == "sequence_gap" for r in rows)
        )
    finally:
        recorder.stop(5)
        publisher.close()
    rows = records(path)
    gap = next(r for r in rows if r["record_type"] == "sequence_gap")
    assert (gap["first_missing_sequence"], gap["last_missing_sequence"]) == (0, 1)
    assert recorder.sequence_gaps == 1 and recorder.error_records == 1
    assert len([r for r in rows if r["record_type"] == "kv_event_batch"]) == 1


def test_event_projection_preserves_reset_and_binary_hash_identity():
    assert event_record({"type": "AllBlocksCleared"}) == {"type": "AllBlocksCleared"}
    event = event_record(
        {
            "type": "BlockStored",
            "block_hashes": [b"\x00\xff"],
            "parent_block_hash": b"\x00\x01",
            "group_idx": 2,
            "token_ids": [99],
            "kv_cache_spec_kind": "mamba",
            "medium": "GPU",
        }
    )
    assert event["block_hashes"] == ["00ff"]
    assert event["parent_block_hash"] == "0001" and event["group_idx"] == 2
    assert "token_ids" not in event


def test_telemetry_enables_recorder_for_kv_routing_and_can_opt_out(tmp_path):
    from serve_llm_autoscaling.config import load_config
    from serve_llm_autoscaling.telemetry import TelemetrySession

    config = load_config("experiments/h100_workday_curve_mooncake_decay_400s.yaml")
    session = TelemetrySession(config, tmp_path, kv_event_discover=lambda: [])
    assert any(isinstance(c, GPUKVEventRecorder) for c in session.collectors)
    config.analysis.record_gpu_kv_events = False
    session = TelemetrySession(config, tmp_path, kv_event_discover=lambda: [])
    assert not any(isinstance(c, GPUKVEventRecorder) for c in session.collectors)


def test_discovery_uses_running_replica_actor_name_and_filters_application(monkeypatch):
    import ray
    from types import SimpleNamespace
    from ray.serve._private.common import DeploymentID, ReplicaID, RunningReplicaInfo
    from serve_llm_autoscaling.kv_event_recorder import discover_kv_event_sources

    replica = RunningReplicaInfo(
        replica_id=ReplicaID("original", DeploymentID("LLMServer:m", "app")),
        node_id="node",
        node_ip="127.0.0.1",
        availability_zone=None,
        actor_name="replica-actor",
        max_ongoing_requests=100,
        routing_stats={
            "kv_event_metadata": {
                "endpoint": "tcp://127.0.0.1:1234",
                "replay_endpoint": "tcp://127.0.0.1:1235",
                "dp_rank": 0,
            }
        },
    )
    controller = SimpleNamespace(
        _all_running_replicas=SimpleNamespace(remote=lambda: "ref")
    )
    actor = SimpleNamespace(_actor_id=SimpleNamespace(hex=lambda: "actor-id"))
    monkeypatch.setattr(
        ray,
        "get_actor",
        lambda name, namespace: actor if name == "replica-actor" else controller,
    )
    monkeypatch.setattr(
        ray,
        "get",
        lambda ref, timeout: {
            DeploymentID("LLMServer:m", "other-app"): [replica],
            DeploymentID("LLMServer:m", "app"): [replica],
        },
    )
    sources = discover_kv_event_sources("app")
    assert len(sources) == 1
    assert (
        sources[0]["actor_id"] == "actor-id" and sources[0]["replica_id"] == "original"
    )


def test_installed_vllm_publisher_wire_compatibility(tmp_path):
    # vLLM's real publisher and event codec, with no model, Ray job or GPU work.
    events = pytest.importorskip("vllm.distributed.kv_events")
    publisher = events.ZmqEventPublisher(
        0, endpoint="tcp://*:0", replay_endpoint="tcp://*:0"
    )
    config = publisher.get_publisher_config()
    source = {
        "replica_id": "vllm-real",
        "actor_id": "cpu-test",
        "application": "app",
        "deployment": "LLMServer:m",
        "dp_rank": 0,
        "endpoint": config.endpoint,
        "replay_endpoint": config.replay_endpoint,
    }
    path = tmp_path / "real-vllm.jsonl"
    recorder = GPUKVEventRecorder(
        path, "app", discover=lambda: [source], interval_s=0.05
    )
    try:
        publisher.publish(
            events.KVEventBatch(
                ts=123.5,
                events=[
                    events.BlockRemoved(block_hashes=[123], medium="GPU", group_idx=2)
                ],
            )
        )
        recorder.start()
        rows = wait_for(
            path, lambda rows: any(r["record_type"] == "kv_event_batch" for r in rows)
        )
    finally:
        recorder.stop(5)
        publisher.shutdown()
    batch = next(r for r in rows if r["record_type"] == "kv_event_batch")
    assert batch["timestamp_ns"] == 123500000000
    assert batch["events"][0] == {
        "type": "BlockRemoved",
        "medium": "GPU",
        "group_idx": 2,
        "block_hashes": ["123"],
        "block_count": 1,
    }
    assert recorder.error_records == 0
