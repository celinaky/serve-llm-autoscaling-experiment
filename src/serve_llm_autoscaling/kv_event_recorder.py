"""Debug log of each replica's GPU KV-cache block events.

Saved to ``<run>/telemetry/gpu_kv_events.jsonl`` when
``analysis.record_gpu_kv_events`` is on (the default) and routing is
``kv_aware``. No scorecard or plot reads it; it is there to answer questions
such as "was this session's context evicted before it came back?".

Persists vLLM's existing KV-event streams without touching router consumers.

Only this driver-side thread owns the ZMQ sockets. It follows Serve's running
replicas, replays from sequence zero on discovery and periodically catches up
through each publisher's replay buffer. JSONL records retain publisher time,
replica identity, cache group and hashes, but never prompt token IDs.
"""
from __future__ import annotations

import json
import threading
import time
from pathlib import Path
from typing import Any, Callable


def discover_kv_event_sources(application: str) -> list[dict[str, Any]]:
    """Read the same running-replica metadata used by Ray's KV-aware router."""
    import ray
    from ray.serve._private.constants import SERVE_CONTROLLER_NAME, SERVE_NAMESPACE

    controller = ray.get_actor(SERVE_CONTROLLER_NAME, namespace=SERVE_NAMESPACE)
    deployments = ray.get(controller._all_running_replicas.remote(), timeout=2)
    sources = []
    for deployment, replicas in deployments.items():
        if deployment.app_name != application:
            continue
        for replica in replicas:
            metadata = replica.routing_stats.get("kv_event_metadata")
            if not metadata or not metadata.get("endpoint"):
                continue
            # RunningReplicaInfo advertises an actor name, not an actor handle.
            actor = ray.get_actor(replica.actor_name, namespace=SERVE_NAMESPACE)
            sources.append(
                {
                    "application": application,
                    "deployment": deployment.name,
                    "replica_id": replica.replica_id.unique_id,
                    "replica_full_id": replica.replica_id.to_full_id_str(),
                    "actor_id": actor._actor_id.hex(),
                    **metadata,
                }
            )
    return sources


def _hash(value: Any) -> str | None:
    if value is None:
        return None
    return value.hex() if isinstance(value, bytes) else str(value)


def event_record(event: dict[str, Any]) -> dict[str, Any]:
    """Keep block identity and tier metadata without saving prompt contents."""
    result = {
        key: event[key]
        for key in (
            "type",
            "medium",
            "group_idx",
            "locality",
            "ownership",
            "block_size",
            "kv_cache_spec_kind",
            "kv_cache_spec_sliding_window",
        )
        if key in event
    }
    hashes = event.get("block_hashes")
    if hashes is not None:
        result["block_hashes"] = [_hash(value) for value in hashes]
        result["block_count"] = len(hashes)
    if "parent_block_hash" in event:
        result["parent_block_hash"] = _hash(event["parent_block_hash"])
    return result


class GPUKVEventRecorder:
    """A telemetry collector; dependency/transport failures become audit rows.

    Each ZMQ batch is written once, even when live and replay deliveries overlap.
    A BlockRemoved denotes a prefix index removal, not request preemption. For hybrid models, group counts are retained
    separately and must not be summed into unique prompt-token counts.
    """

    def __init__(
        self,
        path: Path,
        application: str,
        *,
        discover: Callable[[], list[dict[str, Any]]] | None = None,
        interval_s: float = 1,
    ):
        self.path = path
        self._discover = discover or (lambda: discover_kv_event_sources(application))
        self.interval_s = interval_s
        self._stop = threading.Event()
        self._ready = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name="gpu-kv-event-recorder", daemon=True
        )
        self._fh = None
        self.records_written = 0
        self.error_records = 0
        self.batches_written = 0
        self.sequence_gaps = 0
        self._sources: dict[tuple, dict[str, Any]] = {}

    def start(self) -> None:
        self._fh = self.path.open("w")
        self._thread.start()
        # Establish initial subscriptions/replay before the load generator starts.
        self._ready.wait(5)

    def stop(self, timeout_s: float | None = None) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        if not self._thread.is_alive() and self._fh is not None:
            self._fh.close()

    def _write(self, kind: str, **fields: Any) -> None:
        row = {
            "schema_version": 1,
            "record_type": kind,
            "received_timestamp_ns": time.time_ns(),
            **fields,
        }
        self._fh.write(json.dumps(row, separators=(",", ":")) + "\n")
        self._fh.flush()
        self.records_written += 1
        if row.get("error"):
            self.error_records += 1

    @staticmethod
    def _identity(source: dict[str, Any]) -> dict[str, Any]:
        return source["metadata"]

    def _advance(self, source: dict[str, Any]) -> None:
        while source["next"] in source["pending"]:
            source["pending"].remove(source["next"])
            source["next"] += 1

    def _accept(
        self, source: dict[str, Any], sequence: int, payload: bytes, delivery: str
    ) -> None:
        if sequence < source["next"] or sequence in source["pending"]:
            return
        # EventBatch is an array-like msgspec.Struct: [ts, events, dp_rank?].
        batch = self._msgpack.decode(payload)
        stamp, events = batch[:2]
        rank = batch[2] if len(batch) > 2 else source["metadata"].get("dp_rank")
        self._write(
            "kv_event_batch",
            **self._identity(source),
            sequence=sequence,
            delivery=delivery,
            timestamp_ns=round(stamp * 1e9),
            data_parallel_rank=rank,
            events=[event_record(event) for event in events],
        )
        self.batches_written += 1
        source["pending"].add(sequence)
        self._advance(source)

    def _request_replay(self, source: dict[str, Any]) -> None:
        if source["replay"] is None or source["replaying"]:
            return
        source["replay"].send_multipart([b"", source["next"].to_bytes(8, "big")])
        source["replaying"] = True
        source["replay_max"] = None
        source["replay_started"] = time.monotonic()

    def _finish_replay(self, source: dict[str, Any]) -> None:
        # A replay that starts after a missing sequence means that prefix of
        # the publisher's bounded buffer is gone. Report the hole explicitly.
        limit = source["replay_max"]
        while limit is not None and source["next"] <= limit:
            following = min(n for n in source["pending"] if n <= limit)
            self._write(
                "sequence_gap",
                **self._identity(source),
                first_missing_sequence=source["next"],
                last_missing_sequence=following - 1,
                error="Missing batches were unavailable from the publisher replay buffer",
            )
            self.sequence_gaps += 1
            source["next"] = following
            self._advance(source)
        source["replaying"] = False

    def _reconcile(self, sources: list[dict[str, Any]]) -> None:
        active = set()
        for metadata in sources:
            key = (
                metadata["replica_id"],
                metadata.get("actor_id"),
                metadata["endpoint"],
                metadata.get("dp_rank", 0),
            )
            active.add(key)
            if key in self._sources:
                continue
            sub = self._ctx.socket(self._zmq.SUB)
            sub.setsockopt(self._zmq.SUBSCRIBE, b"")
            sub.setsockopt(self._zmq.RCVHWM, 100_000)
            sub.connect(metadata["endpoint"])
            self._poller.register(sub, self._zmq.POLLIN)
            replay = None
            if metadata.get("replay_endpoint"):
                replay = self._ctx.socket(self._zmq.DEALER)
                replay.connect(metadata["replay_endpoint"])
                self._poller.register(replay, self._zmq.POLLIN)
            source = dict(
                metadata=metadata,
                sub=sub,
                replay=replay,
                next=0,
                pending=set(),
                replaying=False,
                replay_max=None,
                departed=False,
            )
            self._sources[key] = source
            self._write(
                "source_added",
                **metadata,
                error=None
                if replay is not None
                else "No replay endpoint; slow-joiner/tail loss cannot be recovered",
            )
            self._request_replay(source)
        for key, source in self._sources.items():
            if key not in active and not source["departed"]:
                source["departed"] = True
                # Keep draining the connection through graceful replica removal.
                self._write("source_departed", **self._identity(source))
                self._request_replay(source)

    def _poll(self) -> None:
        available = dict(self._poller.poll(50))
        for source in self._sources.values():
            for socket, delivery in (
                (source["sub"], "live"),
                (source["replay"], "replay"),
            ):
                if socket is None or socket not in available:
                    continue
                # Bound draining so a busy replica cannot starve discovery/shutdown.
                for _ in range(1000):
                    try:
                        frames = socket.recv_multipart(flags=self._zmq.NOBLOCK)
                    except self._zmq.Again:
                        break
                    try:
                        if len(frames) != (4 if delivery == "replay" else 3):
                            raise ValueError(
                                f"Unexpected {delivery} frame count: {len(frames)}"
                            )
                        _, sequence_bytes, payload = frames[-3:]
                        if sequence_bytes == (-1).to_bytes(8, "big", signed=True):
                            self._finish_replay(source)
                            continue
                        sequence = int.from_bytes(sequence_bytes, "big")
                        self._accept(source, sequence, payload, delivery)
                        if delivery == "replay":
                            source["replay_max"] = max(
                                sequence, source["replay_max"] or 0
                            )
                    except Exception as exc:
                        self._write(
                            "receive_error",
                            **self._identity(source),
                            error=f"{type(exc).__name__}: {exc}",
                        )

    def _run(self) -> None:
        try:
            import msgspec
            import zmq

            self._zmq, self._msgpack = zmq, msgspec.msgpack
            self._ctx = zmq.Context()
            self._poller = zmq.Poller()
            self._write(
                "recorder_started",
                timestamp_semantics="publisher batch time; not per-block allocator instruction time",
            )
            next_discovery = 0.0
            while not self._stop.is_set():
                if time.monotonic() >= next_discovery:
                    try:
                        self._reconcile(self._discover())
                    except Exception as exc:
                        self._write(
                            "discovery_error", error=f"{type(exc).__name__}: {exc}"
                        )
                    next_discovery = time.monotonic() + self.interval_s
                    for source in self._sources.values():
                        if source["departed"]:
                            continue
                        if (
                            source["replaying"]
                            and time.monotonic() - source["replay_started"] > 3
                        ):
                            self._write(
                                "replay_timeout",
                                **self._identity(source),
                                error="Publisher did not complete replay within 3 seconds",
                            )
                            # Reconnect instead of mixing late replies from old requests.
                            self._poller.unregister(source["replay"])
                            source["replay"].close(linger=0)
                            source["replay"] = self._ctx.socket(zmq.DEALER)
                            source["replay"].connect(
                                source["metadata"]["replay_endpoint"]
                            )
                            self._poller.register(source["replay"], zmq.POLLIN)
                            source["replaying"] = False
                        self._request_replay(source)
                self._poll()
                if self._sources and all(
                    not s["replaying"] for s in self._sources.values()
                ):
                    self._ready.set()
            # Complete any old replay before requesting a fresh final snapshot.
            deadline = time.monotonic() + 1
            while (
                any(s["replaying"] for s in self._sources.values())
                and time.monotonic() < deadline
            ):
                self._poll()
            # Last replay catches a tail lost on PUB even without a later sequence.
            for source in self._sources.values():
                if not source["departed"]:
                    self._request_replay(source)
            deadline = time.monotonic() + 1.5
            while (
                any(s["replaying"] for s in self._sources.values())
                and time.monotonic() < deadline
            ):
                self._poll()
            for source in self._sources.values():
                self._write(
                    "source_finished",
                    **self._identity(source),
                    next_sequence=source["next"],
                    pending_sequences=sorted(source["pending"]),
                    replay_incomplete=source["replaying"],
                    error=(
                        "Replay did not complete"
                        if source["replaying"]
                        else "Unrecovered sequence gap"
                        if source["pending"]
                        else None
                    ),
                )
            self._write(
                "recorder_stopped",
                batches=self.batches_written,
                source_count=len(self._sources),
                sequence_gaps=self.sequence_gaps,
                error_records=self.error_records,
                error="No KV event sources were discovered"
                if not self._sources
                else None,
            )
        except Exception as exc:
            self._write("recorder_error", error=f"{type(exc).__name__}: {exc}")
        finally:
            self._ready.set()
            for source in self._sources.values():
                for socket in (source["sub"], source["replay"]):
                    if socket is not None:
                        socket.close(linger=0)
            if hasattr(self, "_ctx"):
                self._ctx.term()
