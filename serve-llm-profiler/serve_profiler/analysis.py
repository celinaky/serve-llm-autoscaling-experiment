"""Reconstruct remaining prefill work and estimated first-execution waits.

Uses the original scheduler JSONL schema. Cache hits are discovered at admission,
so arrival loads are an offline reconstruction, not an online observable.
"""
from __future__ import annotations

import bisect
import csv
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path

BIN_WIDTH = 20_000
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def external_id(event: dict) -> str:
    value = event.get("external_request_id") or event["request_id"]
    match = ENGINE_ID.match(value)
    return match.group(1) if match else value


def event_time(event: dict, field: str) -> float:
    value = float(event.get(field, event["monotonic_s"]))
    if not math.isfinite(value):
        raise ValueError("non-finite timestamp")
    return value


def read_clients(root: Path) -> dict:
    clients = {}
    paths = sorted(root.glob("workload/*/profile_export.jsonl"))
    # Allow offline analysis of historical harness runs too.
    paths += sorted(root.glob("benchmark/*/profile_export.jsonl"))
    for path in paths:
        with path.open() as stream:
            for line in stream:
                if not line.strip():
                    continue
                record = json.loads(line)
                meta = record.get("metadata") or {}
                if meta.get("benchmark_phase") != "profiling":
                    continue
                rid = meta.get("x_request_id")
                if rid:
                    if rid in clients:
                        raise ValueError(f"duplicate client request ID: {rid}")
                    clients[rid] = {
                        "phase": path.parent.name,
                        "client_error": bool(record.get("error")),
                    }
    return clients


def read_engines(paths: list[Path]) -> tuple[dict, dict]:
    engines = defaultdict(list)
    quality = {
        "trace_files": [str(p.resolve()) for p in paths],
        "malformed_lines": 0,
        "dropped_events": 0,
    }
    for path in paths:
        # host_pid.0000.jsonl and host_pid.0001.jsonl are one engine stream.
        engine = re.sub(r"\.\d+\.jsonl$", "", path.name)
        with path.open() as stream:
            for number, line in enumerate(stream, 1):
                if not line.strip():
                    continue
                try:
                    event = json.loads(line)
                    event_time(event, "monotonic_s")
                    kind = event["event"]
                except (ValueError, KeyError, TypeError):
                    quality["malformed_lines"] += 1
                    continue
                if kind == "trace_events_dropped":
                    quality["dropped_events"] += int(event["count"])
                engines[engine].append(event)
    quality["event_index_gaps"] = 0
    for events in engines.values():
        indices = [e["event_index"] for e in events if "event_index" in e]
        # Legacy traces have no event_index. New traces expose dropped events
        # even if a drop marker could not enter the bounded writer queue.
        if indices:
            quality["event_index_gaps"] += int(indices[0] != 1)
            quality["event_index_gaps"] += sum(
                following != previous + 1
                for previous, following in zip(indices, indices[1:])
            )
    return dict(engines), quality


def reconstruct_engine(engine: str, events: list[dict], clients: dict | None):
    queued, admissions, scheduled, completed, first, finished = {}, {}, {}, {}, {}, {}
    chunks, preemptions = defaultdict(list), defaultdict(list)
    skipped = Counter()
    invalid_reasons = []
    for event in events:
        kind = event["event"]
        if kind == "request_queued":
            rid = event["request_id"]
            if rid in queued:
                invalid_reasons.append("duplicate_request_arrival")
            queued[rid] = event
        elif kind == "request_admitted":
            admissions.setdefault(event["request_id"], event)
        elif kind == "request_first_token":
            first.setdefault(event["request_id"], event)
        elif kind == "request_finished":
            finished[event["request_id"]] = event
        elif kind == "request_preempted":
            preemptions[event["request_id"]].append(event)
        elif kind == "batch_scheduled":
            batch = int(event["batch_index"])
            if batch in scheduled:
                invalid_reasons.append("duplicate_batch_index_or_engine_restart")
            scheduled[batch] = event_time(event, "scheduled_at_monotonic_s")
            for request in event["requests"]:
                if request["phase"] == "prefill" and request["scheduled_tokens"] > 0:
                    chunks[request["request_id"]].append(
                        (batch, int(request["scheduled_tokens"]))
                    )
        elif kind == "batch_completed":
            batch = int(event["batch_index"])
            if batch in completed:
                invalid_reasons.append("duplicate_batch_completion")
            completed[batch] = event_time(event, "completed_at_monotonic_s")

    # Never silently omit an unresolved request's contribution to other points.
    # Exclude every arrival after its arrival: its workload is not reconstructible.
    contaminated_from = math.inf
    incomplete = 0
    deltas = []
    for rid, event in queued.items():
        at = event_time(event, "queued_at_monotonic_s")
        cs = sorted(chunks.get(rid, []))
        if rid not in first or not cs or any(b not in completed for b, _ in cs):
            contaminated_from = min(contaminated_from, at)
            incomplete += 1
            continue
        starts = [(0, at)] + sorted(
            (int(p["batch_index"]), event_time(p, "monotonic_s"))
            for p in preemptions.get(rid, [])
        )
        # Add observed prefill chunks at original arrival, then add recompute
        # chunks at preemption. Decode-time recomputation contributes load too.
        for index, (batch, start) in enumerate(starts):
            until = starts[index + 1][0] if index + 1 < len(starts) else math.inf
            deltas.append((start, sum(t for b, t in cs if batch <= b < until)))
        deltas.extend((completed[b], -tokens) for b, tokens in cs)
    unknown = set(chunks) - set(queued)
    if unknown:
        invalid_reasons.append("chunks_without_arrival")
    if invalid_reasons:
        contaminated_from = -math.inf
    changes = defaultdict(int)
    for at, delta in deltas:
        changes[at] += delta
    times, levels, level = [], [], 0
    for at, delta in sorted(changes.items()):
        level += delta
        times.append(at)
        levels.append(level)
        if level < 0:
            invalid_reasons.append("negative_reconstructed_load")
    if invalid_reasons:
        contaminated_from = -math.inf

    points = []
    for rid, arrival in queued.items():
        client_id = external_id(arrival)
        client = (
            clients.get(client_id) if clients is not None else {"phase": "trace-only"}
        )
        if client is None:
            continue
        if client.get("client_error"):
            skipped["client_error"] += 1
            continue
        at = event_time(arrival, "queued_at_monotonic_s")
        if at >= contaminated_from:
            skipped["incomplete_engine_history"] += 1
            continue
        admission, token = admissions.get(rid), first.get(rid)
        if admission is None or token is None:
            skipped["missing_admission_or_first_token"] += 1
            continue
        batch = min(b for b, _ in chunks[rid])
        if int(admission["batch_index"]) != batch:
            skipped["first_prefill_batch_mismatch"] += 1
            continue
        if batch - 1 not in completed:
            skipped["no_previous_batch"] += 1
            continue
        start = max(scheduled[batch], completed[batch - 1])
        if start < at or completed[batch] < start:
            skipped["invalid_batch_timing"] += 1
            continue
        if any(int(p["batch_index"]) <= batch for p in preemptions.get(rid, [])):
            skipped["preempted_before_first_execution"] += 1
            continue
        index = bisect.bisect_left(times, at)  # excludes its own arrival increment
        initial_chunks = sum(
            t for b, t in chunks[rid] if b <= int(token["batch_index"])
        )
        points.append(
            {
                "request_id": client_id,
                "engine": engine,
                "phase": client["phase"],
                "engine_arrival_monotonic_s": at,
                "load_tokens": levels[index - 1] if index else 0,
                "queue_ms": (start - at) * 1000,
                "scheduling_queue_ms": (scheduled[batch] - at) * 1000,
                "first_prefill_batch": batch,
                "estimated_execution_start_monotonic_s": start,
                "prompt_tokens": int(admission["prompt_tokens"]),
                "cached_prefix_tokens": int(admission["cached_prefix_tokens"]),
                "own_uncached_tokens": int(admission["prompt_tokens"])
                - int(admission["cached_prefix_tokens"]),
                "observed_initial_prefill_tokens": initial_chunks,
                "preemptions": len(preemptions.get(rid, [])),
            }
        )
    return points, {
        "engine": engine,
        "requests": len(queued),
        "incomplete_requests": incomplete,
        "preemptions": sum(map(len, preemptions.values())),
        "invalid_reasons": sorted(set(invalid_reasons)),
        "excluded": dict(skipped),
    }


def linear_fit(points: list[dict]) -> dict:
    result = {
        "status": "unavailable",
        "samples": len(points),
        "target": "estimated_first_execution_queue_ms",
        "input": "remaining_uncached_prefill_tokens_at_engine_arrival",
    }
    if len(points) < 2:
        return {**result, "reason": "Need at least two points to identify a line"}
    xs, ys = [p["load_tokens"] for p in points], [p["queue_ms"] for p in points]
    xmean, ymean = sum(xs) / len(xs), sum(ys) / len(ys)
    xx = sum((x - xmean) ** 2 for x in xs)
    if xx == 0:
        return {**result, "reason": "All observed loads are identical"}
    slope = sum((x - xmean) * (y - ymean) for x, y in zip(xs, ys)) / xx
    intercept = ymean - slope * xmean
    errors = [y - (intercept + slope * x) for x, y in zip(xs, ys)]
    sse = sum(e * e for e in errors)
    yy = sum((y - ymean) ** 2 for y in ys)
    finite = all(math.isfinite(v) for v in (slope, intercept, sse))
    if not finite:
        return {**result, "reason": "Non-finite fit"}
    return {
        **result,
        "status": "fitted" if slope > 0 else "nonpositive_slope",
        "slope_ms_per_token": slope,
        "intercept_ms": intercept,
        "rmse_ms": math.sqrt(sse / len(points)),
        "mae_ms": sum(abs(e) for e in errors) / len(points),
        "r_squared": 1 - sse / yy if yy else None,
        "observed_load_min_tokens": min(xs),
        "observed_load_max_tokens": max(xs),
        "policy_conversion": "(target_queue_ms - intercept_ms) / slope_ms_per_token",
        "caveat": "Targets below intercept are infeasible; do not extrapolate beyond observed loads.",
    }


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def coverage(points: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for point in points:
        load = point["load_tokens"]
        index = 0 if load == 0 else math.ceil(load / BIN_WIDTH)
        groups[index].append(point["queue_ms"])
    highest = max(groups, default=0)
    rows = []
    for index in range(highest + 1):
        values = groups[index]
        rows.append(
            {
                "bin": "0"
                if index == 0
                else f"({(index-1)*BIN_WIDTH},{index*BIN_WIDTH}]",
                "requests": len(values),
                "mean_queue_ms": sum(values) / len(values) if values else "",
            }
        )
    return rows


def plot(out: Path, points: list[dict], fit: dict) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(13, 5), layout="constrained")
    for phase in sorted({p["phase"] for p in points}):
        subset = [p for p in points if p["phase"] == phase]
        axes[0].scatter(
            [p["load_tokens"] / 1000 for p in subset],
            [p["queue_ms"] for p in subset],
            s=9,
            alpha=0.4,
            label=phase,
        )
    if "slope_ms_per_token" in fit:
        xs = [
            min(p["load_tokens"] for p in points),
            max(p["load_tokens"] for p in points),
        ]
        axes[0].plot(
            [x / 1000 for x in xs],
            [fit["intercept_ms"] + fit["slope_ms_per_token"] * x for x in xs],
            color="black",
            label="OLS fit",
        )
        axes[1].scatter(
            [p["load_tokens"] / 1000 for p in points],
            [
                p["queue_ms"]
                - fit["intercept_ms"]
                - fit["slope_ms_per_token"] * p["load_tokens"]
                for p in points
            ],
            s=9,
            alpha=0.4,
        )
    axes[0].set_ylabel("Estimated initial execution queue (ms)")
    axes[1].set_ylabel("Fit residual (ms)")
    axes[1].axhline(0, color="black", lw=1)
    for ax in axes:
        ax.set_xlabel("Remaining uncached prefill work at arrival (K tokens)")
        ax.grid(alpha=0.2)
    if points:
        axes[0].legend(fontsize=8)
    fig.suptitle(
        f"AgentX prefill queue calibration · {len(points)} points · CPU-callback estimate"
    )
    fig.savefig(out / "queue_vs_load.png", dpi=150)
    plt.close(fig)


def analyze(
    root: Path, traces: Path | None = None, *, trace_only: bool = False
) -> dict:
    root = root.resolve()
    out = root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    directory = traces or root / "traces"
    paths = sorted(directory.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"No scheduler traces at {directory}")
    clients = None if trace_only else read_clients(root)
    if clients == {}:
        raise ValueError(
            "No profiling client records; use --trace-only explicitly for historical traces"
        )
    engines, quality = read_engines(paths)
    points, reports = [], []
    for engine, events in engines.items():
        rows, report = reconstruct_engine(engine, events, clients)
        points.extend(rows)
        reports.append(report)
    # Event loss can change every reconstructed workload. Keep the evidence but
    # do not publish misleading coefficients or points from a broken stream.
    if (
        quality["dropped_events"]
        or quality["malformed_lines"]
        or quality["event_index_gaps"]
    ):
        quality["points_rejected_for_trace_loss"] = len(points)
        points = []
    fit = linear_fit(points)
    quality.update(
        engines=reports,
        client_records=None if clients is None else len(clients),
        trace_only=trace_only,
        points=len(points),
    )
    if clients is not None:
        seen = {
            external_id(e)
            for es in engines.values()
            for e in es
            if e["event"] == "request_queued"
        }
        quality["client_requests_missing_from_traces"] = len(set(clients) - seen)
    fit["per_phase"] = {
        phase: linear_fit([p for p in points if p["phase"] == phase])
        for phase in sorted({p["phase"] for p in points})
    }
    write_json(out / "quality.json", quality)
    write_json(out / "fit.json", fit)
    fields = (
        list(points[0])
        if points
        else ["request_id", "engine", "phase", "load_tokens", "queue_ms"]
    )
    write_csv(out / "points.csv", points, fields)
    write_csv(
        out / "coverage.csv", coverage(points), ["bin", "requests", "mean_queue_ms"]
    )
    plot(out, points, fit)
    return {"output": str(out), "fit": fit, "quality": quality}
