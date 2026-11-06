#!/usr/bin/env python3
"""Join router reservations, native labels, and independent scheduler traces."""

import argparse
import bisect
import csv
import html
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
BANDS = [
    (0, 705, "≤704", "#c23b80"),
    (705, 4000, "705–4K", "#2a78d6"),
    (4000, 16000, "4K–16K", "#eb6834"),
    (16000, 48000, "16K–48K", "#1baf7a"),
    (48000, math.inf, "≥48K", "#8a5cd1"),
]


def key(record):
    value = record.get("external_request_id", record["request_id"])
    match = ENGINE_ID.match(value)
    return match.group(1) if match else value


def read_traces(directory):
    groups = defaultdict(
        lambda: dict(
            queued={},
            chunks=defaultdict(list),
            completed={},
            preempted=defaultdict(list),
        )
    )
    admitted, queued, dropped = {}, {}, 0
    for path in sorted(directory.glob("*.jsonl")):
        group = groups[path.name.rsplit(".", 2)[0]]
        for line in path.open():
            try:
                event = json.loads(line)
            except ValueError:
                continue
            kind = event["event"]
            if kind == "request_queued":
                group["queued"][event["request_id"]] = event["wall_time_s"]
                queued[key(event)] = event
            elif kind == "request_admitted":
                admitted.setdefault(key(event), event)
            elif kind == "batch_scheduled":
                for request in event["requests"]:
                    if request["phase"] == "prefill":
                        group["chunks"][request["request_id"]].append(
                            (event["batch_index"], request["scheduled_tokens"])
                        )
            elif kind == "batch_completed":
                group["completed"][event["batch_index"]] = event["wall_time_s"]
            elif kind == "request_preempted":
                group["preempted"][event["request_id"]].append(
                    (event["batch_index"], event["wall_time_s"])
                )
            elif kind == "trace_events_dropped":
                dropped += event["count"]
    replay, incomplete = {}, 0
    for name, group in groups.items():
        deltas = []
        for rid, arrival in group["queued"].items():
            chunks = group["chunks"].get(rid, [])
            if not chunks or any(
                batch not in group["completed"] for batch, _ in chunks
            ):
                incomplete += 1
                continue
            starts = [(0, arrival)] + sorted(group["preempted"].get(rid, []))
            for index, (first_batch, at) in enumerate(starts):
                next_batch = (
                    starts[index + 1][0] if index + 1 < len(starts) else math.inf
                )
                deltas.append(
                    (
                        at,
                        sum(
                            tokens
                            for batch, tokens in chunks
                            if first_batch <= batch < next_batch
                        ),
                    )
                )
            deltas.extend(
                (group["completed"][batch], -tokens) for batch, tokens in chunks
            )
        deltas.sort()
        times, levels, level = [], [], 0
        for at, delta in deltas:
            level += delta
            times.append(at)
            levels.append(level)
        replay[name] = (times, levels)
    return (
        replay,
        admitted,
        queued,
        dict(dropped_trace_events=dropped, incomplete_trace_requests=incomplete),
    )


def load_before(replay, engine, at):
    times, levels = replay[engine]
    index = bisect.bisect_left(times, at)
    return levels[index - 1] if index else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--traces", required=True, type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    out = root / "analysis/router-trace-comparison"
    out.mkdir(parents=True, exist_ok=True)
    replay, admitted, queued, quality = read_traces(args.traces)
    samples = {
        r["request_id"]: r
        for r in csv.DictReader(
            (root / "analysis/prefill-calibration/samples.csv").open()
        )
    }
    traced = list(csv.DictReader((root / "analysis/load_requests.csv").open()))
    rows = []
    for trace in traced:
        rid = trace["request_id"]
        if rid not in samples or rid not in admitted or rid not in queued:
            continue
        sample, admission, arrival = samples[rid], admitted[rid], queued[rid]
        engine = trace["engine"]
        routed = float(sample["routed_at_s"])
        rows.append(
            dict(
                request_id=rid,
                worker_id=sample["worker_id"],
                engine=engine,
                routed_at_s=routed,
                engine_arrival_at_s=arrival["wall_time_s"],
                routing_to_engine_ms=(arrival["wall_time_s"] - routed) * 1000,
                router_outstanding_tokens=float(sample["workload_tokens"]),
                engine_remaining_at_routing_tokens=load_before(replay, engine, routed),
                engine_remaining_at_engine_arrival_tokens=float(
                    trace["engine_load_tokens"]
                ),
                replay_remaining_at_engine_arrival_tokens=load_before(
                    replay, engine, arrival["wall_time_s"]
                ),
                router_own_effective_tokens=float(
                    sample["router_effective_prefill_tokens"]
                ),
                engine_own_uncached_tokens=float(trace["own_uncached"]),
                engine_local_cached_tokens=admission.get("local_cached_prefix_tokens"),
                engine_external_cached_tokens=admission.get(
                    "external_cached_prefix_tokens"
                ),
                native_initial_queue_ms=float(sample["queue_time_ms"]),
                trace_initial_queue_ms=admission["queue_wait_ms"],
                trace_total_wait_ms=float(trace["queue_ms"]),
                trace_prefill_step_time_ms=float(trace["prefill_ms"]),
                trace_preemptions_before_first_token=int(
                    float(trace["preemptions_before_first_token"])
                ),
            )
        )
    if not rows:
        raise ValueError(
            "No completed requests joined across router, native, and scheduler records"
        )
    with (out / "points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    # Same cohort in all three columns; preemption/recompute is kept in CSV.
    plotted = [r for r in rows if not r["trace_preemptions_before_first_token"]]
    if not plotted:
        raise ValueError("All matched requests were preempted")
    columns = [
        (
            "engine_remaining_at_engine_arrival_tokens",
            "trace_total_wait_ms",
            "Trace: wait before + between prefill steps",
        ),
        (
            "engine_remaining_at_engine_arrival_tokens",
            "native_initial_queue_ms",
            "Trace work → native first-scheduling queue",
        ),
        (
            "router_outstanding_tokens",
            "native_initial_queue_ms",
            "Router estimate → native first-scheduling queue",
        ),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(20, 11), sharey="row")
    for row_index, maximum in enumerate([None, 100000]):
        row_top = 1.0
        for ax, (xfield, yfield, title) in zip(axes[row_index], columns):
            selected = [r for r in plotted if maximum is None or r[xfield] <= maximum]
            for low, high, label, color in BANDS:
                points = [
                    r for r in selected if low <= r["engine_own_uncached_tokens"] < high
                ]
                ax.scatter(
                    [r[xfield] / 1000 for r in points],
                    [r[yfield] / 1000 for r in points],
                    color=color,
                    s=13,
                    alpha=0.5,
                    label=f"{label} own uncached",
                )
            row_top = max(row_top, max((r[yfield] / 1000 for r in selected), default=1))
            ax.axhline(1, color="#555", ls="--", lw=1)
            ax.set_title(
                title + ("\n0–100K zoom" if maximum else "\nFull range"), fontsize=11
            )
            ax.set_xlabel(
                (
                    "Router reservations before routing"
                    if xfield.startswith("router")
                    else "Reconstructed remaining engine work at arrival"
                )
                + "\n(thousands of tokens)"
            )
            ax.set_xlim(left=0, right=maximum / 1000 if maximum else None)
            ax.grid(alpha=0.2)
            ax.spines[["top", "right"]].set_visible(False)
        axes[row_index, 0].set_ylim(-row_top * 0.04, row_top * 1.08)
        axes[row_index, 0].set_ylabel("Queue / waiting time (seconds)")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        f"Same traced H100 run — {len(plotted)} matched requests, {len(rows) - len(plotted)} preempted requests excluded from plot",
        fontsize=14,
    )
    fig.text(
        0.03,
        0.015,
        "Left column subtracts every prefill step involving the request. Middle/right stop at first scheduling. Router and engine arrival timestamps differ; CSV also includes engine work at routing time.",
        fontsize=9,
    )
    fig.tight_layout(rect=(0.01, 0.04, 1, 0.89))
    for suffix in ["png", "pdf"]:
        fig.savefig(out / f"comparison.{suffix}", dpi=160)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, xfield, yfield, title, xlabel, ylabel in [
        (
            axes[0],
            "engine_remaining_at_routing_tokens",
            "router_outstanding_tokens",
            "Backlog at the same routing timestamp",
            "Trace remaining work (tokens)",
            "Router reservations (tokens)",
        ),
        (
            axes[1],
            "engine_own_uncached_tokens",
            "router_own_effective_tokens",
            "Per-request reservation versus computed work",
            "Trace uncached prompt tokens",
            "Router effective tokens (tier-weighted)",
        ),
        (
            axes[2],
            "trace_initial_queue_ms",
            "native_initial_queue_ms",
            "Validate native queue-time labels",
            "Trace queue-to-first-scheduling (ms)",
            "Native queue-to-first-scheduling (ms)",
        ),
    ]:
        ax.scatter(
            [r[xfield] for r in rows], [r[yfield] for r in rows], s=12, alpha=0.4
        )
        upper = max(max(r[xfield], r[yfield]) for r in rows)
        ax.plot([0, upper], [0, upper], color="black", ls="--", lw=1)
        ax.set(title=title, xlabel=xlabel, ylabel=ylabel)
        ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(out / "validation.png", dpi=160)
    plt.close(fig)
    queue_delta = [
        abs(r["native_initial_queue_ms"] - r["trace_initial_queue_ms"]) for r in rows
    ]
    report = dict(
        **quality,
        native_samples=len(samples),
        traced_completed_requests=len(traced),
        matched_requests=len(rows),
        preempted_requests_excluded_from_plot=len(rows) - len(plotted),
        native_vs_trace_initial_queue_abs_error_ms=dict(
            median=float(np.median(queue_delta)),
            p95=float(np.quantile(queue_delta, 0.95)),
            maximum=max(queue_delta),
        ),
        engine_arrival_replay_vs_original_max_token_difference=max(
            abs(
                r["replay_remaining_at_engine_arrival_tokens"]
                - r["engine_remaining_at_engine_arrival_tokens"]
            )
            for r in rows
        ),
        note="Ground-truth replay is retrospective and uses completed chunks. Missing trace requests or dropped events limit completeness. Effective router reservations may include cache-tier weights.",
    )
    (out / "summary.json").write_text(json.dumps(report, indent=2))
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Router versus scheduler traces</title><style>img{width:100%}body{font:16px system-ui;max-width:1800px;margin:auto}</style><h1>Router versus scheduler traces</h1><pre>'
        + html.escape(json.dumps(report, indent=2))
        + '</pre><a href="points.csv">Paired points CSV</a><img src="comparison.png"><img src="validation.png">'
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
