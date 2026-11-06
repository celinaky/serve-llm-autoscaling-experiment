#!/usr/bin/env python3
"""Compare saved router learning samples with traced offline H100 profiles.

The offline x value is reconstructed remaining engine work. The online x
value is the selected worker's undecayed router reservation before routing.
They are intentionally labeled separately. No missing engine work is inferred.

Example:
  python tools/compare_online_queue_profile.py ONLINE_RUN OFFLINE_RUN ...
"""

import argparse
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


BANDS = [
    (0, 705, "≤704", "#c23b80"),
    (705, 4000, "705–4K", "#2a78d6"),
    (4000, 16000, "4K–16K", "#eb6834"),
    (16000, 48000, "16K–48K", "#1baf7a"),
    (48000, math.inf, "≥48K", "#8a5cd1"),
]
ENGINE_ID = re.compile(r"^(?:chatcmpl-|cmpl-)?(.+)-[0-9a-f]{8}$")
SURFACE, GRID = "#fcfcfb", "#e4e3df"


def canonical_id(request_id):
    match = ENGINE_ID.match(request_id)
    return match.group(1) if match else request_id


def band(value):
    return next(label for low, high, label, _ in BANDS if low <= value < high)


def read_online(root):
    benchmark = {}
    for path in sorted((root / "benchmark").glob("*/profile_export.jsonl")):
        for line in path.open():
            row = json.loads(line)
            if row.get("error"):
                continue
            metrics = row.get("metrics", {})
            benchmark[row["metadata"]["x_request_id"]] = metrics
    rows = []
    with (root / "analysis/prefill-calibration/samples.csv").open() as stream:
        for sample in csv.DictReader(stream):
            metrics = benchmark[sample["request_id"]]
            prompt = metrics["input_sequence_length"]["value"]
            cached = metrics["usage_prompt_cache_read_tokens"]["value"]
            own = prompt - cached
            if own < 0:
                raise ValueError("Reported cached tokens exceed prompt length")
            rows.append(
                dict(
                    run=root.name,
                    request_id=sample["request_id"],
                    worker_id=sample["worker_id"],
                    routed_at_s=float(sample["routed_at_s"]),
                    observed_at_s=float(sample["observed_at_s"]),
                    workload_tokens=float(sample["workload_tokens"]),
                    initial_queue_ms=float(sample["queue_time_ms"]),
                    prompt_tokens=prompt,
                    reported_cached_tokens=cached,
                    own_uncached_tokens=own,
                    size_band=band(own),
                )
            )
    if len({row["request_id"] for row in rows}) != len(rows):
        raise ValueError("Duplicate online sample IDs")
    return rows


def read_offline(roots):
    rows, trace_dirs = [], set()
    for root in roots:
        config = json.loads((root / "engine.json").read_text())
        trace_dirs.add(Path(config["trace_dir"]))
        with (root / "analysis/load_requests.csv").open() as stream:
            for record in csv.DictReader(stream):
                if int(float(record["preemptions_before_first_token"])):
                    continue  # Same exclusion as the original profiling graph.
                own = float(record["own_uncached"])
                rows.append(
                    dict(
                        run=root.name,
                        request_id=record["request_id"],
                        worker_id=record["engine"],
                        workload_tokens=float(record["engine_load_tokens"]),
                        original_queue_ms=float(record["queue_ms"]),
                        own_uncached_tokens=own,
                        size_band=band(own),
                    )
                )
    wanted = {row["request_id"] for row in rows}
    initial = {}
    trace_files = []
    for directory in sorted(trace_dirs):
        for path in sorted(directory.glob("*.jsonl")):
            trace_files.append(str(path))
            for line in path.open():
                # Batch records dominate the traces; decode only admission
                # records, whose queue_wait_ms stops at the first schedule.
                if '"request_admitted"' not in line:
                    continue
                record = json.loads(line)
                key = canonical_id(
                    record.get("external_request_id", record["request_id"])
                )
                if key not in wanted:
                    continue
                value = record.get("queue_wait_ms")
                if value is not None:
                    initial.setdefault(key, float(value))
    missing = wanted - initial.keys()
    if missing:
        raise ValueError(
            f"Missing first-scheduling queue labels for {len(missing)} offline IDs"
        )
    for row in rows:
        row["initial_queue_ms"] = initial[row["request_id"]]
    return rows, trace_files


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def summarize(rows, dataset, field, width, maximum):
    cells = defaultdict(list)
    for row in rows:
        x = row["workload_tokens"]
        index = 0 if x == 0 else min(math.ceil(x / width), int(maximum / width) + 1)
        for size in [row["size_band"], "All sizes"]:
            cells[size, index].append(row[field])
    result = []
    for (size, index), values in sorted(cells.items()):
        label = (
            "0"
            if index == 0
            else f">{maximum / 1000:g}K"
            if index > maximum / width
            else f"{(index - 1) * width / 1000:g}–{index * width / 1000:g}K"
        )
        result.append(
            dict(
                dataset=dataset,
                size_band=size,
                bin_index=index,
                bin_label=label,
                requests=len(values),
                mean_queue_ms=float(np.mean(values)),
                median_queue_ms=float(np.median(values)),
                p95_queue_ms=float(np.quantile(values, 0.95)),
            )
        )
    return result


def format_axes(ax):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="y", color=GRID)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def binned_line(ax, cells, size, color, label=None):
    values = sorted(
        (r for r in cells if r["size_band"] == size), key=lambda r: r["bin_index"]
    )
    xs = [r["bin_index"] for r in values]
    ys = [r["mean_queue_ms"] / 1000 for r in values]
    # Thin lines retain sparse observations; hollow markers flag n < 3.
    ax.plot(xs, ys, color=color, alpha=0.35, lw=1)
    dense = [r for r in values if r["requests"] >= 3]
    sparse = [r for r in values if r["requests"] < 3]
    ax.scatter(
        [r["bin_index"] for r in dense],
        [r["mean_queue_ms"] / 1000 for r in dense],
        color=color,
        s=36,
        label=label,
        zorder=3,
    )
    ax.scatter(
        [r["bin_index"] for r in sparse],
        [r["mean_queue_ms"] / 1000 for r in sparse],
        facecolors="none",
        edgecolors=color,
        s=36,
        zorder=3,
    )
    if not dense and label:
        ax.plot([], [], color=color, label=label)


def save_figure(fig, destination):
    for suffix in ["png", "pdf"]:
        fig.savefig(destination.with_suffix("." + suffix), dpi=160, facecolor=SURFACE)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("online", type=Path)
    parser.add_argument("offline", type=Path, nargs="+")
    parser.add_argument("--out", type=Path)
    parser.add_argument("--bin-width", type=int, default=5000)
    parser.add_argument("--max-load", type=int, default=100000)
    args = parser.parse_args()
    if args.bin_width <= 0 or args.max_load <= 0 or args.max_load % args.bin_width:
        parser.error("max-load must be a positive multiple of bin-width")
    out = args.out or args.online / "analysis/offline-profile-comparison"
    out.mkdir(parents=True, exist_ok=True)
    online = read_online(args.online)
    offline, trace_files = read_offline(args.offline)
    write_csv(out / "online_points.csv", online)
    write_csv(out / "offline_points.csv", offline)
    tables = {}
    for name, rows, field in [
        ("Offline original", offline, "original_queue_ms"),
        ("Offline initial", offline, "initial_queue_ms"),
        ("Online initial", online, "initial_queue_ms"),
    ]:
        tables[name] = summarize(rows, name, field, args.bin_width, args.max_load)
    write_csv(out / "binned_summary.csv", [r for rows in tables.values() for r in rows])
    last = args.max_load // args.bin_width
    ticks = list(range(last + 1))
    labels = ["0"] + [
        f"{(i - 1) * args.bin_width / 1000:g}–{i * args.bin_width / 1000:g}K"
        for i in ticks[1:]
    ]
    within = {k: [r for r in v if r["bin_index"] <= last] for k, v in tables.items()}
    fig, axes = plt.subplots(2, 3, figsize=(21, 12), sharey="row", facecolor=SURFACE)
    titles = [
        "Offline: original profiling definition\nWait before + between prefill steps\nExcludes all of the request's prefill-step time",
        "Offline: first-scheduling queue time\nQueue admission → first scheduled",
        "Online: first-scheduling queue time\nQueue admission → first scheduled",
    ]
    zoom_max = min(30000, args.max_load)
    for row_index, maximum in enumerate([args.max_load, zoom_max]):
        end_bin = maximum // args.bin_width
        row_cells = {
            name: [r for r in cells if r["bin_index"] <= end_bin]
            for name, cells in within.items()
        }
        for ax, (name, cells), title in zip(axes[row_index], row_cells.items(), titles):
            for _, _, size, color in BANDS:
                binned_line(ax, cells, size, color, f"{size} own uncached")
            ax.axhline(1, color="#555555", ls="--", lw=1)
            prefix = f"0–{maximum / 1000:g}K zoom · " if row_index else ""
            ax.set_title(prefix + title, fontsize=12)
            ax.set_xticks(
                ticks[: end_bin + 1],
                labels[: end_bin + 1],
                rotation=65,
                ha="right",
                fontsize=8,
            )
            n = sum(r["requests"] for r in cells if r["size_band"] == "All sizes")
            source = (
                "Router-estimated outstanding prefill tokens"
                if name.startswith("Online")
                else "Reconstructed remaining engine prefill tokens"
            )
            ax.set_xlabel(
                f"{source} at arrival\n{n:,} requests in the displayed 0–{maximum / 1000:g}K range",
                fontsize=10,
            )
            ax.set_xlim(-0.4, end_bin + 0.4)
            format_axes(ax)
        row_top = (
            max(
                1,
                max(r["mean_queue_ms"] for cells in row_cells.values() for r in cells)
                / 1000,
            )
            * 1.12
        )
        axes[row_index, 0].set_ylim(0, row_top)
        axes[row_index, 0].set_ylabel("Mean queue time per token-load bin (seconds)")
    comparison_top = (
        max(r["mean_queue_ms"] for cells in within.values() for r in cells)
        / 1000
        * 1.12
    )
    handles, legend = axes[0, 0].get_legend_handles_labels()
    fig.legend(
        handles,
        legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.95),
        ncol=5,
        frameon=False,
    )
    fig.suptitle(
        "H100 queue profile comparison — same bins and request-size bands",
        fontsize=16,
        y=0.99,
    )
    fig.text(
        0.02,
        0.025,
        "Hollow markers: fewer than 3 requests in the cell. Dashed horizontal line: 1 second. No regression is used in these curves.\n"
        "Offline: 5 mixed-workload runs, no CPU KV offload. Online: 16-session ramp/hold, native CPU KV offload (64 GiB).\n"
        "Online size bands use post-completion reported prompt minus cached tokens; the online model's x feature remains the router reservation estimate.",
        fontsize=10,
    )
    fig.tight_layout(rect=(0.01, 0.09, 1, 0.92), h_pad=3)
    save_figure(fig, out / "offline_vs_online")

    fig, (scatter, means) = plt.subplots(2, 1, figsize=(14, 11), facecolor=SURFACE)
    for _, _, size, color in BANDS:
        rows = [r for r in online if r["size_band"] == size]
        scatter.scatter(
            [r["workload_tokens"] / 1000 for r in rows],
            [r["initial_queue_ms"] / 1000 for r in rows],
            color=color,
            s=28,
            alpha=0.65,
            label=f"{size} own uncached (n={len(rows)})",
        )
        binned_line(
            means, tables["Online initial"], size, color, f"{size} own uncached"
        )
    scatter.set(
        xlabel="Selected replica's router-estimated outstanding prefill tokens at routing (thousands)",
        ylabel="Initial engine queue time (seconds)",
        title=f"All {len(online)} collected requests — individual measurements",
    )
    scatter.legend(frameon=False, fontsize=9)
    means.set_xticks(
        ticks + [last + 1],
        labels + [f">{args.max_load / 1000:g}K"],
        rotation=60,
        ha="right",
        fontsize=9,
    )
    means.set(
        xlabel="Router-estimated outstanding prefill tokens on selected replica before adding this request",
        ylabel="Mean initial engine queue time (seconds)",
        title="Same binning as the offline profile; hollow markers have n < 3",
    )
    for ax in [scatter, means]:
        ax.axhline(1, color="#555555", ls="--", lw=1)
        ax.set_ylim(bottom=0)
        format_axes(ax)
    fig.suptitle(
        "Online H100 run — outstanding prefill estimate versus measured first-scheduling queue time",
        fontsize=13,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    save_figure(fig, out / "online_all_points")

    fig, axes = plt.subplots(2, 3, figsize=(18, 10), sharey=True, facecolor=SURFACE)
    for ax, size in zip(axes.flat, [b[2] for b in BANDS] + ["All sizes"]):
        binned_line(
            ax,
            within["Offline initial"],
            size,
            "#2a78d6",
            "Offline: engine remaining work",
        )
        binned_line(
            ax, within["Online initial"], size, "#eb6834", "Online: router estimate"
        )
        ax.axhline(1, color="#555555", ls="--", lw=1)
        ax.set_title(f"{size} own uncached" if size != "All sizes" else size)
        ax.set_xticks(ticks[::2], labels[::2], rotation=60, ha="right", fontsize=8)
        ax.set_ylim(bottom=0)
        ax.set_ylabel("Mean initial queue time (s)")
        ax.set_xlabel("Outstanding prefill tokens at arrival (5K bins)")
        format_axes(ax)
    axes.flat[0].set_ylim(0, comparison_top)
    for ax in axes.flat:
        ax.set_xlim(-0.4, last + 0.4)
    axes.flat[0].legend(frameon=False, fontsize=9)
    fig.suptitle(
        "Matched queue boundary: admission → first scheduled\nX-axis sources differ: offline engine reconstruction versus online router estimate",
        fontsize=14,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    save_figure(fig, out / "aligned_queue_overlay")

    report = dict(
        online_run=str(args.online.resolve()),
        offline_runs=[str(p.resolve()) for p in args.offline],
        trace_files=trace_files,
        online_samples=len(online),
        online_workers=sorted({r["worker_id"] for r in online}),
        offline_samples=len(offline),
        online_samples_in_0_100k=sum(
            r["workload_tokens"] <= args.max_load for r in online
        ),
        offline_samples_in_0_100k=sum(
            r["workload_tokens"] <= args.max_load for r in offline
        ),
        bin_width=args.bin_width,
        max_displayed_load=args.max_load,
        zoom_max_displayed_load=zoom_max,
        missing_online_benchmark_joins=0,
        missing_offline_admission_joins=0,
        notes=[
            "Online x: selected replica's undecayed router reservations before incoming request admission; excludes the incoming request.",
            "Offline x: existing trace reconstruction of actual remaining uncached prefill work at scheduler arrival; excludes the incoming request.",
            "Original offline y: engine TTFT minus the full measured duration of every prefill step involving this request, including its first step. Counts waiting before and between its prefill steps, plus any residual engine overhead not assigned to those steps; these are scheduler batch wall-time measurements, not isolated CUDA kernel timings.",
            "Aligned offline y: trace request_admitted.queue_wait_ms, stopping at first scheduling. Online y: native vLLM scheduled_ts minus queued_ts.",
            "Online engine scheduler traces were disabled; actual remaining engine work and total inter-chunk queue time cannot be reconstructed for this run.",
            "Online own-size bands are post-hoc ISL minus server-reported cached prompt tokens; these are grouping labels, not the model feature.",
            "Offline profiles include side probes and different offered load; online is one 16-session ramp/hold. Offline no CPU KV offload; online native 64 GiB CPU KV offload.",
            "Offline preempted requests excluded, matching the original profiling plot. All online collected samples retained, without a preemption exclusion because none was captured in their audit schema.",
        ],
    )
    (out / "comparison_summary.json").write_text(json.dumps(report, indent=2))
    (out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Offline and online H100 queue profiles</title>'
        "<style>body{font:16px system-ui;max-width:1600px;margin:30px auto;padding:0 20px}img{width:100%}li{margin-bottom:8px}</style>"
        f"<h1>Offline versus online H100 queue profiles</h1><p>{len(online)} online requests; {len(offline)} offline requests.</p>"
        "<ul>"
        + "".join("<li>" + html.escape(note) + "</li>" for note in report["notes"])
        + "</ul>"
        + '<p><a href="online_points.csv">All online points (CSV)</a> · <a href="offline_points.csv">Offline points (CSV)</a> · '
        '<a href="binned_summary.csv">Bin statistics and counts</a> · <a href="comparison_summary.json">Provenance</a></p>'
        + "".join(
            f'<h2>{title}</h2><a href="{name}.png">Open PNG</a> · <a href="{name}.pdf">PDF</a><img src="{name}.png">'
            for name, title in [
                ("offline_vs_online", "Original and aligned comparison"),
                ("online_all_points", "Every online sample"),
                ("aligned_queue_overlay", "Aligned queue time by request size"),
            ]
        )
    )
    print(
        json.dumps(
            {k: v for k, v in report.items() if k not in ("notes", "trace_files")},
            indent=2,
        )
    )
    print(out / "index.html")


if __name__ == "__main__":
    main()
