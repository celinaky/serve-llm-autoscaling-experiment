#!/usr/bin/env python3
"""Plot a reproducible snapshot of native queue-time learning samples."""

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter


def regression(rows):
    x = np.array([r["workload_tokens"] for r in rows], dtype=float)
    y = np.array([r["queue_time_ms"] for r in rows], dtype=float)
    a, b = np.linalg.lstsq(np.column_stack((np.ones(len(x)), x)), y, rcond=None)[0]
    r2 = 1 - np.sum((y - a - b * x) ** 2) / np.sum((y - y.mean()) ** 2)
    return float(a), float(b), float(r2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    root = args.run.resolve()
    manifest = json.loads((root / "manifest.json").read_text())
    source = Path(manifest["prefill_calibration"]["directory"])
    out = root / "analysis/native-queue-preview"
    snapshot = out / "audit_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    samples, latest, latest_rows = [], None, []
    for source_file in sorted(source.glob("*.jsonl")):
        # Freeze the bytes before parsing so the figure and exported samples
        # describe the same snapshot while the producer continues appending.
        data = source_file.read_bytes()
        (snapshot / source_file.name).write_bytes(data)
        for line in data.splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                continue  # The writer may be partway through its final line.
            if event["event"] == "sample_accepted":
                samples.append(dict(router_id=event["router_id"], **event["sample"]))
            if event["event"] == "fit_snapshot" and event.get("candidate"):
                latest = event["candidate"]
                latest_rows = event["samples"]
    if len(samples) < 2:
        raise ValueError("Not enough paired samples to plot")
    if len({s["router_id"] for s in samples}) != 1:
        raise ValueError("This preview expects one router-local learner")
    all_fit = regression(samples)
    zoom = [s for s in samples if s["workload_tokens"] <= 100000]
    zoom_fit = regression(zoom)
    with (out / "snapshot_points.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(samples[0]))
        writer.writeheader()
        writer.writerows(samples)
    start = min(s["routed_at_s"] for s in samples)
    elapsed = [(s["routed_at_s"] - start) / 60 for s in samples]
    norm = plt.Normalize(0, max(elapsed))
    fig, axes = plt.subplots(1, 2, figsize=(16, 7), facecolor="#fcfcfb")
    for ax, rows, maximum, title in [
        (
            axes[0],
            samples,
            max(s["workload_tokens"] for s in samples),
            "Full collected load range",
        ),
        (axes[1], zoom, 100000, "Zoom: 0–100K outstanding tokens"),
    ]:
        points = ax.scatter(
            [s["workload_tokens"] for s in rows],
            [s["queue_time_ms"] / 1000 for s in rows],
            c=[(s["routed_at_s"] - start) / 60 for s in rows],
            cmap="viridis",
            norm=norm,
            s=12,
            alpha=0.45,
            linewidths=0,
        )
        x = np.linspace(0, maximum, 200)
        ax.plot(
            x,
            (all_fit[0] + all_fit[1] * x) / 1000,
            color="black",
            ls="--",
            lw=1.8,
            label="OLS fit to all collected samples",
        )
        if latest:
            ax.plot(
                x,
                (latest["intercept_ms"] + latest["slope_ms_per_token"] * x) / 1000,
                color="#ef8b20",
                lw=2,
                label=f"Latest online model (n={latest['sample_count']})",
            )
        if ax is axes[1]:
            ax.plot(
                x,
                (zoom_fit[0] + zoom_fit[1] * x) / 1000,
                color="#2878bd",
                ls=":",
                lw=2,
                label="OLS fit to 0–100K subset only",
            )
        score = all_fit[2] if ax is axes[0] else zoom_fit[2]
        ax.set_title(
            f"{title}\nn={len(rows):,}; diagnostic OLS R²={score:.3f}", fontsize=12
        )
        ax.axhline(1, color="#b83131", ls=":", lw=1, label="1-second queue target")
        ax.set_xlim(0, maximum * 1.02)
        ax.set_xlabel(
            "Router-estimated outstanding prefill tokens\non selected replica, before routing this request"
        )
        ax.set_ylabel("Engine queue admission → first scheduled (seconds)")
        ax.xaxis.set_major_formatter(
            FuncFormatter(
                lambda v, _: f"{v / 1e6:g}M" if v >= 1e6 else f"{v / 1000:g}K"
            )
        )
        ax.set_facecolor("#fcfcfb")
        ax.grid(alpha=0.2)
        ax.set_axisbelow(True)
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(loc="upper left", fontsize=8)
    fig.suptitle(
        f"Long H100 native-metrics run — {len(samples):,} paired requests collected",
        fontsize=15,
    )
    fig.subplots_adjust(left=0.06, right=0.91, bottom=0.20, top=0.83, wspace=0.24)
    fig.colorbar(
        points,
        ax=axes,
        label="Minutes since first sampled routing",
        fraction=0.025,
        pad=0.03,
    )
    fig.text(
        0.06,
        0.04,
        "Live snapshot; run continues. Every dot is a collected request. X is the router reservation estimate, not reconstructed remaining engine work.\n"
        "Black/blue lines are diagnostic fits computed for this plot; orange is the actual published online model. Negative fit predictions are shown unchanged.",
        fontsize=9,
    )
    for extension in ["png", "pdf"]:
        fig.savefig(
            out / f"live_samples.{extension}", dpi=170, facecolor=fig.get_facecolor()
        )
    plt.close(fig)
    summary = dict(
        snapshot_at_s=time.time(),
        run_status=manifest["status"],
        samples=len(samples),
        nonzero_workload_samples=sum(s["workload_tokens"] > 0 for s in samples),
        all_sample_ols=dict(
            intercept_ms=all_fit[0], slope_ms_per_token=all_fit[1], r_squared=all_fit[2]
        ),
        zoom_sample_count=len(zoom),
        zoom_ols=dict(
            intercept_ms=zoom_fit[0],
            slope_ms_per_token=zoom_fit[1],
            r_squared=zoom_fit[2],
        ),
        latest_online_model=latest,
        latest_model_window_r_squared=regression(latest_rows)[2]
        if latest_rows
        else None,
    )
    (out / "snapshot_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    print(out / "live_samples.png")


if __name__ == "__main__":
    main()
