"""Reproduce queue-model fits from saved router audits; no live cluster needed."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import html
import json
import math
import shutil
from pathlib import Path


def fit(rows):
    x = [r["workload_tokens"] for r in rows]
    y = [r["queue_time_ms"] for r in rows]
    mx, my = math.fsum(x) / len(x), math.fsum(y) / len(y)
    var = math.fsum((v - mx) ** 2 for v in x)
    if not var:
        return None
    b = math.fsum((v - mx) * (w - my) for v, w in zip(x, y)) / var
    return my - b * mx, b


def collect_and_render(root: Path, source: Path, target_ms: float):
    destination = root / "telemetry" / "prefill-calibration"
    destination.mkdir(parents=True, exist_ok=True)
    for path in source.glob("*.jsonl"):
        if path.resolve() != (destination / path.name).resolve():
            shutil.copy2(path, destination / path.name)
    return render(root, target_ms)


def render(root: Path, target_ms: float):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    destination = root / "analysis" / "prefill-calibration"
    destination.mkdir(parents=True, exist_ok=True)
    samples, summaries, links, warnings = [], [], [], []
    history = {}
    paths = sorted((root / "telemetry" / "prefill-calibration").glob("*.jsonl"))
    for path in paths:
        attempts, accepted, closed = [], [], False
        for line in path.read_text().splitlines():
            try:
                event = json.loads(line)
            except ValueError:
                warnings.append(f"Malformed record in {path.name}")
                continue
            if event["event"] == "audit_closed":
                closed = True
                if event["dropped"]:
                    warnings.append(f"{path.name}: {event['dropped']} dropped records")
            if event["event"] == "sample_accepted":
                accepted.append(event["sample"]["sample_id"])
                samples.append(dict(router_id=event["router_id"], **event["sample"]))
            if event["event"] != "fit_snapshot":
                continue
            attempts.append(event["attempt_id"])
            rows, router = event["samples"], event["router_id"]
            candidate = event["candidate"]
            verified = None
            if candidate:
                actual = fit(rows)
                verified = actual is not None and all(
                    math.isclose(a, b, rel_tol=1e-8, abs_tol=1e-8)
                    for a, b in zip(
                        actual,
                        [candidate["intercept_ms"], candidate["slope_ms_per_token"]],
                    )
                )
                if not verified:
                    warnings.append(f"Fit mismatch: {router}/{event['attempt_id']}")
            fig, ax = plt.subplots(figsize=(9, 6))
            if rows:
                scatter = ax.scatter(
                    [r["workload_tokens"] for r in rows],
                    [r["queue_time_ms"] for r in rows],
                    s=12,
                    alpha=0.55,
                    c=[r.get("prompt_tokens", 0) for r in rows],
                )
                fig.colorbar(scatter, ax=ax, label="Incoming prompt tokens")
            upper = max([r["workload_tokens"] for r in rows] + [1])
            for model, name, style in [
                (history.get(router), "Previous fit", "--"),
                (candidate, "Candidate fit", "-"),
            ]:
                if model:
                    ax.plot(
                        [0, upper],
                        [
                            model["intercept_ms"],
                            model["intercept_ms"] + upper * model["slope_ms_per_token"],
                        ],
                        style,
                        label=name,
                    )
            budget = None
            if candidate:
                budget = (target_ms - candidate["intercept_ms"]) / candidate[
                    "slope_ms_per_token"
                ]
                if budget > 0:
                    ax.axvline(
                        max(1, math.floor(budget)),
                        color="orange",
                        label="Derived token budget",
                    )
                history[router] = candidate
            ax.axhline(target_ms, color="red", linestyle=":", label="Queue SLO")
            details = "No candidate fit"
            if candidate:
                details = (
                    f"intercept={candidate['intercept_ms']:.3g} ms; "
                    f"slope={candidate['slope_ms_per_token']:.5g} ms/token"
                )
            stamp = datetime.fromtimestamp(
                event["wall_time_s"], timezone.utc
            ).isoformat(timespec="seconds")
            version = event["published_model"].get("prefill_queue_model_version", 0)
            ax.set(
                xlabel="Undecayed reservations before routing (tokens)",
                ylabel="Engine queue time (ms)",
                title=f"{router[:8]} • attempt {event['attempt_id']} • model v{version} • {event['reason']} • n={len(rows)}\n"
                f"{stamp}\n{details}",
            )
            ax.legend()
            fig.tight_layout()
            name = f"{router}-{event['attempt_id']:05d}.png"
            fig.savefig(destination / name, dpi=120)
            plt.close(fig)
            links.append(
                f'<figure><figcaption>{html.escape(name)}</figcaption><img loading="lazy" width="900" src="{name}"></figure>'
            )
            summaries.append(
                dict(
                    router_id=router,
                    attempt_id=event["attempt_id"],
                    wall_time_s=event["wall_time_s"],
                    reason=event["reason"],
                    sample_count=len(rows),
                    verified=verified,
                    derived_budget=budget,
                    candidate=candidate,
                    published_model=event["published_model"],
                )
            )
        if not closed:
            warnings.append(
                f"{path.name}: producer still running or missing audit closure"
            )
        for values, label in [(attempts, "fit"), (accepted, "sample")]:
            if values and values != list(range(1, values[-1] + 1)):
                warnings.append(f"{path.name}: missing or duplicate {label} records")
    if not paths:
        warnings.append("No router audit files found")
    with (destination / "samples.csv").open("w") as stream:
        keys = sorted({k for sample in samples for k in sample})
        if keys:
            writer = csv.DictWriter(stream, fieldnames=keys)
            writer.writeheader()
            writer.writerows(samples)
    (destination / "fits.json").write_text(json.dumps(summaries, indent=2))
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax, replicas_ax = axes
    controller_points = []
    model_deployments = set()
    replica_candidates = []
    metric_path = root / "telemetry" / "serve_metrics.jsonl"
    if metric_path.exists():
        for line in metric_path.read_text().splitlines():
            try:
                record = json.loads(line)
            except ValueError:
                continue
            timestamp = record["timestamp_ns"] / 1e9
            for sample in record.get("samples", []):
                metric = sample.get("normalized_name", "")
                labels = sample.get("labels", {})
                label = labels.get("metric", labels.get("metric_name"))
                deployment = (labels.get("application"), labels.get("deployment"))
                if (
                    metric == "serve_autoscaling_policy_metric"
                    and label == "learned_prefill_tokens_per_replica"
                ):
                    controller_points.append((timestamp, sample["value"]))
                    model_deployments.add(deployment)
                if metric in (
                    "serve_autoscaling_target_replicas",
                    "serve_autoscaling_desired_replicas",
                ):
                    replica_candidates.append((timestamp, sample["value"], deployment))
    replica_points = [
        (timestamp, value)
        for timestamp, value, deployment in replica_candidates
        if deployment in model_deployments
    ]
    if replica_points:
        replicas_ax.scatter(*zip(*replica_points), s=5, color="black")
    (destination / "controller_timeline.json").write_text(
        json.dumps(
            {
                "model_deployments": sorted(model_deployments, key=str),
                "budget_points": controller_points,
                "replica_target_points": replica_points,
            },
            indent=2,
        )
    )
    if controller_points:
        controller_points.sort()
        ax.step(
            *zip(*controller_points),
            where="post",
            color="black",
            linewidth=2,
            label="Policy-selected budget",
        )
    replicas_ax.set(xlabel="Epoch seconds", ylabel="Replica target")

    for router in {r["router_id"] for r in summaries}:
        rows = [
            r
            for r in summaries
            if r["router_id"] == router and r["derived_budget"] is not None
        ]
        ax.step(
            [r["wall_time_s"] for r in rows],
            [r["derived_budget"] for r in rows],
            where="post",
            label=router[:8],
        )
    ax.set(xlabel="Epoch seconds", ylabel="Per-router derived token budget")
    if summaries or controller_points:
        ax.legend()
    fig.tight_layout()
    fig.savefig(destination / "budgets.png")
    plt.close(fig)
    report = dict(
        status="incomplete" if warnings else "verified",
        warnings=warnings,
        fits=len(summaries),
        samples=len(samples),
        target_queue_time_ms=target_ms,
    )
    (destination / "verification.json").write_text(json.dumps(report, indent=2))
    (destination / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Queue calibration</title>'
        "<h1>Queue calibration audit</h1><pre>"
        + html.escape(json.dumps(report, indent=2))
        + '</pre><img src="budgets.png"><p><a href="samples.csv">Samples</a> · <a href="fits.json">Fits</a></p>'
        + "".join(links)
    )
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--target-queue-time-ms", type=float, required=True)
    args = parser.parse_args()
    print(json.dumps(render(args.run, args.target_queue_time_ms), indent=2))
