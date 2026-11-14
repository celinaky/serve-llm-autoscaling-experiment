"""Join observed router bookings to engine cache reuse; never infer missing credit.

python -m serve_llm_autoscaling.diagnostics.router_cache_credit RUN [--host-credit .75]
"""
from __future__ import annotations

import argparse
import csv
import json
import re
from collections import defaultdict
from pathlib import Path

import yaml


def _read(paths):
    for path in sorted(paths):
        for line in path.open():
            try:
                yield path, json.loads(line)
            except ValueError:
                continue


def _client_id(value, selections):
    if value in selections:
        return value
    value = re.sub(r"^(?:chatcmpl-|cmpl-)", "", value or "")
    if value in selections:
        return value
    value = re.sub(r"-[0-9a-f]{8}$", "", value)
    return value if value in selections else None


def audit(run: Path, traces: Path, host_credit: float | None = None):
    if host_credit is not None and not 0 <= host_credit <= 1:
        raise ValueError("host_credit must be between 0 and 1")
    selections, lifecycle, admissions, chunks, first = (
        defaultdict(list),
        {},
        defaultdict(list),
        defaultdict(list),
        {},
    )
    snapshot_errors, snapshot_count, dropped = [], 0, 0
    router_files = list((traces / "router-request-audit").glob("*.jsonl"))
    for path, r in _read(router_files):
        event = r.get("event")
        if event == "router_selection":
            selections[r["request_id"]].append({**r, "source": str(path)})
        elif event == "router_prefill_complete":
            lifecycle.setdefault(r["request_id"], r["wall_time_s"])
        elif event == "router_autoscaling_snapshot":
            snapshot_count += 1
            difference = abs(
                sum(x["remaining_estimate_tokens"] for x in r["requests"])
                - r["outstanding_effective_prefill_tokens"]
            )
            if difference > 1.000001:
                snapshot_errors.append(
                    dict(file=str(path), time=r["wall_time_s"], difference=difference)
                )
        elif event == "trace_events_dropped":
            dropped += r["count"]
    for path, r in _read(traces.glob("*.jsonl")):
        event = r.get("event")
        rid = r.get("request_id")
        key = (path.stem.rsplit(".", 1)[0], rid)
        if event == "request_admitted":
            client = _client_id(r.get("external_request_id") or rid, selections)
            if client:
                admissions[client].append((key, r))
        elif event == "batch_scheduled":
            for request in r.get("requests", []):
                if request["phase"] == "prefill":
                    k = (key[0], request["request_id"])
                    chunks[k].append((r["batch_index"], request["scheduled_tokens"]))
        elif event == "request_first_token":
            first[key] = r
        elif event == "trace_events_dropped":
            dropped += r["count"]
    rows, unmatched, ambiguous = [], [], []
    for rid, bookings in selections.items():
        if not admissions.get(rid):
            unmatched.append(rid)
            continue
        # Retry/preemption ambiguity must not masquerade as a clean comparison.
        if len(bookings) != 1 or len(admissions[rid]) != 1:
            ambiguous.append(rid)
            continue
        booking = bookings[0]
        key, engine = admissions[rid][0]
        p = engine["prompt_tokens"]
        local = engine.get("local_cached_prefix_tokens")
        external = engine.get("external_cached_prefix_tokens")
        compute = engine.get("initial_gpu_prefill_tokens")
        split_known = all(v is not None for v in (local, external, compute))
        consistent = (
            split_known
            and local + external + compute == p
            and booking["prompt_tokens"] == p
        )
        ft = first.get(key)
        scheduled = sum(
            n for b, n in chunks[key] if ft is not None and b <= ft["batch_index"]
        )
        row = dict(
            request_id=rid,
            engine=key[0],
            router_replica_id=booking.get("replica_id"),
            router_selected_wall_time_s=booking["wall_time_s"],
            engine_admitted_wall_time_s=engine["wall_time_s"],
            engine_first_token_wall_time_s=ft["wall_time_s"] if ft else None,
            router_prefill_complete_wall_time_s=lifecycle.get(rid),
            router_prompt_tokens=booking["prompt_tokens"],
            engine_prompt_tokens=p,
            router_gpu_overlap_tokens=booking["overlap"].get("gpu"),
            router_cpu_overlap_tokens=booking["overlap"].get("cpu"),
            router_effective_prefill_tokens=booking["effective_prefill_tokens"],
            engine_local_cached_tokens=local,
            engine_external_cached_tokens=external,
            engine_initial_gpu_prefill_tokens=compute,
            engine_scheduled_prefill_tokens_before_first_output=scheduled
            if ft
            else None,
            accounting_consistent=consistent,
            router_initial_excess_vs_gpu_tokens=booking["effective_prefill_tokens"]
            - compute
            if consistent
            else None,
            assumed_host_credit=host_credit,
            router_initial_excess_after_transfer_allowance=(
                booking["effective_prefill_tokens"]
                - (compute + (1 - host_credit) * external)
                if consistent and host_credit is not None
                else None
            ),
        )
        rows.append(row)
    valid = [r for r in rows if r["accounting_consistent"]]
    candidates = sorted(
        (
            r
            for r in valid
            if r["engine_external_cached_tokens"] > 0
            and r["router_initial_excess_vs_gpu_tokens"] > 0
        ),
        key=lambda r: r["router_initial_excess_vs_gpu_tokens"],
        reverse=True,
    )
    report = dict(
        status="available" if valid else "unavailable",
        run=run.name,
        traces_directory=str(traces),
        router_files=[str(p) for p in router_files],
        routed_requests=len(selections),
        matched_rows=len(rows),
        valid_rows=len(valid),
        missing_engine_requests=unmatched,
        ambiguous_requests=ambiguous,
        router_snapshot_count=snapshot_count,
        snapshot_sum_errors=snapshot_errors,
        dropped_trace_events=dropped,
        assumed_host_credit=host_credit,
        external_reuse_with_initial_router_excess=len(candidates),
        largest_initial_disagreements=candidates[:20],
        interpretation="A candidate is an observed estimate/computation disagreement, not automatic proof of a routing bug. Compare GPU overlap, external reuse, timing, and intentional transfer-cost weighting.",
        reason=None
        if valid
        else "No unambiguous router selections with separately recorded engine local/external cache counts. Historical combined cached-token counts cannot establish router credit.",
    )
    return report, rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    parser.add_argument("--traces-dir", type=Path)
    parser.add_argument("--host-credit", type=float)
    args = parser.parse_args()
    config = yaml.safe_load((args.run / "resolved.yaml").read_text())
    directory = args.traces_dir or Path(config["analysis"]["scheduler_trace_dir"])
    report, rows = audit(args.run, directory, args.host_credit)
    out = args.run / "analysis/router_cache_credit"
    out.mkdir(parents=True, exist_ok=True)
    (out / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    csv_path = out / "requests.csv"
    csv_path.unlink(missing_ok=True)
    if rows:
        with csv_path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
