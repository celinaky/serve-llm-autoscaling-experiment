"""Put several runs' autoscaling scorecards side by side."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import _fmt, _pct, _read_json

# (header, width, extractor)
COLUMNS: list[tuple[str, int, Any]] = [
    ("policy", 18, lambda e: _policy(e)),
    ("SLO", 5, lambda e: {True: "PASS", False: "FAIL"}.get(_requests(e).get("slo_met"), "-")),
    ("TTFT pN", 9, lambda e: _fmt(_requests(e)["ttft"].get("slo_percentile_ms"), ".0f", "ms")),
    ("ITL pN", 8, lambda e: _fmt(_requests(e)["itl"].get("slo_percentile_ms"), ".0f", "ms")),
    ("TTFT req", 9, lambda e: _pct(_requests(e)["ttft"]["slo_attainment"])),
    ("ITL req", 8, lambda e: _pct(_requests(e)["itl"]["slo_attainment"])),
    ("TTFT mean", 10, lambda e: _fmt(_requests(e)["ttft"]["mean_ms"], ".0f", "ms")),
    ("ITL mean", 9, lambda e: _fmt(_requests(e)["itl"]["mean_ms"], ".0f", "ms")),
    ("failed", 7, lambda e: str(_requests(e)["failed_requests"])),
    ("GPU-h", 7, lambda e: _fmt(_load(e).get("gpu_hours"), ".2f")),
    ("mean", 5, lambda e: _fmt(_load(e)["mean_running_replicas"], ".1f")),
    ("max", 4, lambda e: _fmt(_load(e)["max_running_replicas"])),
    ("up/down", 8, lambda e: f"{e['scaling']['upscales']}/{e['scaling']['downscales']}"),
    ("1st up", 7, lambda e: _fmt(e["scaling"]["first_upscale_at_s"], ".0f", "s")),
    ("lag", 6, lambda e: _fmt(e["scaling"]["mean_upscale_lag_s"], ".0f", "s")),
    ("model ready", 12, lambda e: _fmt((e.get("sla_policy") or {}).get("model_ready_at_s"), ".0f", "s")),
]


def _policy(evaluation: dict[str, Any]) -> str:
    name = evaluation["autoscaling_policy"]
    kwargs = evaluation.get("autoscaling_policy_kwargs") or {}
    return f"{name} (advisory)" if kwargs.get("advisory") else name


def _requests(evaluation: dict[str, Any]) -> dict[str, Any]:
    return evaluation["slo_attainment"]["steady"]["requests"]


def _windows(evaluation: dict[str, Any]) -> dict[str, Any]:
    return evaluation["slo_attainment"]["steady"]["windows"]


def _load(evaluation: dict[str, Any]) -> dict[str, Any]:
    return evaluation["cost"].get("load") or evaluation["cost"]["whole_run"]


def format_comparison(roots: list[Path]) -> str:
    """One row per run from its ``autoscaling_evaluation.json``.

    Attainment columns are the steady-state figures: ``req`` is the share of
    requests within the SLO, ``time`` the share of rolling windows whose
    mean met it. Cost is over the load period.
    """
    width = max(len(root.name) for root in roots)
    header = f"{'run':<{width}}  " + "".join(f"{h:>{w}}" for h, w, _ in COLUMNS)
    lines = [header]
    for root in roots:
        evaluation = _read_json(root / "analysis" / "autoscaling_evaluation.json")
        if not evaluation:
            lines.append(f"{root.name:<{width}}  (no autoscaling_evaluation.json; run analyze first)")
            continue
        lines.append(
            f"{root.name:<{width}}  "
            + "".join(f"{extract(evaluation):>{w}}" for _, w, extract in COLUMNS)
        )
    slos = {
        (e["slo"]["ttft_ms"], e["slo"]["itl_ms"], e["slo"].get("percentile", 0.95))
        for root in roots if (e := _read_json(root / "analysis" / "autoscaling_evaluation.json"))
    }
    if len(slos) == 1:
        ttft, itl, pct = slos.pop()
        pct = f"p{round(pct * 100)}"
        lines.append(
            f"SLO ({pct} of requests): TTFT <= {_fmt(ttft, 'g', 'ms')}, "
            f"ITL <= {_fmt(itl, 'g', 'ms')}; pN columns are the measured {pct}, "
            "req columns the share of requests within the target; GPU-h over the load period"
        )
    elif slos:
        lines.append("WARNING: runs were scored against different SLOs")
    return "\n".join(lines)
