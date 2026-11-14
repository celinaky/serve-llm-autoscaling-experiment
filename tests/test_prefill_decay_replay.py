"""Validate capacity sharing, observed cleanup, and exact mean integration."""

import numpy as np
import pytest

from tools.prefill_decay_replay import individual_curves, replay_fifo, step_curve


def request(rid, queued, terminal, tokens):
    return {"request_id": rid, "queued": queued, "terminal": terminal, "uncached": tokens}


def test_shared_capacity_goes_to_oldest_request():
    requests = [request("a", 0, 30, 100000), request("b", 1, 30, 100000)]
    curve, _ = replay_fifo(requests, 10000)
    individual = individual_curves(requests, 10000)
    assert curve.value(5) == pytest.approx(150000)
    assert individual["a"].value(5) == pytest.approx(50000)
    assert individual["b"].value(5) == pytest.approx(100000)
    assert individual["b"].value(12) == pytest.approx(80000)


def test_observed_completion_clears_residual_and_unblocks_next():
    requests = [request("a", 0, 3, 100000), request("b", 1, 20, 100000)]
    curve, diagnostics = replay_fifo(requests, 10000)
    assert diagnostics[0]["residual_at_terminal"] == pytest.approx(70000)
    assert curve.value(3) == pytest.approx(100000)
    assert curve.value(5) == pytest.approx(80000)
    assert curve.value(20) == 0


def test_idle_capacity_is_not_banked():
    curve, _ = replay_fifo([request("a", 10, 30, 100000)], 10000)
    assert curve.value(9) == 0
    assert curve.value(10) == 100000
    assert curve.value(15) == 50000
    assert curve.value(20) == 0
    assert curve.integral(30) == pytest.approx(500000)


def test_estimated_exhaustion_can_precede_first_token():
    curve, diagnostics = replay_fifo([request("a", 0, 20, 100000)], 10000)
    assert diagnostics[0]["estimated_zero_at"] == 10
    assert curve.value(15) == 0
    assert curve.mean(20, 20) == pytest.approx(25000)
    assert curve.mean(30, 10) == 0


def test_individual_contributions_sum_to_aggregate_with_out_of_order_completions():
    requests = [request("a", 0, 4, 100000), request("b", 1, 2, 100000), request("c", 3, 30, 100000)]
    curve, _ = replay_fifo(requests, 10000)
    individual = individual_curves(requests, 10000)
    grid = np.linspace(-1, 40, 1000)
    np.testing.assert_allclose(sum(c.value(grid) for c in individual.values()), curve.value(grid), atol=1e-7)
    assert np.min(curve.value(grid)) >= 0
    assert curve.value(40) == 0


def test_engines_have_independent_capacity():
    a, _ = replay_fifo([request("a", 0, 30, 100000)], 10000)
    b, _ = replay_fifo([request("b", 0, 30, 100000)], 10000)
    assert a.value(5) + b.value(5) == 100000


def test_step_mean_integrates_event_times_instead_of_scrape_samples():
    curve = step_curve([(0.2, 100000), (0.4, -100000)])
    assert curve.mean(1, 1) == pytest.approx(20000)
    assert curve.value(1) == 0

