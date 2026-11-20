import json
from pathlib import Path

import pytest

from conftest import short_config
from serve_llm_autoscaling import runner
from serve_llm_autoscaling.analysis import analyze_run, lifecycle_periods
from serve_llm_autoscaling.artifacts import format_run_summary, write_json
from test_analysis import S, T0, make_agentx_run
from test_runner import ROW

# --- Runner -----------------------------------------------------------------


def _phase_manifest(root: Path, end_s: float = 225):
    ramp = root / "benchmark" / "agentx-concurrency-ramp"
    ramp.mkdir(parents=True, exist_ok=True)
    write_json(ramp / "phase_manifest.json", {"phases": [
        {"phase_kind": "profiling", "start_ns": T0, "end_ns": T0 + int(end_s * S)},
    ]})


def _clock(monkeypatch, sleeps):
    """A fake epoch clock that advances only when the harness sleeps."""
    now = [T0 - 20 * S]
    monkeypatch.setattr(runner.time, "time_ns", lambda: now[0])

    def sleep(seconds):
        sleeps.append(seconds)
        now[0] += int(seconds * S)

    monkeypatch.setattr(runner.time, "sleep", sleep)
    return now


def test_observation_runs_with_telemetry_after_aiperf(fake_cluster, experiment, monkeypatch):
    config, path = experiment(post_load_observation_s=90)
    sleeps = []
    now = _clock(monkeypatch, sleeps)
    alive_during_sleep = []

    def aiperf(self):
        _phase_manifest(self.benchmark_root.parent)
        now[0] = T0 + 240 * S  # AIPerf exits after its drain and export
        return dict(ROW)

    advance = runner.time.sleep

    def sleep(seconds):
        alive_during_sleep.append(fake_cluster.collectors_alive())
        advance(seconds)

    monkeypatch.setattr(runner.time, "sleep", sleep)
    monkeypatch.setattr(runner.AIPerfRunner, "run", aiperf)
    monkeypatch.setattr(runner, "run_analysis", lambda root: {"warnings": []})
    root = runner.run_experiment(config, path)

    assert sleeps == [90]
    assert alive_during_sleep == [[True, True]]
    lifecycle = json.loads((root / "benchmark" / "lifecycle.json").read_text())
    assert lifecycle == {
        "schema_version": 1,
        "clock": "epoch_ns",
        "aiperf_launched_ns": T0 - 20 * S,
        "profiling_start_ns": T0,
        "ramp_end_ns": None,
        "load_end_ns": T0 + 200 * S,
        "profiling_phase_end_ns": T0 + 225 * S,
        "aiperf_exited_ns": T0 + 240 * S,
        "post_load_observation_s": 90,
        "observation_end_ns": T0 + 330 * S,
        "observation_skipped": None,
    }


def test_lifecycle_record_without_a_phase_manifest(tmp_path: Path):
    record = runner.lifecycle_record(
        short_config(), tmp_path, aiperf_launched_ns=1, aiperf_exited_ns=2,
    )
    assert (record["aiperf_launched_ns"], record["aiperf_exited_ns"]) == (1, 2)
    assert record["profiling_start_ns"] is None and record["load_end_ns"] is None


# --- Analysis ---------------------------------------------------------------


def _lifecycle(root: Path, **fields):
    record = {
        "schema_version": 1, "clock": "epoch_ns", "aiperf_launched_ns": T0 - 20 * S,
        "profiling_start_ns": T0, "ramp_end_ns": None, "load_end_ns": T0 + 200 * S,
        "profiling_phase_end_ns": T0 + 225 * S, "aiperf_exited_ns": T0 + 240 * S,
        "post_load_observation_s": 90, "observation_end_ns": T0 + 330 * S,
        "observation_skipped": None, **fields,
    }
    write_json(root / "benchmark" / "lifecycle.json", record)


@pytest.fixture
def observed_run(tmp_path: Path):
    config = short_config(post_load_observation_s=90)
    root = make_agentx_run(tmp_path / "run", config)
    # AIPerf closed the profiling phase after draining for 25s.
    _phase_manifest(root, end_s=225)
    _lifecycle(root)
    return root


def test_plot_data_separates_drain_from_idle(observed_run: Path):
    analyze_run(observed_run, plot=False)
    data = json.loads((observed_run / "analysis" / "plot_data.json").read_text())
    lifecycle = data["lifecycle"]
    assert lifecycle["source"] == "lifecycle.json"
    assert lifecycle["markers"] == {
        "ramp_end_s": None, "load_end_s": 200, "drain_end_s": 225.0,
        "aiperf_exit_s": 240.0, "observation_end_s": 330.0,
    }
    assert lifecycle["periods"] == [
        {"name": "load", "start_s": 0.0, "end_s": 200},
        {"name": "drain", "start_s": 200, "end_s": 225.0},
        {"name": "idle", "start_s": 225.0, "end_s": 330.0},
    ]
    assert data["experiment"]["post_load_observation_s"] == 90
    assert data["experiment"]["routing_policy"] is None


def test_request_windows_cover_idle_observation(observed_run: Path):
    analyze_run(observed_run, plot=False)
    timeseries = json.loads(
        (observed_run / "analysis" / "request_timeseries.json").read_text())
    windows = timeseries["windows"]
    assert windows[-1]["window_end_s"] == 330
    idle = [w for w in windows if w["window_start_s"] >= 240]
    assert idle and all(w["offered_requests"] == 0 and w["in_flight_at_end"] == 0
                        for w in idle)


def test_concurrency_is_zero_after_load_ends(observed_run: Path):
    analyze_run(observed_run, plot=False)
    result = json.loads((observed_run / "analysis" / "concurrency_qps.json").read_text())
    after = [w for w in result["windows"] if w["window_start_s"] >= 200]
    assert after and all(w["configured_session_concurrency"] == 0 for w in after)
    assert all(w["requests_per_second_per_configured_session"] is None for w in after)


def test_lifecycle_falls_back_to_config_without_record():
    periods = lifecycle_periods(short_config(), T0, T0 + 215 * S, None)
    assert periods["source"] == "config"
    assert periods["periods"] == [
        {"name": "load", "start_s": 0.0, "end_s": 200},
        {"name": "drain", "start_s": 200, "end_s": 215.0},
    ]


def test_summary_lists_observed_periods(observed_run: Path):
    result = analyze_run(observed_run)
    write_json(observed_run / "manifest.json",
               {"name": "agentx-ramp", "status": "succeeded", "analysis": result})
    summary = format_run_summary(observed_run)
    assert "Lifecycle: load 0-200s · drain 200-225s · idle 225-330s" in summary
