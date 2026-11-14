import json
from pathlib import Path

from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import load_config
from serve_llm_autoscaling.diagnostics import pipeline


def config():
    return load_config('experiments/b200_workday_curve_prefill_mean.yaml')


def make_inputs(root):
    artifact = root / 'benchmark/agentx-concurrency-ramp'
    artifact.mkdir(parents=True)
    (artifact / 'profile_export.jsonl').write_text(json.dumps({
        'metadata': {'benchmark_phase': 'profiling', 'x_request_id': 'this-request'},
    }) + '\n')
    traces = root / 'traces'
    traces.mkdir()
    (traces / 'matched.jsonl').write_text(json.dumps({
        'event': 'request_queued', 'request_id': 'chatcmpl-this-request-abcdef12',
    }) + '\n')
    (traces / 'other-run.jsonl').write_text(json.dumps({
        'event': 'request_queued', 'request_id': 'cmpl-unrelated-request-abcdef12',
    }) + '\n')
    return traces


def test_trace_collection_is_unique_and_forwarded_without_changing_input(tmp_path, monkeypatch):
    monkeypatch.delenv(pipeline.TRACE_ENV, raising=False)
    original = config()
    original.analysis.scheduler_trace_dir = tmp_path / 'shared'
    prepared = pipeline.prepare_tracing(original, tmp_path / 'runs/run-a')
    other = pipeline.prepare_tracing(original, tmp_path / 'runs/run-b')
    assert prepared.analysis.scheduler_trace_dir == tmp_path / 'shared/run-a'
    assert other.analysis.scheduler_trace_dir == tmp_path / 'shared/run-b'
    assert original.analysis.scheduler_trace_dir == tmp_path / 'shared'
    assert RayServeBackend(prepared).deployment_spec()['runtime_env']['env_vars'][pipeline.TRACE_ENV] == str(tmp_path / 'shared/run-a')
    original.analysis.diagnostic_plots = False
    assert pipeline.prepare_tracing(original, tmp_path / 'disabled') is original
    assert not (tmp_path / 'shared/disabled').exists()


def test_trace_selection_excludes_other_runs_even_in_same_directory(tmp_path):
    traces = make_inputs(tmp_path)
    assert pipeline.select_traces(tmp_path, config(), traces) == [traces / 'matched.jsonl']


def test_no_traces_is_explicit_and_does_not_claim_empty_plots_succeeded(tmp_path):
    make_inputs(tmp_path)
    (tmp_path / 'analysis').mkdir()
    (tmp_path / 'analysis/arrival_stats_stack180_0_1860.png').write_bytes(b'stale chart')
    (tmp_path / 'analysis/load_requests.csv').write_text('stale request data')
    (tmp_path / 'analysis/session_lanes.csv').write_text('stale session data')
    report = pipeline.generate_diagnostics(tmp_path, config(), traces_dir=tmp_path / 'missing')
    assert report['status'] == 'skipped'
    assert len(report['plots']) == 4
    assert all(p['status'] == 'skipped' for p in report['plots'].values())
    assert report['warnings'] and 'No scheduler traces matched' in report['warnings'][0]
    assert not (tmp_path / 'analysis/arrival_stats_stack180_0_1860.png').exists()
    assert not (tmp_path / 'analysis/load_requests.csv').exists()
    assert not (tmp_path / 'analysis/session_lanes.csv').exists()
    page = (tmp_path / 'analysis/index.html').read_text()
    assert 'No scheduler traces matched' in page
    assert 'href="load_requests.csv"' not in page
    assert 'href="session_lanes.csv"' not in page
    assert json.loads((tmp_path / 'analysis/diagnostic_plots.json').read_text())['status'] == 'skipped'


def test_one_plot_failure_does_not_prevent_other_plots_or_gallery(tmp_path, monkeypatch):
    traces = make_inputs(tmp_path)
    calls = []

    def run(root, name, args, output):
        calls.append(name)
        if name == 'arrival_stats_stack':
            return {'status': 'failed', 'reason': 'render failed', 'log': 'arrival.log'}
        output.write_bytes(b'artifact')
        return {'status': 'generated', 'file': output.name, 'log': 'generation.log'}

    monkeypatch.setattr(pipeline, '_run_plot', run)
    report = pipeline.generate_diagnostics(tmp_path, config(), traces_dir=traces)
    assert calls[0] == 'load_breakdown'
    assert set(calls[1:]) == {'arrival_stats_stack', 'replica_percentile_lines', 'policy_vs_engine_mean', 'session_lanes'}
    assert report['status'] == 'partial'
    assert report['plots']['arrival_stats_stack']['status'] == 'failed'
    assert report['plots']['session_lanes']['status'] == 'generated'
    page = (tmp_path / 'analysis/index.html').read_text()
    assert 'render failed' in page and 'session_lanes_0_1860.png' in page


def test_request_join_failure_keeps_independent_policy_plot(tmp_path, monkeypatch):
    traces = make_inputs(tmp_path)
    calls = []

    def run(root, name, args, output):
        calls.append(name)
        if name == 'load_breakdown':
            return {'status': 'failed', 'reason': 'join failed'}
        output.write_bytes(b'chart')
        return {'status': 'generated', 'file': output.name}

    monkeypatch.setattr(pipeline, '_run_plot', run)
    report = pipeline.generate_diagnostics(tmp_path, config(), traces_dir=traces)
    assert calls == ['load_breakdown', 'policy_vs_engine_mean']
    assert report['status'] == 'partial'
    assert report['plots']['policy_vs_engine_mean']['status'] == 'generated'
    assert report['plots']['session_lanes']['status'] == 'skipped'
