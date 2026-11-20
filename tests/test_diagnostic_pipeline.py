import json
from pathlib import Path

from conftest import TOKEN_POLICY, build_config
from serve_llm_autoscaling.diagnostics import pipeline


def config():
    return build_config(deployment=TOKEN_POLICY, duration_s=1860)


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


def test_trace_selection_excludes_other_runs_even_in_same_directory(tmp_path):
    traces = make_inputs(tmp_path)
    assert pipeline.select_traces(tmp_path, config(), traces) == [traces / 'matched.jsonl']


def test_no_traces_is_explicit_and_does_not_claim_empty_plots_succeeded(tmp_path):
    make_inputs(tmp_path)
    (tmp_path / 'analysis').mkdir()
    (tmp_path / 'analysis/arrival_stats_stack180_0_1860.png').write_bytes(b'stale chart')
    (tmp_path / 'analysis/load_requests.csv').write_text('stale request data')
    report = pipeline.generate_diagnostics(tmp_path, config(), traces_dir=tmp_path / 'missing')
    assert report['status'] == 'skipped'
    assert set(report['plots']) == {'arrival_stats_stack', 'policy_vs_engine_mean'}
    assert all(p['status'] == 'skipped' for p in report['plots'].values())
    assert report['warnings'] and 'No scheduler traces matched' in report['warnings'][0]
    assert not (tmp_path / 'analysis/arrival_stats_stack180_0_1860.png').exists()
    assert not (tmp_path / 'analysis/load_requests.csv').exists()
    page = (tmp_path / 'analysis/index.html').read_text()
    assert 'No scheduler traces matched' in page
    assert 'href="load_requests.csv"' not in page
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
    assert calls[1:] == ['arrival_stats_stack', 'policy_vs_engine_mean']
    assert report['status'] == 'partial'
    assert report['plots']['arrival_stats_stack']['status'] == 'failed'
    assert report['plots']['policy_vs_engine_mean']['status'] == 'generated'
    page = (tmp_path / 'analysis/index.html').read_text()
    assert 'render failed' in page and 'policy_vs_engine_mean60_0_1860.png' in page
    assert 'href="load_requests.csv"' in page


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
    assert report['plots']['arrival_stats_stack']['status'] == 'skipped'
