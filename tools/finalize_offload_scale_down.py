#!/usr/bin/env python3
"""Wait for the scale-down experiment, generate its plots and a static report.

This postprocessor never deploys or changes Serve. It can run detached so the
report is produced even if the interactive connection closes.
"""
import argparse
import csv
import datetime
import html
import json
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

TITLE = 'H100s, 16 session concurrency, KV Offloading with scale down'
REPO = Path(__file__).resolve().parents[1]
BASELINE = REPO / 'runs/20261103T233308Z-h100-workday-curve-prefill-mean'
TRACES = Path('/mnt/cluster_storage/serve-llm-autoscaling-experiment/vllm-scheduler-traces')


def read(path):
    return json.loads(path.read_text())


def request_stats(evaluation):
    return evaluation.get('slo_attainment', {}).get('steady', {}).get('requests', {})


def fmt(number, digits=2):
    return 'unavailable' if number is None else f'{number:,.{digits}f}'


def root_transitions(run):
    """Join observed root-turn engine changes to actual prefill and TTFT."""
    load_path = run / 'analysis/load_requests.csv'
    origin_path = run / 'analysis/plot_data.json'
    if not load_path.exists() or not origin_path.exists():
        return []
    with load_path.open() as stream:
        requests = {row['request_id']: row for row in csv.DictReader(stream)}
    t0 = read(origin_path)['time_origin']['profiling_start_ns']
    roots = defaultdict(list)
    for path in run.glob('benchmark/*/profile_export.jsonl'):
        with path.open() as stream:
            for line in stream:
                meta = json.loads(line)['metadata']
                req = requests.get(meta['x_request_id'])
                if (meta.get('benchmark_phase') == 'profiling'
                    and meta.get('agent_depth', 0) == 0 and req and req.get('engine')):
                    roots[meta['root_correlation_id']].append((meta, req))
    changes = []
    for turns in roots.values():
        turns.sort(key=lambda turn: turn[0]['request_start_ns'])
        for (previous, prev_req), (meta, req) in zip(turns, turns[1:]):
            if prev_req['engine'] != req['engine']:
                changes.append(dict(trace_id=meta.get('source_trace_id', '')[-6:],
                    time_s=(meta['request_start_ns']-t0)/1e9,
                    request_id=meta['x_request_id'], from_engine=prev_req['engine'],
                    to_engine=req['engine'], turn_index=meta.get('turn_index'),
                    input_tokens=int(req['isl']), computed_prefill_tokens=int(req['own_uncached']),
                    client_ttft_ms=float(req['client_ttft_ms'])))
    return sorted(changes, key=lambda change: change['time_s'])


def request_errors(run):
    origin_path = run / 'analysis/plot_data.json'
    t0 = read(origin_path)['time_origin']['profiling_start_ns'] if origin_path.exists() else 0
    errors = []
    for path in run.glob('benchmark/*/profile_export.jsonl'):
        with path.open() as stream:
            for line in stream:
                record = json.loads(line)
                if not record.get('error'):
                    continue
                meta = record['metadata']
                if meta.get('benchmark_phase') != 'profiling':
                    continue
                errors.append(dict(request_id=meta['x_request_id'],
                    trace_id=meta.get('source_trace_id', '')[-6:],
                    type=record['error'].get('type', 'unknown'),
                    start_s=(meta['request_start_ns']-t0)/1e9,
                    end_s=(meta['request_end_ns']-t0)/1e9,
                    ended_at_utc=datetime.datetime.fromtimestamp(
                        meta['request_end_ns']/1e9, datetime.timezone.utc).isoformat()))
    return sorted(errors, key=lambda row: row['end_s'])


def build_report(run, site, plot_errors):
    manifest = read(run / 'manifest.json')
    evaluation_path = run / 'analysis/autoscaling_evaluation.json'
    evaluation = read(evaluation_path) if evaluation_path.exists() else {}
    baseline = read(BASELINE / 'analysis/autoscaling_evaluation.json')
    stats, before = request_stats(evaluation), request_stats(baseline)
    events = evaluation.get('scale_events', [])
    down = [e for e in events if e['direction'] == 'down']
    load_down = [e for e in down if e['time_s'] < 1050]
    after_cost = evaluation.get('cost', {}).get('load', {})
    before_cost = baseline['cost']['load']
    summary = dict(title=TITLE, run=run.name, status=manifest['status'],
        config='experiments/h100_workday_curve_prefill_mean_offload_scale_down.yaml',
        downscale_delay_s=200, cpu_cache_gib_per_replica=64,
        downscales_during_load=len(load_down), downscales_after_load=len(down)-len(load_down),
        baseline_run=BASELINE.name, scale_events=events,
        scaling=evaluation.get('scaling', {}), requests=stats, baseline_requests=before,
        load_cost=after_cost, baseline_load_cost=before_cost,
        plot_errors=plot_errors, error=manifest.get('error'))
    (run / 'analysis/scale_down_report.json').write_text(json.dumps(summary, indent=2)+'\n')
    dist = site / 'dist'; assets = dist / 'assets'; assets.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run / 'analysis/scale_down_report.json', assets / 'report.json')
    shutil.copy2(run / 'resolved.yaml', assets / 'experiment.yaml')
    charts = [
        ('scaling', 'Scaling timeline', 'autoscaling_timeline.png',
         'Replica counts, traffic, and latency over load, drain, and idle observation. Downscale decisions are not guaranteed merely by setting a 200s delay.'),
        ('arrival', 'Arrival statistics · 180s windows', 'arrival_stats_stack180_0_1050.png',
         'p95, p90 and mean of uncached work queued ahead at arrival, engine queue time, and client TTFT. Counts are shown under each point.'),
        ('replicas', 'Replica percentiles · 180s windows', 'replica_percentiles180_0_1050.png',
         'Per-replica p95, p90 and mean arrival load. Windows with fewer than 10 requests on a replica are blank.'),
        ('policy', 'Policy vs engine · 60s mean', 'policy_vs_engine_mean60_0_1050.png',
         'Purple is the router/policy mean; green is remaining uncached prefill work reconstructed from scheduler traces. The load phase is 0–1050s.'),
        ('sessions', 'Session lanes and replica placement', 'session_lanes_0_1050.png',
         'Bars show observed request spans, not verified session completion. Colors identify the first root replica. The session table lists every replica that served root turns, exposing migrations.'),
    ]
    available = []
    for chart in charts:
        source = run / 'analysis' / chart[2]
        if source.exists():
            shutil.copy2(source, assets / chart[2]); available.append(chart)
    for name in ['session_lanes.csv', 'arrival_stats_stack180_generation.log']:
        if (run / 'analysis' / name).exists(): shutil.copy2(run / 'analysis' / name, assets / name)
    session_rows = []
    if (assets / 'session_lanes.csv').exists():
        session_rows = list(csv.DictReader((assets / 'session_lanes.csv').open()))
        (assets / 'sessions.json').write_text(json.dumps(session_rows, indent=2))
    migrated = [row for row in session_rows if int(row.get('root_replica_count', 0)) > 1]
    transitions = root_transitions(run)
    summary['root_replica_transitions'] = transitions
    errors = request_errors(run)
    summary['request_errors'] = errors
    summary['request_errors_by_type'] = dict(Counter(row['type'] for row in errors))
    summary['baseline_request_errors_by_type'] = dict(Counter(
        row['type'] for row in request_errors(BASELINE)))
    first_down = min((row['time_s'] for row in down), default=float('inf'))
    failed_before_down = sum(row['end_s'] < first_down for row in errors)
    summary['errors_completed_before_first_downscale'] = failed_before_down
    error_rows = ''.join(f"<tr><td>{row['trace_id']}</td><td>{row['type']}</td>"
        f"<td>{row['start_s']:.1f}s</td><td>{row['end_s']:.1f}s</td>"
        f"<td>{html.escape(row['ended_at_utc'])}</td></tr>" for row in errors)
    if not error_rows:
        error_rows = '<tr><td colspan="5">No request errors recorded.</td></tr>'
    failure_note = (f"{len(errors)} failed request records; {failed_before_down} completed before the first downscale. "
        "The effective HAProxy configuration has hard-stop-after 120s. Retired proxy worker logs explicitly record "
        "forced connection closures after reloads, with timestamps matching multiple groups of incomplete streams. "
        "One additional stream ended at the removed replica’s 35s shutdown deadline (09:19:13 UTC). "
        "This is evidence of connection/drain effects; do not attribute all failures to KV offloading or scale-down alone.")
    evidence = run / 'analysis/haproxy-evidence/summary.json'
    if evidence.exists():
        summary['haproxy_evidence'] = read(evidence)
        shutil.copy2(evidence, assets / 'haproxy-evidence.json')
        failure_note += ' The evidence summary is available with the report data.'
    transition_rows = ''.join(
        f"<tr><td>{row['trace_id']}</td><td>{row['time_s']:.1f}s</td>"
        f"<td>{html.escape(row['from_engine'])} → {html.escape(row['to_engine'])}</td>"
        f"<td>{row['computed_prefill_tokens']:,} / {row['input_tokens']:,}</td>"
        f"<td>{row['client_ttft_ms']/1000:.2f}s</td></tr>" for row in transitions)
    if not transition_rows:
        transition_rows = '<tr><td colspan="5">No root-turn replica changes were observed in joined scheduler traces.</td></tr>'
    summary['sessions_with_multiple_root_replicas'] = len(migrated)
    summary['observed_sessions'] = len(session_rows)
    (run / 'analysis/scale_down_report.json').write_text(json.dumps(summary, indent=2)+'\n')
    shutil.copy2(run / 'analysis/scale_down_report.json', assets / 'report.json')
    if load_down:
        finding = (f"Observed {len(load_down)} downscale decision(s) during the 1050s load phase "
                   f"and {len(down)-len(load_down)} afterward. "
                   f"{len(migrated)} of {len(session_rows)} observed sessions had root turns served by multiple replicas. "
                   "Replica changes can trigger prefix recomputation because native CPU caches are private to an engine.")
    else:
        finding = (f"No downscale decision occurred during the 1050s load phase; {len(down)} occurred afterward. "
                   "This run therefore does not demonstrate the latency impact of removing a replica while new turns are arriving.")
    peak_running = after_cost.get('max_running_replicas')
    if peak_running is not None:
        finding += f" Peak running replicas during load: {peak_running:g}. Targets can exceed the running count while engines initialize."
    request_summary = []
    for label, metric, unit in [('TTFT p50', 'p50_ms', 'ms'), ('TTFT p95', 'p95_ms', 'ms'), ('TTFT p99', 'p99_ms', 'ms')]:
        request_summary.append((label, fmt(before.get('ttft', {}).get(metric)), fmt(stats.get('ttft', {}).get(metric)), unit))
    request_summary += [('Failed requests', str(before.get('failed_requests', '?')), str(stats.get('failed_requests', '?')), ''),
        ('Load GPU-hours', fmt(before_cost.get('gpu_hours'), 3), fmt(after_cost.get('gpu_hours'), 3), 'GPU-h'),
        ('Starting-replica time during load', fmt(before_cost.get('starting_replica_seconds'), 1), fmt(after_cost.get('starting_replica_seconds'), 1), 'replica-s'),
        ('Mean running replicas during load', fmt(before_cost.get('mean_running_replicas')), fmt(after_cost.get('mean_running_replicas')), '')]
    comparisons = ''.join(f'<tr><td>{html.escape(label)}</td><td>{old} {unit}</td><td>{new} {unit}</td></tr>' for label, old, new, unit in request_summary)
    event_rows = ''.join(f"<tr><td>{e['direction']}</td><td>{e['time_s']:.1f}s</td><td>{e['from_replicas']:g} → {e['to_replicas']:g}</td><td>{fmt(e.get('running_matched_at_s'), 1)}s</td></tr>" for e in events)
    if not event_rows: event_rows = '<tr><td colspan="4">No scale decisions recorded.</td></tr>'
    cards = f'''<div><span class="stat-label">DOWNSCALE DELAY</span><strong>200 <small>s</small></strong><span>64 GiB CPU cache per replica</span></div>
    <div><span class="stat-label">DOWNSCALES OBSERVED</span><strong>{len(down)}</strong><span>{len(load_down)} during load · {len(down)-len(load_down)} afterward</span></div>
    <div><span class="stat-label">TTFT P95</span><strong>{fmt(stats.get('ttft',{}).get('p95_ms',0)/1000)} <small>s</small></strong><span>Successful requests during load</span></div>
    <div><span class="stat-label">LOAD GPU-HOURS</span><strong>{fmt(after_cost.get('gpu_hours'),3)}</strong><span>Baseline: {fmt(before_cost.get('gpu_hours'),3)}</span></div>'''
    sections = []
    nav = ['<a href="#overview">00 Overview</a>']
    viewer_charts = []
    for index, (anchor, title, filename, description) in enumerate(available):
        nav.append(f'<a href="#{anchor}">{index+1:02d} {html.escape(title.split(" · ")[0])}</a>')
        viewer_charts.append([title, filename])
        sections.append(f'''<section class="chart-section" id="{anchor}"><div class="section-heading"><div><span class="section-number">{index+1:02d} / EXPERIMENT ANALYSIS</span><h2>{html.escape(title)}</h2></div><button class="expand" data-chart="{index}">Enlarge ↗</button></div><p class="section-description">{html.escape(description)}</p><figure><button class="image-button" data-chart="{index}"><img src="assets/{filename}" alt="{html.escape(title)}" loading="lazy"></button><figcaption><span>{html.escape(run.name)}</span><a href="assets/{filename}" target="_blank" rel="noopener">Original PNG ↗</a></figcaption></figure></section>''')
    if session_rows:
        sections[-1] += '''<details class="session-details"><summary>Inspect session assignments and migrations</summary><div class="table-toolbar"><label for="session-filter">Find a session</label><input id="session-filter" type="search" placeholder="Trace ID or replica…"></div><div class="table-wrap"><table><thead><tr><th>Lane / row</th><th>Trace</th><th>First root replica</th><th>First request</th><th>Last request end*</th><th>Root turns</th></tr></thead><tbody id="session-rows"></tbody></table></div><div class="table-wrap"><table><thead><tr><th>Trace</th><th>All root replicas</th><th>Lane source</th></tr></thead><tbody>'''+''.join(f"<tr><td>{r['trace_id']}</td><td>{r.get('root_replicas','')}</td><td>{html.escape(r.get('lane_source','inferred'))}</td></tr>" for r in session_rows)+'''</tbody></table></div></details><a class="download" href="assets/session_lanes.csv" download>Download session assignments CSV ↓</a>'''
    full_policy = sorted((run / 'analysis').glob('policy_vs_engine_mean60_0_*.png'))
    for image in full_policy:
        if image.name != 'policy_vs_engine_mean60_0_1050.png':
            shutil.copy2(image,assets/image.name)
            sections.append(f'<p><a href="assets/{image.name}" target="_blank" rel="noopener">Policy/engine chart including drain and idle observation ↗</a></p>')
    warnings = '<p class="correction">'+html.escape('; '.join(plot_errors))+'</p>' if plot_errors else ''
    document = f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{TITLE}</title><meta name="description" content="H100 16-session native CPU KV offloading experiment with a 200-second scale-down delay, plots, and comparison with the baseline."><link rel="stylesheet" href="style.css"></head><body>
    <aside class="sidebar"><a class="brand" href="#overview">EXPERIMENT NOTES</a><div class="sidebar-title">H100 / Scale down</div><div class="sidebar-sub">16 sessions · Native KV offload</div><nav>{''.join(nav)}</nav><div class="sidebar-footer">Completed experiment<br>{manifest['status']}<p>Static report; no live cluster connection.</p></div></aside>
    <main><header id="overview"><div class="eyebrow">RAY SERVE LLM / 200s DOWNSCALE DELAY</div><h1 style="font-size:36px;letter-spacing:-1px;max-width:1000px">{TITLE}</h1><p class="intro">Same 1→16-session workday curve, seed, engine, and mean token policy as the offloading baseline. Downscale delay changes from 1800s to 200s; idle observation extends from 180s to 300s.</p><div class="run-id">RUN <code>{run.name}</code></div><div class="stats">{cards}</div><div class="reading-note"><strong>What scale-down tests</strong><p>CPU cache belongs to each engine. A removed replica takes its CPU cache with it; rerouted sessions can need prefill again. A 200s delay permits scale-down only while the policy continuously asks for fewer replicas.</p></div></header>
    {warnings}<div class="reading-note"><strong>Observed result</strong><p>{html.escape(finding)}</p></div><h2>Compared with the offloading baseline</h2><p class="section-description">Baseline run: {BASELINE.name}. Compare load-period metrics; idle GPU-hours use different observation durations. Request outcomes and routing can change between runs. Latency percentiles include successful requests; failures are listed separately.</p><div class="table-wrap"><table><thead><tr><th>Metric</th><th>1800s delay baseline</th><th>200s delay experiment</th></tr></thead><tbody>{comparisons}</tbody></table></div><h2 style="margin-top:28px">Scale decisions</h2><div class="table-wrap"><table><thead><tr><th>Direction</th><th>Decision</th><th>Replica target</th><th>Running count matched</th></tr></thead><tbody>{event_rows}</tbody></table></div><h2 style="margin-top:28px">Request failures and drain effects</h2><p class="section-description">{html.escape(failure_note)}</p><details><summary>Inspect failed request timing</summary><div class="table-wrap"><table><thead><tr><th>Trace</th><th>Error</th><th>Arrival</th><th>End</th><th>End time UTC</th></tr></thead><tbody>{error_rows}</tbody></table></div></details>
    {''.join(sections)}<h2>First successful root turn showing a replica change</h2><p class="section-description">Observed transitions between consecutive successful root turns with matched engine traces. Computed / input tokens shows how much prefill the receiving engine performed. Failed turns without a joined timing record are excluded; an earlier failed turn can already have warmed the receiving engine’s cache. This establishes routing changes and recomputation, without attributing every change to scale-down.</p><div class="table-wrap"><table><thead><tr><th>Trace</th><th>Arrival</th><th>Engine change</th><th>Computed / input tokens</th><th>TTFT</th></tr></thead><tbody>{transition_rows}</tbody></table></div><footer><span>{TITLE}</span><p>Native offloading retains KV within an engine. This experiment does not test a shared cache or Mooncake. Plot gaps and missing replica windows are not proof of inactivity.</p><a href="assets/report.json" download>Report JSON ↓</a><a href="assets/experiment.yaml" download>Config ↓</a></footer></main>
    <dialog id="chart-dialog" aria-labelledby="dialog-title"><div class="dialog-header"><div><span class="section-number">CHART VIEWER</span><h2 id="dialog-title"></h2></div><button id="close-dialog">Close ✕</button></div><div class="dialog-tools"><button id="previous-chart" aria-label="Previous chart">←</button><button id="next-chart" aria-label="Next chart">→</button><span class="divider"></span><button id="zoom-out" aria-label="Zoom out">−</button><output id="zoom-level">100%</output><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-reset">Fit width</button><a id="dialog-original" target="_blank" rel="noopener">Original PNG ↗</a></div><div id="image-scroll"><img id="dialog-image" alt=""></div><p class="viewer-note">Scroll to explore. + / − to zoom, ← / → to switch charts, Esc to close.</p></dialog><script src="app.js"></script></body></html>'''
    (dist / 'index.html').write_text(document)
    js=(site/'dist/app.js').read_text()
    end=js.index('];')+2
    js='const charts = '+json.dumps(viewer_charts)+';'+js[end:]
    (dist/'app.js').write_text(js)
    report_lines = [TITLE, f"Run: {run}", f"Status: {manifest['status']}",
        finding, failure_note, f"Observed downscales: {len(down)}", f"Scale events: {json.dumps(events)}"]
    report_lines += [f'{label}: baseline {old} {unit}; scale-down {new} {unit}' for label,old,new,unit in request_summary]
    report_lines += [f'Plot errors: {plot_errors}', f'Web report source: {dist}']
    (run/'analysis/report.txt').write_text('\n'.join(report_lines)+'\n')
    print('\n'.join(report_lines),flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',type=Path,required=True)
    p.add_argument('--site-dir',type=Path,required=True)
    a=p.parse_args();run=a.run_dir.resolve();site=a.site_dir.resolve()
    deadline=time.monotonic()+3*3600
    while read(run/'manifest.json').get('status')=='running':
        if time.monotonic()>deadline:raise TimeoutError('Experiment still running after three hours')
        time.sleep(10)
    # Manifest succeeds before finally performs Serve teardown. Wait for its
    # event so cache-loss observations are not confused with final cleanup.
    for _ in range(60):
        events=(run/'deployment/events.jsonl').read_text()
        if any(event in events for event in ('teardown_completed','teardown_failed','teardown_skipped')):break
        time.sleep(2)
    errors=[]
    if read(run/'manifest.json')['status']=='succeeded':
        m=read(run/'manifest.json')
        start=datetime.datetime.fromisoformat(m['started_at']).timestamp()
        stop=datetime.datetime.fromisoformat(m['finished_at']).timestamp()+30
        links=run/'analysis/trace-inputs';links.mkdir(exist_ok=True)
        selected=[]
        for trace in TRACES.glob('*.jsonl'):
            if start<=trace.stat().st_mtime<=stop:
                target=links/trace.name
                if not target.exists():target.symlink_to(trace)
                selected.append(str(trace))
        (run/'analysis/scheduler_trace_inputs.json').write_text(json.dumps(selected,indent=2))
        evaluation=read(run/'analysis/autoscaling_evaluation.json')
        full_end=int(evaluation['periods']['whole_run']['end_s'])
        jobs=[('load_breakdown.py',['--traces',str(links)]),
              ('arrival_stats_stack.py',['--window','180','--end','1050']),
              ('replica_percentile_lines.py',['--window','180','--min-samples','10','--end','1050']),
              ('policy_vs_engine_mean.py',['--lookback','60','--end','1050','--traces',str(links)]),
              ('policy_vs_engine_mean.py',['--lookback','60','--end',str(full_end),'--traces',str(links)]),
              ('session_lanes.py',[])]
        for i,(tool,args) in enumerate(jobs):
            log=run/'analysis'/f'{tool[:-3]}_{i}_generation.log'
            with log.open('w') as f:
                result=subprocess.run([sys.executable,str(REPO/'tools'/tool),str(run),*args],cwd=REPO,stdout=f,stderr=subprocess.STDOUT)
            if result.returncode:errors.append(f'{tool} failed; inspect {log.name}')
    build_report(run,site,errors)


if __name__=='__main__':main()
