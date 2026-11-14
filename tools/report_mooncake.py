#!/usr/bin/env python3
"""Build an evidence-backed static Mooncake experiment report after a run."""
import argparse
import collections
import csv
import html
import json
import os
import re
import shutil
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import yaml


def read(path):
    return json.loads(path.read_text())


def number(value, places=2):
    return 'unavailable' if value is None else f'{value:,.{places}f}'


def replica_ids(driver_log, application):
    pattern = re.compile(re.escape(application)+r'_LLMServer:\S+\s+([a-z0-9]{8})\s')
    return set(pattern.findall(driver_log.read_text()))


def operation_data(run, ids, t0):
    """Deltas per full metric series; discard stale exports from other runs."""
    previous = {}
    grouped = collections.defaultdict(lambda: collections.defaultdict(float))
    per_replica = collections.defaultdict(lambda: collections.defaultdict(float))
    totals = collections.defaultdict(float)
    names = {
        'ray_vllm_mooncake_store_operation_bytes_total': 'bytes',
        'ray_vllm_mooncake_store_operation_total': 'operations',
        'ray_vllm_mooncake_store_operation_failed_keys_total': 'failed_keys',
        'ray_vllm_mooncake_store_operation_time_seconds_sum': 'seconds',
        'ray_vllm_mooncake_store_operation_time_seconds_count': 'timed_operations',
        'ray_vllm_external_prefix_cache_hits_total': 'external_hits',
        'ray_vllm_external_prefix_cache_queries_total': 'external_queries',
        'ray_vllm_prefix_cache_hits_total': 'gpu_hits',
        'ray_vllm_prefix_cache_queries_total': 'gpu_queries',
        'ray_vllm_prompt_tokens_by_source_total': 'computed_prefill_tokens',
    }
    for line in (run/'telemetry/serve_metrics.jsonl').open():
        row = json.loads(line)
        if row.get('error'): continue
        at = (row.get('scrape_finished_at_ns', row['timestamp_ns'])-t0)/1e9
        for sample in row.get('samples', []):
            labels = sample['labels']; rid = labels.get('ReplicaId')
            if rid not in ids or sample['name'] not in names or sample['value'] is None: continue
            family = names[sample['name']]
            if family == 'computed_prefill_tokens' and labels.get('source') != 'local_compute': continue
            metric = family
            if labels.get('operation'): metric = labels['operation']+'_'+metric
            if labels.get('status'): metric += '_'+labels['status']
            key = (sample['name'], tuple(sorted(labels.items())))
            value = sample['value']; prior = previous.get(key, 0)
            delta = value-prior if value >= prior else value
            previous[key] = value
            if at < 0: continue
            bucket = int(at//10)*10
            grouped[bucket][metric] += delta
            totals[metric] += delta
            per_replica[rid][metric] += delta
    return dict(totals), dict(grouped), dict(per_replica)


def pool_data(run, t0):
    points = []
    path = run/'telemetry/mooncake_metrics.jsonl'
    if not path.exists(): return {}, []
    for line in path.open():
        row = json.loads(line)
        if row.get('error') or row.get('component') != 'master':continue
        values = {s['name']: s['value'] for s in row['samples'] if not s.get('labels')}
        points.append({'time_s': (row['timestamp_ns']-t0)/1e9, **values})
    summary = {}
    for name in ['master_allocated_bytes','master_total_capacity_bytes']:
        values = [p[name] for p in points if p.get(name) is not None]
        if values:summary[name] = {'initial': values[0], 'peak': max(values), 'last': values[-1]}
    for name in ['master_evicted_key_count','master_evicted_size_bytes','master_successful_evictions_total']:
        # The Python Prometheus parser adds _total to C++ counters whose
        # on-the-wire name omits the conventional suffix.
        values = [p[name] if name in p else p[name+'_total'] for p in points
                  if p.get(name) is not None or p.get(name+'_total') is not None]
        if values:summary[name] = max(0, values[-1]-values[0])
    return summary, points


def cache_plot(run, grouped, pool_points):
    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True, constrained_layout=True)
    fig.patch.set_facecolor('#fcfcfb')
    times = sorted(grouped)
    xs = [t+5 for t in times]
    def values(name, divisor=1):return [grouped[t].get(name, 0)/divisor for t in times]
    axes[0].plot(xs, values('load_get_bytes_ok',10*1024**2), label='Mooncake GET',color='#2a78d6')
    axes[0].plot(xs, values('save_put_bytes_ok',10*1024**2), label='Mooncake PUT',color='#eb6834')
    axes[0].set_ylabel('MiB/s · 10s bins');axes[0].legend()
    axes[1].plot(xs, [100*grouped[t].get('external_hits',0)/grouped[t]['external_queries']
        if grouped[t].get('external_queries',0)>0 else float('nan') for t in times], color='#6250d6')
    axes[1].set_ylabel('External hit tokens /\nexternal query tokens (%)');axes[1].set_ylim(0,105)
    for operation, label, color in [('load_get','GET','#2a78d6'),('save_put','PUT','#eb6834')]:
        axes[2].plot(xs, [1000*grouped[t].get(operation+'_seconds_ok',0)/grouped[t][operation+'_timed_operations_ok']
            if grouped[t].get(operation+'_timed_operations_ok',0)>0 else float('nan') for t in times],label=label,color=color)
    axes[2].set_ylabel('Approximate exported mean\noperation latency (ms)');axes[2].legend()
    px=[p['time_s'] for p in pool_points]
    axes[3].plot(px,[p.get('master_allocated_bytes',float('nan'))/1024**3 for p in pool_points],label='Allocated',color='#1baf7a')
    axes[3].plot(px,[p.get('master_total_capacity_bytes',float('nan'))/1024**3 for p in pool_points],label='Pool capacity',color='#52514e',ls=':')
    axes[3].set_ylabel('Shared pool GiB');axes[3].legend();axes[3].set_xlabel('Seconds since profiling start')
    for ax in axes:
        ax.set_facecolor('#fcfcfb');ax.grid(alpha=.2);ax.spines[['top','right']].set_visible(False)
    fig.suptitle('Mooncake requester transfers and standalone DRAM pool\nOnly this run’s replica IDs; pool occupancy includes all namespaces. Latency is bucket-approximated: >4s observations become 8s.',fontsize=11)
    path=run/'analysis/mooncake_cache.png';fig.savefig(path,dpi=150);plt.close(fig)
    return path


def errors(run, t0):
    records=[]
    for path in run.glob('benchmark/*/profile_export.jsonl'):
        for line in path.open():
            row=json.loads(line);meta=row['metadata']
            if meta.get('benchmark_phase')=='profiling' and row.get('error'):
                start=meta.get('request_start_ns') or meta.get('credit_issued_ns')
                end=meta.get('request_end_ns')
                records.append({'request_id':meta.get('x_request_id'),
                    'time_s':(start-t0)/1e9 if start is not None else None,
                    'end_s':(end-t0)/1e9 if end is not None else None,'error':row['error']})
    return records


def build(run, site, baseline, smoke_path, driver_log, title, previous_attempt=None):
    manifest=read(run/'manifest.json')
    if manifest['status']!='succeeded':raise RuntimeError(f"Experiment status {manifest['status']}: {manifest.get('error')}")
    evaluation=read(run/'analysis/autoscaling_evaluation.json')
    prior=read(baseline/'analysis/autoscaling_evaluation.json')
    data=read(run/'analysis/plot_data.json');t0=data['time_origin']['profiling_start_ns']
    config=yaml.safe_load((run/'resolved.yaml').read_text())
    baseline_config=yaml.safe_load((baseline/'resolved.yaml').read_text())
    decay_rate=config['deployment'].get('routing',{}).get('request_router_kwargs',{}).get('prefill_decay_tokens_per_second',0)
    ids=replica_ids(driver_log,config['deployment']['application_name'])
    if not ids:raise RuntimeError('Could not identify current model replicas; refusing stale metric aggregation')
    totals,buckets,per_replica=operation_data(run,ids,t0);pool,pool_points=pool_data(run,t0)
    cache_image=cache_plot(run,buckets,pool_points)
    diagnostic=read(run/'analysis/diagnostic_plots.json')
    if diagnostic['status']!='generated':raise RuntimeError(f"Required plots incomplete: {diagnostic['status']}")
    smoke=read(smoke_path)
    if smoke['status']!='PASS':raise RuntimeError('Cross-replica smoke is not PASS')
    request_errors=errors(run,t0)
    service_path=run/'deployment/mooncake_services.json'
    services=read(service_path) if service_path.exists() else {}
    previous_failure=None
    if previous_attempt:
        path=previous_attempt/'analysis/monitoring/lease_failure_summary.json'
        previous_failure=read(path) if path.exists() else None
    retest_path=run/'analysis/monitoring/lease_retest_success.json'
    retest=read(retest_path) if retest_path.exists() else None
    routing_path=run/'analysis/monitoring/root_request_routing.json'
    root_routing=read(routing_path) if routing_path.exists() else None
    tree_audit_path=run/'analysis/monitoring/session_tree_audit.json'
    tree_audit=read(tree_audit_path) if tree_audit_path.exists() else None
    scale_up_path=run/'analysis/monitoring/scale_up_shared_hit.json'
    scale_up_hit=read(scale_up_path) if scale_up_path.exists() else None
    post_removal_path=run/'analysis/monitoring/post_removal_shared_hits.json'
    post_removal=read(post_removal_path) if post_removal_path.exists() else None
    port_issue_path=run/'analysis/monitoring/event_publisher_port_collision.json'
    port_issue=read(port_issue_path) if port_issue_path.exists() else None
    concurrency_path=run/'analysis/monitoring/concurrency_limit_events.json'
    concurrency=read(concurrency_path) if concurrency_path.exists() else None
    outcome_path=run/'analysis/monitoring/run_outcome.json'
    outcome=read(outcome_path) if outcome_path.exists() else None
    policy_path=run/'analysis/monitoring/policy_estimate_audit.json'
    policy_audit=read(policy_path) if policy_path.exists() else None
    error_audit_path=run/'analysis/haproxy-evidence/full_error_correlation.json'
    if not error_audit_path.exists():
        error_audit_path=run/'analysis/monitoring/proxy_error_correlation.json'
    error_audit=read(error_audit_path) if error_audit_path.exists() else {}
    downscale_path=run/'analysis/monitoring/post_load_downscale.json'
    downscale_audit=read(downscale_path) if downscale_path.exists() else None
    validation_assets=[]
    for relative,filename,label in [
        ('deployment/schedule_validation.json','schedule-validation.json','Validated schedule and early observed starts'),
        ('deployment/engine_compatibility_audit.json','engine-compatibility.json','Observed model revision, cache dtype and TP compatibility'),
        ('deployment/node_topology_audit.json','node-topology.json','Observed replica node placement'),
        ('analysis/monitoring/runtime_decay_config.json','runtime-decay.json','Applied router decay and autoscaling settings'),
        ('analysis/monitoring/run_outcome.json','run-outcome.json','Final run outcome and error evidence'),
        ('analysis/monitoring/first_evictions.json','first-evictions.json','First observed shared-pool evictions'),
        ('analysis/monitoring/early_root_replacements.json','early-replacements.json','Observed session-slot reuse during the hold'),
        ('analysis/monitoring/engine_startup_failure_summary.json','startup-failures.json','Deduplicated replica initialization failures'),
        ('analysis/monitoring/concurrency_limit_events.json','concurrency-limit-events.json','Observed concurrency limits, live sessions and debt'),
        ('analysis/monitoring/all_conversation_routing.json','request-routing.json','All request routing, including failed requests'),
        ('analysis/monitoring/session_tree_audit.json','session-tree-audit.json','Initial sessions, fresh roots and recorded retirements'),
        ('analysis/monitoring/policy_estimate_audit.json','policy-estimate-audit.json','Published policy input versus reconstructed engine prefill'),
        ('analysis/haproxy-evidence/full_error_correlation.json','transport-error-correlation.json','Per-request transport error timing and retained proxy evidence'),
        ('analysis/monitoring/latency_export_audit.json','latency-export-audit.json','Why exported operation latencies saturate at 8 seconds'),
        ('analysis/monitoring/proxy_error_correlation.json','transport-error-correlation.json','Request error timing and retained proxy hard-stop evidence'),
        ('analysis/monitoring/post_load_downscale.json','post-load-downscale.json','Automatic downscale decision and completed replica removals'),
        ('analysis/monitoring/request_credit_reconciliation.json','credit-reconciliation.json','Sent credits, exported requests and unexported cancellations'),
    ]:
        path=run/relative
        if path.exists():validation_assets.append((path,filename,label))
    stats=evaluation['slo_attainment']['steady']['requests'];before=prior['slo_attainment']['steady']['requests']
    assignments=list(csv.DictReader((run/'analysis/session_lanes.csv').open()))
    from serve_llm_autoscaling.diagnostics.curve import phase_points
    phases=phase_points(config['benchmark']['workload'].get('concurrency_series') or [])
    phase_starts=[]
    for left,right in zip(phases,phases[1:]):
        phase='ramp up' if right['sessions']>left['sessions'] else 'ramp down' if right['sessions']<left['sessions'] else 'hold'
        count=sum(left['time_s']<=float(row['first_request_s'])<right['time_s'] for row in assignments)
        phase_starts.append({'phase':phase,'start_s':left['time_s'],'end_s':right['time_s'],'new_observed_trees':count})
    peak_sessions=config['benchmark']['workload'].get('target_concurrency',0)
    session_summary={'observed_trees':len(assignments),'peak_target_slots':peak_sessions,
        'minimum_slot_reuses':max(0,len(assignments)-peak_sessions),'new_trees_by_phase':phase_starts,
        'note':'New root correlation IDs establish new session trees. Counts above peak target slots establish slot reuse, but do not identify the previous tree or actual lane owner. Phase counts use first request times; polling can place a boundary start just into the next phase, so a hold-phase start is not necessarily a replacement. Bar ends are last observed request ends, not verified tree completion.'}
    session_note=(f"{len(assignments)} observed session trees across a peak target of {peak_sessions} slots; "
                  f"at least {session_summary['minimum_slot_reuses']} session starts reused an existing slot. "
                  +'; '.join(f"{p['phase']}: {p['new_observed_trees']} new trees" for p in phase_starts)+'. '
                  +session_summary['note']+' Chart placement: '
                  +'; '.join(sorted({row.get('lane_source','unavailable') for row in assignments}))+'.')
    if tree_audit:
        initial=tree_audit.get('initial_profile_roots_identified',tree_audit.get('initial_source_lane_count'))
        fresh=tree_audit.get('fresh_root_starts_by_phase')
        source_note=(f"The source audit identified {initial} initial trees and "
                     f"{tree_audit['fresh_root_starts_beyond_initial']} fresh starts")
        if fresh is not None:
            source_note+=f": {fresh.get('hold',0)} during the hold and {fresh.get('ramp_down',0)} during ramp-down"
        session_note=source_note+'. '+session_note
    summary=dict(title=title,run=run.name,status=manifest['status'],baseline=baseline.name,
        evaluation=evaluation,baseline_evaluation=prior,replica_ids=sorted(ids),
        mooncake_operations=totals,mooncake_per_replica=per_replica,shared_pool=pool,
        smoke_result=smoke,failed_requests=request_errors,shared_configuration=manifest.get('shared_kv_cache'),
        service_configuration=services,prior_setup_failure=previous_failure,
        successful_long_read_retest=retest,root_request_routing=root_routing,
        workload_scale_up_shared_hit=scale_up_hit,experiment_configuration=config,
        baseline_configuration=baseline_config,session_summary=session_summary,
        post_removal_shared_hits=post_removal,event_publisher_port_collision=port_issue,
        observed_concurrency=concurrency,session_tree_audit=tree_audit,
        final_run_outcome=outcome,policy_estimate_audit=policy_audit,
        post_load_downscale=downscale_audit)
    (run/'analysis/mooncake_report.json').write_text(json.dumps(summary,indent=2)+'\n')
    dist=site/'dist';assets=dist/'assets';assets.mkdir(parents=True,exist_ok=True)
    shutil.copy2(run/'analysis/mooncake_report.json',assets/'report.json')
    shutil.copy2(run/'resolved.yaml',assets/'experiment.yaml')
    shutil.copy2(smoke_path,assets/'smoke.json')
    if service_path.exists():shutil.copy2(service_path,assets/'services.json')
    if previous_failure:
        (assets/'setup-failure.json').write_text(json.dumps(previous_failure,indent=2)+'\n')
    if retest:shutil.copy2(retest_path,assets/'long-read-retest.json')
    if root_routing:shutil.copy2(routing_path,assets/'root-request-routing.json')
    if scale_up_hit:shutil.copy2(scale_up_path,assets/'scale-up-shared-hit.json')
    if post_removal:shutil.copy2(post_removal_path,assets/'post-removal-shared-hits.json')
    if port_issue:shutil.copy2(port_issue_path,assets/'event-publisher-port-collision.json')
    for path,filename,_ in validation_assets:shutil.copy2(path,assets/filename)
    validation_links=' · '.join(f'<a href="assets/{filename}">{html.escape(label)} ↗</a>'
                                for _,filename,label in validation_assets)
    initialization_note=''
    transport_note=''
    policy_note=''
    credit_note=''
    telemetry_note=''
    downscale_note=''
    if downscale_audit:
        downscale_note=(f"<p>The automatic target changed "
            f"{downscale_audit['from_target']} → {downscale_audit['to_target']} at "
            f"{html.escape(downscale_audit['target_change_at_utc'])}, "
            f"{downscale_audit['after_sending_stop_s']:.1f}s after sending stopped. "
            "The configured downscale delay measures sustained below-target demand; "
            "its timer can begin before the final request is sent. "
            f"All four removed replicas stopped by {html.escape(downscale_audit['all_four_stopped_by_utc'])}. "
            "No forced termination was observed. No further traffic was sent after this removal, "
            "so this workload does not establish new prefix reuse after removal. "
            '<a href="assets/post-load-downscale.json">Decision and removal evidence ↗</a></p>')
    if outcome and outcome.get('credit_reconciliation'):
        credits=outcome['credit_reconciliation']
        credit_note=(f"<p>AIPerf sent {credits['sent_credits']} credits. "
            f"{credits['exported_profiles']} requests have exported profiles: "
            f"{credits['exported_successes']} successes and {credits['exported_errors']} errors. "
            f"{credits['unexported_cancelled_credits']} additional credits were cancelled without "
            "exported request records when the grace period ended. These credits are separate from "
            "the observed-request failure denominator and latency samples. "
            '<a href="assets/run-outcome.json">Generator and profile reconciliation ↗</a></p>')
    if outcome and outcome.get('telemetry_warning'):
        telemetry_note='<p>'+html.escape(outcome['telemetry_warning'])+'</p>'
    if outcome and outcome.get('transport_error_summary'):
        failure=outcome['transport_error_summary']
        proxy_failures=failure.get('consistent_with_proxy_reload_plus120s',
            failure.get('timing_consistent_with_reload_plus120s',0))
        forcekill_failures=failure.get('same_replica_forcekill_confirmed',0)
        unattributed_failures=failure.get('unattributed_transport_failure',
            failure.get('unattributed_errors',0))
        retained=error_audit.get('proxy_matches_with_retained_worker_stderr_confirmation')
        retained_note=(f"Retained worker stderr directly confirms hard-stop activity for {retained} of these. "
                       if retained is not None else
                       f"{len(error_audit.get('retained_direct_hardstop_stderr',[]))} retained proxy worker stderr logs "
                       "confirm hard-stop activity in the error clusters. ")
        transport_note=(f"<p>{proxy_failures} transport failures were timing-consistent "
            "with HAProxy reloads followed by the 120-second hard stop. "+retained_note+
            f"{forcekill_failures} other failures matched forced termination of the "
            "same serving replica. "
            f"{unattributed_failures} transport failures remain unattributed. "
            "Timing correlation alone does not prove a cause. No Mooncake GET failures or lease errors were observed. "
            '<a href="assets/transport-error-correlation.json">Inspect request-level error evidence ↗</a></p>')
    if policy_audit:
        policy_note=(f"<p>The published prefill signal averaged "
            f"{policy_audit['published_policy_mean_tokens_time_weighted']/1000:.1f}K tokens during load, "
            f"versus about {policy_audit['engine_remaining_prefill_mean_tokens_from_plot_log']/1000:.1f}K "
            "remaining prefill tokens reconstructed from scheduler traces. The purple line uses the actual "
            "published input; the raw-gauge reconstruction in the plot log is a different series. "
            "The green curve excludes cache transfer and decode work, so it does not establish an ideal replica count. "
            '<a href="assets/policy-estimate-audit.json">Policy signal audit ↗</a></p>')
    concurrency_note=''
    if concurrency:
        expected=concurrency.get('expected_update_count',len(concurrency.get('expected_limits',[])))
        concurrency_text=f"Observed {concurrency['observed_update_count']} of {expected} ordered limit updates. "
        if 'endpoint_update_seconds_after_sending_stop' in concurrency:
            concurrency_text+=(
            f"Maximum timing deviation: {concurrency['upward_max_timing_deviation_seconds']:.3f}s up and "
            f"{concurrency['downward_max_timing_deviation_seconds']:.3f}s down. The limit-one update occurred "
            f"{1000*concurrency['endpoint_update_seconds_after_sending_stop']:.0f}ms after HTTP sending stopped, "
            f"with {concurrency['actual_live_tree_count_at_endpoint_update']} trees still draining "
            f"(debt {concurrency['debt_at_endpoint_update']}). "
            f"There were {concurrency['explicit_idle_root_retirement_count']} explicit idle-root retirement log entries. "
            "The observed limit curve and the number of live trees are separate measurements.")
        else:
            concurrency_text+=("This run retains the original linearly interpolated ramp and hold. "
                "Nearest-integer interpolation advances the first added slot at about 15 seconds, "
                "then one per 30 seconds; there is no concurrency down-ramp.")
        concurrency_note='<p>'+html.escape(concurrency_text)+'</p>'
    if port_issue:
        initialization_note=('<div class="reading-note"><strong>Replica initialization failures during scaling</strong>'
            '<p>New vLLM engines failed to bind KV-event publishers, including port 5560, which an active serving engine still owned. '
            'The controller reassigned that surviving replica\'s local rank while its publisher kept the original port. '
            'The base-plus-rank assignment then reused the address for replacement replicas. Some requested capacity could not '
            'start during these attempts. This is an event-publisher lifecycle issue in the current Ray branch; '
            'the standalone Mooncake storage services continued running. The experiment kept its configuration unchanged.</p>'
            '<a href="assets/event-publisher-port-collision.json">Socket ownership, controller rank mapping and source evidence ↗</a></div>')
        if outcome:
            startup=outcome.get('publisher_startup_failure_summary',{})
            initialization_note=initialization_note.replace('<p>New vLLM engines',
                f"<p>{startup.get('count',0)} distinct engine initialization attempts failed "
                f"(ports: {html.escape(str(startup.get('ports',{})))}). New vLLM engines")
    (assets/'sessions.json').write_text(json.dumps(assignments,indent=2))
    shutil.copy2(run/'analysis/session_lanes.csv',assets/'session_lanes.csv')
    charts=[('scaling','Scaling timeline','autoscaling_timeline.png','Replica counts and traffic through load, drain and idle observation.'),
        ('cache','Mooncake transfers and shared pool',cache_image.name,
         'Requester GET/PUT traffic, external hits, approximate exported operation latency, and master-reported pool occupancy. '
         'Ray reconstructs histogram sums from bucket midpoints; every observation above the 4-second final boundary becomes 8 seconds. '
         'Flat 8-second segments do not establish an actual latency ceiling. Request logs retain individual transfer durations.')]
    ordered_plots=[('arrival','arrival_stats_stack'),('replicas','replica_percentile_lines'),
                   ('policy','policy_vs_engine_mean'),('sessions','session_lanes')]
    for anchor,name in ordered_plots:
        step=diagnostic['plots'][name]
        description=step['description']
        if name=='replica_percentile_lines':
            description+=' The full measured range is shown. Replica numbers identify successive engine processes across churn; they are not simultaneous GPU counts.'
        if name=='session_lanes' and any(float(row['last_request_end_s'])>config['benchmark']['duration_s'] for row in assignments):
            description+=' The figure is limited to the traffic window; the table and CSV retain later completion times during drain.'
        charts.append((anchor,step['title'],step['file'],description))
    nav=['<a href="#overview">00 Overview</a>'];sections=[]
    for index,(anchor,label,filename,description) in enumerate(charts):
        shutil.copy2(run/'analysis'/filename,assets/filename)
        nav.append(f'<a href="#{anchor}">{index+1:02d} {html.escape(label)}</a>')
        sections.append(f'''<section class="chart-section" id="{anchor}"><div class="section-heading"><div><span class="section-number">{index+1:02d} / EXPERIMENT ANALYSIS</span><h2>{html.escape(label)}</h2></div><button class="expand" data-chart="{index}">Enlarge ↗</button></div><p class="section-description">{html.escape(description)}</p><figure><button class="image-button" data-chart="{index}"><img src="assets/{filename}" alt="{html.escape(label)}" loading="lazy"></button><figcaption><span>{html.escape(run.name)}</span><a href="assets/{filename}" target="_blank" rel="noopener">Original PNG ↗</a></figcaption></figure></section>''')
    comparisons=[]
    for name,key in [('TTFT p50','p50_ms'),('TTFT p95','p95_ms'),('TTFT p99','p99_ms')]:
        comparisons.append((name,number(before['ttft'][key])+' ms',number(stats['ttft'][key])+' ms'))
    comparisons += [('Failed / total requests',f"{before['failed_requests']} / {before['requests']}",f"{stats['failed_requests']} / {stats['requests']}"),
        ('Allocated load GPU-hours',number(prior['cost']['load']['gpu_hours'],3),number(evaluation['cost']['load']['gpu_hours'],3)),
        ('Mean allocated GPUs during load',number(prior['cost']['load']['gpu_hours']*3600/baseline_config['benchmark']['duration_s'],2),number(evaluation['cost']['load']['gpu_hours']*3600/config['benchmark']['duration_s'],2)),
        ('Starting replica-seconds',number(prior['cost']['load']['starting_replica_seconds'],1),number(evaluation['cost']['load']['starting_replica_seconds'],1)),
        ('Peak running replicas',str(prior['cost']['whole_run']['max_running_replicas']),str(evaluation['cost']['whole_run']['max_running_replicas']))]
    comparison_rows=''.join(f'<tr><td>{a}</td><td>{b}</td><td>{c}</td></tr>' for a,b,c in comparisons)
    proof_rows=''.join(f"<tr><td>{label}</td><td>{smoke[key]['replica']['replica_id']}</td><td>{smoke[key]['computed_prefill_tokens']:,.0f}</td><td>{smoke[key]['external_hit_tokens']:,.0f}</td><td>{smoke[key]['gpu_hit_tokens']:,.0f}</td><td>{smoke[key]['store_get_bytes']/1024**2:.2f} MiB</td></tr>" for label,key in [('Cold producer','producer'),('Different replica','consumer'),('Fresh replica after producer removal','fresh_consumer')])
    event_rows=''.join(f"<tr><td>{e['direction']}</td><td>{e['time_s']:.1f}s</td><td>{e['from_replicas']:g} → {e['to_replicas']:g}</td><td>{number(e.get('running_matched_at_s'),1)}s</td></tr>" for e in evaluation['scale_events'])
    duration=config['benchmark']['duration_s']
    load_down=sum(e['direction']=='down' and e['time_s']<duration for e in evaluation['scale_events'])
    after_down=sum(e['direction']=='down' and e['time_s']>=duration for e in evaluation['scale_events'])
    scale_note=(f"Downscale target decisions: {load_down} during load, {after_down} after load. "
                f"Peak actual running replicas: {evaluation['cost']['whole_run']['max_running_replicas']}. "
                "Target changes can cancel starting replicas and do not establish how many replicas were running.")
    if root_routing:
        matched_roots=root_routing['root_request_count']-root_routing['unmatched_root_requests']
        failed_receivers=sum(bool(row.get('profile_error')) for row in root_routing['all_root_replica_transitions'])
        scale_note+=(f" Scheduler events matched {matched_roots} of {root_routing['root_request_count']} root requests "
                     f"across {root_routing['root_sessions']} sessions, including failed requests; "
                     f"{root_routing['unmatched_root_requests']} were unmatched and "
                     f"{len(root_routing['all_root_replica_transitions'])} root replica transitions were observed "
                     f"({failed_receivers} receiving requests failed). ")
        if load_down:
            scale_note+=("Replica removal during load was observed. Reuse after removal requires "
                         "matching producer and consumer requests to different replicas, external GETs, "
                         "and actual computed prefill; inspect the request evidence alongside these decisions.")
        else:
            scale_note+=("The separate smoke test establishes reuse after producer removal; this workload's "
                         "idle scale-down alone does not establish reuse by a request after removal.")
    shared=config['deployment']['engine']['shared_kv_cache'];pool_peak=pool.get('master_allocated_bytes',{}).get('peak')
    evictions=pool.get('master_evicted_key_count')
    failed_load_keys=sum(value for name,value in totals.items() if name.startswith('load_get_failed_keys'))
    failed_put_keys=sum(value for name,value in totals.items() if name.startswith('save_put_failed_keys'))
    service_note=(f"Master read lease: {services.get('lease_ttl','unavailable')}; "
                  f"cache-load failure policy: {shared.get('load_failure_policy','unavailable')}. "
                  f"Failed GET keys: {failed_load_keys:,.0f}; failed PUT keys: {failed_put_keys:,.0f}. "
                  "Counts include partial-failure statuses, not just operations labeled error.")
    high_watermark=services.get('observed_master_runtime_defaults',{}).get('values',{}).get('eviction_high_watermark_ratio')
    if high_watermark is not None:
        service_note+=f" The running master's eviction high-water mark is {100*high_watermark:g}% of pool capacity."
    setup_note=''
    workload_proof=''
    post_removal_proof=''
    if post_removal:
        case_rows=[]
        for case in post_removal.get('cases',[])[:8]:
            if (not case.get('started_after_producer_stopped') or case.get('profile_error')
                    or case.get('actual_prefill_before_first_token') is None):
                continue
            get_bytes=sum(int(match.group(1)) for line in case.get('get_log_lines',[])
                          if (match:=re.search(r"'memory':\s*(\d+)",line)))
            if not get_bytes:continue
            role='root' if case['agent_depth']==0 else 'subagent'
            previous_error=(case.get('previous_profile_error') or {}).get('type','none recorded')
            case_rows.append(f"<tr><td>{role} {html.escape(case['trace_id'])}, turn {case['turn_index']}</td>"
                f"<td><code>{html.escape(case['from_replica_id'])}</code> → <code>{html.escape(case['to_replica_id'])}</code></td>"
                f"<td>{case['actual_prefill_before_first_token']:,} / {case['prompt_tokens']:,}</td>"
                f"<td>{get_bytes/1024**3:.3f} GiB</td><td>{case['client_ttft_ms']/1000:.3f}s</td>"
                f"<td>{html.escape(previous_error)}</td></tr>")
        if case_rows:
            post_removal_proof=('<h2>Verified reuse after a replica stopped</h2>'
                '<p>These receiving requests began after their previous producer replicas stopped, completed successfully, '
                'logged positive Mooncake DRAM GET bytes, and scheduled materially fewer prefill tokens than their input length. '
                'Admission cache credits can combine local and external hits; the GET logs establish shared-tier use without '
                'assigning every cached token to Mooncake. The preceding interrupted turn remains visible.</p>'
                '<div class="table-wrap"><table><thead><tr><th>Request</th><th>Producer → consumer</th>'
                '<th>Computed / input tokens</th><th>Mooncake GET</th><th>Client TTFT</th><th>Previous turn error</th>'
                '</tr></thead><tbody>'+''.join(case_rows)+'</tbody></table></div>'
                '<p>Less prefill work does not guarantee a proportionate TTFT improvement. Cache transfer and queuing still take time; '
                'the engine prefill plots do not measure transfer duration. '
                '<a href="assets/post-removal-shared-hits.json">Request IDs, stopped times, scheduler chunks and GET logs ↗</a></p>')
    if scale_up_hit and scale_up_hit.get('actual_computed_prefill_tokens') is not None:
        workload_proof=(f'<div class="reading-note"><strong>Shared-cache reuse on the newly added third replica</strong>'
            f'<p>A subagent conversation first ran on replica 2, then continued on new replica 3 '
            f'(<code>{html.escape(scale_up_hit["consumer_replica_id"])}</code>). The consumer loaded '
            f'{scale_up_hit["consumer_get_bytes"]/1024**3:.3f} GiB from Mooncake, reused '
            f'{scale_up_hit["consumer_cached_tokens"]:,} prefix tokens, and computed '
            f'{scale_up_hit["actual_computed_prefill_tokens"]:,} of '
            f'{scale_up_hit["consumer_input_tokens"]:,} input tokens. Both requests completed successfully. '
            'This is a subagent request transition; root sessions stayed on their original replicas.</p>'
            '<a href="assets/scale-up-shared-hit.json">Request IDs, scheduler assignments, computed tokens and GET-log evidence ↗</a></div>')
    if previous_failure:
        setup_note='''<div class="reading-note"><strong>Setup correction before this completed run</strong><p>An earlier attempt exposed a 158K-token TCP load exceeding the upstream 10-second read lease. It was intentionally stopped, with evidence preserved. This run uses a 120-second read lease and recomputation after cache-load failures. The initial short-prefix sharing proof passed separately; these service settings address long-context transfers.</p><a href="assets/setup-failure.json">Read the preserved failure evidence summary ↗</a></div>'''
        if retest:
            setup_note+='<p><a href="assets/long-read-retest.json">Successful 102-key, 5.24 GB read taking 11.1 seconds: direct connector-log evidence ↗</a></p>'
    snippet='''engine:
  enable_prefix_caching: true
  shared_kv_cache:
    enabled: true  # false disables the shared tier
    config_path: /mnt/cluster_storage/mooncake-kv/requester.json
    cache_prefix: MODEL_AND_EXACT_REVISION/experiment-name
    package: mooncake-transfer-engine-cuda13==0.3.14'''
    if decay_rate:
        snippet+='\nrouting:\n  policy: kv_aware\n  request_router_kwargs:\n    prefill_decay_tokens_per_second: '+str(decay_rate)
    baseline_shared=baseline_config['deployment']['engine'].get('shared_kv_cache')
    if baseline_shared and baseline_shared.get('enabled',True):
        baseline_capacity='The baseline also used a standalone Mooncake pool. This run starts with a fresh namespace and empty pool.'
    else:
        native_size=baseline_config['deployment']['engine'].get('kv_offloading_size')
        baseline_capacity=f'Native baseline capacity was {native_size} GiB per replica.'
    current_workload=config['benchmark']['workload'];prior_workload=baseline_config['benchmark']['workload']
    schedule_note=(f"Load duration: {config['benchmark']['duration_s']:g}s; peak target session concurrency: "
        f"{current_workload.get('target_concurrency','unavailable')}. Upscale delay: "
        f"{config['deployment']['autoscaling']['upscale_delay_s']:g}s; downscale delay: "
        f"{config['deployment']['autoscaling']['downscale_delay_s']:g}s. "
        f"Prefill estimate decay: {decay_rate:g} tokens/sec per replica. "
        "Decay spends each replica's estimated service capacity on its oldest request first; "
        "it changes the token autoscaling signal, while Dynamo's selection load remains unchanged. "
        "Topology: model replicas and the standalone pool share one head node with eight H100 GPUs, using TCP.")
    if phases:
        schedule_note+=' Schedule: '+', '.join(
            f"{right['time_s']-left['time_s']:g}s "
            +('ramp up' if right['sessions']>left['sessions'] else 'ramp down' if right['sessions']<left['sessions'] else 'hold')
            +f" ({left['sessions']} → {right['sessions']} target sessions)"
            for left,right in zip(phases,phases[1:]))+'.'
    comparison_note=(f"Baseline peak concurrency: {prior_workload.get('target_concurrency','unavailable')}; "
                     f"baseline load duration: {baseline_config['benchmark']['duration_s']:g}s. "
                     "Different workload curves, delays or decay settings prevent attributing the result to one change. "
                     "Allocated replica GPU-hours do not measure billed cost for a fixed eight-GPU head node.")
    error_counts=collections.Counter(row['error'].get('type','unknown') for row in request_errors)
    warning=(f"{len(request_errors)} failed request records: {dict(error_counts)}. Latency percentiles use successful requests only; compare failure counts alongside them. "
             "The selected KV router was not changed to understand shared residency. Proxy reload/drain effects and routing estimates can influence the comparison; this is one experimental run, not an isolated causal estimate.")
    document=f'''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{html.escape(title)}</title><link rel="stylesheet" href="style.css"></head><body>
<aside class="sidebar"><a class="brand" href="#overview">EXPERIMENT NOTES</a><div class="sidebar-title">H100 / Shared KV</div><div class="sidebar-sub">Mooncake · TCP · Scale down</div><nav>{''.join(nav)}</nav><div class="sidebar-footer">Completed experiment<p>Static report; no live cluster connection.</p></div></aside>
<main><header id="overview"><div class="eyebrow">RAY SERVE LLM / MOONCAKE SHARED DRAM</div><h1 style="font-size:36px;letter-spacing:-1px">{html.escape(title)}</h1><p class="intro">A separate controlled test verified shared-cache reuse after replica removal. This workload measures request latency, scaling and shared-cache traffic with the configured router.</p><div class="run-id">RUN <code>{run.name}</code></div><div class="stats"><div><span class="stat-label">CROSS-REPLICA PROOF</span><strong>PASS</strong><span>Also passed after producer removal</span></div><div><span class="stat-label">TTFT P95</span><strong>{stats['ttft']['p95_ms']/1000:.2f} <small>s</small></strong><span>Successful load-period requests</span></div><div><span class="stat-label">MOONCAKE GET</span><strong>{totals.get('load_get_bytes_ok',0)/1024**3:.2f} <small>GiB</small></strong><span>This run’s replica counters</span></div><div><span class="stat-label">POOL PEAK</span><strong>{number(pool_peak/1024**3 if pool_peak is not None else None)} <small>GiB</small></strong><span>Evicted keys: {number(evictions,0)}</span></div></div></header>
<div class="reading-note"><strong>Experiment settings</strong><p>{html.escape(schedule_note)}</p><p>The harness translates the shared-cache setting into the connector and worker runtime environment. Remove native <code>kv_offloading_size</code> / <code>kv_offloading_backend</code> when enabling it. Pool capacity belongs to the standalone client; it is not allocated per replica.</p><pre style="overflow:auto">{html.escape(snippet)}</pre><p><a href="assets/experiment.yaml">Download the complete resolved experiment configuration ↗</a></p></div>
<h2>Verified sharing before the workload run</h2><p>Same pinned Qwen revision, TP=1, separate H100 workers. Both consumers had zero local GPU hits, positive Mooncake GET bytes, materially less prefill, and identical greedy output. The final consumer started after the producer was removed while the standalone services remained alive.</p><div class="table-wrap"><table><thead><tr><th>Request</th><th>Replica</th><th>Computed tokens</th><th>External-hit tokens</th><th>GPU-hit tokens</th><th>Mooncake GET</th></tr></thead><tbody>{proof_rows}</tbody></table></div>
{workload_proof}{post_removal_proof}{setup_note}<h2 style="margin-top:32px">Compared with the previous experiment</h2><p>Baseline: <code>{baseline.name}</code>. Compare load-period metrics. The shared pool is {pool.get('master_total_capacity_bytes',{}).get('peak',0)/1024**3:g} GiB total. {html.escape(baseline_capacity)} Separate namespaces prevent the smoke test from warming this workload.</p><p>{html.escape(service_note)}</p><div class="table-wrap"><table><thead><tr><th>Metric</th><th>Baseline</th><th>Mooncake</th></tr></thead><tbody>{comparison_rows}</tbody></table></div><div class="reading-note"><strong>Interpretation limits</strong><p>{html.escape(warning)}</p><p>{html.escape(comparison_note)}</p></div>
{credit_note}{transport_note}{telemetry_note}{initialization_note}<h2>Session turnover</h2>{concurrency_note}<p>{html.escape(session_note)}</p><h2>Replica target decisions</h2><p>{html.escape(scale_note)}</p>{downscale_note}{policy_note}<p>{validation_links}</p><div class="table-wrap"><table><thead><tr><th>Direction</th><th>Decision time</th><th>Target</th><th>Running count matched</th></tr></thead><tbody>{event_rows}</tbody></table></div>
{''.join(sections)}<details class="session-details"><summary>Inspect session assignments</summary><div class="table-toolbar"><label for="session-filter">Find a session</label><input id="session-filter" type="search" placeholder="Trace ID or replica…"></div><div class="table-wrap"><table><thead><tr><th>Lane / row</th><th>Trace</th><th>First root replica</th><th>First request</th><th>Last observed request end</th><th>Root turns</th></tr></thead><tbody id="session-rows"></tbody></table></div></details>
<footer><span>{html.escape(title)}</span><p>Session bars show observed spans, not verified session completion. The CSV lists all observed root replicas. The store survives Serve replica removal but is lost if its owner or the Workspace stops.</p><a href="assets/report.json" download>Report and raw summary JSON ↓</a><a href="assets/smoke.json" download>Cross-replica proof JSON ↓</a><a href="assets/session_lanes.csv" download>Session assignments ↓</a></footer></main>
<dialog id="chart-dialog" aria-labelledby="dialog-title"><div class="dialog-header"><div><span class="section-number">CHART VIEWER</span><h2 id="dialog-title"></h2></div><button id="close-dialog">Close ✕</button></div><div class="dialog-tools"><button id="previous-chart" aria-label="Previous chart">←</button><button id="next-chart" aria-label="Next chart">→</button><span class="divider"></span><button id="zoom-out" aria-label="Zoom out">−</button><output id="zoom-level">100%</output><button id="zoom-in" aria-label="Zoom in">+</button><button id="zoom-reset">Fit width</button><a id="dialog-original" target="_blank" rel="noopener">Original PNG ↗</a></div><div id="image-scroll"><img id="dialog-image" alt=""></div><p class="viewer-note">Scroll to explore. + / − to zoom, ← / → to switch charts, Esc to close.</p></dialog><script src="app.js"></script></body></html>'''
    (dist/'index.html').write_text(document)
    js=(dist/'app.js').read_text();end=js.index('];')+2
    (dist/'app.js').write_text('const charts = '+json.dumps([[label,file] for _,label,file,_ in charts])+';'+js[end:])
    lines=[title,f'Run: {run}',f'Status: {manifest["status"]}',f'Mooncake metrics: {json.dumps(totals)}',f'Pool: {json.dumps(pool)}',warning]
    lines.extend(f'{label}: baseline {before}; Mooncake {after}' for label,before,after in comparisons)
    (run/'analysis/mooncake_report.txt').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines),flush=True)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('run',type=Path);parser.add_argument('--site-dir',type=Path,required=True)
    parser.add_argument('--baseline',type=Path,required=True);parser.add_argument('--smoke',type=Path,required=True)
    parser.add_argument('--driver-log',type=Path,required=True)
    parser.add_argument('--title',default='H100s, 16 session concurrency, Mooncake shared KV cache with scale down')
    parser.add_argument('--wait',action='store_true');parser.add_argument('--driver-pid',type=int)
    parser.add_argument('--previous-attempt',type=Path)
    args=parser.parse_args();run=args.run.resolve()
    if args.wait:
        deadline=time.monotonic()+3*3600
        while read(run/'manifest.json')['status']=='running':
            if args.driver_pid:
                try:
                    os.kill(args.driver_pid,0)
                    if Path(f'/proc/{args.driver_pid}/stat').read_text().split()[2]=='Z':raise ProcessLookupError()
                except (ProcessLookupError,FileNotFoundError):
                    raise RuntimeError('Driver stopped while manifest remained running; inspect saved artifacts')
            if time.monotonic()>deadline:raise TimeoutError('Report wait exceeded three hours')
            time.sleep(10)
        for _ in range(60):
            events=(run/'deployment/events.jsonl').read_text()
            if any(event in events for event in ['teardown_completed','teardown_skipped','teardown_failed']):break
            time.sleep(2)
    build(run,args.site_dir.resolve(),args.baseline.resolve(),args.smoke.resolve(),args.driver_log.resolve(),args.title,
          args.previous_attempt.resolve() if args.previous_attempt else None)


if __name__=='__main__':main()
