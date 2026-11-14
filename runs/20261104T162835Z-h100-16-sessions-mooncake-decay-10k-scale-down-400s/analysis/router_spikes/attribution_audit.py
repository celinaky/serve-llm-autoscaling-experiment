"""Offline attribution of the observed router/GPU-prefill discrepancy."""
from pathlib import Path
from collections import Counter
from datetime import datetime
import importlib.util
import json
import re

import numpy as np

OUT = Path(__file__).resolve().parent
RUN = OUT.parent.parent
REPO = RUN.parent.parent
evidence = json.loads((OUT/'evidence.json').read_text())
requests = evidence['requests']
by_id = {r['request_id']:r for r in requests}
t0 = evidence['time_origin_epoch_s']
log = re.sub(r'\x1b\[[0-9;]*m','', (REPO/'findings-h100/mooncake-decay-16-400s/run.log').read_text())
selected = {}
index_warnings = []
for line in log.splitlines():
    if 'Selected worker ' in line:
        rid = re.search(r'request_id="([^"]+)"',line)
        if rid and rid[1] in by_id:
            selected[rid[1]] = {key:float(re.search(key+r'=([0-9.]+)',line)[1])
                for key in ('host_pinned_blocks','disk_blocks','logit')}
    if 'ParentBlockNotFound' in line:
        stamp = re.search(r'2026-\d\d-\d\dT\S+Z',line)
        if stamp:
            index_warnings.append(datetime.fromisoformat(stamp[0].replace('Z','+00:00')).timestamp()-t0)

gets = [r for r in requests if r['get_bytes']]
assert len(selected) == len(requests) == 364
assert len(gets) == 51 and all(r['request_id'] in selected for r in gets)
assert all(selected[r['request_id']]['host_pinned_blocks'] == 0 for r in gets)
assert all(selected[r['request_id']]['disk_blocks'] == 0 for r in gets)
assert sum(r['get_bytes'] for r in gets) == 130248867840

def group_summary(rows):
    queues = [r['queue_wait_s'] for r in rows if r['queue_wait_s'] is not None]
    return dict(requests=len(rows), total_prompt_tokens=sum(r['prompt_tokens'] for r in rows),
        gpu_prefill_tokens_before_first_output=sum(r['computed_before_first_token'] for r in rows),
        get_bytes=sum(r['get_bytes'] for r in rows),
        sum_before_first_prefill_wait_s=sum(queues),
        median_before_first_prefill_wait_s=float(np.median(queues)),
        note='Wait includes async KV lookup/load plus scheduling; it is not isolated transfer duration.')

# Import the saved offline FIFO simulator, with no Ray/vLLM/model initialization.
spec = importlib.util.spec_from_file_location('fifo_replay',REPO/'tools/prefill_decay_replay.py')
replay = importlib.util.module_from_spec(spec)
spec.loader.exec_module(replay)
grid = np.arange(0,1050.001,.1)
curves = {}
for label in ('oracle_gpu_tokens_no_decay','oracle_gpu_tokens_decay_10k',
              'assume_get_full_prompt_decay_10k','assume_get_full_prompt_decay_20k'):
    grouped = {}
    for r in requests:
        terminal = r['first_token_s'] if r['first_token_s'] is not None else r['finish_s']
        if terminal is None:
            continue
        # Hold placement and observed completion times fixed. This measures only
        # accounting; a changed policy would change placement/latency in a real run.
        n = r['prompt_tokens'] if label.startswith('assume_get') and r['get_bytes'] else r['computed_before_first_token']
        grouped.setdefault(r['engine'],[]).append(dict(request_id=r['request_id'],
            queued=r['start_s'], terminal=terminal, uncached=n))
    series = np.zeros_like(grid)
    for rows in grouped.values():
        if label.endswith('no_decay'):
            deltas = [(r['queued'],r['uncached']) for r in rows]+[(r['terminal'],-r['uncached']) for r in rows]
            curve = replay.step_curve(deltas)
        else:
            curve,_ = replay.replay_fifo(rows,20000 if label.endswith('20k') else 10000)
        series += curve.mean(grid,60)
    curves[label] = series

engine = replay.step_curve([d for r in requests for d in r['engine_deltas']])
peaks = []
for s in evidence['spikes']:
    at = s['peak_time_s']
    peaks.append(dict(time_s=at, observed_router_60s_mean=s['published_mean_tokens'],
        observed_engine_60s_mean=float(engine.mean(at,60)),
        illustrative_accounting_only={label:float(np.interp(at,grid,v)) for label,v in curves.items()}))

get_summary = group_summary(gets)
no_get_summary = group_summary([r for r in requests if not r['get_bytes']])
probe = json.loads((OUT/'cpu_event_compatibility_probe.json').read_text())
probe_summary = []
for scenario in probe['scenarios']:
    after = scenario['after_gpu_removal']
    probe_summary.append(dict(name=scenario['name'], medium=scenario['cpu_medium'],
        cpu_cache_spec_kind=scenario['cpu_cache_spec_kind'],
        host_extension_blocks=after['overlap_scores']['workers'][0]['host_pinned_extension_blocks'],
        effective_prefill_tokens=after['effective_prefill_tokens']))
assert probe_summary[0]['host_extension_blocks'] == 0
assert probe_summary[1]['host_extension_blocks'] == 4
assert probe_summary[0]['effective_prefill_tokens'] == 64
assert probe_summary[1]['effective_prefill_tokens'] == 16
result = dict(run=RUN.name, observed=dict(
    requests=len(requests),selection_log_coverage=len(selected),
    get_requests=get_summary,no_get_requests=no_get_summary,
    get_fraction_of_sum_before_first_prefill_wait=get_summary['sum_before_first_prefill_wait_s']/(
        get_summary['sum_before_first_prefill_wait_s']+no_get_summary['sum_before_first_prefill_wait_s']),
    get_requests_with_zero_host_and_disk_credit=len(gets),
    get_requests_on_original_replica=sum('pid-4108718.' in r['engine'] for r in gets),
    first_get_request_arrival_s=min(r['start_s'] for r in gets),
    index_parent_missing_warning_count=len(index_warnings),
    first_index_parent_missing_warning_s=min(index_warnings),
    index_parent_missing_warning_times_s=index_warnings,
    prefill_complete_hook_failures=log.count('KV lifecycle hook on_prefill_complete failed'),
    request_completed_hook_failures=log.count('KV lifecycle hook on_request_completed failed'),
    lifecycle_transport_drop_warnings=log.count('Dropping KV lifecycle events')+log.count('KV lifecycle events dropped'),
    peak_comparisons=peaks), synthetic_cpu_only_probe=probe_summary,
    conclusions=[
        'The installed Mooncake store worker emits medium="cpu". A CPU-only event-contract probe reproduced zero host extension after GPU removal with this label; changing only the label to "CPU" yielded four host extension blocks and reduced effective prefill from 64 to 16 tokens.',
        'No configured global Mooncake lookup in the selector (startup shared_cache_type=none); all 51 confirmed GET requests had host_pinned_blocks=0 and disk_blocks=0 in the selection logs, i.e. no additional host/disk credit beyond the selected GPU overlap.',
        'The connector DOES emit CPU BlockStored events. The event medium spelling is an observed interoperability problem in the installed versions, rather than proof that no events are emitted.',
        'Even recognized CPU hits are not zero cost: the installed selector defaults to 0.75 host-cache hit credit, leaving a 25% token-equivalent load in the synthetic probe. This is different from GPU tokens actually computed.',
        'The Ray 10K/s FIFO decay only subtracts work and applies one shared budget per worker; it does not alter Dynamo routing reservations.',
        'There is no early correction of the original router estimate at external-hit discovery or per-prefill-chunk completion. RequestTokenTracker signals prefill completion on first output token.',
        'The green curve uses retrospective GPU chunk work; KV transfer duration/bytes are not expressed as GPU-prefill tokens.',
        'The 60s mean spreads/retains overestimates; 400s downscale delay changes scaling decisions rather than token measurements.',
        'ParentBlockNotFound is an additional index-consistency concern; it can reduce cache credit but its contribution cannot be quantified from these logs.',
        'Completion-hook errors occur after local autoscaling state is removed; no prefill-completion failures or lifecycle-transport-drop warnings were observed.',
    ],
    limits=[
        'Selection-time per-request effective_prefill_tokens and GPU overlap were not logged. Zero host/disk credit does NOT imply zero GPU overlap.',
        'There is no valid exact percentage attribution between missing external credit, GPU-index inconsistency, sampling/lifecycle latency, and fixed-rate estimation.',
        'Illustrative replays use observed placements/times with hypothetical initial cache credit, not recovered router estimates or predictions of a changed deployment.',
        'The CPU-only event-contract probe uses one full-attention cache group and synthetic hashes/tokens. It is not an end-to-end Qwen hybrid-model fix validation or a replay of the unsaved historical events.',
    ],site_modified=False,gpu_work_started=False)
(OUT/'attribution_audit.json').write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result['observed'].items() if k!='index_parent_missing_warning_times_s'},indent=2))
