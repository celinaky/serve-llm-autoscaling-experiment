import json
from contextlib import nullcontext
from pathlib import Path

import pytest
from pydantic import ValidationError

from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import ExperimentConfig
from serve_llm_autoscaling.shared_kv import check_shared_kv_support
from serve_llm_autoscaling.telemetry import mooncake_metrics_collector, parse_metrics


def configuration(path=Path('/shared/requester.json'), **engine):
    values = dict(enable_prefix_caching=True, shared_kv_cache={
        'config_path': str(path), 'cache_prefix': 'qwen@exact-revision/tp1/fp8',
        'health_url': 'http://10.0.0.1:9300/health',
    })
    values.update(engine)
    return ExperimentConfig.model_validate({'name': 'shared-kv', 'deployment': {
        'model_id': 'qwen', 'engine': values,
    }})


def test_shared_connector_and_worker_dependencies_are_forwarded_without_native_override(tmp_path):
    config = configuration()
    config.analysis.scheduler_trace_dir = tmp_path
    for gpu in ['H100', 'B200']:
        config.deployment.accelerator_type = gpu
        spec = RayServeBackend(config).deployment_spec()
        assert spec['accelerator_type'] == gpu
        assert 'shared_kv_cache' not in spec['engine_kwargs']
        assert 'kv_offloading_size' not in spec['engine_kwargs']
        assert spec['engine_kwargs']['kv_transfer_config'] == {
            'kv_connector': 'MooncakeStoreConnector', 'kv_role': 'kv_both',
            'kv_load_failure_policy': 'recompute',
            'kv_connector_extra_config': {'cache_prefix': 'qwen@exact-revision/tp1/fp8',
                'load_async': True, 'save_decode_cache': True},
        }
        assert spec['runtime_env']['pip'] == ['mooncake-transfer-engine-cuda13==0.3.14']
        assert spec['runtime_env']['env_vars']['MOONCAKE_CONFIG_PATH'] == '/shared/requester.json'
        assert spec['runtime_env']['env_vars']['PYTHONHASHSEED'] == '0'
        assert spec['runtime_env']['env_vars']['RAY_SERVE_VLLM_SCHEDULER_TRACE_DIR'] == str(tmp_path)


@pytest.mark.parametrize('override', [
    {'enable_prefix_caching': False},
    {'kv_offloading_size': 64},
    {'kv_offloading_backend': 'native'},
    {'kv_transfer_config': {'kv_connector': 'OffloadingConnector'}},
])
def test_conflicting_engine_settings_are_rejected(override):
    with pytest.raises(ValidationError):
        configuration(**override)


def test_disabled_shared_toggle_uses_existing_native_configuration():
    config = configuration(kv_offloading_backend='native', kv_offloading_size=64,
        shared_kv_cache={'enabled': False, 'config_path': '/missing.json', 'cache_prefix': 'disabled'})
    spec = RayServeBackend(config).deployment_spec()
    assert spec['engine_kwargs']['kv_offloading_size'] == 64
    assert 'shared_kv_cache' not in spec['engine_kwargs']
    assert 'kv_transfer_config' not in spec['engine_kwargs']
    assert 'runtime_env' not in spec
    assert check_shared_kv_support(config) is None


def test_image_with_mooncake_can_skip_runtime_install():
    config = configuration()
    config.deployment.engine.shared_kv_cache.package = None
    assert 'pip' not in RayServeBackend(config).deployment_spec()['runtime_env']


def test_preflight_records_real_requester_configuration(tmp_path, monkeypatch):
    path = tmp_path/'requester.json'
    requester = dict(mode='standalone-store', global_segment_size=0,
        metadata_server='http://10.0.0.1:8080/metadata', master_server_address='10.0.0.1:50051',
        protocol='tcp', local_buffer_size='1GB')
    path.write_text(json.dumps(requester))
    monkeypatch.setattr('serve_llm_autoscaling.shared_kv.socket.create_connection',
                        lambda *args, **kwargs: nullcontext())
    class Response:
        def raise_for_status(self): pass
        def json(self): return {'status': 'healthy'}
    monkeypatch.setattr('serve_llm_autoscaling.shared_kv.requests.get', lambda *a, **k: Response())
    result = check_shared_kv_support(configuration(path))
    assert result['requester'] == requester
    assert len(result['config_sha256']) == 64
    assert result['router_modified'] is False


@pytest.mark.parametrize('mode,amount', [('embedded', 0), ('standalone-store', 1024)])
def test_preflight_rejects_embedded_or_per_rank_pool(tmp_path, mode, amount):
    path = tmp_path/'bad.json';path.write_text(json.dumps({'mode': mode,'global_segment_size': amount}))
    with pytest.raises(ValueError, match='external client owns'):
        check_shared_kv_support(configuration(path))


def test_mooncake_operation_labels_survive_ray_metric_selection():
    samples, _ = parse_metrics('''# TYPE ray_vllm_mooncake_store_operation_bytes_total counter
ray_vllm_mooncake_store_operation_bytes_total{ReplicaId="consumer",operation="load_get",status="ok"} 1234
''')
    assert len(samples) == 1
    assert samples[0]['value'] == 1234
    assert samples[0]['labels']['operation'] == 'load_get'


def test_pool_telemetry_records_failure_separately_from_success(tmp_path):
    def fetch(url):
        if url.endswith('client'):raise TimeoutError('slow endpoint')
        return '# TYPE master_allocated_bytes gauge\nmaster_allocated_bytes 100\n'
    collector = mooncake_metrics_collector(tmp_path/'pool.jsonl', 2,
        {'master': 'http://cache/master', 'client': 'http://cache/client'}, fetch)
    rows = collector._sample(100, 0)
    assert rows[0]['samples'][0]['name'] == 'master_allocated_bytes'
    assert rows[0]['samples'][0]['value'] == 100
    assert rows[0]['error'] is None
    assert rows[1]['error'] == 'TimeoutError: slow endpoint'
