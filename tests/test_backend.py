from serve_llm_autoscaling.backend import RayServeBackend
from serve_llm_autoscaling.config import ExperimentConfig, load_config


def test_deployment_spec_matches_baseline():
    config = load_config("experiments/baseline.yaml")
    spec = RayServeBackend(config).deployment_spec()
    assert spec["model_loading_config"] == {
        "model_id": "Qwen/Qwen3-0.6B-FP8",
        "model_source": "Qwen/Qwen3-0.6B-FP8",
    }
    assert spec["engine_kwargs"]["max_model_len"] == 10000
    assert spec["engine_kwargs"]["enable_prefix_caching"] is False
    assert spec["engine_kwargs"]["enable_prompt_tokens_details"] is True
    assert spec["deployment_config"]["autoscaling_config"]["min_replicas"] == 1
    assert spec["deployment_config"]["autoscaling_config"]["max_replicas"] == 1


def test_deployment_spec_leaves_max_ongoing_requests_to_serve_llm_by_default():
    config = load_config("experiments/baseline.yaml")
    spec = RayServeBackend(config).deployment_spec()
    # Ray Serve LLM's own default (1e9) applies unless the experiment caps it.
    assert "max_ongoing_requests" not in spec["deployment_config"]

    config.deployment.max_ongoing_requests = 128
    spec = RayServeBackend(config).deployment_spec()
    assert spec["deployment_config"]["max_ongoing_requests"] == 128


def test_deployment_spec_forwards_only_set_engine_and_autoscaling_keys():
    config = ExperimentConfig.model_validate({
        "name": "defaults", "deployment": {"model_id": "m"},
    })
    spec = RayServeBackend(config).deployment_spec()
    # Unset keys fall back to vLLM's and Serve's own defaults.
    assert spec["engine_kwargs"] == {"enable_prompt_tokens_details": True}
    assert set(spec["deployment_config"]["autoscaling_config"]) == {"policy"}
    assert "experimental_configs" not in spec
    assert "request_router_config" not in spec["deployment_config"]

    config = ExperimentConfig.model_validate({
        "name": "set", "deployment": {
            "model_id": "m",
            "engine": {"max_num_seqs": 64, "kv_cache_dtype": "fp8"},
            "autoscaling": {"max_replicas": 4, "target_ongoing_requests": 8},
            "experimental_configs": {"stream_batching_interval_ms": 0},
        },
    })
    spec = RayServeBackend(config).deployment_spec()
    assert spec["engine_kwargs"] == {
        "max_num_seqs": 64, "kv_cache_dtype": "fp8", "enable_prompt_tokens_details": True,
    }
    autoscaling = spec["deployment_config"]["autoscaling_config"]
    assert autoscaling["max_replicas"] == 4 and autoscaling["target_ongoing_requests"] == 8
    assert "min_replicas" not in autoscaling
    assert spec["experimental_configs"] == {"stream_batching_interval_ms": 0}
