# P/D token autoscaling prototype

Companion to `design.md` (current state, options, trade-offs). This file
describes what the prototype on the `celina/prefill-token-autoscaling-policy`
working tree does, how to configure it, how to test it, and what the GPU smoke
run showed.

## What it does

With `KVAwareRouter` on **both** the prefill and the decode `LLMConfig` of a
`build_pd_openai_app` deployment (direct streaming, HAProxy, body forwarding,
`ai-dynamo` installed):

1. The decode deployment's `LLMRouter` builds a second `KVTokenTracker` that
   follows the **prefill** deployment (its replicas, their KV events).
2. For every request it routes to a decode replica, the router also scores the
   prefill replicas on KV overlap and load, reserves the request on the prefill
   tracker under the request's Serve id, and returns the chosen prefill replica
   to HAProxy in the `x-serve-router-pd-prefill-replica` request header.
3. The decode orchestrator reads that header and dispatches its prefill call
   to exactly that replica (`choose_replica(..., pinned_replica_id=...)`,
   honored by the prefill handle's `KVAwareRouter`, which has no tracker in the
   decode process). A stale pin falls back to a load-balanced pick.
4. Prefill engines tag their lifecycle events with their deployment id, so the
   prefill tracker is drained by the real prefill engine (`on_prefill_complete`
   and `on_request_completed` arrive together: the prefill request is one
   token). Decode engines do the same for the decode tracker.
5. Each `LLMServer` replica's `record_autoscaling_stats` asks every router
   replica for the snapshot of **its own** deployment's tracker and relays the
   per-key maximum. So the prefill deployment's custom metrics are the prefill
   tracker's view and the decode deployment's are the decode tracker's.
6. `LLMTokenAutoscalingPolicy` accepts either target alone: the prefill
   deployment is configured with `target_prefill_tokens_per_replica`, the
   decode deployment with `target_decode_requests_per_replica`.

Without `KVAwareRouter` on the prefill config nothing changes: no prefill
tracker, no pin, prefill selection is the prefill handle's configured router,
and the prefill deployment can only autoscale on Serve's built-in
`ongoing_requests`.

Snapshot keys (per tracker): `outstanding_effective_prefill_tokens` (uncached
prompt tokens of requests not yet past their first token),
`active_decode_requests` (past first token, not finished) and the new
`pending_prefill_requests` (requests not yet past their first token; on the
decode tracker this is the remote-prefill-and-transfer wait, reported for
diagnostics and not used by either scaling term).

## Files changed in the Ray checkout

| File | Change |
| --- | --- |
| `routing_policies/kv_aware/constants.py` | `PINNED_REPLICA_ID_KWARG`, `PD_PREFILL_REPLICA_HEADER`, decode port bases 15557 / 17557. |
| `routing_policies/kv_aware/kv_token_tracker.py` | `pending_prefill_requests` counter and snapshot key; `deployment_id` on reservation broadcasts; `select_replica()` (select + reserve among all tracked replicas, returns the full replica id); tracker registry keyed by deployment id (`set_kv_token_tracker(tracker, deployment_id)`, `get_kv_token_tracker(deployment_id)`, single-tracker fallback). |
| `routing_policies/kv_aware/kv_aware_router.py` | Binds to the tracker for its own deployment id; without a tracker honors `pinned_replica_id`. |
| `routing_policies/kv_aware/vllm/token_tracking.py` | `LifecycleEventForwarder` tags batches with the engine's deployment id. |
| `core/ingress/router.py` | Optional prefill tracker; `_select_prefill_replica` and the pin header on `/internal/route`; `on_lifecycle_events(batch, deployment_id)`, `on_reservations_created` and `get_autoscaling_snapshot(deployment_id)` dispatch per deployment; events for an unknown deployment are dropped, never applied to the wrong tracker. |
| `core/ingress/builder.py` | `_build_openai_ingress_request_router(prefill_server=, prefill_config=)`. |
| `core/server/llm_server.py` | `record_autoscaling_stats` asks for the replica's own deployment and merges every numeric key. |
| `serving_patterns/prefill_decode/builder.py` | `_configure_pd_kv_aware_routing`: validates both-KV-aware + direct streaming, shifts decode ports, binds the prefill deployment to the router. |
| `serving_patterns/prefill_decode/pd_server.py` | `_pinned_prefill_replica` and the pinned `choose_replica` + `dispatch` path in `_pd_handle_request`. |
| `serve/experimental/llm_token_autoscaling_policy.py` | Optional terms, `aggregation="max"|"sum"`, `pending_prefill_requests` in the state. |

Tests: `python/ray/llm/tests/serve/cpu/deployments/routers/kv/test_pd_token_autoscaling.py`
(new, 26 tests), updates to `test_request_lifecycle_tracking.py` and
`python/ray/serve/tests/unit/test_llm_token_autoscaling_policy.py`.

## Configuration

```python
from ray.serve.llm import LLMConfig, build_pd_openai_app
from ray.serve.llm.request_router import KVAwareRouter

POLICY = "ray.serve.experimental.llm_token_autoscaling_policy:LLMTokenAutoscalingPolicy"

def config(policy_kwargs):
    return LLMConfig(
        model_loading_config={"model_id": "m", "model_source": "Qwen/Qwen3.6-27B-FP8"},
        engine_kwargs={"kv_transfer_config": {"kv_connector": "NixlConnector", "kv_role": "kv_both"}},
        deployment_config={
            "request_router_config": {"request_router_class": KVAwareRouter},
            "autoscaling_config": {
                "min_replicas": 1, "max_replicas": 4,
                "upscale_delay_s": 0, "downscale_delay_s": 300,
                "metrics_interval_s": 2, "look_back_period_s": 10,
                "policy": {"policy_function": POLICY, "policy_kwargs": policy_kwargs},
            },
        },
    )

app = build_pd_openai_app({
    "prefill_config": config({"target_prefill_tokens_per_replica": 60000}),
    "decode_config": config({"target_decode_requests_per_replica": 8}),
})
```

Environment (driver and controller): `RAY_SERVE_ENABLE_HA_PROXY=1`,
`RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1`,
`RAY_SERVE_INGRESS_REQUEST_ROUTER_FORWARD_BODY=1`, plus a larger
`RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_BUFSIZE` for long prompts.

## Running the tests

The checkout's `python/ray` has no compiled `_raylet`, so pytest must not put
it on `sys.path`. From a directory outside the checkout:

```bash
cd ~/default
python -m pytest --import-mode=importlib -c /dev/null -o asyncio_mode=auto -p no:cacheprovider -q \
  /mnt/local_storage/ray/python/ray/llm/tests/serve/cpu/deployments/routers/kv/ \
  /mnt/local_storage/ray/python/ray/llm/tests/serve/cpu/deployments/prefill_decode_disagg/test_prefill_decode_disagg.py
python -m pytest --import-mode=append -p no:cacheprovider -q \
  /mnt/local_storage/ray/python/ray/serve/tests/unit/test_llm_token_autoscaling_policy.py
RAY_SERVE_ENABLE_HA_PROXY=1 RAY_SERVE_LLM_ENABLE_DIRECT_STREAMING=1 \
python -m pytest --import-mode=importlib -c /dev/null -o asyncio_mode=auto -p no:cacheprovider -q \
  /mnt/local_storage/ray/python/ray/llm/tests/serve/cpu/deployments/prefill_decode_disagg/test_pd_direct_streaming.py
```

(`pytest-asyncio` was installed and `site-packages/ray/tests` symlinked to the
checkout for this; both are dev-only.)

## GPU smoke test

`scripts/pd_token_smoke.py` deploys the configuration above with the cached
`Qwen/Qwen3.6-27B-FP8` (1 prefill + 1 decode replica, max 2 each, prefill
target 60K tokens, decode target 4 requests, upscale delay 0, downscale delay
60s), drives 12 concurrent ~16K-token cold prompts with 400 output tokens for
4 minutes, idles for 2.5 minutes, and records every 2s the per-deployment
router snapshots and Serve replica counts to `timeline.jsonl`.

```bash
cd ~/default/serve-llm-autoscaling-experiment
python findings-pd-token-policy/scripts/pd_token_smoke.py --out-dir runs/pd-token-smoke
```

Results: see `smoke-results.md`.
