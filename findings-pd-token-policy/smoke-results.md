# GPU smoke test of the P/D token policy, 2026-10-26

Run: `runs/pd-token-smoke/` (`driver.log`, `timeline.jsonl` sampled every 2s,
`summary.txt`). Script: `scripts/pd_token_smoke.py`. A first attempt
(`runs/pd-token-smoke-attempt1-mamba-layout/`) failed at engine start because
Qwen3.6-27B is a hybrid Mamba model and NIXL transfer of its conv state needs
`VLLM_SSM_CONV_STATE_LAYOUT=DS`; the script now sets it in both engines'
`runtime_env`.

## Setup

- `Qwen/Qwen3.6-27B-FP8`, 1 B200 per replica, `max_model_len` 32768,
  `max_num_batched_tokens` 8192, prefix caching on, NIXL `kv_both`.
- `KVAwareRouter` on both configs; direct streaming, HAProxy, body
  forwarding, 8 MiB router buffer.
- Prefill deployment: `LLMTokenAutoscalingPolicy(target_prefill_tokens_per_replica=60000)`.
  Decode deployment: `LLMTokenAutoscalingPolicy(target_decode_requests_per_replica=4)`.
  Both: min 1, max 2, `upscale_delay_s` 0, `downscale_delay_s` 60,
  `metrics_interval_s` 2, `look_back_period_s` 10.
- Load: 12 concurrent clients, each sending back-to-back cold ~16K-token
  prompts (random words, no shared prefix) with 400 output tokens, for 245s,
  then 150s idle.

## What happened

| t (s) | event |
| --- | --- |
| 0 | 1 prefill + 1 decode replica RUNNING, both trackers empty |
| 6 | load starts |
| 8.2 | prefill tracker: 204,985 outstanding uncached tokens over 11 pending requests (> 60K target) |
| 10.2 | prefill replica 2 STARTING (prefill term) |
| 12.2 | decode tracker: 5 active decodes (> 4 target) |
| 18.3 | decode replica 2 STARTING (decode term) |
| 20-180 | steady at 1P+1D: mean 132K outstanding prefill tokens (7.1 requests queued/prefilling), 4.8 active decodes, 7 requests waiting for prefill+transfer on the decode side |
| 186-188 | both second replicas RUNNING (3-minute model load) |
| 195-245 | steady at 2P+2D: mean 57K outstanding prefill tokens (3.1 pending), 8.6 active decodes |
| 251 | load ends; both trackers read 0 within one sample |
| 307 / 313 | prefill, then decode, back to 1 replica after the 60s downscale delay |

344 requests, 0 failures; TTFT p50 6.0s, p90 6.1s, max 10.5s (16K-token cold
prefill on one B200 plus queueing).

## What this validates

- **Per-deployment signals.** The prefill deployment scaled on the prefill
  tracker's `outstanding_effective_prefill_tokens` 2s after the threshold was
  crossed; the decode deployment scaled on the decode tracker's
  `active_decode_requests` 6s after its threshold. Each term fired only for
  its own deployment.
- **Reservations drain.** Both trackers returned to exactly 0 after load,
  so prefill-engine lifecycle events reached the prefill tracker (deployment
  tagging), matched the reservations the router made under HAProxy's request
  id (id propagation through the decode replica's handle call), and the
  decode tracker was never freed by prefill completions (the case-B hazard in
  `design.md` 1.2). No "dropping it", "prefill selection failed" or stale
  eviction warnings in the log.
- **Both trackers registered their own workers**: the router logged two
  deployments' KV event workers with block size 1568 (the hybrid model's
  effective attention block), decode on the shifted port base.
- **Scale-down** followed the 60s delay on both deployments independently.
- The expected one-line warning "No KVTokenTracker in this process" appears
  once per decode replica for the prefill handle's `KVAwareRouter`, which is
  where the pin takes over.

## What it does not validate

- Cache-awareness of the prefill signal (every prompt was cold) and the
  benefit of KV-aware prefill selection on multi-turn traffic; the aggregated
  sweep's AgentX workload would show both. The harness has no P/D mode yet.
- That the pin was honored on the real path: the unit tests cover it, and the
  drained trackers show the request-id path, but the prefill replica that
  served each request is not visible to the client under direct streaming.
  A vLLM prefix-cache hit-rate comparison on a warm workload would show it.
- Target values. 60K / 4 were chosen to trigger within the run, not tuned.

Noise in the log: the workspace's `TelemetryAgent` fails calling
`ServeController._get_next_routing_config_generation`, which this checkout's
controller no longer has; unrelated to the prototype.
