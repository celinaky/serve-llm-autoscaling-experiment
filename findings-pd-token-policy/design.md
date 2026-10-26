# Token autoscaling for P/D disaggregation: current state, options, prototype

Written 2026-10-26 against `celina/prefill-token-autoscaling-policy`
(48c0441d2e, one commit over master 35661db2a3 of 2026-10-22) and
`origin/master` ed1d83ac1f (2026-10-25). Everything in section 1 was checked
by reading the code paths named, not the docs.

## 1. Current state

### 1.1 What is merged

**P/D is decode-as-orchestrator** (`serving_patterns/prefill_decode/pd_server.py`,
`builder.py`). The graph is `ingress -> Decode:<model> -> Prefill:<model>`.
`PDDecodeServer.chat()` deep-copies the request, clamps it to one token,
calls `prefill_handle.chat.remote(...)` (a Serve handle call made *inside the
decode replica*), takes `kv_transfer_params` from the first prefill chunk, and
runs the real request on its own engine. Under direct streaming the decode
server is the ASGI ingress and `LLMRouter` is attached as the ingress request
router **for the decode deployment only**:
`_build_openai_ingress_request_router(server=decode_deployment,
llm_config=pd_config.decode_config)` (builder.py:981-985). The ingress never
sees the prefill deployment; prefill replica selection happens in the decode
replica's process through the prefill handle's request router (default
`PowerOfTwoChoicesRequestRouter`).

**KV-aware routing** (`routing_policies/kv_aware/`) builds exactly one
`KVTokenTracker` per `LLMRouter` replica, for the deployment the router's
handle targets (`router.py:LLMRouter.__init__`). The tracker owns a Dynamo
`SelectionService` fed by that deployment's KV events, books a reservation
per selected request (`select_worker`), and is drained by engine lifecycle
events (`on_prefill_complete` at the first output token,
`on_request_completed` at the end) that every engine replica broadcasts to
the deployment named `LLMRouter` (`vllm/token_tracking.py`). Both the KV
events and the lifecycle events are enabled per engine by
`is_kv_aware(llm_config)` of *that engine's* config
(`vllm_engine.py:619-645`, `kv_aware/utils.py`).

**Your branch** adds two incremental counters to that tracker
(`outstanding_effective_prefill_tokens`, `active_decode_requests`), a
`get_autoscaling_snapshot` RPC on `LLMRouter`, a relay in
`LLMServer.record_autoscaling_stats` that broadcasts to every router replica
and takes the max, and `LLMTokenAutoscalingPolicy`, which takes the max over
reporters and asks for `max(ceil(P/target_P), ceil(D/target_D), 1)`.

**Docs** (`doc/source/serve/llm/user-guides/kv-aware-routing.md`): "No
prefill-decode disaggregation. The router doesn't yet support disaggregated
prefill and decode." That is accurate for master and for your branch.

### 1.2 What actually happens if you configure KVAwareRouter in a P/D app today

Traced, not run (an experiment would show the same, see 1.4 for a test).

**Case A: `KVAwareRouter` on `decode_config` only.**

- Decode selection is KV-aware against the *decode* engines' KV index. After
  a NIXL transfer the decode engine holds the prompt's blocks, so "effective
  prefill tokens" on the decode tracker means *tokens whose KV is not yet on
  that decode replica*, i.e. transfer work, not prefill compute.
- The decode tracker's lifecycle: `on_prefill_complete` fires at the decode
  engine's first generated token, which is after remote prefill *and* KV
  transfer. So your branch's `outstanding_effective_prefill_tokens` on the
  decode deployment in P/D counts "requests between routing and first token,
  weighted by tokens not cached at D" (prefill queue + prefill + transfer),
  and `active_decode_requests` is the count of generating requests. Those are
  coherent signals, but both describe the decode deployment.
- The prefill deployment gets nothing: `PDPrefillServer` inherits
  `record_autoscaling_stats`, whose gate `is_kv_aware(self._llm_config)` is
  False for the prefill config, so it returns `{}`, the prefill deployment has
  no custom metrics, and `LLMTokenAutoscalingPolicy` holds
  `ctx.target_num_replicas` forever. Prefill never scales.
- Prefill selection is power-of-two over prefill replicas with no cache
  awareness, so a session's follow-up turns land on different prefill
  replicas and recompute the whole context. On the AgentX workload that is
  the dominant cost. (Session affinity through `ConsistentHashRouter` on the
  prefill config does work: `pd_server.py:287-290` forwards the session id.)

**Case B: `KVAwareRouter` on both configs.** Do not do this on master or on
your branch:

- Prefill engines now emit lifecycle events (`is_kv_aware(prefill_config)`)
  to `LLMRouter`, which applies them to its *decode* tracker. The request id
  is the same on both legs: HAProxy stamps `x-request-id`, the decode replica
  adopts it (`replica.py:3209`), and the handle call to prefill copies the
  caller's `request_id` into the new `RequestMetadata`
  (`serve/_private/default_impl.py:118`). So prefill's
  `on_prefill_complete` marks the decode reservation prefill-complete before
  the transfer has started, and prefill's `on_request_completed` (the prefill
  request is one token, so it fires within milliseconds) **frees the decode
  reservation** while the decode replica is still generating. The decode
  tracker's load view and `active_decode_requests` collapse toward zero and
  decode routing degrades to herding.
- The prefill handle's `KVAwareRouter` runs in the decode replica's process,
  finds no tracker, and "degrades to load-balanced selection", which is
  `random.choice` (`kv_aware_router.py:112`), worse than the default
  power-of-two.
- Colocated prefill and decode replicas both bind KV-events port
  `5557 + local_rank` and token-channel port `7557 + local_rank`
  (`vllm/kv_events.py`), so one engine's socket bind fails. PR #66445 moved
  decode to 15557 / 17557 for this reason.

### 1.3 PR #66445 / branch `pd-kv-aware-v2`

- Author Jeffrey Wang, draft, 4 commits dated 2026-09-18..24 on a 2026-09-15
  master. Not merged. Stale-bot pinged it 2026-10-08. On 2026-10-16 the
  author commented: "Evaluating the new router application architecture
  #66506. Will adopt that for PD." #66506 (1/5) and #66507 (2/5) of that
  series are merged (2026-10-20/21); #66519 (3/5), #67087 (4/5) and #67218
  (5/5, "Preserve KV-aware application routing") are open. 5/5 gives the
  shared router one tokenizer and one `KVTokenTracker` per KV-aware target
  deployment, keyed by deployment id, with lifecycle events routed back to
  the right tracker. So the P/D routing work is going to be re-done on that
  foundation; the branch as it stands is a design reference, not something
  to rebase.
- Design: **ingress-as-orchestrator**. `LLMPDRouter` owns two trackers; the
  prefill one runs Dynamo in prefill-router mode (`track_prefill_tokens`),
  the decode one with `overlap_score_credit: 0` and `assume_kv_reuse: False`.
  `/internal/route` tokenizes, calls `prefill.prefill.remote(...)` and
  **waits for the whole prefill inside the HAProxy routing call**, then
  selects decode and returns `kv_transfer_params` to HAProxy as a header;
  `PDDecodeServer` in the decode stage reads the header and skips its own
  orchestration. Lifecycle events carry `deployment_id`. The HAProxy Lua
  template was changed to stop retrying a router after bytes were sent (a
  retry would run prefill twice). Costs: HAProxy's router-call timeout is
  `RAY_SERVE_HAPROXY_INGRESS_REQUEST_ROUTER_TIMEOUT_S` = 5 s, so a 64K cold
  prompt needs it raised and every prefill sits on HAProxy's decision path;
  text-only prompts; no data parallelism; both configs must be KV-aware.
- Autoscaling: nothing in the branch, but its two trackers are exactly the
  per-stage source of truth the policy wants. The engine-side changes it
  needs (lifecycle events keyed by deployment, a `routing_request_id`
  context var) are the same ones the prototype below needs.

### 1.4 Source of truth

| Quantity | Engine-local (per replica) | Router tracker | Serve built-in |
| --- | --- | --- | --- |
| Uncached prefill tokens outstanding on prefill | Sum of prompt tokens of in-flight prefill requests. **Cache-blind for queued requests**: vLLM only learns a request's prefix-cache hit when it schedules it (`KVCacheManager.get_computed_blocks`), and `RequestOutput.num_cached_tokens` arrives with the first output, after the work is done. `SchedulerStats` (vLLM 0.31) has `num_waiting_reqs`, `num_running_reqs`, `prefix_cache_stats` but no pending-token figure. | Dynamo index over the prefill engines' KV events gives the predicted uncached count **at selection time**, which is what your aggregated branch uses. Needs a tracker for the prefill deployment inside the ingress and needs to know which prefill replica each request went to. | `ongoing_requests` on the prefill deployment: a request count, not tokens. |
| Active decode requests | Requests past their first token and not finished, counted in the decode engine's `generate()` stream. Exact, cheap, no router dependency. | Decode tracker after `on_prefill_complete`; eventually consistent. | `ongoing_requests` on the decode deployment, which also counts requests still waiting for remote prefill or transfer. |

So: decode needs no new infrastructure; any of the three columns works and
they differ only in whether the pre-first-token wait is counted. Prefill
tokens need either the router's KV index (cache-aware) or an acceptance of
cache-blindness. On a multi-turn workload where follow-up turns carry a
100K context and ~1K new tokens, a cache-blind backlog overstates prefill
work by ~100x for every warm turn, so the KV index is the only source worth
scaling prefill on for the AgentX-style traffic you measure.

## 2. Options for a P/D token policy

**A. Replica-local engine counters, one policy per deployment (sum across
replicas).** Wrap the engine's `generate()` on both servers; prefill reports
its in-flight prompt tokens, decode its generating requests;
`record_autoscaling_stats` returns the replica's own numbers; the policy sums
them. Works on the merged flow, with any prefill router, without Dynamo or
direct streaming. Cost: cache-blind prefill backlog (above). Right fallback
when KV-aware routing is off; wrong primary signal for cached workloads.

**B. Port `pd-kv-aware-v2`.** Two trackers in the ingress, cache-aware
prefill signal and KV-aware prefill routing for free. Cost: ~2K lines that
conflict with a month of router changes (LoRA, Anthropic messages, text-only
staging, tokenize-once removal), prefill on HAProxy's decision path with a
5 s default timeout, an author who is re-architecting it, and a routing
redesign bundled into an autoscaling change.

**C. Hybrid: router-selected prefill, decode-orchestrated transfer (the
prototype).** Keep the merged request flow and data path. Add a *prefill*
`KVTokenTracker` to `LLMRouter` next to the decode one. When the router
routes a request to a decode replica it also asks the prefill tracker to
select and reserve a prefill replica for the same request id and token ids,
and returns that replica id to HAProxy as an extra request header. The
decode orchestrator reads the header and dispatches the prefill request to
exactly that replica (`choose_replica(..., pinned_replica_id=...)` honored by
`KVAwareRouter` when it has no tracker in its process, which is precisely the
decode-process case). Prefill engines broadcast lifecycle events tagged with
their deployment id, so the prefill tracker is drained by the real prefill
engine. Each deployment's `record_autoscaling_stats` asks the router for *its
own* tracker's snapshot. One policy class with optional terms serves both:
prefill scales on `outstanding_effective_prefill_tokens`, decode on
`active_decode_requests`.

Why C over A and B: it is the only option that gives a cache-aware prefill
signal on the merged flow, it also fixes the prefill routing hole in 1.2
(prefill becomes KV-aware instead of random), it changes nothing in HAProxy
or the client path, it keeps prefill off the routing-call deadline, and its
tracker-per-deployment shape is the one #67218 introduces, so the autoscaling
pieces carry over when routing moves into the router application.

Trade-offs and decisions inside C:

- *Reservation semantics on prefill.* The prefill engine's request is one
  token, so `on_prefill_complete` and `on_request_completed` arrive together
  and the reservation is freed as soon as prefill finishes: the tracker's
  `outstanding_effective_prefill_tokens` is exactly "uncached tokens routed to
  prefill replicas that have not finished prefilling". `active_decode_requests`
  on the prefill tracker is always ~0 and is ignored.
- *Decode phase definition.* On the decode tracker, the time between routing
  and the decode engine's first token is remote prefill plus transfer. The
  prototype reports it as `pending_decode_requests` in the snapshot so it can
  be graphed, and scales decode on `active_decode_requests` only, which is
  the quantity your sweep found predicts ITL.
- *Pin is advisory.* If the pinned prefill replica has gone away by the time
  the decode orchestrator dispatches, the prefill handle's router falls back
  to its load-balanced pick and the prefill tracker's stale reservation is
  evicted by the existing TTL sweep. No request fails because of the pin.
- *Scoring mode.* The prefill tracker uses the same `select_and_reserve` as
  the aggregated router (overlap credit plus active prefill/decode load).
  PR #66445 used Dynamo's prefill-router override; that is a one-dict change
  (`router_config_override`) once the mode's semantics are checked against
  `ai-dynamo` 1.5.0, and is not needed for the signal.
- *Aggregation.* Router snapshots are global, so the policy still takes the
  max across reporting replicas; replica-local counters (option A) would
  need a sum. The policy takes an `aggregation` kwarg so the same class can
  consume either source later.
- *Relay.* Each deployment keeps at least one replica (`min_replicas >= 1`),
  as on the aggregated branch, because the deployment's own replicas are the
  relay for its snapshot.
- *Ports.* Decode defaults to KV-events port base 15557 and token-channel
  base 17557 when both stages are KV-aware, as in #66445.
- *Requirements.* `KVAwareRouter` on both configs, direct streaming, body
  forwarding, `ai-dynamo`. Without KV-aware prefill the router simply has no
  prefill tracker, prefill selection stays as it was, and the prefill
  deployment can only autoscale on Serve's built-in `ongoing_requests`.

## 3. Prototype

Implemented as option C on the working tree of
`celina/prefill-token-autoscaling-policy` (uncommitted). `README.md` in this
folder lists the files, configuration and test commands; `smoke-results.md`
has the GPU validation. In short: 12 files, 642 insertions; 26 new unit tests
plus the existing KV router and P/D suites (212 tests) and the direct
streaming e2e tests pass; on 4 B200s with Qwen3.6-27B-FP8 the prefill
deployment scaled on outstanding prefill tokens 2s after crossing its target
and the decode deployment on active decodes 6s after its own, both trackers
drained to zero after load, and both scaled back down independently.

Follow-ups, in order of value:

1. Add a P/D mode to the harness (`prefill_config`/`decode_config` in
   `DeploymentConfig`, `build_pd_openai_app`) and rerun the AgentX workday ramp
   with the sweep's `P=60K, D=8` to measure the cache-aware prefill signal and
   KV-aware prefill selection on warm turns.
2. Try Dynamo's prefill-router scoring mode for the prefill tracker
   (`router_config_override={"track_prefill_tokens": True}` as in #66445) and
   compare prefill replica balance.
3. When #67218 lands, move the prefill tracker and pin into the router
   application; the per-deployment snapshot relay and the policy need no
   change.
