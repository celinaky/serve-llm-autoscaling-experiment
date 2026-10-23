# Token autoscaling policy: target sweep on the B200 workday ramp

Experiment: `experiments/b200_workday_ramp_token.yaml` (32 AgentX sessions
ramped over 600s, every session starting cold, 5 min hold, 4x B200).
SLO: p95 TTFT <= 3000ms and p95 ITL <= 50ms over the whole load period.
Policy: `LLMTokenAutoscalingPolicy` (branch
`celina/prefill-token-autoscaling-policy`), replicas =
`max(ceil(prefill_tokens / P), ceil(decode_requests / D), 1)`.

Goal: the loosest (cheapest) `P`, `D` that meets the SLO.

| run | P (prefill tokens/replica) | D (decodes/replica) | TTFT p95 | ITL p95 | SLO | GPU-h | max replicas | notes |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
