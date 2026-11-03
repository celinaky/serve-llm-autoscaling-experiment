# Gauge gap investigation (2026-11-02)

Why the KV router's `outstanding_effective_prefill_tokens` gauge, as seen by the token
autoscaling policy, runs above the engines' remaining uncached prefill rebuilt from the vLLM
scheduler traces, for run `20261102T203807Z-b200-workday-curve-prefill-mean`.

- `claude_trace.md` — the Claude Code session that did the analysis: every question, answer,
  command and (trimmed) output, in order.
- `*.py` — the one-off scripts the session ran, copied from its scratchpad. They expect to run
  from one directory (they write and read `gauge_decomp_state.json`, `aiperf_recs.json`,
  `pipeline_sim.json`, `sim_best.json`, `eng_deltas.json` next to themselves) and hard-code the
  run directory and the trace directory `/mnt/cluster_storage/.../vllm-scheduler-traces`.
  Run order: `gauge_decomp.py <run>` first (writes the state file), then the others.
- Plots produced during the session live one level up: `policy_vs_engine_rebuild60_*.png`.
  The reusable tool is `tools/policy_vs_engine_rebuild.py` in the repo root.

Headline: policy mean 15.4K vs engine 8.2K over the load phase. 5.2K is the router holding a
request's whole uncached prompt until first token instead of burning it down per chunk; the
remainder splits into timing (admission ~0.1–0.4s before engine queue, decrement at first token),
the hybrid-Mamba model making the engine unable to use some prefix hits the router's index can
see (router lower), and the router missing roughly a sixth of long-lived sessions' cached blocks
after the first replica join (router higher). The raw gauge scrapes are a 10s sample-and-hold
of Ray's metrics export; use the policy's published mean, not an average of raw scrapes.
