# Side probes added mid-run

`benchmark/side-probe/profile_export.jsonl` holds 625 extra requests from
`tools/side_probe.py --rate 0.3`, sent alongside AIPerf from ~1555s into the
load window to its end (bursts 10-18; bursts 1-9 had none). Each is a shared
~40K-token prefix (cached after the first) plus a unique 4-16K-token suffix,
8 output tokens, Poisson arrivals. They exist because the closed-loop replay
almost never sends 4-16K-uncached turns while a prefill backlog is draining
(running sessions stall), so that row of `tools/state_breakdown.py` was
empty above ~20K load after the first bursts. They add ~3K uncached
tokens/s on average, which is part of the load every request saw.
`load_requests.csv` includes them; tell them apart by joining
`request_id` to that file.

6 requests were preempted before their first token (KV usage reached 100%
at some step); exclude rows with `preemptions_before_first_token > 0`.
