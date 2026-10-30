# Extension of the burst sweep, with side probes throughout

The first 9 bursts of `b200_burst_sweep_single_replica.yaml` with seed 43,
meant to be pooled with runs/20261030T074300Z-b200-burst-sweep-single-replica
(same deployment and design). `benchmark/side-probe/profile_export.jsonl`
holds 576 requests from `tools/side_probe.py --rate 0.3`, sent for the whole
load window (shared ~40K cached prefix + unique 4-16K suffix, 8 output
tokens); see that run's NOTES.md for why. 1 request was preempted before its
first token; exclude rows with `preemptions_before_first_token > 0`.

Requests per (own uncached band, 20K load bin), the two runs pooled, no
preemptions (tools/bucket_counts.py): every cell has >= 12 except >=48K at
180-200K load, which has 6. A >=48K arrival near ~175K lifts the load past
~220K by itself, so that bin is crossed in jumps.
