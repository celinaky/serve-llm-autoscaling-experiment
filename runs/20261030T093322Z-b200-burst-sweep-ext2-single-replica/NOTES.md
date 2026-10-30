# Second extension of the burst sweep, with side probes throughout

The first 9 bursts of `b200_burst_sweep_single_replica.yaml` with seed 44,
pooled with runs/20261030T074300Z-b200-burst-sweep-single-replica (seed 42)
and runs/20261030T085155Z-b200-burst-sweep-ext-single-replica (seed 43).
`benchmark/side-probe/profile_export.jsonl` holds 576 requests from
`tools/side_probe.py --rate 0.3` over the whole load window; see the first
run's NOTES.md for why. No preemptions before first token; KV peaked at 87%.

Requests per (own uncached band, 20K load bin), the three runs pooled, no
preemptions (tools/bucket_counts.py): every cell has >= 19 except >=48K at
180-200K load, which has 7 (2, 4 and 1 per run). Every run spends time in
that bin (24-50 arrivals of all sizes each), but >=48K arrivals skip it: one
arriving near ~170K lifts the load past ~220K by itself, and the load comes
back down through the bin during drains, when bursting sessions have
stopped sending turn 1s.
