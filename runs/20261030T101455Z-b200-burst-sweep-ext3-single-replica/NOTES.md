# Third extension of the burst sweep, with mid-size and large side probes

The first 9 bursts of `b200_burst_sweep_single_replica.yaml` with seed 45,
pooled with the seed 42, 43 and 44 burst-sweep runs (20261030T074300Z,
20261030T085155Z, 20261030T093322Z). Two side probers ran for the whole load
window (see the first run's NOTES.md for the mid-size one):

- `benchmark/side-probe/`: 576 requests, `tools/side_probe.py --rate 0.3`
  (shared ~40K cached prefix + unique 4-16K suffix).
- `benchmark/side-probe-large/`: 137 requests, `--rate 0.07 --min-new 48000
  --max-new 90000 --prefix 0 --seed 9` (cold 48-90K prompts), added to fill
  >=48K own uncached at 180-200K load, which the replay crosses in jumps.

No preemptions before first token; KV peaked at 80%.

Requests per (own uncached band, 20K load bin), the four runs pooled, no
preemptions (tools/bucket_counts.py): every cell has >= 12. The >=48K /
180-200K cell has 12: 9 from the replay across the four runs and 3 from
this run's large probes.
