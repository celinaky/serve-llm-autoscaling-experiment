#!/usr/bin/env bash
# Publish the B200 workspace's migration files through this workspace's own
# S3-compatible bucket and print presigned download commands (valid 7 days)
# for the H100 workspace, which needs no credentials to use them.
#
# Run on the old (B200, nscale-osl1) workspace:
#   bash findings-prefill-arrival-p95/publish_migration.sh
# Optional: BUCKET=s3://other-bucket PROFILE=rkn to use another bucket/profile.
set -euo pipefail

CS=/mnt/cluster_storage/serve-llm-autoscaling-experiment
M=$CS/workspace-migration-2026-11-02
BUCKET=${BUCKET:-s3://anyscale-nscale-osl1-system-storage}
PREFIX=${PREFIX:-celina-workspace-migration-2026-11-02}
EXPIRES=${EXPIRES:-604800}   # 7 days, the SigV4 maximum
AWS=(aws ${PROFILE:+--profile "$PROFILE"})

FILES=(
  "$M/RESTORE.md"
  "$M/pip-freeze.txt"
  "$M/gpus.txt"
  "$M/keep_alive.py"
  "/mnt/cluster_storage/kvfinal4/ai_dynamo_runtime-1.7.0-cp311-abi3-manylinux_2_35_x86_64.whl"
  "$M/claude-home.tar.gz"
  "$CS/sessions/2026-10-30_to_11-02_prefill-arrival-p95-session_c79a47ff.jsonl.gz"
  "$M/runs-raw-streams.tar.gz"
  "$M/vllm-scheduler-traces.tar.gz"
)

tmp=$(mktemp -d)
for f in "${FILES[@]}"; do
  [ -f "$f" ] || { echo "missing: $f" >&2; exit 1; }
done
( cd "$tmp" && for f in "${FILES[@]}"; do sha256sum "$f" | sed "s| .*/| |" ; done > SHA256SUMS )

echo "# uploading to $BUCKET/$PREFIX/" >&2
for f in "${FILES[@]}" "$tmp/SHA256SUMS"; do
  "${AWS[@]}" s3 cp "$f" "$BUCKET/$PREFIX/$(basename "$f")" --only-show-errors
done

out=$tmp/download_migration.sh
{
  echo "#!/usr/bin/env bash"
  echo "# Generated $(date -u +%FT%TZ); links expire in $((EXPIRES/86400)) days. Run on the H100 workspace."
  echo "set -e; mkdir -p /home/ray/default/migration && cd /home/ray/default/migration"
  for f in "${FILES[@]}" "$tmp/SHA256SUMS"; do
    n=$(basename "$f")
    url=$("${AWS[@]}" s3 presign "$BUCKET/$PREFIX/$n" --expires-in "$EXPIRES")
    printf 'curl -fL --retry 3 -o %q %q\n' "$n" "$url"
  done
  echo "sha256sum -c SHA256SUMS"
} > "$out"
cp "$out" /home/ray/default/download_migration.sh
echo "# wrote /home/ray/default/download_migration.sh" >&2
echo "# paste its contents into a shell on the H100 workspace (or copy the file there)." >&2
cat "$out"
