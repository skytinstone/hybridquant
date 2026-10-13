#!/usr/bin/env bash
# Pull the repo and mirror results/ to ~/Downloads/hybridquant_results/<YYYY-MM-DD>/ (Mac side).
# Usage: scripts/sync_results.sh [repo_dir] [dest_root]
#   HQ_SYNC_DRY_RUN=1 makes rsync a dry run (used by tests).
set -u
REPO=${1:-"$(cd "$(dirname "$0")/.." && pwd)"}
DEST_ROOT=${2:-"$HOME/Downloads/hybridquant_results"}
DAY=$(date +%F)
DEST="$DEST_ROOT/$DAY"
mkdir -p "$DEST"
cd "$REPO" || { echo "repo not found: $REPO"; exit 1; }

if git pull -q --ff-only origin main 2>>"$DEST_ROOT/sync_errors.log"; then
  PULL="ok ($(git rev-parse --short HEAD))"
else
  PULL="failed (see sync_errors.log)"
fi

RSYNC_FLAGS=(-a --delete-excluded --exclude '.gitkeep')
[ "${HQ_SYNC_DRY_RUN:-0}" = "1" ] && RSYNC_FLAGS+=(--dry-run)
rsync "${RSYNC_FLAGS[@]}" results/ "$DEST/"
N=$(find "$DEST" -type f | wc -l | tr -d ' ')
echo "$(date -u +%FT%TZ) pull=$PULL copied results/ -> $DEST ($N files)" | tee "$DEST_ROOT/last_sync.txt"
