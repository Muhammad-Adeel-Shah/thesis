#!/usr/bin/env bash
# P5 backup to Drive (rclone, approved 2026-11-22), run by the session wrapper's exit trap
# BEFORE the stop, from the repo checkout:    export P5_BACKUP_CMD="bash env/aws_p5_backup.sh"
# Small files only: the session ledger and logs, the small report folders and the cloud lock.
# Never the clips. A source that does not exist yet is SKIPPED and reported, never a failure
# (the first session's backup failed on folders its job had not created, 2026-11-24); a
# failed rclone copy still fails, so the wrapper prints BACKUP FAILED.
set -uo pipefail
REMOTE="${P5_BACKUP_REMOTE:-gdrive:dbp-p5}"
RCLONE="${P5_RCLONE:-rclone}"
status=0
copy() {   # copy <source> <destination under the remote> [rclone filter args...]
  local src="$1" dst="$2"; shift 2
  if [ ! -e "$src" ]; then
    echo "backup: skipped $src (does not exist yet)"
    return 0
  fi
  if $RCLONE copy "$src" "$REMOTE/$dst" "$@"; then
    echo "backup: copied $src -> $REMOTE/$dst"
  else
    echo "backup: FAILED for $src"
    status=1
  fi
}
copy "${P5_STATE:-/data/p5/session}" session
copy reports/p5_train_recon reports/p5_train_recon
copy reports/p5_train_reals reports/p5_train_reals
copy reports/p5_cloud reports/p5_cloud
copy env/lock env/lock --include "cloud_p5_*"
exit "$status"
