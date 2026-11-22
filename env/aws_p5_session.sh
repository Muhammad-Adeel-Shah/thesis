#!/usr/bin/env bash
# P5 AWS session wrapper -- every job on the instance runs through this (user rules,
# 2026-11-21). On the instance, from the repo checkout:
#
#     bash env/aws_p5_session.sh --label recon --estimate-hours 6 -- \
#         python scripts/p5_pilot.py --phase reconstruct ...
#
# 1. a HARD CAP is scheduled at session start: shutdown in ceil(estimate x 1.5 x 60) min;
# 2. the IDLE WATCHDOG + spend cap (scripts/p5_cloud_guard.py) is armed beside the job;
# 3. a SHUTDOWN TRAP fires on EXIT and on ERR (and on INT/TERM): the session record is
#    written, then the instance stops (shutdown behaviour = stop, set at launch);
# 4. everything is logged under $P5_STATE (the data volume), never in the repo tree.
#
# Settings (environment): P5_STATE (default /data/p5/session), P5_RATE (1.217 = $1.212 instance + $0.005 public IPv4),
# P5_VOLUME_CREATED (YYYY-MM-DD, required), P5_EBS_PER_DAY (0.5333 = 200 GB x $0.08/30),
# P5_CAP_USD (75), P5_IDLE_MIN (20), P5_POLL (60), P5_SHUTDOWN (default "sudo shutdown -h";
# the dry-run tests replace it with a recorder), P5_BACKUP_CMD (optional, run in the exit
# trap BEFORE the stop: e.g. `rclone copy /data/p5/session gdrive:dbp-p5/session`, so the
# ledger is readable once the instance is stopped). Nothing here reads or prints a secret.
set -Eeuo pipefail

LABEL="" ; EST=""
while [ $# -gt 0 ]; do
  case "$1" in
    --label) LABEL="$2"; shift 2 ;;
    --estimate-hours) EST="$2"; shift 2 ;;
    --) shift; break ;;
    *) echo "usage: aws_p5_session.sh --label L --estimate-hours H -- command..."; exit 2 ;;
  esac
done
[ -n "$LABEL" ] && [ -n "$EST" ] && [ $# -gt 0 ] || {
  echo "usage: aws_p5_session.sh --label L --estimate-hours H -- command..."; exit 2; }
# (no apostrophe in this message: inside "${...:?}" one swallowed the lines below; P5-C test)
: "${P5_VOLUME_CREATED:?set P5_VOLUME_CREATED=YYYY-MM-DD, the creation date of the data volume}"
STATE="${P5_STATE:-/data/p5/session}"
SHUTDOWN="${P5_SHUTDOWN:-sudo shutdown -h}"
PY="${P5_PYTHON:-python}"
REPO="$(cd "$(dirname "$0")/.." && pwd)"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$STATE/logs"
LOG="$STATE/logs/${LABEL}_${STAMP}.log"
RECORD="$STATE/sessions.jsonl"
CAP_MIN="$(awk -v h="$EST" 'BEGIN { m = h * 1.5 * 60; printf "%d", (m == int(m)) ? m : int(m) + 1 }')"
JOB_PID="" ; GUARD_PID="" ; STATUS="running"

finish() {
  local code=$?
  trap - EXIT ERR INT TERM
  [ -n "$JOB_PID" ] && kill "$JOB_PID" 2>/dev/null || true
  [ -n "$GUARD_PID" ] && kill "$GUARD_PID" 2>/dev/null || true
  printf '{"label":"%s","start":"%s","end":"%s","estimate_h":%s,"cap_min":%s,"exit":%s,"status":"%s","log":"%s"}\n' \
    "$LABEL" "$STAMP" "$(date -u +%Y%m%dT%H%M%SZ)" "$EST" "$CAP_MIN" "$code" "$STATUS" "$LOG" >> "$RECORD"
  if [ -n "${P5_BACKUP_CMD:-}" ]; then        # small files only (ledger, records, logs)
    bash -c "$P5_BACKUP_CMD" >> "$LOG" 2>&1 || echo "BACKUP FAILED (see $LOG); stopping anyway" | tee -a "$LOG"
  fi
  echo "session $LABEL ended (exit $code, $STATUS): stopping the instance" | tee -a "$LOG"
  $SHUTDOWN now
  exit "$code"
}
trap finish EXIT
trap 'STATUS="error"; exit 1' ERR
trap 'STATUS="interrupted"; exit 130' INT TERM

echo "session $LABEL: estimate ${EST} h -> hard cap in ${CAP_MIN} min" | tee -a "$LOG"
$SHUTDOWN "+${CAP_MIN}"                     # 1. the hard cap, scheduled first
"$@" >> "$LOG" 2>&1 &                       # the job
JOB_PID=$!
"$PY" "$REPO/scripts/p5_cloud_guard.py" --state "$STATE" --label "$LABEL" \
  --rate "${P5_RATE:-1.217}" --volume-created "$P5_VOLUME_CREATED" \
  --ebs-usd-per-day "${P5_EBS_PER_DAY:-0.5333}" --cap-usd "${P5_CAP_USD:-75}" \
  --watch-pid "$JOB_PID" --idle-minutes "${P5_IDLE_MIN:-20}" --poll "${P5_POLL:-60}" \
  --grace 5 --stop-command "$SHUTDOWN now" >> "$STATE/logs/guard_${STAMP}.log" 2>&1 &
GUARD_PID=$!                                # 2. the watchdog + spend cap
echo "job pid $JOB_PID, guard pid $GUARD_PID; log $LOG" | tee -a "$LOG"
if wait "$JOB_PID"; then STATUS="ok"; else STATUS="failed"; JOB_PID=""; exit 1; fi
JOB_PID=""
