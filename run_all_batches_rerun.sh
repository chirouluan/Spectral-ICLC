#!/usr/bin/env bash
# Full four-batch rerun with the original (baseline) prompt protocol:
# every available experiment, prompt.txt + train/val/test JSON read verbatim,
# no variant hints, no transforms, full-length sample ids.
# Selection uses index_current.csv (300 experiments in the current package).
#
# Launched detached:
#   setsid nohup bash run_all_batches_rerun.sh </dev/null >/dev/null 2>&1 &
set -u
cd "$(dirname "$0")"

OUT="outputs/four_batches_strata_rerun"
LOG="$OUT/run.log"
PIDFILE="$OUT/run.pid"
mkdir -p "$OUT"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running: PID $(cat "$PIDFILE")" >&2
  exit 1
fi
echo $$ >"$PIDFILE"
rm -f "$OUT/run.finished"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$*" >>"$LOG"; }

log "===== ALL-BATCH RERUN BEGIN $(stamp) ====="
timeout 21600 python3 run_variants.py \
  --phase ALL \
  --variant four_batches_strata_rerun \
  --output-root outputs \
  --no-thinking \
  --format plain \
  --max-tokens 16384 \
  --concurrency 2 >>"$LOG" 2>&1
rc=$?
log "===== ALL-BATCH RERUN END rc=$rc $(stamp) ====="
if [ "$rc" -eq 0 ]; then
  touch "$OUT/run.finished"
fi
