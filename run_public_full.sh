#!/usr/bin/env bash
# Full public run with original (full-length) sample ids. Deleted datasets
# (honey_adulteration, fruit_purees) are skipped automatically at selection
# time, so P4 selects the six remaining public datasets (120 experiments).
# Designed to be launched detached:
#   setsid nohup bash run_public_full.sh </dev/null >/dev/null 2>&1 &
# so it survives closing the editor / Claude Code session.
set -u
cd "$(dirname "$0")"

OUT="outputs/four_batches_strata_variants"
LOG="$OUT/public_full.log"
PIDFILE="$OUT/public_full.pid"
mkdir -p "$OUT"

if [ -f "$PIDFILE" ] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; then
  echo "Already running: PID $(cat "$PIDFILE")" >&2
  exit 1
fi
echo $$ >"$PIDFILE"
rm -f "$OUT/public_full.finished"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }
log() { echo "$*" >>"$LOG"; }

run_phase() {
  local name="$1"
  shift
  log "===== START $name $(stamp) ====="
  timeout 14400 python3 run_variants.py "$@" >>"$LOG" 2>&1
  local rc=$?
  log "===== DONE $name rc=$rc $(stamp) ====="
  return 0
}

log "##### PUBLIC FULL RUN BEGIN $(stamp) #####"

# 1) All feasible public datasets (6 datasets x 20 experiments = 120).
run_phase P4_public \
  --phase P4 --variant p4_public --no-thinking \
  --max-tokens 16384 --concurrency 2 --resume

# 2) Thinking-on arm for the weak datasets (full_reference test acc < 70%).
#    Same prompts/transforms as p4_public; only enable_thinking differs.
run_phase P4_weak_thinking \
  --phase P4 --datasets turmeric_levels,beef --variant p4_weak_thinking --thinking \
  --max-tokens 16384 --concurrency 2 --resume

log "##### PUBLIC FULL RUN END $(stamp) #####"
touch "$OUT/public_full.finished"
