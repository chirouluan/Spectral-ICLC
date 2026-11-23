#!/usr/bin/env bash
# One-shot progress report for the detached public full run.
set -u
cd "$(dirname "$0")"
OUT="outputs/four_batches_strata_variants"

echo "== state =="
if [ -f "$OUT/public_full.finished" ]; then
  echo "FINISHED (all phases attempted)"
fi
if [ -f "$OUT/public_full.pid" ]; then
  pid="$(cat "$OUT/public_full.pid")"
  if kill -0 "$pid" 2>/dev/null; then
    echo "running: PID $pid"
  else
    echo "not running (stale PID $pid)"
  fi
else
  echo "no PID file (not started)"
fi

echo "== completed experiments =="
for variant in p4_public p4_weak_thinking; do
  file="$OUT/$variant/predictions.jsonl"
  if [ -f "$file" ]; then
    printf "%-20s %s\n" "$variant" "$(grep -c '"status": "completed"' "$file" 2>/dev/null || true)"
  else
    printf "%-20s %s\n" "$variant" "0"
  fi
done

echo "== log tail =="
tail -n 25 "$OUT/public_full.log" 2>/dev/null || echo "(no log yet)"
