#!/usr/bin/env bash
# Everything on one GPU: round 1 then round 2, then the summary report. Resumable: rerun to continue.
#   nohup bash scripts/run_all.sh > results/run_all_nohup.log 2>&1 &
set -u
cd "$(dirname "$0")/.."
mkdir -p results results/reports
PY=${PYTHON:-python}
LOG=results/run_all.log
log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

if command -v nvidia-smi >/dev/null 2>&1; then
  log "GPU: $(nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv,noheader)"
else
  log "GPU: none (nvidia-smi not found)"
fi
log "estimated time (planning aid):"
$PY scripts/estimate_time.py 2>>"$LOG" | tee -a "$LOG"

bash scripts/run_round1.sh
bash scripts/run_round2.sh

if [ -f analysis/report.py ]; then
  $PY analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md >>"$LOG" 2>&1 \
    && log "summary report: results/reports/summary.md" || log "summary report failed"
  git add results >/dev/null 2>&1; git commit -q -m "results: summary report" 2>/dev/null
  git push origin HEAD 2>>results/push_failures.log || log "final push failed (logged)"
fi
log "run_all finished"
