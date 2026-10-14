#!/usr/bin/env bash
# Round 1 on one GPU: smoke -> 0.8B grid -> superadditivity -> layer sweep -> 4B core.
# Every step is resumable (existing jsonl rows are skipped). After each step: report
# (if analysis/report.py exists), git commit + push of results/, optional Drive rsync.
# Failures do not stop the sequence; see results/round1.log and results/push_failures.log.
set -u
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results results/reports
LOG=results/round1.log
DRIVE=/content/drive/MyDrive/hybridquant_results
PY=${PYTHON:-python}
DTYPE_ARGS=${HQ_DTYPE:+--dtype $HQ_DTYPE}

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

commit_results() {   # $1 = message
  git add results >/dev/null 2>&1 || true
  if git diff --cached --quiet; then log "nothing to commit"; return 0; fi
  git commit -q -m "$1" || { log "commit failed"; return 0; }
  # git prints the remote URL (with any embedded token) on a rejected push; mask it before it reaches the log
  if ! git push origin HEAD 2>&1 | sed -E "s#(https://)[^@/]+@#\1***@#" >> results/push_failures.log; then
    echo "[$(date -u +%FT%TZ)] push failed after: $1" >> results/push_failures.log
    log "push failed (logged to results/push_failures.log), continuing"
  else
    log "pushed: $1"
  fi
  if [ -d "$DRIVE" ]; then rsync -a results/ "$DRIVE/" && log "rsynced results/ to Drive"; fi
}

report() {   # $1 = stage, $2.. = jsonl inputs
  local stage=$1; shift
  if [ -f analysis/report.py ]; then
    $PY analysis/report.py --stage "$stage" --inputs "$@" --out "results/reports/$stage.md" >>"$LOG" 2>&1 \
      && log "report: results/reports/$stage.md" || log "report failed for $stage"
  fi
}

step() {   # $1 = name, $2 = jsonl output, $3.. = command
  local name=$1 out=$2; shift 2
  log "=== step $name: $*"
  if "$@" >>"$LOG" 2>&1; then log "step $name OK"; else log "step $name FAILED (exit $?), continuing"; fi
  report "$name" "$out"
  commit_results "results: round1 $name"
}

log "round 1 start on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
step smoke_0.8b     results/smoke_0.8b.jsonl       $PY scripts/run_grid.py --config configs/smoke_0.8b.yaml $DTYPE_ARGS
step grid_0.8b      results/grid_0.8b.jsonl        $PY scripts/run_grid.py --config configs/grid_0.8b.yaml $DTYPE_ARGS
step superadd_0.8b  results/superadd_0.8b.jsonl    $PY scripts/run_grid.py --config configs/superadd_0.8b.yaml $DTYPE_ARGS
step layer_sweep_0.8b results/layer_sweep_Qwen3.5-0.8B.jsonl $PY scripts/layer_sweep.py --model Qwen/Qwen3.5-0.8B --state-bits 4 --n-docs 8 $DTYPE_ARGS
step grid_4b_core   results/grid_4b_core.jsonl     $PY scripts/run_grid.py --config configs/grid_4b_core.yaml $DTYPE_ARGS
log "round 1 finished"
