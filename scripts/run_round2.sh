#!/usr/bin/env bash
# Round 2: ablation -> C4 -> NIAH 0.8B -> 2B -> GSM8K 4B -> NIAH 4B -> 9B -> Falcon-H1 1.5B.
# Each step is resumable; after each step: report (analysis/report.py), git commit + push, optional Drive rsync.
# A failing step is logged and the sequence continues. Log: results/round2.log
set -u
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results results/reports
LOG=results/round2.log
DRIVE=/content/drive/MyDrive/hybridquant_results
PY=${PYTHON:-python}
DTYPE_ARGS=${HQ_DTYPE:+--dtype $HQ_DTYPE}

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

commit_results() {
  git add results >/dev/null 2>&1 || true
  if git diff --cached --quiet; then log "nothing to commit"; return 0; fi
  git commit -q -m "$1" || { log "commit failed"; return 0; }
  # git prints the remote URL (with any embedded token) on a rejected push; mask it before it reaches the log
  if ! git push origin HEAD 2>&1 | sed -E "s#(https://)[^@/]+@#\1***@#" >> results/push_failures.log; then
    echo "[$(date -u +%FT%TZ)] push failed after: $1" >> results/push_failures.log
    log "push failed (logged), continuing"
  else
    log "pushed: $1"
  fi
  if [ -d "$DRIVE" ]; then rsync -a results/ "$DRIVE/" && log "rsynced results/ to Drive"; fi
}

report() {
  local stage=$1; shift
  if [ -f analysis/report.py ]; then
    $PY analysis/report.py --stage "$stage" --inputs "$@" --out "results/reports/$stage.md" >>"$LOG" 2>&1 \
      && log "report: results/reports/$stage.md" || log "report failed for $stage"
  fi
}

step() {
  local name=$1 out=$2; shift 2
  log "=== step $name: $*"
  if "$@" >>"$LOG" 2>&1; then log "step $name OK"; else log "step $name FAILED (exit $?), continuing"; fi
  report "$name" "$out"
  commit_results "results: round2 $name"
}

log "round 2 start on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"
step r2_ablation_0.8b results/r2_ablation_0.8b.jsonl $PY scripts/run_grid.py --config configs/r2_ablation_0.8b.yaml $DTYPE_ARGS
step r2_c4_0.8b       results/r2_c4_0.8b.jsonl       $PY scripts/run_grid.py --config configs/r2_c4_0.8b.yaml $DTYPE_ARGS
step r2_niah_0.8b     results/r2_niah_0.8b.jsonl     $PY scripts/run_niah.py --config configs/r2_niah_0.8b.yaml $DTYPE_ARGS
step r2_scale_2b      results/r2_scale_2b.jsonl      $PY scripts/run_grid.py --config configs/r2_scale_2b.yaml $DTYPE_ARGS
step r2_gsm8k_4b      results/r2_gsm8k_4b.jsonl      $PY scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml $DTYPE_ARGS
step r2_niah_4b       results/r2_niah_4b.jsonl       $PY scripts/run_niah.py --config configs/r2_niah_4b.yaml $DTYPE_ARGS

# 9B: check VRAM first and let the estimate pick the batch size
BS9=$($PY scripts/vram_check.py --model Qwen/Qwen3.5-9B --batch-size 10 --prefill 256 --decode 256 2>>"$LOG" | tail -1 | $PY -c 'import json,sys; print(json.loads(sys.stdin.read())["batch_size"])' 2>/dev/null || echo 10)
log "9B batch size from vram_check: $BS9"
step r2_scale_9b      results/r2_scale_9b.jsonl      $PY scripts/run_grid.py --config configs/r2_scale_9b.yaml --batch-size "$BS9" $DTYPE_ARGS

# Falcon-H1: only if our hooks cover its cache layers
FALCON=tiiuae/Falcon-H1-1.5B-Instruct
log "=== Falcon-H1 hook pre-check"
if $PY scripts/check_hooks.py --model $FALCON --kv-bits 4 --state-bits 4 --prefill 64 --decode 8 $DTYPE_ARGS >results/falcon_check.log 2>&1; then
  log "Falcon-H1 hooks OK"
  step r2_falcon_1.5b results/r2_falcon_1.5b.jsonl $PY scripts/run_grid.py --config configs/r2_falcon_1.5b.yaml $DTYPE_ARGS
else
  CLASSES=$(grep -m1 "cache layer classes" results/falcon_check.log | sed 's/.*classes: //')
  echo "[$(date -u +%FT%TZ)] Falcon-H1 캐시 클래스 불일치: ${CLASSES:-unknown} (see results/falcon_check.log)" >> results/push_failures.log
  log "Falcon-H1 skipped: 캐시 클래스 불일치: ${CLASSES:-unknown}"
  commit_results "results: round2 falcon check"
fi
log "round 2 finished"
