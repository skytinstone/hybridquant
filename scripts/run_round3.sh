#!/usr/bin/env bash
# Round 3 on one GPU: Falcon-H1 rerun (round-2 file was produced with the batched-decode bug) -> 4B recipe cells
# (KV4,S8), (KV8,S8), (KV4,S4) and their NIAH -> GSM8K S8+EF -> 4B layer sweep -> long-decode checks (1024 decode tokens) on 0.8B and 4B. Same step/report/commit pattern as rounds 1 and 2.
#   nohup bash scripts/run_round3.sh > results/round3_nohup.log 2>&1 &
set -u
set -o pipefail
cd "$(dirname "$0")/.."
mkdir -p results results/reports results/invalid
LOG=results/round3.log
DRIVE=/content/drive/MyDrive/hybridquant_results
PY=${PYTHON:-python}
DTYPE_ARGS=${HQ_DTYPE:+--dtype $HQ_DTYPE}

LOCK=results/run_round3.pid
if [ -f "$LOCK" ] && kill -0 "$(cat "$LOCK" 2>/dev/null)" 2>/dev/null; then
  echo "run_round3.sh is already running (pid $(cat "$LOCK")); not starting a second copy"; exit 0
fi
echo $$ > "$LOCK"; trap 'rm -f "$LOCK"' EXIT

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a "$LOG"; }

commit_results() {
  git add results >/dev/null 2>&1 || true
  if git diff --cached --quiet; then log "nothing to commit"; return 0; fi
  git commit -q -m "$1" || { log "commit failed"; return 0; }
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
  commit_results "results: round3 $name"
}

log "round 3 start on $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'no nvidia-smi')"

# The round-2 Falcon file is invalid (transformers batched single-token decode bug, see docs/decision_log.md 2026-10-14).
# Move it aside so the resumable runner does not skip its rows; the patched load_model makes batch 8 correct.
if [ -f results/r2_falcon_1.5b.jsonl ] && ! grep -q "falcon_h1_gated_norm" results/r2_falcon_1.5b.jsonl 2>/dev/null; then
  mv results/r2_falcon_1.5b.jsonl results/invalid/r2_falcon_1.5b.batchbug.jsonl
  rm -rf results/figures/r2_falcon_1.5b results/reports/r2_falcon_1.5b.md
  log "moved invalid round-2 Falcon results to results/invalid/"
fi
log "=== Falcon-H1 hook pre-check"
if $PY scripts/check_hooks.py --model tiiuae/Falcon-H1-1.5B-Instruct --kv-bits 4 --state-bits 4 --prefill 64 --decode 8 $DTYPE_ARGS >results/falcon_check.log 2>&1; then
  step r2_falcon_1.5b results/r2_falcon_1.5b.jsonl $PY scripts/run_grid.py --config configs/r2_falcon_1.5b.yaml $DTYPE_ARGS
else
  log "Falcon-H1 hook check failed (see results/falcon_check.log); skipped"
fi

step r3_recipe_4b      results/r3_recipe_4b.jsonl           $PY scripts/run_grid.py --config configs/r3_recipe_4b.yaml $DTYPE_ARGS
step r3_niah_recipe_4b results/r3_niah_recipe_4b.jsonl      $PY scripts/run_niah.py --config configs/r3_niah_recipe_4b.yaml $DTYPE_ARGS
step r3_gsm8k_s8ef_4b   results/r3_gsm8k_s8ef_4b.jsonl       $PY scripts/run_gsm8k.py --config configs/r3_gsm8k_s8ef_4b.yaml --batch-size 16 $DTYPE_ARGS
step layer_sweep_4b     results/layer_sweep_Qwen3.5-4B.jsonl $PY scripts/layer_sweep.py --model Qwen/Qwen3.5-4B --state-bits 4 --n-docs 8 $DTYPE_ARGS
step r3_longdecode_0.8b results/r3_longdecode_0.8b.jsonl     $PY scripts/run_grid.py --config configs/r3_longdecode_0.8b.yaml $DTYPE_ARGS
step r3_longdecode_4b   results/r3_longdecode_4b.jsonl       $PY scripts/run_grid.py --config configs/r3_longdecode_4b.yaml $DTYPE_ARGS

if [ -f analysis/report.py ]; then
  $PY analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md >>"$LOG" 2>&1 \
    && log "summary report: results/reports/summary.md" || log "summary report failed"
  commit_results "results: round3 summary"
fi
log "round 3 finished"
