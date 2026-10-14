# Report: r2_ablation_0.8b
generated 2026-10-14T13:50:41+00:00 from r2_ablation_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 셀이 없어 Δ를 계산할 수 없습니다 (N/A (아직 실행 안 됨)).

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: no fp cell, deltas unavailable.

## [KR/EN] 표
N/A (아직 실행 안 됨)

## [KR/EN] 그림 목록
- `results/figures/r2_ablation_0.8b/ablation_Qwen3.5-0.8B.pdf` / `.png`: state 양자화 포맷 ablation: group 크기 × (symmetric, stochastic), 셀별 패널 / State-format ablation: group size x (symmetric, stochastic), one panel per cell

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 1f517a6
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 96
- 입력 파일 (sources): r2_ablation_0.8b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
