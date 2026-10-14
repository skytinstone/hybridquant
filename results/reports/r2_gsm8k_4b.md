# Report: r2_gsm8k_4b
generated 2026-10-14T11:57:47+00:00 from r2_gsm8k_4b.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B GSM8K fp 정답률 0.600 (문제 200개, 95% CI 0.535~0.665), 평균 생성 토큰 2912, 예산 소진율 0.48.
- fp+S8: 정답률 0.312 (fp 대비 -0.287), 토큰 3468, 반복률 0.122.
- KV4+fp: 정답률 0.595 (fp 대비 -0.005), 토큰 3121, 반복률 0.087.

## [EN] Key Findings
- Qwen3.5-4B GSM8K fp accuracy 0.600 (200 problems, 95% CI 0.535~0.665), mean 2912 tokens, budget-exhaustion 0.48.
- fp+S8: accuracy 0.312 (-0.287 vs fp), 3468 tokens, repeat ratio 0.122.
- KV4+fp: accuracy 0.595 (-0.005 vs fp), 3121 tokens, repeat ratio 0.087.

## [KR/EN] 표
### Qwen3.5-4B
| model | cell | error_feedback | kv_bits | state_bits | acc | acc_std_seeds | acc_ci_lo | acc_ci_hi | n | n_seeds | mean_tokens | budget_rate | repeat_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 0.6000 | N/A | 0.5349 | 0.6650 | 200 | 1 | 2912.4450 | 0.4850 | 0.0856 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 0.3125 | N/A | 0.1562 | 0.4688 | 32 | 1 | 3467.5312 | 0.7500 | 0.1222 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 0.5950 | N/A | 0.5250 | 0.6600 | 200 | 1 | 3120.8400 | 0.5550 | 0.0867 |

## [KR/EN] 그림 목록
- `results/figures/r2_gsm8k_4b/gsm8k_acc_Qwen3.5-4B.pdf` / `.png`: 셀별 GSM8K 정답률(문제 부트스트랩 95% CI), EF 색 구분 / GSM8K accuracy per cell (bootstrap 95% CI over problems), colour = EF
- `results/figures/r2_gsm8k_4b/gsm8k_tokens_Qwen3.5-4B.pdf` / `.png`: 셀별 생성 토큰 수 분포 / Generated-token distribution per cell
- `results/figures/r2_gsm8k_4b/gsm8k_behaviour_Qwen3.5-4B.pdf` / `.png`: 예산 소진율과 8-gram 반복률 / Budget-exhaustion rate and 8-gram repeat ratio
- `results/figures/r2_gsm8k_4b/gsm8k_acc_vs_tokens_Qwen3.5-4B.pdf` / `.png`: 정답률 대 평균 생성 토큰 수 / Accuracy versus mean generated tokens

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: N/A
- commit: e33cb9f
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [], 행 (rows): 432
- 입력 파일 (sources): r2_gsm8k_4b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
