# Report: r2_gsm8k_4b
generated 2026-10-14T22:54:42+00:00 from r2_gsm8k_4b.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B GSM8K fp 정답률 0.600 (문제 200개, 95% CI 0.535~0.665), 평균 생성 토큰 2912, 예산 소진율 0.48.
- fp+S4: 정답률 0.045 (fp 대비 -0.555), 토큰 4084, 반복률 0.308.
- fp+S4[EF]: 정답률 0.345 (fp 대비 -0.255), 토큰 3795, 반복률 0.375.
- fp+S6: 정답률 0.105 (fp 대비 -0.495), 토큰 4083, 반복률 0.205.
- fp+S6[EF]: 정답률 0.560 (fp 대비 -0.040), 토큰 3231, 반복률 0.092.
- fp+S8: 정답률 0.345 (fp 대비 -0.255), 토큰 3677, 반복률 0.130.
- KV2+S8: 정답률 0.030 (fp 대비 -0.570), 토큰 4096, 반복률 0.372.
- KV4+fp: 정답률 0.595 (fp 대비 -0.005), 토큰 3121, 반복률 0.087.
- KV4+S6: 정답률 0.100 (fp 대비 -0.500), 토큰 4096, 반복률 0.208.
- KV4+S6[EF]: 정답률 0.555 (fp 대비 -0.045), 토큰 3133, 반복률 0.093.

## [EN] Key Findings
- Qwen3.5-4B GSM8K fp accuracy 0.600 (200 problems, 95% CI 0.535~0.665), mean 2912 tokens, budget-exhaustion 0.48.
- fp+S4: accuracy 0.045 (-0.555 vs fp), 4084 tokens, repeat ratio 0.308.
- fp+S4[EF]: accuracy 0.345 (-0.255 vs fp), 3795 tokens, repeat ratio 0.375.
- fp+S6: accuracy 0.105 (-0.495 vs fp), 4083 tokens, repeat ratio 0.205.
- fp+S6[EF]: accuracy 0.560 (-0.040 vs fp), 3231 tokens, repeat ratio 0.092.
- fp+S8: accuracy 0.345 (-0.255 vs fp), 3677 tokens, repeat ratio 0.130.
- KV2+S8: accuracy 0.030 (-0.570 vs fp), 4096 tokens, repeat ratio 0.372.
- KV4+fp: accuracy 0.595 (-0.005 vs fp), 3121 tokens, repeat ratio 0.087.
- KV4+S6: accuracy 0.100 (-0.500 vs fp), 4096 tokens, repeat ratio 0.208.
- KV4+S6[EF]: accuracy 0.555 (-0.045 vs fp), 3133 tokens, repeat ratio 0.093.

## [KR/EN] 표
### Qwen3.5-4B
| model | cell | error_feedback | kv_bits | state_bits | acc | acc_std_seeds | acc_ci_lo | acc_ci_hi | n | n_seeds | mean_tokens | budget_rate | repeat_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 0.6000 | N/A | 0.5349 | 0.6650 | 200 | 1 | 2912.4450 | 0.4850 | 0.0856 |
| Qwen/Qwen3.5-4B | fp+S4 | False | 16 | 4 | 0.0450 | N/A | 0.0200 | 0.0751 | 200 | 1 | 4083.7300 | 0.9650 | 0.3075 |
| Qwen/Qwen3.5-4B | fp+S4[EF] | True | 16 | 4 | 0.3450 | N/A | 0.2850 | 0.4200 | 200 | 1 | 3795.3450 | 0.8050 | 0.3751 |
| Qwen/Qwen3.5-4B | fp+S6 | False | 16 | 6 | 0.1050 | N/A | 0.0650 | 0.1500 | 200 | 1 | 4082.9300 | 0.9950 | 0.2049 |
| Qwen/Qwen3.5-4B | fp+S6[EF] | True | 16 | 6 | 0.5600 | N/A | 0.4850 | 0.6300 | 200 | 1 | 3230.8550 | 0.5700 | 0.0921 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 0.3450 | N/A | 0.2800 | 0.4100 | 200 | 1 | 3676.6400 | 0.8200 | 0.1295 |
| Qwen/Qwen3.5-4B | KV2+S8 | False | 2 | 8 | 0.0300 | N/A | 0.0100 | 0.0550 | 200 | 1 | 4096.0000 | 1.0000 | 0.3715 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 0.5950 | N/A | 0.5250 | 0.6600 | 200 | 1 | 3120.8400 | 0.5550 | 0.0867 |
| Qwen/Qwen3.5-4B | KV4+S6 | False | 4 | 6 | 0.1000 | N/A | 0.0600 | 0.1400 | 200 | 1 | 4096.0000 | 1.0000 | 0.2082 |
| Qwen/Qwen3.5-4B | KV4+S6[EF] | True | 4 | 6 | 0.5550 | N/A | 0.4900 | 0.6250 | 200 | 1 | 3133.1350 | 0.5300 | 0.0930 |

## [KR/EN] 그림 목록
- `results/figures/r2_gsm8k_4b/gsm8k_acc_Qwen3.5-4B.pdf` / `.png`: 셀별 GSM8K 정답률(문제 부트스트랩 95% CI), EF 색 구분 / GSM8K accuracy per cell (bootstrap 95% CI over problems), colour = EF
- `results/figures/r2_gsm8k_4b/gsm8k_tokens_Qwen3.5-4B.pdf` / `.png`: 셀별 생성 토큰 수 분포 / Generated-token distribution per cell
- `results/figures/r2_gsm8k_4b/gsm8k_behaviour_Qwen3.5-4B.pdf` / `.png`: 예산 소진율과 8-gram 반복률 / Budget-exhaustion rate and 8-gram repeat ratio
- `results/figures/r2_gsm8k_4b/gsm8k_acc_vs_tokens_Qwen3.5-4B.pdf` / `.png`: 정답률 대 평균 생성 토큰 수 / Accuracy versus mean generated tokens

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: N/A
- commit: 96b56bd, e33cb9f
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [], 행 (rows): 2000
- 입력 파일 (sources): r2_gsm8k_4b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
