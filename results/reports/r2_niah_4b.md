# Report: r2_niah_4b
generated 2026-10-14T13:51:08+00:00 from r2_niah_4b.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B NIAH 컨텍스트 4096: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.94.
- Qwen3.5-4B NIAH 컨텍스트 16384: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.92.

## [EN] Key Findings
- Qwen3.5-4B NIAH context 4096: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.94.
- Qwen3.5-4B NIAH context 16384: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.92.

## [KR/EN] 표
### Qwen3.5-4B
| model | cell | error_feedback | kv_bits | state_bits | context_len | acc | ci_lo | ci_hi | n |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4 | False | 16 | 4 | 4096 | 0.9600 | 0.9000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4 | False | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4[EF] | True | 16 | 4 | 4096 | 0.9600 | 0.9000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4[EF] | True | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8[EF] | True | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8[EF] | True | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+fp | False | 2 | 16 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+fp | False | 2 | 16 | 16384 | 0.9200 | 0.8400 | 0.9800 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8 | False | 2 | 8 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8 | False | 2 | 8 | 16384 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8[EF] | True | 2 | 8 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8[EF] | True | 2 | 8 | 16384 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4 | False | 4 | 4 | 4096 | 0.9800 | 0.9400 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4 | False | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4[EF] | True | 4 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4[EF] | True | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |

## [KR/EN] 그림 목록
- `results/figures/r2_niah_4b/niah_heatmap_Qwen3.5-4B.pdf` / `.png`: 셀 × 컨텍스트 길이 NIAH 정확도 / NIAH accuracy, cell x context length
- `results/figures/r2_niah_4b/niah_depth_Qwen3.5-4B.pdf` / `.png`: 깊이 × 셀 정확도(길이별 패널) / Accuracy by needle depth x cell, one panel per context length
- `results/figures/r2_niah_4b/niah_kv_vs_state_Qwen3.5-4B.pdf` / `.png`: KV-only 대 state-only: 회수 정확도(왼쪽)와 같은 셀의 ΔNLL(오른쪽), 태스크별 민감도 비교 / KV-only versus state-only: retrieval accuracy (left) and dNLL for the same cells (right)

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: N/A
- commit: 935f3ad
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [], 행 (rows): 1100
- 입력 파일 (sources): r2_niah_4b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
