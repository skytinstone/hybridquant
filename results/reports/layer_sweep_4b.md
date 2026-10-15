# Report: layer_sweep_4b
generated 2026-10-15T04:42:44+00:00 from layer_sweep_Qwen3.5-4B.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B: 레이어 하나만 state int4로 양자화했을 때 ΔNLL 최대는 레이어 0 (+0.3939), 최소는 레이어 6 (-0.0076), 전체 합 +0.4951.
- n_half(오차 반감 토큰 수)와 ΔNLL의 Spearman 상관 +0.13 (순열 p=0.535).

## [EN] Key Findings
- Qwen3.5-4B: quantizing one layer's state to int4, the largest dNLL is layer 0 (+0.3939) and the smallest layer 6 (-0.0076); sum over layers +0.4951.
- Spearman correlation between n_half (error half-life) and dNLL: +0.13 (permutation p=0.535).

## [KR/EN] 표
### Qwen3.5-4B
| layer_idx | delta_nll | mean_alpha | n_half | mean_beta |
|---|---|---|---|---|
| 0 | 0.3939 | 0.7787 | 1.3570 | 0.5554 |
| 1 | 0.0113 | 0.6481 | 0.8108 | 0.4399 |
| 2 | 0.0012 | 0.6869 | 0.9766 | 0.4314 |
| 4 | 0.0065 | 0.7326 | 1.3925 | 0.3491 |
| 5 | 0.0127 | 0.8060 | 2.0309 | 0.4376 |
| 6 | -0.0076 | 0.9044 | 3.1697 | 0.4544 |
| 8 | -0.0010 | 0.8787 | 1.6561 | 0.4269 |
| 9 | -0.0009 | 0.9435 | 8.0703 | 0.3517 |
| 10 | 0.0037 | 0.9342 | 6.3429 | 0.3458 |
| 12 | 0.0140 | 0.9335 | 6.0242 | 0.3319 |
| 13 | 0.0021 | 0.9280 | 7.5905 | 0.3650 |
| 14 | -0.0041 | 0.9176 | 3.2294 | 0.4058 |
| 16 | 0.0115 | 0.9493 | 11.0441 | 0.3508 |
| 17 | 0.0063 | 0.9207 | 6.5636 | 0.4357 |
| 18 | 0.0312 | 0.9363 | 8.6882 | 0.3818 |
| 20 | 0.0089 | 0.9329 | 8.0199 | 0.3573 |
| 21 | 0.0054 | 0.8780 | 2.7010 | 0.4694 |
| 22 | 0.0020 | 0.9082 | 5.7291 | 0.4476 |
| 24 | -0.0019 | 0.8879 | 4.2636 | 0.5374 |
| 25 | -0.0004 | 0.8847 | 4.3157 | 0.6774 |
| 26 | 0.0013 | 0.7851 | 2.1033 | 0.6284 |
| 28 | -0.0015 | 0.8071 | 1.8322 | 0.5085 |
| 29 | 0.0012 | 0.7913 | 1.8909 | 0.4765 |
| 30 | -0.0005 | 0.8423 | 2.1072 | 0.4204 |

## [KR/EN] 그림 목록
- `results/figures/layer_sweep_4b/layer_bars_Qwen3.5-4B.pdf` / `.png`: 레이어 하나씩만 state 양자화했을 때의 ΔNLL / dNLL when only one layer's state is quantized
- `results/figures/layer_sweep_4b/layer_halflife_Qwen3.5-4B.pdf` / `.png`: 오차 반감 토큰 수 n_half 대 ΔNLL 산점도, 회귀선, Spearman/Pearson(순열 p값) / Error half-life n_half versus dNLL with regression line; Spearman/Pearson with permutation p-values
- `results/figures/layer_sweep_4b/layer_alpha_beta_Qwen3.5-4B.pdf` / `.png`: 레이어별 평균 α와 평균 β 산점도, 색은 ΔNLL / Per-layer mean alpha versus mean beta, colour = dNLL

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: N/A
- commit: 3bf7c5a
- 날짜 (dates): 2026-10-15
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [8], 행 (rows): 25
- 입력 파일 (sources): layer_sweep_Qwen3.5-4B.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
