# Report: layer_sweep_0.8b
generated 2026-10-14T22:56:21+00:00 from layer_sweep_Qwen3.5-0.8B.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B: 레이어 하나만 state int4로 양자화했을 때 ΔNLL 최대는 레이어 0 (+0.2175), 최소는 레이어 9 (-0.0085), 전체 합 +0.5747.
- n_half(오차 반감 토큰 수)와 ΔNLL의 Spearman 상관 -0.08 (순열 p=0.783).

## [EN] Key Findings
- Qwen3.5-0.8B: quantizing one layer's state to int4, the largest dNLL is layer 0 (+0.2175) and the smallest layer 9 (-0.0085); sum over layers +0.5747.
- Spearman correlation between n_half (error half-life) and dNLL: -0.08 (permutation p=0.783).

## [KR/EN] 표
### Qwen3.5-0.8B
| layer_idx | delta_nll | mean_alpha | n_half | mean_beta |
|---|---|---|---|---|
| 0 | 0.2175 | 0.7714 | 0.9636 | 0.4520 |
| 1 | 0.0100 | 0.6046 | 0.7367 | 0.4189 |
| 2 | 0.0313 | 0.7448 | 1.3552 | 0.3785 |
| 4 | 0.0307 | 0.8277 | 2.1971 | 0.3539 |
| 5 | 0.0162 | 0.9092 | 5.0683 | 0.3810 |
| 6 | 0.0055 | 0.8674 | 3.5119 | 0.4618 |
| 8 | 0.0311 | 0.9282 | 7.7287 | 0.4343 |
| 9 | -0.0085 | 0.9439 | 9.8564 | 0.2476 |
| 10 | 0.0567 | 0.9174 | 6.1371 | 0.4727 |
| 12 | 0.0374 | 0.9623 | 14.4020 | 0.2358 |
| 13 | 0.0256 | 0.9084 | 5.7655 | 0.3712 |
| 14 | 0.0456 | 0.9264 | 7.6561 | 0.4154 |
| 16 | 0.0386 | 0.8589 | 3.2087 | 0.3928 |
| 17 | 0.0078 | 0.9446 | 9.9908 | 0.4169 |
| 18 | -0.0040 | 0.9054 | 5.1884 | 0.5782 |
| 20 | 0.0081 | 0.8525 | 2.3100 | 0.5947 |
| 21 | 0.0109 | 0.8886 | 4.6544 | 0.5541 |
| 22 | 0.0143 | 0.7654 | 1.2789 | 0.5595 |

## [KR/EN] 그림 목록
- `results/figures/layer_sweep_0.8b/layer_bars_Qwen3.5-0.8B.pdf` / `.png`: 레이어 하나씩만 state 양자화했을 때의 ΔNLL / dNLL when only one layer's state is quantized
- `results/figures/layer_sweep_0.8b/layer_halflife_Qwen3.5-0.8B.pdf` / `.png`: 오차 반감 토큰 수 n_half 대 ΔNLL 산점도, 회귀선, Spearman/Pearson(순열 p값) / Error half-life n_half versus dNLL with regression line; Spearman/Pearson with permutation p-values
- `results/figures/layer_sweep_0.8b/layer_alpha_beta_Qwen3.5-0.8B.pdf` / `.png`: 레이어별 평균 α와 평균 β 산점도, 색은 ΔNLL / Per-layer mean alpha versus mean beta, colour = dNLL

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: N/A
- commit: 569541d
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [8], 행 (rows): 19
- 입력 파일 (sources): layer_sweep_Qwen3.5-0.8B.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
