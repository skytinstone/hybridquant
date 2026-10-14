# Report: superadd_0.8b
generated 2026-10-14T22:56:20+00:00 from superadd_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 2.974 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.049 nats (95% CI +1.001~+1.094).
- KV 4bit (state fp): fp 대비 +0.028 nats (95% CI +0.024~+0.032).
- KV4+S4 초가산: 실측 Δ +1.105 대 단독합 +1.076, 초과분 +0.029 nats (문서 대응 95% CI +0.009~+0.049).
- KV 노이즈 대조(같은 RMS의 가우시안 노이즈): Δ +1.061 대 실제 양자화 +1.105 nats.
- state 노이즈 대조(같은 RMS의 가우시안 노이즈): Δ +10.033 대 실제 양자화 +1.105 nats.

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 2.974 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.049 nats over fp (95% CI +1.001~+1.094).
- KV 4-bit (state fp): +0.028 nats over fp (95% CI +0.024~+0.032).
- KV4+S4 superadditivity: measured +1.105 vs sum of single effects +1.076; excess +0.029 nats (paired 95% CI +0.009~+0.049).
- KV noise control (Gaussian noise of equal RMS): +1.061 vs real quantization +1.105 nats.
- state noise control (Gaussian noise of equal RMS): +10.033 vs real quantization +1.105 nats.

## [KR/EN] 표
### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.9742 | ± 0.0494 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 20 |
| fp+S4 | 4.0227 | ± 0.0931 (seeds) | +1.0485 | +1.001~+1.094 | 8.65 | 3 | 20 |
| KV4+fp | 3.0017 | ± 0.0489 (seeds) | +0.0275 | +0.024~+0.032 | 11.01 | 3 | 20 |
| KV4+S4 | 4.0794 | ± 0.0717 (seeds) | +1.1053 | +1.059~+1.147 | 3.93 | 3 | 20 |
| KV4+S4[Sgn] | 13.0067 | ± 0.0735 (seeds) | +10.0325 | +9.918~+10.147 | 3.93 | 3 | 20 |
| KV4+S4[KVgn] | 4.0356 | ± 0.0837 (seeds) | +1.0614 | +1.015~+1.108 | 3.93 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/superadd_0.8b/superadd_bars_Qwen3.5-0.8B.pdf` / `.png`: 셀별 ΔNLL 막대(95% CI)와 fp+ΔKV+ΔS 예측값(점선), 차이가 초과분 / dNLL per cell (95% CI) with the additive prediction fp+dKV+dS (dotted); the gap is the excess
- `results/figures/superadd_0.8b/superadd_hist_Qwen3.5-0.8B.pdf` / `.png`: 문서별 초과분 히스토그램 / Histogram of the per-document excess
- `results/figures/superadd_0.8b/superadd_noise_Qwen3.5-0.8B.pdf` / `.png`: noise_match 조건과 실제 양자화 조건의 문서별 NLL 대응(대각선 = 동일) / Per-document NLL, noise-match versus real quantization (diagonal = equal)

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 9bb281c
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 18
- 입력 파일 (sources): superadd_0.8b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
