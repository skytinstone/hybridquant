# Report: grid_0.8b
generated 2026-10-14T13:50:10+00:00 from grid_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 2.974 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.049 nats (95% CI +1.001~+1.094).
- KV 4bit (state fp): fp 대비 +0.028 nats (95% CI +0.024~+0.032).
- state 8bit: fp 대비 +0.015 nats (95% CI +0.010~+0.020).
- KV 2bit: fp 대비 +0.439 nats (95% CI +0.416~+0.464).
- KV4+S4 초가산: 실측 Δ +1.105 대 단독합 +1.076, 초과분 +0.029 nats (문서 대응 95% CI +0.009~+0.049).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.222 nats (20개 셀, 범위 -0.634~+0.001).

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 2.974 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.049 nats over fp (95% CI +1.001~+1.094).
- KV 4-bit (state fp): +0.028 nats over fp (95% CI +0.024~+0.032).
- state 8-bit: +0.015 nats over fp (95% CI +0.010~+0.020).
- KV 2-bit: +0.439 nats over fp (95% CI +0.416~+0.464).
- KV4+S4 superadditivity: measured +1.105 vs sum of single effects +1.076; excess +0.029 nats (paired 95% CI +0.009~+0.049).
- error feedback: mean change in dNLL versus RTN at the same bits -0.222 nats (20 cells, range -0.634~+0.001).

## [KR/EN] 표
### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.9742 | ± 0.0494 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 20 |
| fp+S8 | 2.9891 | ± 0.0505 (seeds) | +0.0149 | +0.010~+0.020 | 11.01 | 3 | 20 |
| fp+S8[EF] | 2.9753 | ± 0.0492 (seeds) | +0.0012 | -0.000~+0.003 * | 11.01 | 3 | 20 |
| fp+S6 | 3.1092 | ± 0.0522 (seeds) | +0.1351 | +0.122~+0.148 | 9.83 | 3 | 20 |
| fp+S6[EF] | 3.0245 | ± 0.0522 (seeds) | +0.0504 | +0.044~+0.057 | 9.83 | 3 | 20 |
| fp+S5 | 3.3457 | ± 0.0611 (seeds) | +0.3715 | +0.347~+0.395 | 9.24 | 3 | 20 |
| fp+S5[EF] | 3.1203 | ± 0.0548 (seeds) | +0.1462 | +0.133~+0.158 | 9.24 | 3 | 20 |
| fp+S4 | 4.0227 | ± 0.0931 (seeds) | +1.0485 | +1.001~+1.094 | 8.65 | 3 | 20 |
| fp+S4[EF] | 3.4985 | ± 0.0841 (seeds) | +0.5243 | +0.495~+0.553 | 8.65 | 3 | 20 |
| KV8+fp | 2.9739 | ± 0.0493 (seeds) | -0.0003 | -0.001~+0.000 * | 12.58 | 3 | 20 |
| KV8+S8 | 2.9908 | ± 0.0498 (seeds) | +0.0167 | +0.012~+0.021 | 7.86 | 3 | 20 |
| KV8+S8[EF] | 2.9757 | ± 0.0478 (seeds) | +0.0015 | -0.000~+0.003 * | 7.86 | 3 | 20 |
| KV8+S6 | 3.1115 | ± 0.0508 (seeds) | +0.1374 | +0.124~+0.150 | 6.68 | 3 | 20 |
| KV8+S6[EF] | 3.0214 | ± 0.0525 (seeds) | +0.0472 | +0.041~+0.054 | 6.68 | 3 | 20 |
| KV8+S5 | 3.3370 | ± 0.0565 (seeds) | +0.3628 | +0.340~+0.387 | 6.09 | 3 | 20 |
| KV8+S5[EF] | 3.1261 | ± 0.0478 (seeds) | +0.1519 | +0.139~+0.165 | 6.09 | 3 | 20 |
| KV8+S4 | 4.0196 | ± 0.0816 (seeds) | +1.0454 | +0.999~+1.092 | 5.51 | 3 | 20 |
| KV8+S4[EF] | 3.4883 | ± 0.0735 (seeds) | +0.5141 | +0.487~+0.541 | 5.51 | 3 | 20 |
| KV4+fp | 3.0017 | ± 0.0489 (seeds) | +0.0275 | +0.024~+0.032 | 11.01 | 3 | 20 |
| KV4+S8 | 3.0172 | ± 0.0493 (seeds) | +0.0430 | +0.035~+0.050 | 6.29 | 3 | 20 |
| KV4+S8[EF] | 3.0036 | ± 0.0485 (seeds) | +0.0294 | +0.025~+0.034 | 6.29 | 3 | 20 |
| KV4+S6 | 3.1391 | ± 0.0461 (seeds) | +0.1649 | +0.151~+0.179 | 5.11 | 3 | 20 |
| KV4+S6[EF] | 3.0505 | ± 0.0548 (seeds) | +0.0764 | +0.069~+0.084 | 5.11 | 3 | 20 |
| KV4+S5 | 3.3777 | ± 0.0523 (seeds) | +0.4035 | +0.378~+0.430 | 4.52 | 3 | 20 |
| KV4+S5[EF] | 3.1614 | ± 0.0595 (seeds) | +0.1873 | +0.172~+0.201 | 4.52 | 3 | 20 |
| KV4+S4 | 4.0794 | ± 0.0717 (seeds) | +1.1053 | +1.059~+1.147 | 3.93 | 3 | 20 |
| KV4+S4[EF] | 3.5397 | ± 0.0899 (seeds) | +0.5655 | +0.537~+0.595 | 3.93 | 3 | 20 |
| KV3+fp | 3.1094 | ± 0.0521 (seeds) | +0.1353 | +0.125~+0.145 | 10.62 | 3 | 20 |
| KV3+S8 | 3.1247 | ± 0.0530 (seeds) | +0.1505 | +0.139~+0.163 | 5.90 | 3 | 20 |
| KV3+S8[EF] | 3.1134 | ± 0.0508 (seeds) | +0.1392 | +0.129~+0.150 | 5.90 | 3 | 20 |
| KV3+S6 | 3.2636 | ± 0.0542 (seeds) | +0.2894 | +0.272~+0.308 | 4.72 | 3 | 20 |
| KV3+S6[EF] | 3.1703 | ± 0.0549 (seeds) | +0.1961 | +0.181~+0.212 | 4.72 | 3 | 20 |
| KV3+S5 | 3.5056 | ± 0.0656 (seeds) | +0.5314 | +0.502~+0.560 | 4.13 | 3 | 20 |
| KV3+S5[EF] | 3.2938 | ± 0.0488 (seeds) | +0.3197 | +0.299~+0.340 | 4.13 | 3 | 20 |
| KV3+S4 | 4.2840 | ± 0.0953 (seeds) | +1.3098 | +1.258~+1.359 | 3.54 | 3 | 20 |
| KV3+S4[EF] | 3.7066 | ± 0.0953 (seeds) | +0.7324 | +0.700~+0.768 | 3.54 | 3 | 20 |
| KV2+fp | 3.4136 | ± 0.0613 (seeds) | +0.4394 | +0.416~+0.464 | 10.22 | 3 | 20 |
| KV2+S8 | 3.4222 | ± 0.0589 (seeds) | +0.4481 | +0.425~+0.475 | 5.51 | 3 | 20 |
| KV2+S8[EF] | 3.4236 | ± 0.0562 (seeds) | +0.4495 | +0.424~+0.477 | 5.51 | 3 | 20 |
| KV2+S6 | 3.6044 | ± 0.0560 (seeds) | +0.6302 | +0.601~+0.660 | 4.33 | 3 | 20 |
| KV2+S6[EF] | 3.4954 | ± 0.0647 (seeds) | +0.5213 | +0.495~+0.549 | 4.33 | 3 | 20 |
| KV2+S5 | 3.8869 | ± 0.0592 (seeds) | +0.9127 | +0.878~+0.950 | 3.74 | 3 | 20 |
| KV2+S5[EF] | 3.6323 | ± 0.0461 (seeds) | +0.6582 | +0.627~+0.690 | 3.74 | 3 | 20 |
| KV2+S4 | 4.7861 | ± 0.0933 (seeds) | +1.8120 | +1.759~+1.864 | 3.15 | 3 | 20 |
| KV2+S4[EF] | 4.1517 | ± 0.0753 (seeds) | +1.1775 | +1.142~+1.215 | 3.15 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/grid_0.8b/heatmap_Qwen3.5-0.8B.pdf` / `.png`: Qwen3.5-0.8B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-0.8B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/grid_0.8b/lines_Qwen3.5-0.8B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/grid_0.8b/pareto_Qwen3.5-0.8B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/grid_0.8b/curves_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/grid_0.8b/violin_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 527bfa3, b944556
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 135
- 입력 파일 (sources): grid_0.8b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
