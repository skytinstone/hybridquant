# Report: r2_scale_2b
generated 2026-10-14T22:56:36+00:00 from r2_scale_2b.jsonl

## [KR] 핵심 요약
- Qwen3.5-2B / wikitext2: fp 베이스라인 mean NLL 2.605 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.808 nats (95% CI +1.725~+1.896).
- KV 4bit (state fp): fp 대비 +0.011 nats (95% CI +0.009~+0.014).
- state 8bit: fp 대비 +0.010 nats (95% CI +0.006~+0.013).
- KV4+S4 초가산: 실측 Δ +1.867 대 단독합 +1.820, 초과분 +0.047 nats (문서 대응 95% CI +0.021~+0.073).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.440 nats (6개 셀, 범위 -1.255~-0.008).

## [EN] Key Findings
- Qwen3.5-2B / wikitext2: fp baseline mean NLL 2.605 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.808 nats over fp (95% CI +1.725~+1.896).
- KV 4-bit (state fp): +0.011 nats over fp (95% CI +0.009~+0.014).
- state 8-bit: +0.010 nats over fp (95% CI +0.006~+0.013).
- KV4+S4 superadditivity: measured +1.867 vs sum of single effects +1.820; excess +0.047 nats (paired 95% CI +0.021~+0.073).
- error feedback: mean change in dNLL versus RTN at the same bits -0.440 nats (6 cells, range -1.255~-0.008).

## [KR/EN] 표
### Qwen3.5-2B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.6054 | ± 0.0483 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 20 |
| fp+S8 | 2.6151 | ± 0.0480 (seeds) | +0.0097 | +0.006~+0.013 | 11.01 | 3 | 20 |
| fp+S8[EF] | 2.6073 | ± 0.0473 (seeds) | +0.0018 | +0.000~+0.003 | 11.01 | 3 | 20 |
| fp+S6 | 2.6953 | ± 0.0431 (seeds) | +0.0899 | +0.080~+0.100 | 9.83 | 3 | 20 |
| fp+S6[EF] | 2.6297 | ± 0.0500 (seeds) | +0.0243 | +0.019~+0.029 | 9.83 | 3 | 20 |
| fp+S4 | 4.4139 | ± 0.0341 (seeds) | +1.8085 | +1.725~+1.896 | 8.65 | 3 | 20 |
| fp+S4[EF] | 3.1827 | ± 0.0608 (seeds) | +0.5773 | +0.539~+0.612 | 8.65 | 3 | 20 |
| KV4+fp | 2.6169 | ± 0.0461 (seeds) | +0.0114 | +0.009~+0.014 | 11.01 | 3 | 20 |
| KV4+S6 | 2.7115 | ± 0.0419 (seeds) | +0.1061 | +0.096~+0.117 | 5.11 | 3 | 20 |
| KV4+S6[EF] | 2.6460 | ± 0.0500 (seeds) | +0.0406 | +0.035~+0.047 | 5.11 | 3 | 20 |
| KV4+S4 | 4.4722 | ± 0.0723 (seeds) | +1.8667 | +1.787~+1.951 | 3.93 | 3 | 20 |
| KV4+S4[EF] | 3.2174 | ± 0.0665 (seeds) | +0.6120 | +0.570~+0.650 | 3.93 | 3 | 20 |
| KV2+S8 | 3.0652 | ± 0.0702 (seeds) | +0.4597 | +0.432~+0.484 | 5.51 | 3 | 20 |
| KV2+S8[EF] | 3.0526 | ± 0.0705 (seeds) | +0.4472 | +0.417~+0.477 | 5.51 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r2_scale_2b/heatmap_Qwen3.5-2B.pdf` / `.png`: Qwen3.5-2B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-2B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r2_scale_2b/lines_Qwen3.5-2B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r2_scale_2b/pareto_Qwen3.5-2B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r2_scale_2b/curves_Qwen3.5-2B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r2_scale_2b/violin_Qwen3.5-2B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-2B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: cae4f7a
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 42
- 입력 파일 (sources): r2_scale_2b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
