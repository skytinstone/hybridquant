# Report: smoke_0.8b
generated 2026-10-14T13:49:58+00:00 from smoke_0.8b_mps.jsonl, smoke_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 3.116 (시드 1개, 문서 8개/시드).
- state 4bit (KV fp): fp 대비 +1.072 nats (95% CI +1.051~+1.149).
- KV 4bit (state fp): fp 대비 +0.031 nats (95% CI +0.003~+0.033).
- state 8bit: fp 대비 -0.251 nats (95% CI +0.007~+0.038).
- KV 2bit: fp 대비 +0.234 nats (95% CI +0.460~+0.551).
- KV4+S4 초가산: 실측 Δ +1.095 대 단독합 +1.103, 초과분 -0.008 nats (문서 대응 95% CI N/A).

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 3.116 (1 seeds, 8 docs per seed).
- state 4-bit (KV fp): +1.072 nats over fp (95% CI +1.051~+1.149).
- KV 4-bit (state fp): +0.031 nats over fp (95% CI +0.003~+0.033).
- state 8-bit: -0.251 nats over fp (95% CI +0.007~+0.038).
- KV 2-bit: +0.234 nats over fp (95% CI +0.460~+0.551).
- KV4+S4 superadditivity: measured +1.095 vs sum of single effects +1.103; excess -0.008 nats (paired 95% CI N/A).

## [KR/EN] 표
### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 3.1164 | ± 0.3259 (docs) | +0.0000 | +0.000~+0.000 * | 15.73 | 1 | 8 |
| fp+S8 | 2.8658 | ± 0.2802 (docs) | -0.2506 | +0.007~+0.038 | 11.01 | 1 | 8 |
| fp+S4 | 4.1884 | ± 0.3186 (docs) | +1.0720 | +1.051~+1.149 | 8.65 | 1 | 8 |
| KV8+fp | 2.8431 | ± 0.2704 (docs) | -0.2732 | -0.002~+0.000 * | 12.58 | 1 | 8 |
| KV8+S8 | 2.8665 | ± 0.2871 (docs) | -0.2499 | +0.006~+0.040 | 7.86 | 1 | 8 |
| KV8+S4 | 3.9624 | ± 0.3011 (docs) | +0.8460 | +1.077~+1.156 | 5.51 | 1 | 8 |
| KV4+fp | 3.1472 | ± 0.3381 (docs) | +0.0308 | +0.003~+0.033 | 11.01 | 1 | 8 |
| KV4+S8 | 2.8799 | ± 0.2655 (docs) | -0.2364 | +0.019~+0.053 | 6.29 | 1 | 8 |
| KV4+S4 | 4.2109 | ± 0.3061 (docs) | +1.0945 | +1.057~+1.178 | 3.93 | 1 | 8 |
| KV2+fp | 3.3499 | ± 0.2686 (docs) | +0.2335 | +0.460~+0.551 | 10.22 | 1 | 8 |
| KV2+S8 | 3.3685 | ± 0.2703 (docs) | +0.2521 | +0.476~+0.574 | 5.51 | 1 | 8 |
| KV2+S4 | 4.7317 | ± 0.2047 (docs) | +1.6153 | +1.809~+1.968 | 3.15 | 1 | 8 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/smoke_0.8b/heatmap_Qwen3.5-0.8B.pdf` / `.png`: Qwen3.5-0.8B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-0.8B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/smoke_0.8b/lines_Qwen3.5-0.8B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/smoke_0.8b/pareto_Qwen3.5-0.8B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/smoke_0.8b/curves_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/smoke_0.8b/violin_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 654fd48
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [2, 8], 행 (rows): 16
- 입력 파일 (sources): smoke_0.8b.jsonl, smoke_0.8b_mps.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
