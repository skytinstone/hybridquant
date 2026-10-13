# Report: smoke
generated 2026-10-13T16:32:42+00:00 from smoke_0.8b_mps.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 3.389 (시드 1개, 문서 2개/시드).
- state 4bit (KV fp): fp 대비 +1.041 nats (95% CI N/A).
- KV 4bit (state fp): fp 대비 +0.044 nats (95% CI N/A).
- KV4+S4 초가산: 실측 Δ +1.068 대 단독합 +1.085, 초과분 -0.017 nats (문서 대응 95% CI N/A).

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 3.389 (1 seeds, 2 docs per seed).
- state 4-bit (KV fp): +1.041 nats over fp (95% CI N/A).
- KV 4-bit (state fp): +0.044 nats over fp (95% CI N/A).
- KV4+S4 superadditivity: measured +1.068 vs sum of single effects +1.085; excess -0.017 nats (paired 95% CI N/A).

## [KR/EN] 표
### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 3.3890 | ± 0.3810 (docs) | +0.0000 | N/A | 15.73 | 1 | 2 |
| fp+S4 | 4.4304 | ± 0.3302 (docs) | +1.0415 | N/A | 8.65 | 1 | 2 |
| KV4+fp | 3.4325 | ± 0.4149 (docs) | +0.0435 | N/A | 11.01 | 1 | 2 |
| KV4+S4 | 4.4569 | ± 0.3203 (docs) | +1.0679 | N/A | 3.93 | 1 | 2 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/smoke/heatmap_Qwen3.5-0.8B.pdf` / `.png`: Qwen3.5-0.8B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-0.8B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/smoke/lines_Qwen3.5-0.8B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/smoke/pareto_Qwen3.5-0.8B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/smoke/curves_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: N/A
- dtype: N/A
- commit: N/A
- 날짜 (dates): N/A
- 시드 수 (seeds): 1, 문서 수/시드 (docs): [2], 행 (rows): 4
- 입력 파일 (sources): smoke_0.8b_mps.jsonl
- 실패/건너뜀 기록 (failures): none logged
