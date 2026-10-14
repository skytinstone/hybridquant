# Report: grid_4b_core
generated 2026-10-14T15:52:51+00:00 from grid_4b_core.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B / wikitext2: fp 베이스라인 mean NLL 2.384 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +0.825 nats (95% CI +0.751~+0.902).
- KV 4bit (state fp): fp 대비 +0.001 nats (95% CI -0.003~+0.004) (구분 불가 / not distinguishable).
- state 8bit: fp 대비 +0.004 nats (95% CI +0.001~+0.007).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.090 nats (5개 셀, 범위 -0.349~+0.000).

## [EN] Key Findings
- Qwen3.5-4B / wikitext2: fp baseline mean NLL 2.384 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +0.825 nats over fp (95% CI +0.751~+0.902).
- KV 4-bit (state fp): +0.001 nats over fp (95% CI -0.003~+0.004) (구분 불가 / not distinguishable).
- state 8-bit: +0.004 nats over fp (95% CI +0.001~+0.007).
- error feedback: mean change in dNLL versus RTN at the same bits -0.090 nats (5 cells, range -0.349~+0.000).

## [KR/EN] 표
### Qwen3.5-4B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.3844 | ± 0.0485 (seeds) | +0.0000 | +0.000~+0.000 * | 41.94 | 3 | 20 |
| fp+S8 | 2.3887 | ± 0.0507 (seeds) | +0.0043 | +0.001~+0.007 | 29.36 | 3 | 20 |
| fp+S8[EF] | 2.3842 | ± 0.0481 (seeds) | -0.0002 | -0.002~+0.001 * | 29.36 | 3 | 20 |
| fp+S6 | 2.4719 | ± 0.0508 (seeds) | +0.0875 | +0.078~+0.097 | 26.21 | 3 | 20 |
| fp+S6[EF] | 2.4217 | ± 0.0501 (seeds) | +0.0373 | +0.030~+0.045 | 26.21 | 3 | 20 |
| fp+S4 | 3.2094 | ± 0.1225 (seeds) | +0.8250 | +0.751~+0.902 | 23.07 | 3 | 20 |
| fp+S4[EF] | 2.8601 | ± 0.0686 (seeds) | +0.4757 | +0.437~+0.518 | 23.07 | 3 | 20 |
| KV4+fp | 2.3850 | ± 0.0452 (seeds) | +0.0006 | -0.003~+0.004 * | 29.36 | 3 | 20 |
| KV4+S6 | 2.4756 | ± 0.0482 (seeds) | +0.0912 | +0.081~+0.101 | 13.63 | 3 | 20 |
| KV4+S6[EF] | 2.4284 | ± 0.0494 (seeds) | +0.0439 | +0.036~+0.053 | 13.63 | 3 | 20 |
| KV2+S8 | 2.6422 | ± 0.0597 (seeds) | +0.2578 | +0.239~+0.279 | 14.68 | 3 | 20 |
| KV2+S8[EF] | 2.6427 | ± 0.0621 (seeds) | +0.2582 | +0.240~+0.278 | 14.68 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/grid_4b_core/heatmap_Qwen3.5-4B.pdf` / `.png`: Qwen3.5-4B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-4B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/grid_4b_core/lines_Qwen3.5-4B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/grid_4b_core/pareto_Qwen3.5-4B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/grid_4b_core/curves_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/grid_4b_core/violin_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: a971d8d
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 36
- 입력 파일 (sources): grid_4b_core.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
