# Report: r2_falcon_1.5b
generated 2026-10-15T05:40:36+00:00 from r2_falcon_1.5b.jsonl

## [KR] 핵심 요약
- Falcon-H1-1.5B-Instruct / wikitext2: fp 베이스라인 mean NLL 2.550 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +0.004 nats (95% CI -0.010~+0.018) (구분 불가 / not distinguishable).
- KV 4bit (state fp): fp 대비 +0.010 nats (95% CI +0.004~+0.015).
- state 8bit: fp 대비 -0.014 nats (95% CI -0.019~-0.010).
- KV4+S4 초가산: 실측 Δ +0.012 대 단독합 +0.014, 초과분 -0.002 nats (문서 대응 95% CI -0.008~+0.005) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 +0.012 nats (6개 셀, 범위 -0.005~+0.028).

## [EN] Key Findings
- Falcon-H1-1.5B-Instruct / wikitext2: fp baseline mean NLL 2.550 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +0.004 nats over fp (95% CI -0.010~+0.018) (구분 불가 / not distinguishable).
- KV 4-bit (state fp): +0.010 nats over fp (95% CI +0.004~+0.015).
- state 8-bit: -0.014 nats over fp (95% CI -0.019~-0.010).
- KV4+S4 superadditivity: measured +0.012 vs sum of single effects +0.014; excess -0.002 nats (paired 95% CI -0.008~+0.005) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits +0.012 nats (6 cells, range -0.005~+0.028).

## [KR/EN] 표
### Falcon-H1-1.5B-Instruct / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.5503 | ± 0.0934 (seeds) | +0.0000 | +0.000~+0.000 * | 50.33 | 3 | 20 |
| fp+S8 | 2.5361 | ± 0.0917 (seeds) | -0.0142 | -0.019~-0.010 | 31.46 | 3 | 20 |
| fp+S8[EF] | 2.5493 | ± 0.0935 (seeds) | -0.0011 | -0.002~+0.000 * | 31.46 | 3 | 20 |
| fp+S6 | 2.5319 | ± 0.0924 (seeds) | -0.0184 | -0.024~-0.012 | 26.74 | 3 | 20 |
| fp+S6[EF] | 2.5520 | ± 0.0911 (seeds) | +0.0017 | -0.001~+0.004 * | 26.74 | 3 | 20 |
| fp+S4 | 2.5548 | ± 0.0874 (seeds) | +0.0045 | -0.010~+0.018 * | 22.02 | 3 | 20 |
| fp+S4[EF] | 2.5498 | ± 0.0829 (seeds) | -0.0005 | -0.010~+0.010 * | 22.02 | 3 | 20 |
| KV4+fp | 2.5599 | ± 0.0939 (seeds) | +0.0096 | +0.004~+0.015 | 40.89 | 3 | 20 |
| KV4+S6 | 2.5413 | ± 0.0968 (seeds) | -0.0090 | -0.016~-0.002 | 17.30 | 3 | 20 |
| KV4+S6[EF] | 2.5622 | ± 0.0923 (seeds) | +0.0119 | +0.006~+0.018 | 17.30 | 3 | 20 |
| KV4+S4 | 2.5626 | ± 0.0859 (seeds) | +0.0123 | -0.003~+0.027 * | 12.58 | 3 | 20 |
| KV4+S4[EF] | 2.5600 | ± 0.0853 (seeds) | +0.0097 | -0.001~+0.019 * | 12.58 | 3 | 20 |
| KV2+S8 | 2.7550 | ± 0.0765 (seeds) | +0.2047 | +0.178~+0.231 | 20.45 | 3 | 20 |
| KV2+S8[EF] | 2.7832 | ± 0.0696 (seeds) | +0.2329 | +0.207~+0.260 | 20.45 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r2_falcon_1.5b/heatmap_Falcon-H1-1.5B-Instruct.pdf` / `.png`: Falcon-H1-1.5B-Instruct KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Falcon-H1-1.5B-Instruct: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r2_falcon_1.5b/lines_Falcon-H1-1.5B-Instruct.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r2_falcon_1.5b/pareto_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r2_falcon_1.5b/curves_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r2_falcon_1.5b/violin_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): tiiuae/Falcon-H1-1.5B-Instruct
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 9540c3b
- 날짜 (dates): 2026-10-15
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 42
- 입력 파일 (sources): r2_falcon_1.5b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
