# Report: r2_falcon_1.5b
generated 2026-10-14T13:51:29+00:00 from r2_falcon_1.5b.jsonl

## [KR] 핵심 요약
- Falcon-H1-1.5B-Instruct / wikitext2: fp 베이스라인 mean NLL 5.890 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 -0.030 nats (95% CI -0.069~+0.010) (구분 불가 / not distinguishable).
- KV 4bit (state fp): fp 대비 -0.010 nats (95% CI -0.019~-0.001).
- state 8bit: fp 대비 +0.000 nats (95% CI -0.006~+0.007) (구분 불가 / not distinguishable).
- KV4+S4 초가산: 실측 Δ -0.040 대 단독합 -0.040, 초과분 +0.000 nats (문서 대응 95% CI -0.016~+0.015) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 +0.017 nats (6개 셀, 범위 +0.003~+0.036).

## [EN] Key Findings
- Falcon-H1-1.5B-Instruct / wikitext2: fp baseline mean NLL 5.890 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): -0.030 nats over fp (95% CI -0.069~+0.010) (구분 불가 / not distinguishable).
- KV 4-bit (state fp): -0.010 nats over fp (95% CI -0.019~-0.001).
- state 8-bit: +0.000 nats over fp (95% CI -0.006~+0.007) (구분 불가 / not distinguishable).
- KV4+S4 superadditivity: measured -0.040 vs sum of single effects -0.040; excess +0.000 nats (paired 95% CI -0.016~+0.015) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits +0.017 nats (6 cells, range +0.003~+0.036).

## [KR/EN] 표
### Falcon-H1-1.5B-Instruct / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 5.8905 | ± 0.0384 (seeds) | +0.0000 | +0.000~+0.000 * | 50.33 | 3 | 20 |
| fp+S8 | 5.8908 | ± 0.0315 (seeds) | +0.0003 | -0.006~+0.007 * | 31.46 | 3 | 20 |
| fp+S8[EF] | 5.8939 | ± 0.0391 (seeds) | +0.0034 | +0.002~+0.005 | 31.46 | 3 | 20 |
| fp+S6 | 5.8869 | ± 0.0250 (seeds) | -0.0036 | -0.018~+0.012 * | 26.74 | 3 | 20 |
| fp+S6[EF] | 5.8987 | ± 0.0360 (seeds) | +0.0082 | +0.002~+0.014 | 26.74 | 3 | 20 |
| fp+S4 | 5.8603 | ± 0.0505 (seeds) | -0.0302 | -0.069~+0.010 * | 22.02 | 3 | 20 |
| fp+S4[EF] | 5.8930 | ± 0.0305 (seeds) | +0.0025 | -0.014~+0.018 * | 22.02 | 3 | 20 |
| KV4+fp | 5.8807 | ± 0.0328 (seeds) | -0.0098 | -0.019~-0.001 | 40.89 | 3 | 20 |
| KV4+S6 | 5.8802 | ± 0.0189 (seeds) | -0.0103 | -0.026~+0.005 * | 17.30 | 3 | 20 |
| KV4+S6[EF] | 5.8905 | ± 0.0307 (seeds) | +0.0000 | -0.011~+0.011 * | 17.30 | 3 | 20 |
| KV4+S4 | 5.8506 | ± 0.0468 (seeds) | -0.0399 | -0.072~-0.008 | 12.58 | 3 | 20 |
| KV4+S4[EF] | 5.8869 | ± 0.0273 (seeds) | -0.0036 | -0.020~+0.013 * | 12.58 | 3 | 20 |
| KV2+S8 | 5.9998 | ± 0.0362 (seeds) | +0.1093 | +0.062~+0.164 | 20.45 | 3 | 20 |
| KV2+S8[EF] | 6.0054 | ± 0.0332 (seeds) | +0.1149 | +0.069~+0.169 | 20.45 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r2_falcon_1.5b/heatmap_Falcon-H1-1.5B-Instruct.pdf` / `.png`: Falcon-H1-1.5B-Instruct KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Falcon-H1-1.5B-Instruct: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r2_falcon_1.5b/lines_Falcon-H1-1.5B-Instruct.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r2_falcon_1.5b/pareto_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/r2_falcon_1.5b/curves_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r2_falcon_1.5b/violin_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): tiiuae/Falcon-H1-1.5B-Instruct
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: cbd6f92
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 42
- 입력 파일 (sources): r2_falcon_1.5b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
