# Report: r3_longdecode_0.8b
generated 2026-10-15T04:43:17+00:00 from r3_longdecode_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 2.772 (시드 3개, 문서 10개/시드).
- state 4bit (KV fp): fp 대비 +1.488 nats (95% CI +1.416~+1.566).
- KV 4bit (state fp): fp 대비 +0.027 nats (95% CI +0.022~+0.031).
- state 8bit: fp 대비 +0.051 nats (95% CI +0.040~+0.062).
- KV4+S4 초가산: 실측 Δ +1.548 대 단독합 +1.515, 초과분 +0.033 nats (문서 대응 95% CI -0.000~+0.068) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.442 nats (5개 셀, 범위 -0.895~-0.047).

## [EN] Key Findings
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 2.772 (3 seeds, 10 docs per seed).
- state 4-bit (KV fp): +1.488 nats over fp (95% CI +1.416~+1.566).
- KV 4-bit (state fp): +0.027 nats over fp (95% CI +0.022~+0.031).
- state 8-bit: +0.051 nats over fp (95% CI +0.040~+0.062).
- KV4+S4 superadditivity: measured +1.548 vs sum of single effects +1.515; excess +0.033 nats (paired 95% CI -0.000~+0.068) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits -0.442 nats (5 cells, range -0.895~-0.047).

## [KR/EN] 표
### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.7722 | ± 0.0881 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 10 |
| fp+S8 | 2.8227 | ± 0.0967 (seeds) | +0.0505 | +0.040~+0.062 | 11.01 | 3 | 10 |
| fp+S8[EF] | 2.7758 | ± 0.0893 (seeds) | +0.0036 | +0.002~+0.005 | 11.01 | 3 | 10 |
| fp+S6 | 3.0247 | ± 0.1062 (seeds) | +0.2525 | +0.229~+0.276 | 9.83 | 3 | 10 |
| fp+S6[EF] | 2.8255 | ± 0.0863 (seeds) | +0.0533 | +0.046~+0.060 | 9.83 | 3 | 10 |
| fp+S4 | 4.2604 | ± 0.1611 (seeds) | +1.4883 | +1.416~+1.566 | 8.65 | 3 | 10 |
| fp+S4[EF] | 3.3854 | ± 0.1282 (seeds) | +0.6132 | +0.579~+0.647 | 8.65 | 3 | 10 |
| KV4+fp | 2.7992 | ± 0.0890 (seeds) | +0.0270 | +0.022~+0.031 | 11.01 | 3 | 10 |
| KV4+S6 | 3.0436 | ± 0.1032 (seeds) | +0.2714 | +0.249~+0.293 | 5.11 | 3 | 10 |
| KV4+S6[EF] | 2.8522 | ± 0.0901 (seeds) | +0.0801 | +0.071~+0.089 | 5.11 | 3 | 10 |
| KV4+S4 | 4.3200 | ± 0.1731 (seeds) | +1.5478 | +1.476~+1.624 | 3.93 | 3 | 10 |
| KV4+S4[EF] | 3.4246 | ± 0.1299 (seeds) | +0.6524 | +0.619~+0.683 | 3.93 | 3 | 10 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r3_longdecode_0.8b/heatmap_Qwen3.5-0.8B.pdf` / `.png`: Qwen3.5-0.8B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-0.8B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r3_longdecode_0.8b/lines_Qwen3.5-0.8B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r3_longdecode_0.8b/pareto_Qwen3.5-0.8B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r3_longdecode_0.8b/curves_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r3_longdecode_0.8b/violin_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 8508a1c
- 날짜 (dates): 2026-10-15
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [10], 행 (rows): 36
- 입력 파일 (sources): r3_longdecode_0.8b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
