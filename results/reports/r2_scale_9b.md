# Report: r2_scale_9b
generated 2026-10-14T22:56:42+00:00 from r2_scale_9b.jsonl

## [KR] 핵심 요약
- Qwen3.5-9B / wikitext2: fp 베이스라인 mean NLL 2.169 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +0.283 nats (95% CI +0.261~+0.303).
- KV 4bit (state fp): fp 대비 +0.006 nats (95% CI +0.004~+0.008).
- state 8bit: fp 대비 +0.003 nats (95% CI -0.000~+0.006) (구분 불가 / not distinguishable).
- KV4+S4 초가산: 실측 Δ +0.287 대 단독합 +0.289, 초과분 -0.002 nats (문서 대응 95% CI -0.009~+0.005) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.045 nats (6개 셀, 범위 -0.127~+0.002).

## [EN] Key Findings
- Qwen3.5-9B / wikitext2: fp baseline mean NLL 2.169 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +0.283 nats over fp (95% CI +0.261~+0.303).
- KV 4-bit (state fp): +0.006 nats over fp (95% CI +0.004~+0.008).
- state 8-bit: +0.003 nats over fp (95% CI -0.000~+0.006) (구분 불가 / not distinguishable).
- KV4+S4 superadditivity: measured +0.287 vs sum of single effects +0.289; excess -0.002 nats (paired 95% CI -0.009~+0.005) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits -0.045 nats (6 cells, range -0.127~+0.002).

## [KR/EN] 표
### Qwen3.5-9B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.1695 | ± 0.0313 (seeds) | +0.0000 | +0.000~+0.000 * | 41.94 | 3 | 20 |
| fp+S8 | 2.1725 | ± 0.0307 (seeds) | +0.0031 | -0.000~+0.006 * | 29.36 | 3 | 20 |
| fp+S8[EF] | 2.1728 | ± 0.0312 (seeds) | +0.0033 | +0.001~+0.005 | 29.36 | 3 | 20 |
| fp+S6 | 2.2374 | ± 0.0360 (seeds) | +0.0679 | +0.056~+0.082 | 26.21 | 3 | 20 |
| fp+S6[EF] | 2.2270 | ± 0.0324 (seeds) | +0.0575 | +0.049~+0.066 | 26.21 | 3 | 20 |
| fp+S4 | 2.4522 | ± 0.0430 (seeds) | +0.2828 | +0.261~+0.303 | 23.07 | 3 | 20 |
| fp+S4[EF] | 2.3256 | ± 0.0447 (seeds) | +0.1562 | +0.140~+0.175 | 23.07 | 3 | 20 |
| KV4+fp | 2.1754 | ± 0.0316 (seeds) | +0.0059 | +0.004~+0.008 | 29.36 | 3 | 20 |
| KV4+S6 | 2.2454 | ± 0.0375 (seeds) | +0.0759 | +0.064~+0.090 | 13.63 | 3 | 20 |
| KV4+S6[EF] | 2.2332 | ± 0.0336 (seeds) | +0.0637 | +0.055~+0.073 | 13.63 | 3 | 20 |
| KV4+S4 | 2.4564 | ± 0.0442 (seeds) | +0.2869 | +0.265~+0.308 | 10.49 | 3 | 20 |
| KV4+S4[EF] | 2.3334 | ± 0.0395 (seeds) | +0.1639 | +0.146~+0.182 | 10.49 | 3 | 20 |
| KV2+S8 | 2.3003 | ± 0.0381 (seeds) | +0.1308 | +0.119~+0.144 | 14.68 | 3 | 20 |
| KV2+S8[EF] | 2.3022 | ± 0.0382 (seeds) | +0.1328 | +0.121~+0.145 | 14.68 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r2_scale_9b/heatmap_Qwen3.5-9B.pdf` / `.png`: Qwen3.5-9B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-9B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r2_scale_9b/lines_Qwen3.5-9B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r2_scale_9b/pareto_Qwen3.5-9B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r2_scale_9b/curves_Qwen3.5-9B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r2_scale_9b/violin_Qwen3.5-9B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-9B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 7fea085
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 42
- 입력 파일 (sources): r2_scale_9b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
