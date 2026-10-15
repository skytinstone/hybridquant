# Report: r3_longdecode_4b
generated 2026-10-15T07:15:53+00:00 from r3_longdecode_4b.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B / wikitext2: fp 베이스라인 mean NLL 2.203 (시드 3개, 문서 10개/시드).
- state 4bit (KV fp): fp 대비 +1.187 nats (95% CI +1.076~+1.294).
- KV 4bit (state fp): fp 대비 +0.001 nats (95% CI -0.002~+0.004) (구분 불가 / not distinguishable).
- state 8bit: fp 대비 +0.011 nats (95% CI +0.006~+0.016).
- KV4+S4 초가산: 실측 Δ +1.190 대 단독합 +1.188, 초과분 +0.003 nats (문서 대응 95% CI -0.019~+0.025) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.283 nats (5개 셀, 범위 -0.624~-0.011).

## [EN] Key Findings
- Qwen3.5-4B / wikitext2: fp baseline mean NLL 2.203 (3 seeds, 10 docs per seed).
- state 4-bit (KV fp): +1.187 nats over fp (95% CI +1.076~+1.294).
- KV 4-bit (state fp): +0.001 nats over fp (95% CI -0.002~+0.004) (구분 불가 / not distinguishable).
- state 8-bit: +0.011 nats over fp (95% CI +0.006~+0.016).
- KV4+S4 superadditivity: measured +1.190 vs sum of single effects +1.188; excess +0.003 nats (paired 95% CI -0.019~+0.025) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits -0.283 nats (5 cells, range -0.624~-0.011).

## [KR/EN] 표
### Qwen3.5-4B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.2032 | ± 0.0634 (seeds) | +0.0000 | +0.000~+0.000 * | 41.94 | 3 | 10 |
| fp+S8 | 2.2146 | ± 0.0660 (seeds) | +0.0114 | +0.006~+0.016 | 29.36 | 3 | 10 |
| fp+S8[EF] | 2.2036 | ± 0.0626 (seeds) | +0.0004 | -0.001~+0.002 * | 29.36 | 3 | 10 |
| fp+S6 | 2.3187 | ± 0.0760 (seeds) | +0.1155 | +0.103~+0.129 | 26.21 | 3 | 10 |
| fp+S6[EF] | 2.2416 | ± 0.0778 (seeds) | +0.0384 | +0.030~+0.046 | 26.21 | 3 | 10 |
| fp+S4 | 3.3898 | ± 0.2123 (seeds) | +1.1866 | +1.076~+1.294 | 23.07 | 3 | 10 |
| fp+S4[EF] | 2.7657 | ± 0.1018 (seeds) | +0.5625 | +0.524~+0.601 | 23.07 | 3 | 10 |
| KV4+fp | 2.2042 | ± 0.0639 (seeds) | +0.0010 | -0.002~+0.004 * | 29.36 | 3 | 10 |
| KV4+S6 | 2.3264 | ± 0.0830 (seeds) | +0.1232 | +0.111~+0.135 | 13.63 | 3 | 10 |
| KV4+S6[EF] | 2.2426 | ± 0.0760 (seeds) | +0.0394 | +0.032~+0.046 | 13.63 | 3 | 10 |
| KV4+S4 | 3.3935 | ± 0.2039 (seeds) | +1.1903 | +1.079~+1.294 | 10.49 | 3 | 10 |
| KV4+S4[EF] | 2.7766 | ± 0.1062 (seeds) | +0.5734 | +0.534~+0.614 | 10.49 | 3 | 10 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r3_longdecode_4b/heatmap_Qwen3.5-4B.pdf` / `.png`: Qwen3.5-4B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-4B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r3_longdecode_4b/lines_Qwen3.5-4B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r3_longdecode_4b/pareto_Qwen3.5-4B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r3_longdecode_4b/curves_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/r3_longdecode_4b/violin_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 3e4cc07
- 날짜 (dates): 2026-10-15
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [10], 행 (rows): 36
- 입력 파일 (sources): r3_longdecode_4b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
