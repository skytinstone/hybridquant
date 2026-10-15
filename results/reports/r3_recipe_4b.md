# Report: r3_recipe_4b
generated 2026-10-15T05:41:40+00:00 from r3_recipe_4b.jsonl

## [KR] 핵심 요약
- Qwen3.5-4B / wikitext2: fp 베이스라인 mean NLL 2.384 (시드 3개, 문서 20개/시드).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.120 nats (3개 셀, 범위 -0.351~-0.004).

## [EN] Key Findings
- Qwen3.5-4B / wikitext2: fp baseline mean NLL 2.384 (3 seeds, 20 docs per seed).
- error feedback: mean change in dNLL versus RTN at the same bits -0.120 nats (3 cells, range -0.351~-0.004).

## [KR/EN] 표
### Qwen3.5-4B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.3844 | ± 0.0485 (seeds) | +0.0000 | +0.000~+0.000 * | 41.94 | 3 | 20 |
| KV8+S8 | 2.3886 | ± 0.0504 (seeds) | +0.0042 | +0.001~+0.007 | 20.97 | 3 | 20 |
| KV8+S8[EF] | 2.3839 | ± 0.0479 (seeds) | -0.0005 | -0.002~+0.001 * | 20.97 | 3 | 20 |
| KV4+S8 | 2.3879 | ± 0.0451 (seeds) | +0.0035 | -0.001~+0.007 * | 16.78 | 3 | 20 |
| KV4+S8[EF] | 2.3839 | ± 0.0478 (seeds) | -0.0005 | -0.004~+0.003 * | 16.78 | 3 | 20 |
| KV4+S4 | 3.2221 | ± 0.1255 (seeds) | +0.8377 | +0.764~+0.917 | 10.49 | 3 | 20 |
| KV4+S4[EF] | 2.8715 | ± 0.0681 (seeds) | +0.4871 | +0.446~+0.529 | 10.49 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

## [KR/EN] 그림 목록
- `results/figures/r3_recipe_4b/heatmap_Qwen3.5-4B.pdf` / `.png`: Qwen3.5-4B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-4B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/r3_recipe_4b/lines_Qwen3.5-4B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/r3_recipe_4b/pareto_Qwen3.5-4B.pdf` / `.png`: 총 메모리 대 ΔNLL 파레토 산점도, T=512/4k/32k 패널, 마커는 KV-only/state-only/joint / Memory versus dNLL Pareto scatter at T = 512, 4k and 32k; marker = KV-only / state-only / joint
- `results/figures/r3_recipe_4b/curves_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents

## 조건
- 모델 (models): Qwen/Qwen3.5-4B
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 82b864f, a971d8d
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [20], 행 (rows): 21
- 입력 파일 (sources): grid_4b_core.jsonl (fp borrowed), r3_recipe_4b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
