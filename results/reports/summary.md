# Report: all
generated 2026-10-14T13:52:06+00:00 from grid_0.8b.jsonl, grid_4b_core.jsonl, layer_sweep_Qwen3.5-0.8B.jsonl, r2_ablation_0.8b.jsonl, r2_c4_0.8b.jsonl, r2_falcon_1.5b.jsonl, r2_gsm8k_4b.jsonl, r2_niah_0.8b.jsonl, r2_niah_4b.jsonl, r2_scale_2b.jsonl, r2_scale_9b.jsonl, smoke_0.8b.jsonl, smoke_0.8b_mps.jsonl, superadd_0.8b.jsonl

## [KR] 핵심 요약
- Qwen3.5-0.8B / c4: fp 베이스라인 mean NLL 3.154 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.031 nats (95% CI +0.975~+1.086).
- KV 4bit (state fp): fp 대비 +0.023 nats (95% CI +0.019~+0.027).
- state 8bit: fp 대비 +0.025 nats (95% CI +0.021~+0.030).
- KV4+S4 초가산: 실측 Δ +1.095 대 단독합 +1.055, 초과분 +0.040 nats (문서 대응 95% CI +0.019~+0.060).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.222 nats (6개 셀, 범위 -0.528~-0.013).
- Qwen3.5-0.8B / wikitext2: fp 베이스라인 mean NLL 2.988 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.046 nats (95% CI +1.022~+1.078).
- KV 4bit (state fp): fp 대비 +0.028 nats (95% CI +0.024~+0.030).
- state 8bit: fp 대비 -0.030 nats (95% CI +0.011~+0.020).
- KV 2bit: fp 대비 +0.403 nats (95% CI +0.425~+0.470).
- KV4+S4 초가산: 실측 Δ +1.098 대 단독합 +1.074, 초과분 +0.024 nats (문서 대응 95% CI N/A).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.215 nats (20개 셀, 범위 -0.608~+0.022).
- KV 노이즈 대조(같은 RMS의 가우시안 노이즈): Δ +1.047 대 실제 양자화 +1.098 nats.
- state 노이즈 대조(같은 RMS의 가우시안 노이즈): Δ +10.018 대 실제 양자화 +1.098 nats.
- Qwen3.5-2B / wikitext2: fp 베이스라인 mean NLL 2.605 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +1.808 nats (95% CI +1.725~+1.896).
- KV 4bit (state fp): fp 대비 +0.011 nats (95% CI +0.009~+0.014).
- state 8bit: fp 대비 +0.010 nats (95% CI +0.006~+0.013).
- KV4+S4 초가산: 실측 Δ +1.867 대 단독합 +1.820, 초과분 +0.047 nats (문서 대응 95% CI +0.021~+0.073).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.440 nats (6개 셀, 범위 -1.255~-0.008).
- Qwen3.5-4B / wikitext2: fp 베이스라인 mean NLL 2.384 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +0.825 nats (95% CI +0.751~+0.902).
- KV 4bit (state fp): fp 대비 +0.001 nats (95% CI -0.003~+0.004) (구분 불가 / not distinguishable).
- state 8bit: fp 대비 +0.004 nats (95% CI +0.001~+0.007).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.090 nats (5개 셀, 범위 -0.349~+0.000).
- Qwen3.5-9B / wikitext2: fp 베이스라인 mean NLL 2.169 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 +0.283 nats (95% CI +0.261~+0.303).
- KV 4bit (state fp): fp 대비 +0.006 nats (95% CI +0.004~+0.008).
- state 8bit: fp 대비 +0.003 nats (95% CI -0.000~+0.006) (구분 불가 / not distinguishable).
- KV4+S4 초가산: 실측 Δ +0.287 대 단독합 +0.289, 초과분 -0.002 nats (문서 대응 95% CI -0.009~+0.005) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 -0.045 nats (6개 셀, 범위 -0.127~+0.002).
- Falcon-H1-1.5B-Instruct / wikitext2: fp 베이스라인 mean NLL 5.890 (시드 3개, 문서 20개/시드).
- state 4bit (KV fp): fp 대비 -0.030 nats (95% CI -0.069~+0.010) (구분 불가 / not distinguishable).
- KV 4bit (state fp): fp 대비 -0.010 nats (95% CI -0.019~-0.001).
- state 8bit: fp 대비 +0.000 nats (95% CI -0.006~+0.007) (구분 불가 / not distinguishable).
- KV4+S4 초가산: 실측 Δ -0.040 대 단독합 -0.040, 초과분 +0.000 nats (문서 대응 95% CI -0.016~+0.015) (구분 불가 / not distinguishable).
- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 +0.017 nats (6개 셀, 범위 +0.003~+0.036).
- Qwen3.5-0.8B: 레이어 하나만 state int4로 양자화했을 때 ΔNLL 최대는 레이어 0 (+0.2175), 최소는 레이어 9 (-0.0085), 전체 합 +0.5747.
- n_half(오차 반감 토큰 수)와 ΔNLL의 Spearman 상관 -0.08 (순열 p=0.783).
- Qwen3.5-4B GSM8K fp 정답률 0.600 (문제 200개, 95% CI 0.535~0.665), 평균 생성 토큰 2912, 예산 소진율 0.48.
- fp+S8: 정답률 0.312 (fp 대비 -0.287), 토큰 3468, 반복률 0.122.
- KV4+fp: 정답률 0.595 (fp 대비 -0.005), 토큰 3121, 반복률 0.087.
- Qwen3.5-0.8B NIAH 컨텍스트 4096: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.32.
- Qwen3.5-0.8B NIAH 컨텍스트 16384: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.54.
- Qwen3.5-4B NIAH 컨텍스트 4096: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.94.
- Qwen3.5-4B NIAH 컨텍스트 16384: fp 정확도 1.00 (문제 50개), 가장 낮은 셀 KV2+fp 0.92.

## [EN] Key Findings
- Qwen3.5-0.8B / c4: fp baseline mean NLL 3.154 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.031 nats over fp (95% CI +0.975~+1.086).
- KV 4-bit (state fp): +0.023 nats over fp (95% CI +0.019~+0.027).
- state 8-bit: +0.025 nats over fp (95% CI +0.021~+0.030).
- KV4+S4 superadditivity: measured +1.095 vs sum of single effects +1.055; excess +0.040 nats (paired 95% CI +0.019~+0.060).
- error feedback: mean change in dNLL versus RTN at the same bits -0.222 nats (6 cells, range -0.528~-0.013).
- Qwen3.5-0.8B / wikitext2: fp baseline mean NLL 2.988 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.046 nats over fp (95% CI +1.022~+1.078).
- KV 4-bit (state fp): +0.028 nats over fp (95% CI +0.024~+0.030).
- state 8-bit: -0.030 nats over fp (95% CI +0.011~+0.020).
- KV 2-bit: +0.403 nats over fp (95% CI +0.425~+0.470).
- KV4+S4 superadditivity: measured +1.098 vs sum of single effects +1.074; excess +0.024 nats (paired 95% CI N/A).
- error feedback: mean change in dNLL versus RTN at the same bits -0.215 nats (20 cells, range -0.608~+0.022).
- KV noise control (Gaussian noise of equal RMS): +1.047 vs real quantization +1.098 nats.
- state noise control (Gaussian noise of equal RMS): +10.018 vs real quantization +1.098 nats.
- Qwen3.5-2B / wikitext2: fp baseline mean NLL 2.605 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +1.808 nats over fp (95% CI +1.725~+1.896).
- KV 4-bit (state fp): +0.011 nats over fp (95% CI +0.009~+0.014).
- state 8-bit: +0.010 nats over fp (95% CI +0.006~+0.013).
- KV4+S4 superadditivity: measured +1.867 vs sum of single effects +1.820; excess +0.047 nats (paired 95% CI +0.021~+0.073).
- error feedback: mean change in dNLL versus RTN at the same bits -0.440 nats (6 cells, range -1.255~-0.008).
- Qwen3.5-4B / wikitext2: fp baseline mean NLL 2.384 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +0.825 nats over fp (95% CI +0.751~+0.902).
- KV 4-bit (state fp): +0.001 nats over fp (95% CI -0.003~+0.004) (구분 불가 / not distinguishable).
- state 8-bit: +0.004 nats over fp (95% CI +0.001~+0.007).
- error feedback: mean change in dNLL versus RTN at the same bits -0.090 nats (5 cells, range -0.349~+0.000).
- Qwen3.5-9B / wikitext2: fp baseline mean NLL 2.169 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): +0.283 nats over fp (95% CI +0.261~+0.303).
- KV 4-bit (state fp): +0.006 nats over fp (95% CI +0.004~+0.008).
- state 8-bit: +0.003 nats over fp (95% CI -0.000~+0.006) (구분 불가 / not distinguishable).
- KV4+S4 superadditivity: measured +0.287 vs sum of single effects +0.289; excess -0.002 nats (paired 95% CI -0.009~+0.005) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits -0.045 nats (6 cells, range -0.127~+0.002).
- Falcon-H1-1.5B-Instruct / wikitext2: fp baseline mean NLL 5.890 (3 seeds, 20 docs per seed).
- state 4-bit (KV fp): -0.030 nats over fp (95% CI -0.069~+0.010) (구분 불가 / not distinguishable).
- KV 4-bit (state fp): -0.010 nats over fp (95% CI -0.019~-0.001).
- state 8-bit: +0.000 nats over fp (95% CI -0.006~+0.007) (구분 불가 / not distinguishable).
- KV4+S4 superadditivity: measured -0.040 vs sum of single effects -0.040; excess +0.000 nats (paired 95% CI -0.016~+0.015) (구분 불가 / not distinguishable).
- error feedback: mean change in dNLL versus RTN at the same bits +0.017 nats (6 cells, range +0.003~+0.036).
- Qwen3.5-0.8B: quantizing one layer's state to int4, the largest dNLL is layer 0 (+0.2175) and the smallest layer 9 (-0.0085); sum over layers +0.5747.
- Spearman correlation between n_half (error half-life) and dNLL: -0.08 (permutation p=0.783).
- Qwen3.5-4B GSM8K fp accuracy 0.600 (200 problems, 95% CI 0.535~0.665), mean 2912 tokens, budget-exhaustion 0.48.
- fp+S8: accuracy 0.312 (-0.287 vs fp), 3468 tokens, repeat ratio 0.122.
- KV4+fp: accuracy 0.595 (-0.005 vs fp), 3121 tokens, repeat ratio 0.087.
- Qwen3.5-0.8B NIAH context 4096: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.32.
- Qwen3.5-0.8B NIAH context 16384: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.54.
- Qwen3.5-4B NIAH context 4096: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.94.
- Qwen3.5-4B NIAH context 16384: fp accuracy 1.00 (50 problems); lowest cell KV2+fp at 0.92.

## [KR/EN] 표
### Qwen3.5-0.8B / c4
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 3.1536 | ± 0.0422 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 20 |
| fp+S8 | 3.1791 | ± 0.0427 (seeds) | +0.0254 | +0.021~+0.030 | 11.01 | 3 | 20 |
| fp+S8[EF] | 3.1560 | ± 0.0414 (seeds) | +0.0024 | +0.001~+0.004 | 11.01 | 3 | 20 |
| fp+S6 | 3.3225 | ± 0.0544 (seeds) | +0.1689 | +0.151~+0.187 | 9.83 | 3 | 20 |
| fp+S6[EF] | 3.1971 | ± 0.0461 (seeds) | +0.0434 | +0.037~+0.049 | 9.83 | 3 | 20 |
| fp+S4 | 4.1851 | ± 0.0860 (seeds) | +1.0314 | +0.975~+1.086 | 8.65 | 3 | 20 |
| fp+S4[EF] | 3.6681 | ± 0.0537 (seeds) | +0.5144 | +0.484~+0.543 | 8.65 | 3 | 20 |
| KV4+fp | 3.1770 | ± 0.0437 (seeds) | +0.0234 | +0.019~+0.027 | 11.01 | 3 | 20 |
| KV4+S6 | 3.3488 | ± 0.0505 (seeds) | +0.1951 | +0.178~+0.212 | 5.11 | 3 | 20 |
| KV4+S6[EF] | 3.2219 | ± 0.0433 (seeds) | +0.0683 | +0.062~+0.074 | 5.11 | 3 | 20 |
| KV4+S4 | 4.2485 | ± 0.0819 (seeds) | +1.0949 | +1.035~+1.154 | 3.93 | 3 | 20 |
| KV4+S4[EF] | 3.7202 | ± 0.0593 (seeds) | +0.5666 | +0.535~+0.598 | 3.93 | 3 | 20 |
| KV2+S8 | 3.5782 | ± 0.0468 (seeds) | +0.4245 | +0.399~+0.452 | 5.51 | 3 | 20 |
| KV2+S8[EF] | 3.5651 | ± 0.0473 (seeds) | +0.4114 | +0.388~+0.438 | 5.51 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

### Qwen3.5-0.8B / wikitext2
| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |
|---|---|---|---|---|---|---|---|
| fp | 2.9884 | ± 0.0740 (seeds) | +0.0000 | +0.000~+0.000 * | 15.73 | 3 | 20 |
| fp+S8 | 2.9588 | ± 0.0021 (seeds) | -0.0296 | +0.011~+0.020 | 11.01 | 3 | 20 |
| fp+S8[EF] | 2.9753 | ± 0.0492 (seeds) | -0.0130 | -0.000~+0.003 * | 11.01 | 3 | 20 |
| fp+S6 | 3.1092 | ± 0.0522 (seeds) | +0.1209 | +0.122~+0.148 | 9.83 | 3 | 20 |
| fp+S6[EF] | 3.0245 | ± 0.0522 (seeds) | +0.0362 | +0.044~+0.057 | 9.83 | 3 | 20 |
| fp+S5 | 3.3457 | ± 0.0611 (seeds) | +0.3573 | +0.347~+0.395 | 9.24 | 3 | 20 |
| fp+S5[EF] | 3.1203 | ± 0.0548 (seeds) | +0.1320 | +0.133~+0.158 | 9.24 | 3 | 20 |
| fp+S4 | 4.0346 | ± 0.1116 (seeds) | +1.0462 | +1.022~+1.078 | 8.65 | 3 | 20 |
| fp+S4[EF] | 3.4985 | ± 0.0841 (seeds) | +0.5101 | +0.495~+0.553 | 8.65 | 3 | 20 |
| KV8+fp | 2.9426 | ± 0.0049 (seeds) | -0.0458 | -0.001~+0.000 * | 12.58 | 3 | 20 |
| KV8+S8 | 2.9605 | ± 0.0029 (seeds) | -0.0278 | +0.013~+0.022 | 7.86 | 3 | 20 |
| KV8+S8[EF] | 2.9757 | ± 0.0478 (seeds) | -0.0127 | -0.000~+0.003 * | 7.86 | 3 | 20 |
| KV8+S6 | 3.1115 | ± 0.0508 (seeds) | +0.1232 | +0.124~+0.150 | 6.68 | 3 | 20 |
| KV8+S6[EF] | 3.0214 | ± 0.0525 (seeds) | +0.0330 | +0.041~+0.054 | 6.68 | 3 | 20 |
| KV8+S5 | 3.3370 | ± 0.0565 (seeds) | +0.3486 | +0.340~+0.387 | 6.09 | 3 | 20 |
| KV8+S5[EF] | 3.1261 | ± 0.0478 (seeds) | +0.1377 | +0.139~+0.165 | 6.09 | 3 | 20 |
| KV8+S4 | 3.9956 | ± 0.0464 (seeds) | +1.0072 | +1.012~+1.099 | 5.51 | 3 | 20 |
| KV8+S4[EF] | 3.4883 | ± 0.0735 (seeds) | +0.4999 | +0.487~+0.541 | 5.51 | 3 | 20 |
| KV4+fp | 3.0166 | ± 0.0746 (seeds) | +0.0282 | +0.024~+0.030 | 11.01 | 3 | 20 |
| KV4+S8 | 2.9849 | ± 0.0070 (seeds) | -0.0035 | +0.035~+0.049 | 6.29 | 3 | 20 |
| KV4+S8[EF] | 3.0036 | ± 0.0485 (seeds) | +0.0152 | +0.025~+0.034 | 6.29 | 3 | 20 |
| KV4+S6 | 3.1391 | ± 0.0461 (seeds) | +0.1507 | +0.151~+0.179 | 5.11 | 3 | 20 |
| KV4+S6 | 3.2882 | ± 0.0553 (seeds) | +0.2998 | +0.294~+0.335 | 5.11 | 3 | 20 |
| KV4+S6 | 3.8254 | ± 0.0820 (seeds) | +0.8370 | +0.815~+0.889 | 5.11 | 3 | 20 |
| KV4+S6 | 4.9960 | ± 0.0694 (seeds) | +2.0077 | +1.950~+2.094 | 5.11 | 3 | 20 |
| KV4+S6 | 3.1391 | ± 0.0461 (seeds) | +0.1507 | +0.155~+0.175 | 5.11 | 3 | 20 |
| KV4+S6 | 3.2882 | ± 0.0553 (seeds) | +0.2998 | +0.294~+0.335 | 5.11 | 3 | 20 |
| KV4+S6 | 3.8254 | ± 0.0820 (seeds) | +0.8370 | +0.815~+0.889 | 5.11 | 3 | 20 |
| KV4+S6 | 4.9960 | ± 0.0694 (seeds) | +2.0077 | +1.950~+2.094 | 5.11 | 3 | 20 |
| KV4+S6 | 3.0358 | ± 0.0509 (seeds) | +0.0474 | +0.053~+0.070 | 5.11 | 3 | 20 |
| KV4+S6 | 3.0833 | ± 0.0619 (seeds) | +0.0949 | +0.097~+0.121 | 5.11 | 3 | 20 |
| KV4+S6 | 3.1955 | ± 0.0572 (seeds) | +0.2071 | +0.207~+0.236 | 5.11 | 3 | 20 |
| KV4+S6 | 3.5772 | ± 0.0607 (seeds) | +0.5888 | +0.574~+0.635 | 5.11 | 3 | 20 |
| KV4+S6 | 3.0701 | ± 0.0501 (seeds) | +0.0817 | +0.086~+0.106 | 5.11 | 3 | 20 |
| KV4+S6 | 3.1521 | ± 0.0531 (seeds) | +0.1637 | +0.163~+0.193 | 5.11 | 3 | 20 |
| KV4+S6 | 3.4080 | ± 0.0492 (seeds) | +0.4196 | +0.411~+0.458 | 5.11 | 3 | 20 |
| KV4+S6 | 4.1512 | ± 0.0836 (seeds) | +1.1629 | +1.132~+1.220 | 5.11 | 3 | 20 |
| KV4+S6[EF] | 3.0505 | ± 0.0548 (seeds) | +0.0622 | +0.069~+0.084 | 5.11 | 3 | 20 |
| KV4+S5 | 3.3777 | ± 0.0523 (seeds) | +0.3893 | +0.378~+0.430 | 4.52 | 3 | 20 |
| KV4+S5[EF] | 3.1614 | ± 0.0595 (seeds) | +0.1731 | +0.172~+0.201 | 4.52 | 3 | 20 |
| KV4+S4 | 4.0794 | ± 0.0717 (seeds) | +1.0911 | +1.059~+1.147 | 3.93 | 3 | 20 |
| KV4+S4 | 6.3156 | ± 0.0750 (seeds) | +3.3272 | +3.225~+3.476 | 3.93 | 3 | 20 |
| KV4+S4 | 8.2221 | ± 0.1129 (seeds) | +5.2337 | +5.115~+5.387 | 3.93 | 3 | 20 |
| KV4+S4 | 10.9108 | ± 0.1141 (seeds) | +7.9224 | +7.750~+8.109 | 3.93 | 3 | 20 |
| KV4+S4 | 4.0866 | ± 0.0835 (seeds) | +1.0983 | +1.082~+1.128 | 3.93 | 3 | 20 |
| KV4+S4[Sgn] | 13.0067 | ± 0.0735 (seeds) | +10.0183 | +9.918~+10.147 | 3.93 | 3 | 20 |
| KV4+S4 | 6.3156 | ± 0.0750 (seeds) | +3.3272 | +3.225~+3.476 | 3.93 | 3 | 20 |
| KV4+S4 | 8.2221 | ± 0.1129 (seeds) | +5.2337 | +5.115~+5.387 | 3.93 | 3 | 20 |
| KV4+S4 | 10.9108 | ± 0.1141 (seeds) | +7.9224 | +7.750~+8.109 | 3.93 | 3 | 20 |
| KV4+S4 | 3.2781 | ± 0.0671 (seeds) | +0.2898 | +0.287~+0.320 | 3.93 | 3 | 20 |
| KV4+S4 | 3.5033 | ± 0.0662 (seeds) | +0.5149 | +0.500~+0.559 | 3.93 | 3 | 20 |
| KV4+S4 | 5.0265 | ± 0.0626 (seeds) | +2.0382 | +1.984~+2.125 | 3.93 | 3 | 20 |
| KV4+S4 | 8.1244 | ± 0.1067 (seeds) | +5.1360 | +4.997~+5.296 | 3.93 | 3 | 20 |
| KV4+S4 | 3.4933 | ± 0.0735 (seeds) | +0.5049 | +0.494~+0.545 | 3.93 | 3 | 20 |
| KV4+S4 | 4.0039 | ± 0.0949 (seeds) | +1.0155 | +0.985~+1.075 | 3.93 | 3 | 20 |
| KV4+S4 | 6.6089 | ± 0.0727 (seeds) | +3.6205 | +3.538~+3.735 | 3.93 | 3 | 20 |
| KV4+S4 | 9.5837 | ± 0.1227 (seeds) | +6.5953 | +6.443~+6.771 | 3.93 | 3 | 20 |
| KV4+S4[KVgn] | 4.0356 | ± 0.0837 (seeds) | +1.0472 | +1.015~+1.108 | 3.93 | 3 | 20 |
| KV4+S4[EF] | 3.5397 | ± 0.0899 (seeds) | +0.5513 | +0.537~+0.595 | 3.93 | 3 | 20 |
| KV3+fp | 3.1094 | ± 0.0521 (seeds) | +0.1211 | +0.125~+0.145 | 10.62 | 3 | 20 |
| KV3+S8 | 3.1247 | ± 0.0530 (seeds) | +0.1363 | +0.139~+0.163 | 5.90 | 3 | 20 |
| KV3+S8[EF] | 3.1134 | ± 0.0508 (seeds) | +0.1250 | +0.129~+0.150 | 5.90 | 3 | 20 |
| KV3+S6 | 3.2636 | ± 0.0542 (seeds) | +0.2752 | +0.272~+0.308 | 4.72 | 3 | 20 |
| KV3+S6[EF] | 3.1703 | ± 0.0549 (seeds) | +0.1819 | +0.181~+0.212 | 4.72 | 3 | 20 |
| KV3+S5 | 3.5056 | ± 0.0656 (seeds) | +0.5172 | +0.502~+0.560 | 4.13 | 3 | 20 |
| KV3+S5[EF] | 3.2938 | ± 0.0488 (seeds) | +0.3054 | +0.299~+0.340 | 4.13 | 3 | 20 |
| KV3+S4 | 4.2840 | ± 0.0953 (seeds) | +1.2956 | +1.258~+1.359 | 3.54 | 3 | 20 |
| KV3+S4[EF] | 3.7066 | ± 0.0953 (seeds) | +0.7182 | +0.700~+0.768 | 3.54 | 3 | 20 |
| KV2+fp | 3.3914 | ± 0.0244 (seeds) | +0.4030 | +0.425~+0.470 | 10.22 | 3 | 20 |
| KV2+S8 | 3.4021 | ± 0.0249 (seeds) | +0.4137 | +0.434~+0.482 | 5.51 | 3 | 20 |
| KV2+S8[EF] | 3.4236 | ± 0.0562 (seeds) | +0.4353 | +0.424~+0.477 | 5.51 | 3 | 20 |
| KV2+S6 | 3.6044 | ± 0.0560 (seeds) | +0.6160 | +0.601~+0.660 | 4.33 | 3 | 20 |
| KV2+S6[EF] | 3.4954 | ± 0.0647 (seeds) | +0.5071 | +0.495~+0.549 | 4.33 | 3 | 20 |
| KV2+S5 | 3.8869 | ± 0.0592 (seeds) | +0.8985 | +0.878~+0.950 | 3.74 | 3 | 20 |
| KV2+S5[EF] | 3.6323 | ± 0.0461 (seeds) | +0.6440 | +0.627~+0.690 | 3.74 | 3 | 20 |
| KV2+S4 | 4.7601 | ± 0.0531 (seeds) | +1.7718 | +1.772~+1.868 | 3.15 | 3 | 20 |
| KV2+S4[EF] | 4.1517 | ± 0.0753 (seeds) | +1.1633 | +1.142~+1.215 | 3.15 | 3 | 20 |

\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.

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

### Qwen3.5-0.8B
| layer_idx | delta_nll | mean_alpha | n_half | mean_beta |
|---|---|---|---|---|
| 0.0000 | 0.2175 | 0.7714 | 0.9636 | 0.4520 |
| 1.0000 | 0.0100 | 0.6046 | 0.7367 | 0.4189 |
| 2.0000 | 0.0313 | 0.7448 | 1.3552 | 0.3785 |
| 4.0000 | 0.0307 | 0.8277 | 2.1971 | 0.3539 |
| 5.0000 | 0.0162 | 0.9092 | 5.0683 | 0.3810 |
| 6.0000 | 0.0055 | 0.8674 | 3.5119 | 0.4618 |
| 8.0000 | 0.0311 | 0.9282 | 7.7287 | 0.4343 |
| 9.0000 | -0.0085 | 0.9439 | 9.8564 | 0.2476 |
| 10.0000 | 0.0567 | 0.9174 | 6.1371 | 0.4727 |
| 12.0000 | 0.0374 | 0.9623 | 14.4020 | 0.2358 |
| 13.0000 | 0.0256 | 0.9084 | 5.7655 | 0.3712 |
| 14.0000 | 0.0456 | 0.9264 | 7.6561 | 0.4154 |
| 16.0000 | 0.0386 | 0.8589 | 3.2087 | 0.3928 |
| 17.0000 | 0.0078 | 0.9446 | 9.9908 | 0.4169 |
| 18.0000 | -0.0040 | 0.9054 | 5.1884 | 0.5782 |
| 20.0000 | 0.0081 | 0.8525 | 2.3100 | 0.5947 |
| 21.0000 | 0.0109 | 0.8886 | 4.6544 | 0.5541 |
| 22.0000 | 0.0143 | 0.7654 | 1.2789 | 0.5595 |

### Qwen3.5-4B
| model | cell | error_feedback | kv_bits | state_bits | acc | acc_std_seeds | acc_ci_lo | acc_ci_hi | n | n_seeds | mean_tokens | budget_rate | repeat_ratio |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 0.6000 | N/A | 0.5349 | 0.6650 | 200 | 1 | 2912.4450 | 0.4850 | 0.0856 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 0.3125 | N/A | 0.1562 | 0.4688 | 32 | 1 | 3467.5312 | 0.7500 | 0.1222 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 0.5950 | N/A | 0.5250 | 0.6600 | 200 | 1 | 3120.8400 | 0.5550 | 0.0867 |

### Qwen3.5-0.8B
| model | cell | error_feedback | kv_bits | state_bits | context_len | acc | ci_lo | ci_hi | n |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-0.8B | fp | False | 16 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp | False | 16 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S4 | False | 16 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S4 | False | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S4[EF] | True | 16 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S4[EF] | True | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S8 | False | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S8 | False | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S8[EF] | True | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | fp+S8[EF] | True | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+fp | False | 2 | 16 | 4096 | 0.3200 | 0.2000 | 0.4600 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+fp | False | 2 | 16 | 16384 | 0.5400 | 0.4000 | 0.6800 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+S8 | False | 2 | 8 | 4096 | 0.4200 | 0.3000 | 0.5600 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+S8 | False | 2 | 8 | 16384 | 0.5400 | 0.4000 | 0.6800 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+S8[EF] | True | 2 | 8 | 4096 | 0.3800 | 0.2600 | 0.5000 | 50 |
| Qwen/Qwen3.5-0.8B | KV2+S8[EF] | True | 2 | 8 | 16384 | 0.5400 | 0.4200 | 0.6800 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+fp | False | 4 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+fp | False | 4 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+S4 | False | 4 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+S4 | False | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+S4[EF] | True | 4 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-0.8B | KV4+S4[EF] | True | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |

### Qwen3.5-4B
| model | cell | error_feedback | kv_bits | state_bits | context_len | acc | ci_lo | ci_hi | n |
|---|---|---|---|---|---|---|---|---|---|
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp | False | 16 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4 | False | 16 | 4 | 4096 | 0.9600 | 0.9000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4 | False | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4[EF] | True | 16 | 4 | 4096 | 0.9600 | 0.9000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S4[EF] | True | 16 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8 | False | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8[EF] | True | 16 | 8 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | fp+S8[EF] | True | 16 | 8 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+fp | False | 2 | 16 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+fp | False | 2 | 16 | 16384 | 0.9200 | 0.8400 | 0.9800 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8 | False | 2 | 8 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8 | False | 2 | 8 | 16384 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8[EF] | True | 2 | 8 | 4096 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV2+S8[EF] | True | 2 | 8 | 16384 | 0.9400 | 0.8600 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+fp | False | 4 | 16 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4 | False | 4 | 4 | 4096 | 0.9800 | 0.9400 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4 | False | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4[EF] | True | 4 | 4 | 4096 | 1.0000 | 1.0000 | 1.0000 | 50 |
| Qwen/Qwen3.5-4B | KV4+S4[EF] | True | 4 | 4 | 16384 | 1.0000 | 1.0000 | 1.0000 | 50 |

## [KR/EN] 그림 목록
- `results/figures/all/superadd_bars_Qwen3.5-0.8B.pdf` / `.png`: 셀별 ΔNLL 막대(95% CI)와 fp+ΔKV+ΔS 예측값(점선), 차이가 초과분 / dNLL per cell (95% CI) with the additive prediction fp+dKV+dS (dotted); the gap is the excess
- `results/figures/all/superadd_hist_Qwen3.5-0.8B.pdf` / `.png`: 문서별 초과분 히스토그램 / Histogram of the per-document excess
- `results/figures/all/superadd_noise_Qwen3.5-0.8B.pdf` / `.png`: noise_match 조건과 실제 양자화 조건의 문서별 NLL 대응(대각선 = 동일) / Per-document NLL, noise-match versus real quantization (diagonal = equal)
- `results/figures/all/ablation_Qwen3.5-0.8B.pdf` / `.png`: state 양자화 포맷 ablation: group 크기 × (symmetric, stochastic), 셀별 패널 / State-format ablation: group size x (symmetric, stochastic), one panel per cell
- `results/figures/all/heatmap_Qwen3.5-0.8B.pdf` / `.png`: Qwen3.5-0.8B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-0.8B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/all/lines_Qwen3.5-0.8B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/all/pareto_Qwen3.5-0.8B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/all/curves_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/all/violin_Qwen3.5-0.8B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)
- `results/figures/all/heatmap_Qwen3.5-2B.pdf` / `.png`: Qwen3.5-2B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-2B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/all/lines_Qwen3.5-2B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/all/pareto_Qwen3.5-2B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/all/curves_Qwen3.5-2B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/all/violin_Qwen3.5-2B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)
- `results/figures/all/heatmap_Qwen3.5-4B.pdf` / `.png`: Qwen3.5-4B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-4B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/all/lines_Qwen3.5-4B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/all/pareto_Qwen3.5-4B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/all/curves_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/all/violin_Qwen3.5-4B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)
- `results/figures/all/heatmap_Qwen3.5-9B.pdf` / `.png`: Qwen3.5-9B KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Qwen3.5-9B: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/all/lines_Qwen3.5-9B.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/all/pareto_Qwen3.5-9B.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/all/curves_Qwen3.5-9B.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/all/violin_Qwen3.5-9B.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)
- `results/figures/all/heatmap_Falcon-H1-1.5B-Instruct.pdf` / `.png`: Falcon-H1-1.5B-Instruct KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on / Falcon-H1-1.5B-Instruct: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on
- `results/figures/all/lines_Falcon-H1-1.5B-Instruct.pdf` / `.png`: state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI / dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI
- `results/figures/all/pareto_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint / Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint
- `results/figures/all/curves_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std / NLL by decode position for core cells (moving average 16); shading = ±1 std across documents
- `results/figures/all/violin_Falcon-H1-1.5B-Instruct.pdf` / `.png`: 핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시) / Per-document dNLL distribution for core cells (violin, mean marked)
- `results/figures/all/scale.pdf` / `.png`: 모델 크기별 ΔNLL, 셀별 선, Falcon-H1은 삼각형 / dNLL versus model size per cell; Falcon-H1 as triangles
- `results/figures/all/layer_bars_Qwen3.5-0.8B.pdf` / `.png`: 레이어 하나씩만 state 양자화했을 때의 ΔNLL / dNLL when only one layer's state is quantized
- `results/figures/all/layer_halflife_Qwen3.5-0.8B.pdf` / `.png`: 오차 반감 토큰 수 n_half 대 ΔNLL 산점도, 회귀선, Spearman/Pearson(순열 p값) / Error half-life n_half versus dNLL with regression line; Spearman/Pearson with permutation p-values
- `results/figures/all/layer_alpha_beta_Qwen3.5-0.8B.pdf` / `.png`: 레이어별 평균 α와 평균 β 산점도, 색은 ΔNLL / Per-layer mean alpha versus mean beta, colour = dNLL
- `results/figures/all/gsm8k_acc_Qwen3.5-4B.pdf` / `.png`: 셀별 GSM8K 정답률(문제 부트스트랩 95% CI), EF 색 구분 / GSM8K accuracy per cell (bootstrap 95% CI over problems), colour = EF
- `results/figures/all/gsm8k_tokens_Qwen3.5-4B.pdf` / `.png`: 셀별 생성 토큰 수 분포 / Generated-token distribution per cell
- `results/figures/all/gsm8k_behaviour_Qwen3.5-4B.pdf` / `.png`: 예산 소진율과 8-gram 반복률 / Budget-exhaustion rate and 8-gram repeat ratio
- `results/figures/all/gsm8k_acc_vs_tokens_Qwen3.5-4B.pdf` / `.png`: 정답률 대 평균 생성 토큰 수 / Accuracy versus mean generated tokens
- `results/figures/all/niah_heatmap_Qwen3.5-0.8B.pdf` / `.png`: 셀 × 컨텍스트 길이 NIAH 정확도 / NIAH accuracy, cell x context length
- `results/figures/all/niah_depth_Qwen3.5-0.8B.pdf` / `.png`: 깊이 × 셀 정확도(길이별 패널) / Accuracy by needle depth x cell, one panel per context length
- `results/figures/all/niah_kv_vs_state_Qwen3.5-0.8B.pdf` / `.png`: KV-only 대 state-only: 회수 정확도(왼쪽)와 같은 셀의 ΔNLL(오른쪽), 태스크별 민감도 비교 / KV-only versus state-only: retrieval accuracy (left) and dNLL for the same cells (right)
- `results/figures/all/niah_heatmap_Qwen3.5-4B.pdf` / `.png`: 셀 × 컨텍스트 길이 NIAH 정확도 / NIAH accuracy, cell x context length
- `results/figures/all/niah_depth_Qwen3.5-4B.pdf` / `.png`: 깊이 × 셀 정확도(길이별 패널) / Accuracy by needle depth x cell, one panel per context length
- `results/figures/all/niah_kv_vs_state_Qwen3.5-4B.pdf` / `.png`: KV-only 대 state-only: 회수 정확도(왼쪽)와 같은 셀의 ΔNLL(오른쪽), 태스크별 민감도 비교 / KV-only versus state-only: retrieval accuracy (left) and dNLL for the same cells (right)
- `results/figures/all/summary_overview.pdf` / `.png`: 대표 그림 8장 개요 / Overview of eight representative figures

## 조건
- 모델 (models): Qwen/Qwen3.5-0.8B, Qwen/Qwen3.5-2B, Qwen/Qwen3.5-4B, Qwen/Qwen3.5-9B, tiiuae/Falcon-H1-1.5B-Instruct
- GPU: NVIDIA A100-SXM4-40GB
- dtype: bfloat16
- commit: 0490bb2, 1f517a6, 4f6674b, 527bfa3, 569541d, 654fd48, 7fea085, 935f3ad, 9bb281c, a971d8d, b944556, cae4f7a, cbd6f92, e33cb9f
- 날짜 (dates): 2026-10-14
- 시드 수 (seeds): 3, 문서 수/시드 (docs): [2, 8, 20], 행 (rows): 3120
- 입력 파일 (sources): grid_0.8b.jsonl, grid_4b_core.jsonl, layer_sweep_Qwen3.5-0.8B.jsonl, r2_ablation_0.8b.jsonl, r2_c4_0.8b.jsonl, r2_falcon_1.5b.jsonl, r2_gsm8k_4b.jsonl, r2_niah_0.8b.jsonl, r2_niah_4b.jsonl, r2_scale_2b.jsonl, r2_scale_9b.jsonl, smoke_0.8b.jsonl, smoke_0.8b_mps.jsonl, superadd_0.8b.jsonl
- 실패/건너뜀 기록 (failures): see results/push_failures.log
