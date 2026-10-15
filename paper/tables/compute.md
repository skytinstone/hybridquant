# Compute (one NVIDIA A100-SXM4-40GB, bf16, pure-PyTorch GDN kernels), wall-clock per stage from results/round*.log

| stage | model | hours | note |
|---|---|---|---|
| smoke_0.8b | 0.8B | <0.01 | all rows already present (skipped) |
| grid_0.8b | 0.8B | 1.5 | first launch 03:04 to 04:35 UTC; a second copy ran concurrently from 03:51 (duplicate launch, rows deduplicated) |
| superadd_0.8b | 0.8B | 0.21 | |
| layer_sweep_0.8b | 0.8B | 0.07 | 1 seed x 8 documents |
| grid_4b_core | 4B | 0.57 | |
| r2_ablation_0.8b | 0.8B | 1.10 | |
| r2_c4_0.8b | 0.8B | 0.48 | |
| r2_niah_0.8b | 0.8B | 0.19 | 1,100 prompts |
| r2_scale_2b | 2B | 0.48 | |
| r2_gsm8k_4b | 4B | 13.2 | 4.25 h for the first 432 generations at batch 8, 8.95 h for the remaining 1,568 at batch 16 |
| r2_niah_4b | 4B | 0.36 | 1,100 prompts |
| r2_scale_9b | 9B | 0.46 | batch 10 |
| r2_falcon_1.5b | Falcon-H1-1.5B | 0.68 | round-3 rerun with both Falcon fixes; two earlier runs (0.96 h, 0.63 h) invalid and excluded |
| r3_recipe_4b | 4B | 0.29 | (KV4,S8), (KV8,S8), (KV4,S4), RTN and EF |
| r3_niah_recipe_4b | 4B | 0.14 | 400 prompts |
| r3_gsm8k_s8ef_4b | 4B | 1.09 | 200 generations, batch 16 |
| layer_sweep_4b | 4B | 0.12 | 1 seed x 8 documents |
| r3_longdecode_0.8b | 0.8B | 0.54 | decode 1024 |
| r3_longdecode_4b | 4B | 1.53 | decode 1024 |
| total (valid runs) | | about 22.6 | rounds 1 to 3; invalid Falcon runs (1.6 h) excluded |
