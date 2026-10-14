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
| r2_gsm8k_4b | 4B | 4.25 + resumed | 432 of 2,000 generations in the first launch (batch 8); resumed with batch 16 [PENDING: final hours] |
| r2_niah_4b | 4B | 0.36 | 1,100 prompts |
| r2_scale_9b | 9B | 0.46 | batch 10 |
| r2_falcon_1.5b | Falcon-H1-1.5B | 0.96 | round-2 run invalid (batched-decode bug); rerun in round 3 [PENDING] |
| total (valid NLL/NIAH stages) | | about 5.4 | excluding gsm8k and the invalid Falcon run |
