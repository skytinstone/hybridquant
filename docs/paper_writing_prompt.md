# Prompt for the manuscript-writing session (Claude Project)

Attach: paper/main.tex (v0 draft), results/reports/*.md, paper/tables/models.md and models.tex, the PNGs listed
under "Figures to place". Paste the block below as the first message.

---

# Task: turn the v0 draft into a complete, data-backed manuscript

You are writing the paper "Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs"
(single author, Minseok Shin). The v0 draft (attached as LaTeX) has all [TBD] placeholders in the Abstract,
Introduction contributions, Table 1, Sections 5 to 9 and Appendices B and C. Your job is to fill every [TBD]
from the attached experiment reports, rewrite the parts that the data contradict, and place the figures.
Write in English, academic register, no em-dashes. Never invent a number: every value you write must come
from the attached reports or from the "Key numbers" section below, and anything still being computed must
be marked [PENDING: ...] rather than guessed.

## Attached inputs (read all before writing)
- paper/main.tex (the v0 draft)
- results/reports/*.md: one report per experiment stage (grid_0.8b, grid_4b_core, r2_scale_2b, r2_scale_9b,
  superadd_0.8b, layer_sweep_0.8b, r2_ablation_0.8b, r2_c4_0.8b, r2_niah_0.8b, r2_niah_4b, r2_gsm8k_4b,
  summary). Each has a "Key Findings" section (use the [EN] block), tables with mean NLL, delta vs fp and
  paired-bootstrap 95% CIs, a figure list, and a "conditions" block (GPU, commit, seeds, docs).
- paper/tables/models.md: Appendix B model table (layer counts, head geometry, |KV|/token, |S|, T*).
- paper/figures/*.png and results/figures/**/*.png: all figures, for you to look at and caption.
  The .pdf versions with identical names are what the LaTeX should include.

## What the experiments established (use these as the spine of Sections 6 to 8)
1. Asymmetry, at every scale (0.8B, 2B, 4B, 9B) and on two corpora (WikiText-2, C4): 4-bit KV is free or
   nearly free; 4-bit state costs about 1 nat; 8-bit state is free; the state needs 6 bits before it is
   usable and 8 bits to be lossless.
2. Error feedback halves the state loss at every bit-width (0.8B S4: +1.05 to +0.52 nats) and is
   essentially free in accuracy at 8 bits.
3. Joint quantization is nearly additive: the super-additive excess at the (KV4, S4) corner is +0.03 to
   +0.05 nats (CI excludes zero) at 0.8B and 2B and on C4, and zero at 9B. State the result as "weakly
   super-additive, vanishing with scale", and note that this means the two memories can be allocated
   almost independently.
4. Noise-matched control: for the KV cache, Gaussian noise of equal RMS costs the same as quantization
   (+1.06 vs +1.11 with S4 fixed), so only the error energy matters. For the state, equal-RMS noise costs
   nine times more (+10.0 vs +1.1 nats). Interpretation to write: round-to-nearest is idempotent, so
   re-storing an already rounded state injects no new error, whereas fresh independent noise is injected
   at every token and accumulates through the recurrence; this is also why the bound in Proposition 1
   is loose.
5. Scale: the state-4-bit penalty is 1.05 (0.8B), 1.81 (2B), 0.83 (4B), 0.28 (9B). It falls with scale
   but 2B is an outlier. Note from the model table that 0.8B and 2B share exactly the same cache
   geometry (same |KV|/token, same |S|), as do 4B and 9B, so the 2B anomaly is not a state-size effect;
   discuss it honestly as unexplained (candidate: the ratio of state capacity to hidden size, 2B has
   twice the hidden size with the same state).
6. Layer sensitivity (0.8B, state int4 on one GDN layer at a time): layer 0 dominates (+0.22 nats, 40%
   of the sum), all others are below +0.06, and the sum of single-layer effects (+0.57) is about half
   the all-layer effect (+1.05). The correlation between a layer's error half-life n_1/2 and its
   sensitivity is NOT there: Spearman -0.08 (permutation p = 0.78), Pearson -0.22 (p = 0.36).
7. Task dependence. Retrieval (NIAH, 4k and 16k contexts) is limited by the KV cache: only KV 2-bit
   hurts (0.8B: 1.00 to 0.32 at 4k and 0.54 at 16k; 4B: 0.94 and 0.92), while state 4-bit stays at
   0.96 to 1.00. Reasoning (GSM8K, thinking mode, 4B) is limited by the state: 4-bit KV is lossless
   (0.595 vs 0.600) while 8-bit state alone halves accuracy (0.31) [PENDING: final n and the EF cells].
8. Practical recipe at 4B (Pareto front at T=512): KV4 + state6 + EF costs +0.04 nats at 32% of the
   bf16 footprint (13.5 MB vs 42 MB); KV4 + state8 is lossless at about 70%. At T = 32k the KV cache
   dominates memory and the state bits barely move the x axis (see the three-panel Pareto figure).

## Required changes to the draft, section by section
- Abstract: fill the TBDs with items 1, 3, 6, 7, 2 and 8 above. Replace "confirm that [layer sensitivity
  correlates with decay]" with an honest statement: the steady-state model explains why the state needs
  more bits than the KV cache, but per-layer sensitivity is not predicted by the decay half-life; a single
  early layer dominates.
- Introduction, contribution bullets: bullet 2 must say the error-propagation model gives a bound whose
  prediction about layer ordering is tested and NOT supported, and that the noise-matched control shows
  why (rounding is idempotent). Do not oversell the theory. Bullets 3, 4, 5: fill from items 3, 7, 8.
- Table 1 (4B headline): rows bf16 / KV-only (KV4) / state-only (S8 with EF) / joint (KV4 + S6 + EF, and
  KV4 + S8 + EF). Columns: memory % at T = 512 from the grid_4b_core table (memory MB column divided by
  the fp row), dNLL from the same table, GSM8K from r2_gsm8k_4b.md [PENDING where the cell is not there yet],
  NIAH (mean of 4k and 16k) from r2_niah_4b.md.
- Section 4.2 / Proposition 1: keep the proposition and proof as they are (they are correct as a bound),
  but change the sentence "Section 7 tests the prediction that layer sensitivity scales with n_1/2" to
  announce that Section 7 finds no such scaling and explains why the bound is loose.
- Section 5 Experimental Setup: write it from the "conditions" blocks: NVIDIA A100-SXM4-40GB, bf16,
  transformers 5.19.0.dev0 (commit 66e0f4a), torch 2.11, pure-PyTorch GDN kernels (no fla / causal-conv1d),
  3 seeds x 20 WikiText-2 windows of 512 tokens (prefill 256, decode 256), paired bootstrap CIs over
  documents (1000 resamples), group size 128 for both memories, EF residual reset after prefill.
  GSM8K: 200-problem subset, thinking mode, 4096-token budget, sampling from the model card.
  NIAH: 2 context lengths x 5 depths x 10 needles, greedy, 32 generated tokens.
- Section 6 Results: subsections in this order: 6.1 the grid (Fig. 3), 6.2 error feedback, 6.3 memory
  vs accuracy and the recipe (Fig. 4), 6.4 scale (Fig. 5), 6.5 second corpus (C4, one paragraph),
  6.6 tasks (Fig. 8, Table 1). Quote deltas with their CIs.
- Section 7 Analysis: 7.1 additivity and the noise control (Fig. 6), 7.2 layer sensitivity (Fig. 7)
  including the negative correlation result, 7.3 state format ablation (Fig. A2; group size, symmetric
  vs asymmetric, stochastic rounding; read the labels from the figure, the table rows are distinguished
  by their label strings).
- Section 8 Discussion and Limitations: the recipe (item 8); limitations: fake quantization (no kernel
  speed), decode length 256 in the language-modelling metric (state accumulation beyond that is only
  probed by GSM8K), one model family at four sizes, the 2B anomaly, uniform group format only.
- Falcon-H1 (Mamba-2 control): the round-2 numbers are INVALID (a batched-decoding bug in the inference
  library) and are being rerun. Everywhere the draft mentions the control, write [PENDING: Falcon-H1
  rerun] and keep the sentences short enough to delete if the rerun is not done. Do not use any Falcon
  number from the summary report.
- Section 9 Conclusion: three sentences: the asymmetry, the recipe, the open question about layer 0 and
  the 2B anomaly.
- Appendix B: paste models.md as a LaTeX table (models.tex is attached), fill hardware/software from
  Section 5, compute hours [PENDING: from logs].
- Appendix C: one full table per model from the grid reports (all cells, EF off and on, delta and CI),
  plus the ablation and C4 tables.
- Reproducibility: the commands are `python scripts/run_grid.py --config configs/<name>.yaml` for every
  table, `python analysis/report.py --stage <name> --inputs results/<name>.jsonl` for every report, and
  `bash scripts/run_all.sh` for everything.

## Figures to place (file names are exact; use the .pdf in LaTeX, look at the .png)
- Fig. 1  paper/figures/overview.pdf (full width): hybrid block, two memories, two store-time hooks.
- Fig. 2  paper/figures/memory_crossover.pdf (single column): KV vs state size against T, T* = 768
  (Qwen3.5) and 1,536 (Falcon-H1).
- Fig. 3  results/figures/grid_0.8b/heatmap_Qwen3.5-0.8B.pdf (full width, EF off and on panels) and
  results/figures/grid_4b_core/heatmap_Qwen3.5-4B.pdf: the main grid.
- Fig. 4  results/figures/grid_4b_core/pareto_Qwen3.5-4B.pdf (full width, three panels T = 512, 4k, 32k).
- Fig. 5  results/figures/all/scale.pdf: dNLL of the core cells against model size.
- Fig. 6  results/figures/superadd_0.8b/superadd_bars_Qwen3.5-0.8B.pdf and
  superadd_noise_Qwen3.5-0.8B.pdf (two panels): additivity and the noise-matched control.
- Fig. 7  results/figures/layer_sweep_0.8b/layer_bars_Qwen3.5-0.8B.pdf and
  layer_halflife_Qwen3.5-0.8B.pdf: per-layer sensitivity and its (absent) relation to n_1/2.
- Fig. 8  results/figures/r2_niah_4b/niah_kv_vs_state_Qwen3.5-4B.pdf and
  results/figures/r2_gsm8k_4b/gsm8k_acc_Qwen3.5-4B.pdf: task dependence.
- Fig. 9 (optional) results/figures/grid_0.8b/curves_Qwen3.5-0.8B.pdf: NLL against decode position.
- Appendix figures: results/figures/r2_ablation_0.8b/ablation_Qwen3.5-0.8B.pdf (format ablation),
  results/figures/r2_c4_0.8b/heatmap_Qwen3.5-0.8B.pdf (C4), results/figures/r2_niah_4b/niah_depth_Qwen3.5-4B.pdf,
  results/figures/r2_gsm8k_4b/gsm8k_behaviour_Qwen3.5-4B.pdf (budget exhaustion and repetition),
  results/figures/grid_*/lines_*.pdf (dNLL vs state bits with CIs).
Captions must say what is plotted, the model, the number of seeds and documents, and what the shading or
error bars are (paired bootstrap 95% CI over documents unless the report says otherwise).

## Key numbers (nats, delta vs bf16, paired bootstrap 95% CI; 3 seeds x 20 docs unless stated)
| model / corpus | fp NLL | state 4-bit | KV 4-bit | state 8-bit | KV 2-bit | (KV4,S4) excess | EF mean effect |
| 0.8B / WikiText-2 | 2.974 | +1.049 [1.001, 1.094] | +0.028 [0.024, 0.032] | +0.015 [0.010, 0.020] | +0.439 [0.416, 0.464] | +0.029 [0.009, 0.049] | -0.222 (20 cells) |
| 0.8B / C4 | 3.154 | +1.031 [0.975, 1.086] | +0.023 [0.019, 0.027] | +0.025 [0.021, 0.030] | - | +0.040 [0.019, 0.060] | -0.222 (6 cells) |
| 2B / WikiText-2 | 2.605 | +1.808 [1.725, 1.896] | +0.011 [0.009, 0.014] | +0.010 [0.006, 0.013] | - | +0.047 [0.021, 0.073] | -0.440 (6 cells) |
| 4B / WikiText-2 | 2.384 | +0.825 [0.751, 0.902] | +0.001 [-0.003, 0.004] n.s. | +0.004 [0.001, 0.007] | - | - | -0.090 (5 cells) |
| 9B / WikiText-2 | 2.169 | +0.283 [0.261, 0.303] | +0.006 [0.004, 0.008] | +0.003 [-0.000, 0.006] n.s. | - | -0.002 [-0.009, 0.005] n.s. | -0.045 (6 cells) |
0.8B grid with EF on: S4 +0.524, S5 +0.146, S6 +0.050, S8 +0.001 (KV fp). 0.8B KV4+S4 with EF: +0.566.
4B Pareto (T=512): fp 42 MB; KV4+S6 13.5 MB +0.09; KV4+S6[EF] 13.5 MB +0.04; KV4+S8 15 MB about 0.
Noise control (0.8B): KV noise +1.061 vs KV quant +1.105; state noise +10.03 vs state quant +1.105.
Layer sweep (0.8B, S4 one layer): max layer 0 +0.2175, min layer 9 -0.0085, sum +0.575; Spearman(n_1/2, dNLL) = -0.08, p = 0.783.
NIAH 4B (acc, n = 50 per length): fp 1.00/1.00; KV4 1.00/1.00; KV2 0.94/0.92; S4 0.96/1.00; S4[EF] 0.96/1.00; S8 1.00/1.00; KV4+S4 0.98/1.00. NIAH 0.8B: fp 1.00/1.00; KV2 0.32/0.54.
GSM8K 4B (partial): fp 0.600 [0.535, 0.665] n = 200, mean 2912 tokens, 48% hit the 4096 budget; KV4 0.595 (n = 200); S8 0.312 [0.16, 0.47] (n = 32 so far) [PENDING].
Model geometry: 0.8B and 2B: L = 24, L_a = 6, L_g = 18, H_kv = 2, d = 256, H = 16, d_k = d_v = 128, |KV|/token = 6,144, |S| = 4.72 M, T* = 768. 4B and 9B: L = 32, L_a = 8, L_g = 24, H_kv = 4, d = 256, H = 32, d_k = d_v = 128, |KV|/token = 16,384, |S| = 12.58 M, T* = 768. Falcon-H1-1.5B: L = 24 hybrid, H_kv = 2, d = 128, H = 48, d_head = 64, d_state = 256, |KV|/token = 12,288, |S| = 18.87 M, T* = 1,536.

## Output
Return the complete main.tex with every placeholder resolved or marked [PENDING: reason], figure
environments with the exact file paths above, and a short change log at the top listing which claims of
v0 you weakened or removed and why.
