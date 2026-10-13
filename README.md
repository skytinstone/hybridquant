# hybridquant

Joint KV-cache and recurrent-state quantization for hybrid linear-attention LLMs (Qwen3.5, Falcon-H1).

English | [한국어](README.ko.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## What this is

Hybrid LLMs keep two kinds of memory during inference:

| memory | produced by | shape | grows with context? |
|---|---|---|---|
| KV cache | softmax attention layers | `[B, H_kv, T, d]` | yes |
| recurrent state | Gated DeltaNet / Mamba-2 layers | `[B, H, d_k, d_v]` | no (fixed size) |

Prior work quantizes each memory in isolation. This repo asks a different question: **given a fixed total memory budget, how should the bits be split between the two, and how do their errors interact?**

Everything here is **fake quantization** (quantize, then dequantize, in plain PyTorch). There are no custom kernels. We measure accuracy and memory, not latency.

## What we measure

- **Decode-time NLL.** The prompt is prefilled in one pass and the continuation is fed one token at a time under a quantized cache (teacher forcing). Only this path lets recurrent-state error accumulate, so it is the primary metric; a plain forward-pass perplexity cannot see the state effect.
- **Round-to-nearest versus error feedback.** Every state cell is run twice: plain RTN, and RTN with error feedback (the store residual is carried into the next store). EF is a comparison axis in every table.
- **Noise-matched control.** For the superadditivity check, one side's quantization is replaced by Gaussian noise with the same error RMS. This separates error magnitude from quantization structure.
- **Statistics.** 20 documents x 3 seeds per cell. Seeds select disjoint document blocks, so seed-to-seed variance is document-sample variance. Differences against fp are reported with paired per-document bootstrap 95% CIs (1000 resamples); a CI that contains zero is marked "not distinguishable".
- **Batched decode does not change results.** Documents are decoded in batches of equal length without padding; `tests/test_quant_ext.py` checks that batched and single-document NLLs match.

## Layout

```
src/hybridquant/
  quant.py     fake-quant primitives: asym/sym, group, stochastic rounding, noise match, error-feedback flag
  hooks.py     store-time hooks on transformers cache layers (KV write, state write, EF residuals), coverage check
  eval.py      batched teacher-forced decode NLL, batched generation, answer scoring, dtype resolution, model loading
  data.py      WikiText-2 / C4 as one token stream cut into disjoint windows per seed
  config.py    YAML schema, CLI overrides, grid or explicit cells, EF crossing, run list
  budget.py    cache sizes (eq. 1) and bit budgets (eq. 5) from a model config
  meta.py      run metadata (git commit, library versions, GPU, timestamp) stored in every result row
configs/
  smoke_0.8b.yaml, grid_0.8b.yaml, superadd_0.8b.yaml, grid_4b_core.yaml         round 1
  r2_ablation_0.8b.yaml, r2_c4_0.8b.yaml, r2_scale_2b.yaml, r2_scale_9b.yaml,     round 2
  r2_falcon_1.5b.yaml, r2_gsm8k_4b.yaml, r2_gsm8k_4b_full.yaml, r2_niah_0.8b.yaml, r2_niah_4b.yaml
scripts/
  run_grid.py       decode-NLL grid or cell list from a YAML config, resumable
  layer_sweep.py    quantize one GDN layer's state at a time; decay statistics per layer
  run_gsm8k.py      GSM8K in thinking mode under a quantized cache
  run_niah.py       single-needle NIAH at 4k / 16k context
  check_hooks.py    verify the hooks fire on a real checkpoint (PASS/FAIL against layer_types)
  vram_check.py     VRAM estimate and batch-size reduction (used for 9B)
  estimate_time.py  rough wall-clock plan for all configs
  run_round1.sh, run_round2.sh, run_all.sh   sequential runners with per-step report, commit and push
  sync_results.sh, install_sync_launchd.sh, uninstall_sync_launchd.sh   Mac-side result sync (launchd, 30 min)
analysis/
  common.py    loading, seed aggregation, paired bootstrap CI, Spearman/Pearson with permutation p-values
  figures.py   all figures (PDF + PNG under results/figures/<stage>/)
  report.py    bilingual stage report (results/reports/<stage>.md) from results files only
notebooks/
  01_setup_colab.ipynb          Colab entry point: tokens, GPU selection, hook check, background run_all
  02_equations_walkthrough.ipynb paper equations (1)-(5) checked against the model and the code
  03_results_dashboard.ipynb    shows every stage report and its figures inline
tests/                           52 tests, CPU only, no checkpoint downloads
docs/                            experiment_log.md, decision_log.md, related_work.md
paper/figures/                   figures from notebook 02 (results figures stay under results/figures/)
results/                         jsonl results, reports/, figures/ (tracked in git)
```

## How the hooks work

`transformers` stores both memories in a `DynamicCache` whose layers are either
`DynamicLayer` (attention), `LinearAttentionLayer` (GDN / Mamba-2), or, for Falcon-H1,
`LinearAttentionAndFullAttentionLayer` (both). We patch two methods:

* `DynamicLayer.update(k, v)`: quantize the new K/V per token before they are appended
* `LinearAttentionLayer.update_recurrent_state(S)`: quantize `S` before it is stored

so the next decode step reads the rounded state. No model code is touched.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8, state_error_feedback=True)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

One detail matters for how results should be read. During prefill the chunked kernel keeps the
intermediate states in full precision and only the final state is quantized. State error
therefore builds up only during token-by-token decoding, which is why `decode_nll` feeds the
continuation one token at a time.

## Checking the hooks

On GPU the `kernels` hub can replace `Qwen3_5GatedDeltaNet.forward` wholesale, which
silently skips the state hook. Check before every experiment (the runners do this for Falcon-H1):

```bash
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
# if it reports FAIL with state_calls=0:
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels
```

`--no-hub-kernels` sets `USE_HUB_KERNELS=NO` before transformers is imported and passes
`from_pretrained(use_kernels=False)`. The reasoning is in `docs/decision_log.md`.

## GPU and dtype

`load_model(dtype="auto")` picks bf16 on compute capability 8.0 and above (A100, H100, L4) and on Apple MPS,
fp16 on older CUDA GPUs (T4), and fp32 on CPU; `HQ_DTYPE` or `--dtype` overrides it. On T4 the experiments
therefore run in fp16. The GDN recurrence, the decay computation and fake quantization run in fp32 inside
transformers and this code, and `decode_nll` raises with a hint to rerun with `--dtype fp32` if a non-finite
NLL appears. The equations notebook (02) uses fp32 on T4 on purpose. On T4 the fast kernels
(flash-linear-attention, causal-conv1d) are skipped and transformers' pure-torch fallback is used.

## Colab workflow

1. Open the badge above, or in Colab use `File > Open notebook > GitHub` and paste the repo URL. Pick a T4, L4, A100 or H100 runtime.
2. **0. Token setup**: a private repo needs `GH_TOKEN`, model downloads use `HF_TOKEN`. Store them under those names in Colab Secrets (key icon in the left sidebar). If a token is missing, the cell only warns and continues, but results can then only be copied to Drive, not pushed.
3. **1. GPU selection**: detects the card and sets the dtype and whether to install the fast kernels.
4. Run the **hook check** cell (expects PASS), optionally mount Drive, then run the **background run_all** cell:
   ```
   nohup bash scripts/run_all.sh > results/run_all_nohup.log 2>&1 &
   ```
   It runs round 1 then round 2. After every step the runner writes `results/reports/<stage>.md` and the figures, commits `results/` and pushes (push failures go to `results/push_failures.log`, and `results/` is also rsynced to Drive if mounted). Every step is resumable: rerun the same cell after a disconnect and finished rows are skipped.
5. Watch progress with the **progress** cell or `notebooks/03_results_dashboard.ipynb`. On a Mac, `scripts/install_sync_launchd.sh` pulls the repo every 30 minutes and copies `results/` to `~/Downloads/hybridquant_results/<date>/`.

Running the notebook cells by hand (hook check, WikiText NLL, error-vs-position curve, small grid) is for debugging only.

## Reproducing the paper

Round 1 and round 2 (resumable, each step followed by report, commit and push):

```bash
bash scripts/run_round1.sh      # smoke_0.8b, grid_0.8b, superadd_0.8b, layer_sweep (0.8B), grid_4b_core
bash scripts/run_round2.sh      # r2_ablation_0.8b, r2_c4_0.8b, r2_niah_0.8b, r2_scale_2b, r2_gsm8k_4b, r2_niah_4b, r2_scale_9b, r2_falcon_1.5b
bash scripts/run_all.sh         # both rounds, then the summary report
```

Round 3 (27B on an external H100) is not defined yet; there is no command for it.

Single configs and tasks:

```bash
python scripts/run_grid.py --config configs/grid_0.8b.yaml
python scripts/layer_sweep.py --model Qwen/Qwen3.5-0.8B --state-bits 4 --n-docs 8
python scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml
python scripts/run_niah.py --config configs/r2_niah_0.8b.yaml
```

Tables and figures are regenerated from the result files only:

```bash
python analysis/report.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl --out results/reports/grid_0.8b.md
python analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md
python analysis/figures.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl       # figures only
```

Outputs: tables go into `results/reports/<stage>.md` (bilingual, with paired bootstrap CIs and memory per cell),
figures into `results/figures/<stage>/` as PDF and PNG. The equation figures from notebook 02 are saved under
`paper/figures/`. Copying result tables and figures into the paper source is a manual step for now; there is no
`paper/tables/` directory yet.

## Local checks on a Mac

CPU only (Intel Mac, or when you do not want MPS):

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest
pip install -e .
pytest tests -q
```

Apple Silicon (MPS): install the default torch wheel instead; `load_model` then picks MPS automatically, and
Qwen3.5-0.8B runs on it as is (the 0.8B smoke grid takes about 5 s per cell on an M3 Pro).

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch
pip install -r requirements.txt pytest
pip install -e .
pytest tests -q
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
```

`requirements.txt` pins transformers to the commit used for rounds 1 and 2 (5.19.0.dev0, `66e0f4a`); it will be
re-pinned at round 3. The optional fast kernels (flash-linear-attention, causal-conv1d) are Linux/CUDA only.

## Status

Current milestone: **M2** (bit-allocation grids, error feedback, superadditivity, layer sweep; round 1 running on Colab).
See [docs/experiment_log.md](docs/experiment_log.md) for results and [docs/decision_log.md](docs/decision_log.md) for design decisions.

## Data

- WikiText-2 (raw): Merity et al., "Pointer Sentinel Mixture Models", 2016. Loaded from `Salesforce/wikitext`, config `wikitext-2-raw-v1`; test split for NLL, train split as NIAH filler.
- C4 (en): Raffel et al., "Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer", 2020. Loaded from `allenai/c4`, validation shard 0, first 5000 documents.
- GSM8K: Cobbe et al., "Training Verifiers to Solve Math Word Problems", 2021. Loaded from `openai/gsm8k`, config `main`, test split; a fixed subset of 200 problems (`results/gsm8k_subset200.json`).

## Citation

```bibtex
@misc{hybridquant2026,
  title   = {Where Do the Bits Go? Joint KV-Cache and Recurrent-State Quantization for Hybrid Linear-Attention LLMs},
  author  = {Minseok (skytinstone)},
  year    = {2026},
  eprint  = {TODO},
  archivePrefix = {arXiv},
  url     = {https://github.com/skytinstone/hybridquant}
}
```

See also `CITATION.cff`.

## License

MIT, see [LICENSE](LICENSE).
