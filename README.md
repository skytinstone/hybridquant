# hybridquant

Code, configurations and raw results for the paper **"Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs"** (Minseok Shin, 2026).

English | [한국어](README.ko.md)

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## What this is

Hybrid LLMs keep two kinds of memory during inference:

| memory | produced by | shape | grows with context? |
|---|---|---|---|
| KV cache | softmax attention layers | `[B, H_kv, T, d]` | yes |
| recurrent state | Gated DeltaNet / Mamba-2 layers | `[B, H, d_k, d_v]` | no (fixed size) |

Softmax attention keeps every past token's key and value and looks them up by similarity, so its memory grows with the context but it can retrieve any token exactly. Linear attention (Gated DeltaNet, Mamba-2) folds the past into one fixed-size matrix that is decayed and rewritten at every token, so its memory is constant but a rounding error written into it is fed back into the next step instead of staying put. This difference in how an error lives on is why the two memories respond so differently to quantization.

Prior work quantizes each memory in isolation. This repo asks a different question: **given a fixed total memory budget, how should the bits be split between the two, and how do their errors interact?**

Everything here is **fake quantization** (quantize, then dequantize, in plain PyTorch). There are no custom kernels. We measure accuracy and memory, not latency.

## Main results

Qwen3.5 (Gated DeltaNet hybrid) at 0.8B, 2B, 4B and 9B, with Falcon-H1-1.5B (Mamba-2 hybrid) as a control. All numbers are measured, 3 seeds x 20 WikiText-2 windows unless stated.

- **The asymmetry is large and stable.** 4-bit KV costs at most 0.03 nats at every size, while a 4-bit state costs 0.3 to 1.8 nats. The state needs 8 bits to come within 0.02 nats of bf16.
- **The delta rule is the likely cause.** In Falcon-H1 a 4-bit Mamba-2 state costs nothing measurable (+0.004 nats, not significant). The comparison supports but does not isolate the delta rule, since the two model families differ in more than the update rule.
- **The two errors are nearly additive.** The joint 4-bit/4-bit cell exceeds the sum of its parts by only +0.03 to +0.05 nats at small scale and not at all at 9B, so the two memories can be allocated almost independently.
- **Rounding is idempotent.** Replacing state rounding with Gaussian noise of equal RMS costs nine times more. Stochastic rounding is five times worse than round-to-nearest for the same reason.
- **One layer dominates.** The first GDN layer carries 38% (0.8B) and 80% (4B) of the summed single-layer sensitivity, unrelated to its decay half-life.
- **The limiting memory depends on the task.** Retrieval (NIAH) is bounded by the KV cache. Reasoning (GSM8K, thinking mode) is bounded by the state: an 8-bit state, lossless on NLL, drops Qwen3.5-4B from 0.600 to 0.345.
- **Error feedback fixes the state.** It restores the 8-bit state on GSM8K to 0.600, identical to bf16, and stops the state error from growing over 1,024 decode tokens.

Recipe on Qwen3.5-4B (memory relative to bf16 at 512 tokens):

| setting | memory | ΔNLL (nats) | GSM8K | NIAH |
|---|---|---|---|---|
| bf16 | 100% | 0 | 0.600 | 1.00 |
| KV4 + S6, error feedback | 32% | +0.044 | 0.555 (n.s.) | 1.00 |
| KV4 + S8, error feedback | 40% | −0.001 | not run | 1.00 |

Full tables are in `paper/tables/` and the per-stage reports in `results/reports/`.

## What we measure

- **Decode-time NLL.** The prompt is prefilled in one pass and the continuation is fed one token at a time under a quantized cache (teacher forcing). Only this path lets recurrent-state error accumulate, so it is the primary metric. A plain forward-pass perplexity cannot see the state effect.
- **Round-to-nearest versus error feedback.** Every state cell is run twice: plain RTN, and RTN with error feedback (the store residual is carried into the next store).
- **Noise-matched control.** For the additivity check, one side's quantization is replaced by Gaussian noise with the same error RMS. This separates error magnitude from quantization structure.
- **Statistics.** 20 documents x 3 seeds per cell. Seeds select disjoint document blocks. Differences against bf16 are reported with paired per-document bootstrap 95% CIs (1,000 resamples). GSM8K differences are tested with an exact McNemar test on paired problems.
- **Batched decode does not change results.** `tests/test_quant_ext.py` and `tests/test_falcon_batch.py` check that batched and single-document NLLs match.

## Layout

```
src/hybridquant/
  quant.py     fake-quant primitives: asym/sym, group, stochastic rounding, noise match, error-feedback flag
  hooks.py     store-time hooks on transformers cache layers (KV write, state write, EF residuals), coverage check
  compat.py    two Falcon-H1 work-arounds for transformers (see "Falcon-H1" below)
  eval.py      batched teacher-forced decode NLL, batched generation, answer scoring, dtype resolution, model loading
  data.py      WikiText-2 / C4 as one token stream cut into disjoint windows per seed
  config.py    YAML schema, CLI overrides, grid or explicit cells, EF crossing, run list
  budget.py    cache sizes and bit budgets from a model config
  meta.py      run metadata (git commit, library versions, GPU, patches applied) stored in every result row
configs/       one YAML per stage: round 1 (smoke, grid, superadd, grid_4b_core), round 2 (r2_*), round 3 (r3_*)
scripts/
  run_grid.py       decode-NLL grid or cell list from a YAML config, resumable
  layer_sweep.py    quantize one GDN layer's state at a time; decay statistics per layer
  run_gsm8k.py      GSM8K in thinking mode under a quantized cache
  run_niah.py       single-needle NIAH at 4k / 16k context
  run_round1.sh, run_round2.sh, run_round3.sh, run_all.sh   sequential runners with per-step report, commit and push
  finish_run.py     wait for runners, dedupe, regenerate all reports, copy to Drive, commit and push
  dedupe_results.py remove duplicate rows from results/*.jsonl
  check_hooks.py    verify the hooks fire on a real checkpoint (PASS/FAIL)
  diag_decode.py    check that cached decoding matches a plain forward pass for a model
  vram_check.py, estimate_time.py, strip_tex_comments.py, Mac sync scripts
analysis/
  common.py, report.py, figures.py          stage reports and figures from results files only
  make_tex_tables.py                         LaTeX tables in paper/tables/ (Table 1 and Appendix C)
  fig_overview.py, fig_memory.py, model_table.py   paper Figures 1 and 2 and the model table
notebooks/     01 Colab entry point, 02 equations walkthrough, 03 results dashboard
tests/         64 tests, CPU only, no checkpoint downloads
docs/          experiment_log.md, decision_log.md, related_work.md, refs_verification.md
paper/         figures/, tables/, refs_verified.bib
results/       jsonl results, reports/, figures/, round logs (tracked in git); results/invalid/ holds excluded runs
```

## How the hooks work

`transformers` stores both memories in a `DynamicCache` whose layers are either `DynamicLayer` (attention), `LinearAttentionLayer` (GDN / Mamba-2), or, for Falcon-H1, `LinearAttentionAndFullAttentionLayer` (both). We patch two methods:

* `DynamicLayer.update(k, v)`: quantize the new K/V per token before they are appended
* `LinearAttentionLayer.update_recurrent_state(S)`: quantize `S` before it is stored

so the next decode step reads the rounded value. No model code is touched.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8, state_error_feedback=True)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

During prefill the chunked kernel keeps the intermediate states in full precision and only the final state is quantized. State error therefore builds up only during token-by-token decoding, which is why `decode_nll` feeds the continuation one token at a time.

## Falcon-H1

Two problems in the transformers version we pin affect Falcon-H1, and `compat.py` works around both. `load_model` applies them automatically, prints which ones it applied, and records them in every result row.

1. `FalconH1RMSNormGated` broadcasts a `[B, dim]` input against a `[B, 1, dim]` gate during single-token decoding, which mixes the rows of a batch.
2. During single-token decoding the Mamba-2 state is updated in place and never passes through `update_recurrent_state`, so the state hook would not see it. The patch re-stores the state through the cache after every token.

Falcon-H1 results produced before these fixes are kept in `results/invalid/` and are not used. See `docs/decision_log.md`.

## Checking the hooks

On GPU the `kernels` hub can replace `Qwen3_5GatedDeltaNet.forward` wholesale, which silently skips the state hook. Check before every experiment:

```bash
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
# if it reports FAIL with state_calls=0:
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels
```

## Environment used for the paper

All paper runs used one NVIDIA A100-SXM4-40GB on Colab Pro+, bf16, PyTorch 2.11 and transformers 5.19.0.dev0 (commit `66e0f4a`, pinned in `requirements.txt`). The GDN and Mamba-2 recurrences ran in transformers' pure-PyTorch path, without flash-linear-attention or causal-conv1d. To reproduce the numbers exactly, skip the fast kernels in the notebook's GPU cell.

`load_model(dtype="auto")` picks bf16 on compute capability 8.0 and above and on Apple MPS, fp16 on older CUDA GPUs (T4), and fp32 on CPU. `HQ_DTYPE` or `--dtype` overrides it.

## Reproducing the paper

The three rounds ran in order on Colab. Each runner is resumable: after every step it writes `results/reports/<stage>.md` and the figures, commits `results/` and pushes.

```bash
bash scripts/run_all.sh        # rounds 1 and 2: 0.8B grid, additivity, layer sweep, 4B core, ablation, C4, 2B, 9B, GSM8K, NIAH
bash scripts/run_round3.sh     # round 3: Falcon-H1 rerun, 4B recipe cells and their NIAH, GSM8K S8+EF,
                               #          4B layer sweep, 1,024-token decoding on 0.8B and 4B
python scripts/finish_run.py   # dedupe, regenerate every report, commit and push
```

Valid runs took about 23 A100 GPU hours in total. Per-stage times are in `paper/tables/compute.md`.

Single stages:

```bash
python scripts/run_grid.py --config configs/grid_0.8b.yaml
python scripts/layer_sweep.py --model Qwen/Qwen3.5-0.8B --state-bits 4 --n-docs 8
python scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml
python scripts/run_niah.py --config configs/r2_niah_4b.yaml
```

Reports, figures and paper tables are regenerated from the result files only:

```bash
python analysis/report.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl --out results/reports/grid_0.8b.md
python analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md
python analysis/make_tex_tables.py      # paper/tables/*.tex
```

## Colab workflow

1. Open the badge above and pick an A100 runtime.
2. **0. Token setup**: store `GH_TOKEN` (push) and `HF_TOKEN` (model downloads) in Colab Secrets. Without a token, results are only copied to Drive.
3. **1. GPU selection** sets the dtype and whether to install the fast kernels.
4. Run the hook check (expects PASS), optionally mount Drive, then start the background runner. After a disconnect, run the same cell again and finished rows are skipped.
5. Watch progress with the monitor cell or `notebooks/03_results_dashboard.ipynb`.

## Local checks on a Mac

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch                 # CPU-only: add --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt pytest
pip install -e .
pytest tests -q
```

On Apple Silicon `load_model` picks MPS automatically and Qwen3.5-0.8B runs as is.

## Status

Complete. All three experiment rounds are finished and every number in the paper is in `results/`. See [docs/experiment_log.md](docs/experiment_log.md) for results and [docs/decision_log.md](docs/decision_log.md) for design decisions.

## Data

- WikiText-2 (raw): Merity et al., 2016. `Salesforce/wikitext`, config `wikitext-2-raw-v1`; test split for NLL, train split as NIAH filler.
- C4 (en): Raffel et al., 2020. `allenai/c4`, validation shard 0, first 5,000 documents.
- GSM8K: Cobbe et al., 2021. `openai/gsm8k`, config `main`, test split; a fixed subset of 200 problems (`results/gsm8k_subset200.json`).

## Citation

```bibtex
@misc{shin2026hybridquant,
  title  = {Joint KV-Cache and Recurrent-State Quantization in Hybrid Linear-Attention LLMs},
  author = {Shin, Minseok},
  year   = {2026},
  url    = {https://github.com/skytinstone/hybridquant}
}
```

See also `CITATION.cff`.

## License

MIT, see [LICENSE](LICENSE).
