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

Prior work quantizes each one on its own. This repo asks a different question: **given a fixed total memory budget, how should the bits be split between the two, and how do their errors interact?**

Everything here is **fake quantization** (quantize, then dequantize, in plain PyTorch). There are no custom kernels. We measure accuracy and memory, not latency.

## Layout

```
src/hybridquant/
  quant.py    fake-quant primitives (asym/sym, group, stochastic rounding)
  hooks.py    store-time hooks on transformers cache layers (KV write, state write) + hook coverage check
  eval.py     decode-time NLL (teacher-forced), generation + behaviour stats, dtype resolution
scripts/
  check_hooks.py  check that the hooks fire on a real checkpoint (PASS/FAIL against layer_types)
  run_grid.py     bit-allocation grid on WikiText-2 -> results/*.jsonl
notebooks/
  01_setup_colab.ipynb  Colab setup + smoke test
tests/
  test_smoke_cpu.py     hook and quantization checks on a tiny random Qwen3.5, CPU only
docs/
  experiment_log.md decision_log.md related_work.md
```

## How the hooks work

`transformers` stores both memories in a `DynamicCache` whose layers are either
`DynamicLayer` (attention) or `LinearAttentionLayer` (GDN/Mamba). We patch two methods:

* `DynamicLayer.update(k, v)`: quantize the new K/V per token before they are appended
* `LinearAttentionLayer.update_recurrent_state(S)`: quantize `S` before it is stored

so the next decode step reads the rounded state. No model code is touched.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

One detail matters here. During prefill the chunked kernel keeps the intermediate
states in full precision and only the final state is quantized. State error
therefore builds up only during token-by-token decoding, which is why
`decode_nll` feeds the continuation one token at a time.

## Checking the hooks

On GPU the `kernels` hub can replace `Qwen3_5GatedDeltaNet.forward` wholesale, which
silently skips the state hook. Check before every experiment:

```bash
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
# if it reports FAIL with state_calls=0:
python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels
```

`--no-hub-kernels` sets `USE_HUB_KERNELS=NO` before transformers is imported and passes
`from_pretrained(use_kernels=False)`. The reasoning is in `docs/decision_log.md`.

## Colab workflow

1. Open the badge above, or in Colab use `File > Open notebook > GitHub` and paste the repo URL.
2. Enter tokens in the **0. Token setup** cell. A private repo needs `GH_TOKEN`; model downloads use `HF_TOKEN`. Store them under the same names in Colab Secrets (key icon in the left sidebar) and the cell picks them up. Missing tokens only produce a warning.
3. The **1. GPU selection** cell detects the card and sets the dtype (fp16 on T4, bf16 otherwise) and whether to install the fast kernels (skipped on T4, which uses the pure-torch fallback). T4, L4, A100 and H100 are supported.
4. Results go to `results/*.jsonl`. Commit them with the last cell or copy them to Drive.
5. To edit the notebook in Colab and save it back: `File > Save a copy in GitHub`.

## Local checks (Mac, CPU or MPS)

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install torch --index-url https://download.pytorch.org/whl/cpu
pip install "transformers @ git+https://github.com/huggingface/transformers.git@main" accelerate datasets pytest
pip install -e .
pytest tests -q
```

On Apple Silicon the 0.8B model runs on MPS as is; `load_model` picks it up by default.

## Status

See `docs/experiment_log.md`. Current milestone: **M1** (environment + hooks + baseline).
