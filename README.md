# hybridquant

Joint KV-cache and recurrent-state quantization for hybrid linear-attention LLMs (Qwen3.5, Falcon-H1).

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/skytinstone/hybridquant/blob/main/notebooks/01_setup_colab.ipynb)

## What this is

Hybrid LLMs keep two kinds of memory during inference:

| memory | produced by | shape | grows with context? |
|---|---|---|---|
| KV cache | softmax attention layers | `[B, H_kv, T, d]` | yes |
| recurrent state | Gated DeltaNet / Mamba-2 layers | `[B, H, d_k, d_v]` | no (fixed size) |

Prior work quantizes each one in isolation. This repo asks: **under a fixed total memory budget, how should bits be split between the two, and how do their errors interact?**

Everything is **fake quantization** (quantize then dequantize in PyTorch). No custom kernels. We measure accuracy and memory, not latency.

## Layout

```
src/hybridquant/
  quant.py    fake-quant primitives (asym/sym, group, stochastic rounding)
  hooks.py    store-time hooks on transformers cache layers (KV write, state write)
  eval.py     decode-time NLL (teacher-forced), generation + behaviour stats
scripts/
  run_grid.py bit-allocation grid on WikiText-2 -> results/*.jsonl
notebooks/
  01_setup_colab.ipynb  one-click Colab setup + smoke test
docs/
  experiment_log.md decision_log.md related_work.md
```

## How the hooks work

`transformers` stores both memories in a `DynamicCache` whose layers are either
`DynamicLayer` (attention) or `LinearAttentionLayer` (GDN/Mamba). We patch:

* `DynamicLayer.update(k, v)` -> quantize new K/V per token before append
* `LinearAttentionLayer.update_recurrent_state(S)` -> quantize `S` before it is stored

so the next decode step reads the rounded state. No model code is modified.

```python
from hybridquant import HybridQuantConfig, load_model, decode_nll
model, tok = load_model("Qwen/Qwen3.5-0.8B")
cfg = HybridQuantConfig.uniform(kv_bits=4, state_bits=8)
ids = tok(text, return_tensors="pt")["input_ids"]
r = decode_nll(model, ids, prefill_len=256, cfg=cfg, max_decode=256)
print(r.ppl, r.stats)
```

Important semantic: during prefill the chunked kernel keeps full-precision
intermediate states; only the final state is quantized. State error therefore
accumulates only during token-by-token decoding, which is why `decode_nll`
feeds the continuation one token at a time.

## Colab workflow

1. Open the badge above (or `File > Open notebook > GitHub` in Colab and paste the repo URL).
2. First cell clones the repo. For a **private** repo, store a GitHub token in Colab
   `Secrets` (left sidebar, key icon) under the name `GH_TOKEN`; the notebook reads it
   with `google.colab.userdata.get("GH_TOKEN")`.
3. Results are written to `results/*.jsonl`. Commit them back with the last cell
   (`git add results && git commit && git push`) or copy to Drive.
4. To edit the notebook in Colab and save back: `File > Save a copy in GitHub`.

## Status

See `docs/experiment_log.md`. Current milestone: **M1** (environment + hooks + baseline).
