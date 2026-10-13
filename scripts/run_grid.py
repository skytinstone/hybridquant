"""Bit-allocation grid: decode-time NLL on WikiText-2 under KV x state bits.

Example (Colab, L4):
    python scripts/run_grid.py --model Qwen/Qwen3.5-0.8B --kv-bits 16 8 4 2 \
        --state-bits 16 8 4 --n-docs 8 --prefill 256 --decode 256 --out results/grid_0.8b.jsonl

Rough cost: each (config, doc) pair = prefill + `decode` single-token forwards.
On a 0.8B model with decode=256 that is ~5-10 s on an L4; a 4x3 grid over 8 docs
is ~10-15 min. 4B is ~3-4x slower.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from hybridquant import HybridQuantConfig, decode_nll, load_model


def load_wikitext_docs(tok, n_docs: int, min_tokens: int, seed: int = 0):
    from datasets import load_dataset

    ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 2000]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=g).tolist()
    docs = []
    for i in order:
        ids = tok(texts[i], return_tensors="pt")["input_ids"]
        if ids.shape[1] >= min_tokens:
            docs.append(ids[:, :min_tokens])
        if len(docs) >= n_docs:
            break
    return docs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--kv-bits", type=int, nargs="+", default=[16, 8, 4, 2])
    p.add_argument("--state-bits", type=int, nargs="+", default=[16, 8, 4])
    p.add_argument("--kv-group", type=int, default=128)
    p.add_argument("--state-group", type=int, default=128)
    p.add_argument("--state-sym", action="store_true")
    p.add_argument("--state-sr", action="store_true", help="stochastic rounding for state")
    p.add_argument("--n-docs", type=int, default=8)
    p.add_argument("--prefill", type=int, default=256)
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--step", type=int, default=1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", default="results/grid.jsonl")
    p.add_argument("--dtype", default="auto", help="bf16 | fp16 | fp32 | auto (see hybridquant.resolve_dtype)")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    model, tok = load_model(args.model, dtype=args.dtype)
    dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"model on {next(model.parameters()).device} dtype={dtype_name}")
    docs = load_wikitext_docs(tok, args.n_docs, args.prefill + args.decode, args.seed)
    print(f"loaded {len(docs)} docs of {args.prefill + args.decode} tokens")

    with open(args.out, "a") as f:
        for kvb in args.kv_bits:
            for sb in args.state_bits:
                cfg = HybridQuantConfig.uniform(kvb, sb, args.kv_group, args.state_group,
                                                state_symmetric=args.state_sym, state_stochastic=args.state_sr)
                t0 = time.time()
                per_doc = []
                for d_i, ids in enumerate(docs):
                    r = decode_nll(model, ids, args.prefill, cfg, max_decode=args.decode, step=args.step)
                    per_doc.append(r)
                    if d_i == 0:
                        print("  hooks:", r.stats)
                mean = sum(r.mean_nll for r in per_doc) / len(per_doc)
                var = sum((r.mean_nll - mean) ** 2 for r in per_doc) / max(1, len(per_doc) - 1)
                row = {
                    "model": args.model, "dtype": dtype_name, "label": cfg.label(), "kv_bits": kvb, "state_bits": sb,
                    "kv_group": args.kv_group, "state_group": args.state_group,
                    "state_sym": args.state_sym, "state_sr": args.state_sr,
                    "prefill": args.prefill, "decode": args.decode, "n_docs": len(docs), "seed": args.seed,
                    "mean_nll": mean, "std_nll": var ** 0.5,
                    "nll_curve": [sum(r.nll_per_pos[i] for r in per_doc) / len(per_doc)
                                  for i in range(min(len(r.nll_per_pos) for r in per_doc))],
                    "sec": time.time() - t0,
                }
                f.write(json.dumps(row) + "\n"); f.flush()
                print(f"{cfg.label():28s} nll={mean:.4f} ±{var**0.5:.4f}  ({row['sec']:.0f}s)")


if __name__ == "__main__":
    main()
