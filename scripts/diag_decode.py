"""Decode-path diagnostic: is a model's decode NLL consistent with a plain forward pass?

For a few WikiText-2 windows it prints, in fp (no quantization):

  full       one forward over the whole window, no cache; NLL on positions [P, T)   <- ground truth
  full+bos   same with the tokenizer's BOS prepended (tests BOS sensitivity)
  chunk      prefill P tokens with cache, then the remaining tokens in ONE chunk    <- cache hand-off only
  decode     prefill P tokens with cache, then one token at a time                   <- our measurement path
  decode+bos decode with BOS prepended
  batch      all docs decoded together in one batch (the run_grid path)             <- batching

If full >> expected for the model, the model or tokenization is off (BOS, dtype).
If full is fine but chunk/decode are high, the cache hand-off is broken for this architecture.
If chunk is fine but decode is high, the per-token recurrent path is broken.
If decode is fine but batch is high, batched decoding is broken for this architecture.

    python scripts/diag_decode.py --model tiiuae/Falcon-H1-1.5B-Instruct --n-docs 3
    python scripts/diag_decode.py --model Qwen/Qwen3.5-0.8B --n-docs 3        # control, all rows should agree
"""
from __future__ import annotations

import argparse
import time

import torch

from hybridquant import HybridQuantConfig, decode_nll, load_model
from hybridquant.eval import decode_nll_batch
from hybridquant.data import load_wikitext_docs


@torch.no_grad()
def full_forward_nll(model, ids: torch.Tensor, prefill: int) -> float:
    device = next(model.parameters()).device
    ids = ids.to(device)
    logits = model(input_ids=ids, use_cache=False).logits.float()
    lp = torch.log_softmax(logits[0, prefill - 1:-1], dim=-1)
    tgt = ids[0, prefill:]
    return float(-lp.gather(1, tgt.unsqueeze(1)).mean())


def with_bos(tok, ids: torch.Tensor) -> torch.Tensor | None:
    bos = getattr(tok, "bos_token_id", None)
    if bos is None:
        return None
    return torch.cat([torch.tensor([[bos]], dtype=ids.dtype), ids], dim=1)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--dtype", default="auto")
    p.add_argument("--n-docs", type=int, default=3)
    p.add_argument("--prefill", type=int, default=256)
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args(argv)

    model, tok = load_model(args.model, dtype=args.dtype)
    print(f"model {args.model} dtype={next(model.parameters()).dtype} device={next(model.parameters()).device}")
    print(f"tokenizer bos_token_id={getattr(tok, 'bos_token_id', None)} ({getattr(tok, 'bos_token', None)!r})")
    docs = load_wikitext_docs(tok, n_docs=args.n_docs, min_tokens=args.prefill + args.decode, seed=args.seed)
    docs = [d[:, :args.prefill + args.decode] for d in docs]
    fp = HybridQuantConfig.uniform(16, 16)

    rows = {"full": [], "full+bos": [], "chunk": [], "decode": [], "decode+bos": []}
    t0 = time.time()
    for i, d in enumerate(docs):
        P = args.prefill
        rows["full"].append(full_forward_nll(model, d, P))
        db = with_bos(tok, d)
        rows["full+bos"].append(full_forward_nll(model, db, P + 1) if db is not None else float("nan"))
        rows["chunk"].append(decode_nll(model, d, P, fp, max_decode=args.decode, step=args.decode).mean_nll)
        rows["decode"].append(decode_nll(model, d, P, fp, max_decode=args.decode, step=1).mean_nll)
        rows["decode+bos"].append(decode_nll(model, db, P + 1, fp, max_decode=args.decode, step=1).mean_nll
                                  if db is not None else float("nan"))
        print(f"doc {i}: " + "  ".join(f"{k}={v[-1]:.3f}" for k, v in rows.items()) + f"  ({time.time() - t0:.0f}s)")

    # all docs at once, as run_grid does (batch_size documents of equal length)
    ids = torch.cat(docs, dim=0)
    rows["batch"] = [r.mean_nll for r in decode_nll_batch(model, ids, args.prefill, fp, max_decode=args.decode, step=1)]
    print(f"batched decode (B={ids.shape[0]}): " + "  ".join(f"doc{i}={v:.3f}" for i, v in enumerate(rows["batch"])))

    print("\nmean over docs")
    for k, v in rows.items():
        vals = [x for x in v if x == x]
        print(f"  {k:11s} {sum(vals) / len(vals):.3f}" if vals else f"  {k:11s} n/a")
    print("\nreading: full is the ground truth. chunk and decode should match full within ~0.02 nats.")


if __name__ == "__main__":
    main()
