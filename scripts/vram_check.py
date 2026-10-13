"""Estimate VRAM for a decode run and pick a batch size that fits (halving until it does).

    python scripts/vram_check.py --model Qwen/Qwen3.5-9B --batch-size 10 --prefill 256 --decode 256
    -> prints a JSON line {"batch_size": ..., "est_gb": ..., "free_gb": ...} on the last line.

Weights: safetensors size from the Hub (fallback: parameter count from the config). Activations:
per sequence, the KV cache and recurrent state (fp32 states), logits [B, vocab] in fp32, plus a
safety factor. Without CUDA the requested batch size is returned unchanged.
"""
from __future__ import annotations

import argparse
import json


def weight_bytes(model: str, dtype_bytes: int = 2) -> float:
    try:
        from huggingface_hub import model_info
        info = model_info(model, files_metadata=True)
        total = sum((s.size or 0) for s in info.siblings if s.rfilename.endswith(".safetensors"))
        if total > 0:
            return float(total)
    except Exception:  # noqa: BLE001
        pass
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model).get_text_config(decoder=True)
    h, L, v = c.hidden_size, c.num_hidden_layers, c.vocab_size
    inter = getattr(c, "intermediate_size", 4 * h)
    approx = 2 * v * h + L * (4 * h * h + 3 * h * inter)
    return float(approx * dtype_bytes)


def estimate(model: str, batch_size: int, T: int) -> dict:
    from transformers import AutoConfig
    from hybridquant.budget import cache_sizes
    c = AutoConfig.from_pretrained(model).get_text_config(decoder=True)
    sizes = cache_sizes(c)
    per_seq = (sizes["kv_per_token"] * T * 2 + sizes["state_elems"] * 4 + c.vocab_size * 4 + c.hidden_size * T * 2 * 4)
    w = weight_bytes(model)
    est = w + batch_size * per_seq * 1.5 + 1.0e9
    return {"weights_gb": w / 1e9, "per_seq_gb": per_seq * 1.5 / 1e9, "est_gb": est / 1e9}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", required=True)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--prefill", type=int, default=256)
    p.add_argument("--decode", type=int, default=256)
    args = p.parse_args(argv)
    import torch
    free_gb = None
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info(0)
        free_gb = free / 1e9
    bs = args.batch_size
    e = estimate(args.model, bs, args.prefill + args.decode)
    while free_gb is not None and e["est_gb"] > free_gb and bs > 1:
        bs //= 2
        e = estimate(args.model, bs, args.prefill + args.decode)
    print(f"{args.model}: weights {e['weights_gb']:.1f} GB, per-seq {e['per_seq_gb']:.3f} GB, "
          f"estimate {e['est_gb']:.1f} GB at batch {bs}, free {free_gb if free_gb is None else round(free_gb, 1)} GB")
    if bs != args.batch_size:
        print(f"batch size reduced {args.batch_size} -> {bs}")
    print(json.dumps({"batch_size": bs, "est_gb": round(e["est_gb"], 2), "free_gb": free_gb}))


if __name__ == "__main__":
    main()
