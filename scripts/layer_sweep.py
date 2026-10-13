"""Per-layer state sensitivity: quantize the recurrent state of ONE GDN layer at a time.

For each linear_attention layer i, run the teacher-forced decode with
HybridQuantConfig(state=QuantSpec(state_bits), gdn_layers={i}) and record the NLL
increase over the fp baseline. During the fp pass, forward hooks on the layer's
``in_proj_a`` / ``in_proj_b`` collect ``a`` and ``b``; the module itself computes
    g    = -exp(A_log) * softplus(a + dt_bias)        (log decay, <= 0)
    beta = sigmoid(b)
(installed transformers/models/qwen3_5/modeling_qwen3_5.py, Qwen3_5GatedDeltaNet.forward),
so we recompute them the same way and report mean alpha = exp(g) and the error
half-life n_half = ln(0.5) / mean(g) tokens (from ||M_j||_2 = alpha_j, paper eq. 4).

    python scripts/layer_sweep.py --model Qwen/Qwen3.5-0.8B --state-bits 4 --n-docs 8

Output: results/layer_sweep_<model>.jsonl, one row per layer plus one fp baseline row
(layer_idx = -1). Resumable on (model, state_bits, layer_idx).
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time

import torch
import torch.nn.functional as F

from hybridquant import HybridQuantConfig, QuantSpec, decode_nll_batch, load_model, run_metadata
from hybridquant.data import load_docs


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--dtype", default="auto")
    p.add_argument("--state-bits", type=int, default=4)
    p.add_argument("--state-group", type=int, default=128)
    p.add_argument("--prefill", type=int, default=256)
    p.add_argument("--decode", type=int, default=256)
    p.add_argument("--n-docs", type=int, default=8)
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset", default="wikitext2")
    p.add_argument("--out", default=None, help="default results/layer_sweep_<model>.jsonl")
    return p.parse_args(argv)


def gdn_layers(model) -> list[int]:
    cfg = model.config.get_text_config(decoder=True)
    return [i for i, t in enumerate(cfg.layer_types) if t == "linear_attention"]


def collect_decay_stats(model, docs, batch_size: int) -> dict[int, dict]:
    """Run a plain forward over the documents with hooks on in_proj_a / in_proj_b and return per-layer stats."""
    captured: dict[int, dict] = {}
    handles = []
    for i in gdn_layers(model):
        m = model.model.layers[i].linear_attn
        handles.append(m.in_proj_a.register_forward_hook(
            lambda mod, inp, out, i=i: captured.setdefault(i, {}).setdefault("a", []).append(out.detach().float().cpu())))
        handles.append(m.in_proj_b.register_forward_hook(
            lambda mod, inp, out, i=i: captured.setdefault(i, {}).setdefault("b", []).append(out.detach().float().cpu())))
    device = next(model.parameters()).device
    with torch.no_grad():
        for j in range(0, len(docs), batch_size):
            ids = torch.cat(docs[j:j + batch_size], 0).to(device)
            model(input_ids=ids, use_cache=False, logits_to_keep=1)
    for h in handles:
        h.remove()
    stats = {}
    for i, d in captured.items():
        m = model.model.layers[i].linear_attn
        a = torch.cat(d["a"], 0); b = torch.cat(d["b"], 0)                       # [B, T, H]
        g = -m.A_log.float().exp().cpu() * F.softplus(a + m.dt_bias.float().cpu())
        alpha = g.exp()
        beta = torch.sigmoid(b)
        mean_g = g.mean().item()
        stats[i] = {
            "mean_alpha": alpha.mean().item(), "alpha_p10": alpha.quantile(0.1).item(),
            "alpha_p50": alpha.quantile(0.5).item(), "alpha_p90": alpha.quantile(0.9).item(),
            "mean_log_alpha": mean_g, "n_half": math.log(0.5) / mean_g if mean_g < 0 else float("inf"),
            "mean_beta": beta.mean().item(),
        }
    return stats


def evaluate(model, docs, prefill, decode, batch_size, cfg, seed):
    per_doc = []
    for j in range(0, len(docs), batch_size):
        ids = torch.cat(docs[j:j + batch_size], 0)
        per_doc += [r.mean_nll for r in decode_nll_batch(model, ids, prefill, cfg, max_decode=decode, noise_seed=seed)]
    mean = sum(per_doc) / len(per_doc)
    return mean, per_doc


def main(argv=None):
    args = parse_args(argv)
    out = args.out or f"results/layer_sweep_{args.model.split('/')[-1]}.jsonl"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    done = set()
    if os.path.exists(out):
        for line in open(out, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                done.add((r["model"], r["state_bits"], r["layer_idx"]))

    meta = run_metadata()
    model, tok = load_model(args.model, dtype=args.dtype)
    docs = load_docs(args.dataset, tok, args.n_docs, args.prefill + args.decode, args.seed)
    layers = gdn_layers(model)
    print(f"{len(layers)} GDN layers: {layers}")

    decay = collect_decay_stats(model, docs, args.batch_size)
    base_cfg = HybridQuantConfig()
    t0 = time.time()
    base_mean, base_docs = evaluate(model, docs, args.prefill, args.decode, args.batch_size, base_cfg, args.seed)
    print(f"fp baseline nll={base_mean:.4f} ({time.time() - t0:.0f}s)")
    common = {"model": args.model, "dataset": args.dataset, "state_bits": args.state_bits, "state_group": args.state_group,
              "prefill": args.prefill, "decode": args.decode, "n_docs": len(docs), "seed": args.seed, "meta": meta}

    with open(out, "a") as f:
        if (args.model, args.state_bits, -1) not in done:
            f.write(json.dumps({**common, "layer_idx": -1, "label": "fp", "mean_nll": base_mean, "delta_nll": 0.0,
                                "per_doc_nll": base_docs, "per_doc_delta": [0.0] * len(base_docs)}) + "\n"); f.flush()
        for i in layers:
            if (args.model, args.state_bits, i) in done:
                print(f"skip (done) layer {i}")
                continue
            cfg = HybridQuantConfig(state=QuantSpec(args.state_bits, args.state_group), gdn_layers=frozenset({i}))
            t0 = time.time()
            mean, per_doc = evaluate(model, docs, args.prefill, args.decode, args.batch_size, cfg, args.seed)
            row = {**common, "layer_idx": i, "label": cfg.label(), "mean_nll": mean, "delta_nll": mean - base_mean,
                   "per_doc_nll": per_doc, "per_doc_delta": [a - b for a, b in zip(per_doc, base_docs)],
                   **decay[i], "sec": time.time() - t0}
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"layer {i:2d}: delta_nll={row['delta_nll']:+.4f} mean_alpha={row['mean_alpha']:.4f} "
                  f"n_half={row['n_half']:.2f} mean_beta={row['mean_beta']:.3f} ({row['sec']:.0f}s)")
    print("done:", out)


if __name__ == "__main__":
    main()
