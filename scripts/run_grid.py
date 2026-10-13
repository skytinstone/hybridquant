"""Bit-allocation grid: decode-time NLL under KV x state bits, one jsonl row per (seed, kv_bits, state_bits).

Settings come from a YAML file (configs/*.yaml), explicit CLI arguments override it,
and anything not given falls back to hybridquant.config.GRID_DEFAULTS.

    python scripts/run_grid.py --config configs/smoke_0.8b.yaml
    python scripts/run_grid.py --config configs/grid_0.8b.yaml --seeds 0 --n-docs 4      # override
    python scripts/run_grid.py --model Qwen/Qwen3.5-0.8B --kv-bits 16 4 --state-bits 16 4 # CLI only

Rough cost: each (cell, doc, seed) = prefill + `decode` single-token forwards. On a 0.8B
model with decode=256 that is about 5 s on an L4 or an M3 Pro, so the 4x3 smoke grid over
8 docs is roughly 10 min. Larger models scale roughly with parameter count.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from hybridquant import HybridQuantConfig, decode_nll, load_model, run_metadata
from hybridquant.config import ALLOWED_DATASETS, iter_runs, load_config, resolve_settings
from hybridquant.data import load_docs


def parse_args(argv=None):
    S = argparse.SUPPRESS   # absent unless given, so only explicit flags override the config
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None, help="YAML settings file (configs/*.yaml)")
    p.add_argument("--model", default=S)
    p.add_argument("--dtype", default=S, help="bf16 | fp16 | fp32 | auto")
    p.add_argument("--kv-bits", dest="kv_bits", type=int, nargs="+", default=S)
    p.add_argument("--state-bits", dest="state_bits", type=int, nargs="+", default=S)
    p.add_argument("--kv-group", dest="kv_group", type=int, default=S)
    p.add_argument("--state-group", dest="state_group", type=int, default=S)
    p.add_argument("--state-sym", dest="state_symmetric", action="store_true", default=S)
    p.add_argument("--state-sr", dest="state_stochastic", action="store_true", default=S,
                   help="stochastic rounding for state")
    p.add_argument("--seeds", type=int, nargs="+", default=S)
    p.add_argument("--seed", type=int, default=S, help="single seed (same as --seeds N)")
    p.add_argument("--n-docs", dest="n_docs", type=int, default=S)
    p.add_argument("--prefill", type=int, default=S)
    p.add_argument("--decode", type=int, default=S)
    p.add_argument("--step", type=int, default=S)
    p.add_argument("--dataset", choices=ALLOWED_DATASETS, default=S)
    p.add_argument("--out", default=S)
    return p.parse_args(argv)


def settings_from_args(args) -> tuple[dict, str | None]:
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    if "seed" in overrides:
        overrides["seeds"] = [overrides.pop("seed")]
    config = load_config(args.config) if args.config else None
    return resolve_settings(config, overrides), args.config


def main(argv=None):
    args = parse_args(argv)
    s, config_path = settings_from_args(args)
    print("settings:", json.dumps(s))

    os.makedirs(os.path.dirname(s["out"]) or ".", exist_ok=True)
    meta = run_metadata()
    meta["config"] = config_path
    print("run metadata:", json.dumps(meta))

    model, tok = load_model(s["model"], dtype=s["dtype"])
    dtype_name = str(next(model.parameters()).dtype).replace("torch.", "")
    print(f"model on {next(model.parameters()).device} dtype={dtype_name}")

    n_tokens = s["prefill"] + s["decode"]
    docs_by_seed: dict[int, list] = {}
    with open(s["out"], "a") as f:
        for run in iter_runs(s):
            seed, kvb, sb = run["seed"], run["kv_bits"], run["state_bits"]
            if seed not in docs_by_seed:
                docs_by_seed[seed] = load_docs(s["dataset"], tok, s["n_docs"], n_tokens, seed)
                print(f"seed {seed}: loaded {len(docs_by_seed[seed])} {s['dataset']} docs of {n_tokens} tokens")
            docs = docs_by_seed[seed]
            torch.manual_seed(seed)   # stochastic rounding, if enabled
            cfg = HybridQuantConfig.uniform(kvb, sb, s["kv_group"], s["state_group"],
                                            state_symmetric=s["state_symmetric"], state_stochastic=s["state_stochastic"])
            t0 = time.time()
            per_doc = []
            for d_i, ids in enumerate(docs):
                r = decode_nll(model, ids, s["prefill"], cfg, max_decode=s["decode"], step=s["step"])
                per_doc.append(r)
                if d_i == 0 and seed == s["seeds"][0]:
                    print("  hooks:", r.stats)
            mean = sum(r.mean_nll for r in per_doc) / len(per_doc)
            var = sum((r.mean_nll - mean) ** 2 for r in per_doc) / max(1, len(per_doc) - 1)
            n_pos = min(len(r.nll_per_pos) for r in per_doc)
            row = {
                "model": s["model"], "dtype": dtype_name, "dataset": s["dataset"],
                "label": cfg.label(), "kv_bits": kvb, "state_bits": sb,
                "kv_group": s["kv_group"], "state_group": s["state_group"],
                "state_sym": s["state_symmetric"], "state_sr": s["state_stochastic"],
                "prefill": s["prefill"], "decode": s["decode"], "step": s["step"],
                "n_docs": len(docs), "seed": seed,
                "mean_nll": mean, "std_nll": var ** 0.5,
                "nll_curve": [sum(r.nll_per_pos[i] for r in per_doc) / len(per_doc) for i in range(n_pos)],
                "sec": time.time() - t0,
                "config": config_path,
                "meta": meta,
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"seed {seed} {cfg.label():28s} nll={mean:.4f} ±{var**0.5:.4f}  ({row['sec']:.0f}s)")


if __name__ == "__main__":
    main()
