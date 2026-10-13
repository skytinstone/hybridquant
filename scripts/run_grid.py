"""Bit-allocation grid: decode-time NLL under KV x state bits, one jsonl row per (seed, cell, error_feedback).

Settings come from a YAML file (configs/*.yaml), explicit CLI arguments override it,
and anything not given falls back to hybridquant.config.GRID_DEFAULTS. Cells are either
the kv_bits x state_bits grid or an explicit ``cells`` list (see hybridquant.config).

    python scripts/run_grid.py --config configs/smoke_0.8b.yaml
    python scripts/run_grid.py --config configs/grid_0.8b.yaml --seeds 0 --n-docs 4      # override
    python scripts/run_grid.py --model Qwen/Qwen3.5-0.8B --kv-bits 16 4 --state-bits 16 4 # CLI only

Resumable: rows already present in ``out`` (same model, label, seed) are skipped.
Documents are decoded in batches of ``batch_size`` (all documents have the same length).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch

from hybridquant import decode_nll_batch, load_model, run_metadata
from hybridquant.config import (ALLOWED_DATASETS, config_from_run, iter_runs, load_config, resolve_settings,
                                run_key)
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
    p.add_argument("--state-ef", dest="state_error_feedback", type=int, nargs="+", default=S,
                   help="error feedback flags to cross with the cells, e.g. 0 1")
    p.add_argument("--seeds", type=int, nargs="+", default=S)
    p.add_argument("--seed", type=int, default=S, help="single seed (same as --seeds N)")
    p.add_argument("--n-docs", dest="n_docs", type=int, default=S)
    p.add_argument("--prefill", type=int, default=S)
    p.add_argument("--decode", type=int, default=S)
    p.add_argument("--step", type=int, default=S)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=S)
    p.add_argument("--dataset", choices=ALLOWED_DATASETS, default=S)
    p.add_argument("--out", default=S)
    p.add_argument("--no-resume", action="store_true", help="recompute rows that already exist in --out")
    return p.parse_args(argv)


def settings_from_args(args) -> tuple[dict, str | None]:
    overrides = {k: v for k, v in vars(args).items() if k not in ("config", "no_resume")}
    if "seed" in overrides:
        overrides["seeds"] = [overrides.pop("seed")]
    config = load_config(args.config) if args.config else None
    return resolve_settings(config, overrides), args.config


def existing_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    keys.add(run_key(r["model"], r["label"], r.get("seed", 0)))
    return keys


def evaluate_cell(model, docs, s: dict, cfg, seed: int) -> dict:
    """Decode every document in batches; return per-document NLLs and mean/std curves."""
    per_doc, curves = [], []
    hooks = None
    for i in range(0, len(docs), s["batch_size"]):
        ids = torch.cat(docs[i:i + s["batch_size"]], dim=0)
        res = decode_nll_batch(model, ids, s["prefill"], cfg, max_decode=s["decode"], step=s["step"], noise_seed=seed)
        hooks = hooks or res[0].stats
        for r in res:
            per_doc.append(r.mean_nll)
            curves.append(r.nll_per_pos)
    n_pos = min(len(c) for c in curves)
    arr = torch.tensor([c[:n_pos] for c in curves], dtype=torch.float64)
    mean = sum(per_doc) / len(per_doc)
    var = sum((v - mean) ** 2 for v in per_doc) / max(1, len(per_doc) - 1)
    return {
        "mean_nll": mean, "std_nll": var ** 0.5, "per_doc_nll": per_doc,
        "nll_curve": arr.mean(0).tolist(),
        "nll_curve_std": (arr.std(0, unbiased=False) if len(curves) > 1 else torch.zeros(n_pos)).tolist(),
        "hooks": hooks,
    }


def main(argv=None):
    args = parse_args(argv)
    s, config_path = settings_from_args(args)
    print("settings:", json.dumps(s))
    runs = iter_runs(s)
    done = set() if args.no_resume else existing_keys(s["out"])
    print(f"{len(runs)} runs ({len(s['seeds'])} seeds); {sum(run_key(s['model'], config_from_run(r).label(), r['seed']) in done for r in runs)} already in {s['out']}")

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
        for run in runs:
            cfg = config_from_run(run)
            seed = run["seed"]
            if run_key(s["model"], cfg.label(), seed) in done:
                print(f"skip (done) seed {seed} {cfg.label()}")
                continue
            if seed not in docs_by_seed:
                docs_by_seed[seed] = load_docs(s["dataset"], tok, s["n_docs"], n_tokens, seed)
            docs = docs_by_seed[seed]
            torch.manual_seed(seed)
            t0 = time.time()
            ev = evaluate_cell(model, docs, s, cfg, seed)
            row = {
                "model": s["model"], "dtype": dtype_name, "dataset": s["dataset"],
                "label": cfg.label(), **{k: run[k] for k in run if k != "seed"},
                "state_sym": run["state_symmetric"], "state_sr": run["state_stochastic"],
                "prefill": s["prefill"], "decode": s["decode"], "step": s["step"], "batch_size": s["batch_size"],
                "n_docs": len(docs), "seed": seed,
                "mean_nll": ev["mean_nll"], "std_nll": ev["std_nll"], "per_doc_nll": ev["per_doc_nll"],
                "nll_curve": ev["nll_curve"], "nll_curve_std": ev["nll_curve_std"],
                "hooks": ev["hooks"], "sec": time.time() - t0,
                "config": config_path, "meta": meta,
            }
            f.write(json.dumps(row) + "\n"); f.flush()
            print(f"seed {seed} {cfg.label():34s} nll={ev['mean_nll']:.4f} ±{ev['std_nll']:.4f}  ({row['sec']:.0f}s)")
    print("done:", s["out"])


if __name__ == "__main__":
    main()
