"""Summarise results/*.jsonl as kv_bits x state_bits tables, one per (model, dataset).

With several seeds a cell shows "mean ± std" over seeds (std of the per-seed means);
with one seed it shows the mean and the per-document std from the row.

    python analysis/make_tables.py                      # all results/*.jsonl
    python analysis/make_tables.py results/grid_0.8b.jsonl --csv docs/tables
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load_rows(paths: list[str]) -> pd.DataFrame:
    rows = []
    for p in paths:
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                r.setdefault("seed", 0)
                r.setdefault("dataset", "wikitext2")
                r["source"] = os.path.basename(p)
                rows.append(r)
    return pd.DataFrame(rows)


def aggregate(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, kv_bits, state_bits): mean over seeds, std over seeds, n_seeds."""
    keys = ["model", "dataset", "kv_bits", "state_bits"]
    g = df.groupby(keys)["mean_nll"]
    out = g.agg(nll_mean="mean", nll_std_seeds="std", n_seeds="count").reset_index()
    doc_std = df.groupby(keys)["std_nll"].mean().rename("doc_std").reset_index()
    return out.merge(doc_std, on=keys)


def format_cell(r) -> str:
    if r.n_seeds > 1:
        return f"{r.nll_mean:.3f} ± {r.nll_std_seeds:.3f}"
    return f"{r.nll_mean:.3f} (doc std {r.doc_std:.3f})"


def pivot_text(agg: pd.DataFrame, model: str, dataset: str) -> pd.DataFrame:
    sub = agg[(agg.model == model) & (agg.dataset == dataset)].copy()
    sub["cell"] = sub.apply(format_cell, axis=1)
    return sub.pivot(index="kv_bits", columns="state_bits", values="cell").sort_index(ascending=False)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="jsonl files (default: results/*.jsonl)")
    p.add_argument("--csv", default=None, help="directory to write one CSV per (model, dataset)")
    args = p.parse_args(argv)

    paths = args.paths or sorted(glob.glob(str(ROOT / "results" / "*.jsonl")))
    if not paths:
        print("WARNING: no results/*.jsonl found; nothing to tabulate")
        return 0
    df = load_rows(paths)
    agg = aggregate(df)
    for (model, dataset), _ in agg.groupby(["model", "dataset"]):
        n_seeds = int(agg[(agg.model == model) & (agg.dataset == dataset)].n_seeds.max())
        print(f"\n## {model} | {dataset} | seeds={n_seeds} | rows index=kv_bits, columns=state_bits | mean NLL")
        table = pivot_text(agg, model, dataset)
        print(table.to_string())
        if args.csv:
            os.makedirs(args.csv, exist_ok=True)
            fn = Path(args.csv) / f"{model.split('/')[-1]}_{dataset}.csv"
            agg[(agg.model == model) & (agg.dataset == dataset)].to_csv(fn, index=False)
            print("wrote", fn)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
