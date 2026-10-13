"""Rough wall-clock estimate for the round-1 and round-2 configs on one GPU.

Per-(cell, document) seconds are taken from measured runs where available (0.8B on A100:
about 12 s per document at prefill 256 + decode 256 with batch 1, which batching reduces) and
scaled with parameter count otherwise. This is a planning aid, not a measurement.

    python scripts/estimate_time.py
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

from hybridquant.config import iter_runs, load_config, resolve_settings

ROOT = Path(__file__).resolve().parents[1]
# seconds per (cell, batch of documents) for prefill 256 + decode 256; batching amortises the per-token loop
SEC_PER_BATCH = {"0.8B": 14, "1.5B": 20, "2B": 24, "4B": 40, "9B": 80}
GEN_TOK_PER_S = {"4B": 180}   # batched thinking-mode generation, tokens per second aggregate


def size_tag(model: str) -> str:
    for k in SEC_PER_BATCH:
        if k in model:
            return k
    return "4B"


def grid_hours(path: str) -> tuple[str, float]:
    s = resolve_settings(load_config(path))
    runs = iter_runs(s)
    n_batches = -(-s["n_docs"] // s["batch_size"])
    scale = (s["prefill"] + s["decode"]) / 512
    return s["model"], len(runs) * n_batches * SEC_PER_BATCH[size_tag(s["model"])] * scale / 3600


def main():
    rows = []
    for path in sorted(glob.glob(str(ROOT / "configs" / "*.yaml"))):
        name = os.path.basename(path)
        if "gsm8k" in name or "niah" in name:
            continue
        model, h = grid_hours(path)
        rows.append((name, model, h))
    rows.append(("r2_gsm8k_4b.yaml (10 cells x 200 problems, ~1500 tok avg)", "Qwen/Qwen3.5-4B", 10 * 200 * 1500 / GEN_TOK_PER_S["4B"] / 3600))
    rows.append(("r2_niah_0.8b.yaml (11 cells x 100 prompts, 4k/16k prefill)", "Qwen/Qwen3.5-0.8B", 11 * 100 * 1.5 / 3600))
    rows.append(("r2_niah_4b.yaml", "Qwen/Qwen3.5-4B", 11 * 100 * 4.0 / 3600))
    total = 0.0
    print(f"{'config':58s} {'model':28s} {'hours':>6s}")
    for name, model, h in rows:
        total += h
        print(f"{name:58s} {model:28s} {h:6.1f}")
    print(f"{'TOTAL (one A100, sequential)':58s} {'':28s} {total:6.1f}")


if __name__ == "__main__":
    main()
