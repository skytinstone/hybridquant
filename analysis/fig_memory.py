"""Figure 2 of the paper: KV-cache size versus recurrent-state size as a function of context length, with the
crossover T* where the two are equal (paper eqs. 1 and 2). Sizes come from the HF configs through
hybridquant.budget.cache_sizes; nothing is measured.

    python analysis/fig_memory.py            # writes paper/figures/memory_crossover.pdf and .png
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridquant.budget import cache_sizes  # noqa: E402

OUT = ROOT / "paper" / "figures"
MODELS = [("Qwen/Qwen3.5-0.8B", "Qwen3.5-0.8B / 2B", "#4C72B0"),      # 0.8B and 2B share the cache geometry
          ("Qwen/Qwen3.5-4B", "Qwen3.5-4B / 9B", "#DD8452"),          # so do 4B and 9B
          ("tiiuae/Falcon-H1-1.5B-Instruct", "Falcon-H1-1.5B", "#55A868")]


def sizes_for(name: str) -> dict:
    from transformers import AutoConfig
    return cache_sizes(AutoConfig.from_pretrained(name).get_text_config(decoder=True))


def main():
    plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})
    fig, ax = plt.subplots(figsize=(4.6, 3.1))
    T = np.logspace(np.log10(64), np.log10(131072), 300)
    for name, label, color in MODELS:
        sz = sizes_for(name)
        kv_mb = sz["kv_per_token"] * T * 2 / 1e6          # bf16 bytes
        s_mb = sz["state_elems"] * 2 / 1e6
        t_star = sz["state_elems"] / sz["kv_per_token"]
        ax.plot(T, kv_mb, color=color, lw=1.4, label=f"{label}: KV cache")
        ax.axhline(s_mb, color=color, lw=1.2, ls="--", label=f"{label}: state ({s_mb:.0f} MB)")
        ax.plot([t_star], [s_mb], "o", color=color, ms=5, mec="k", mew=0.5)
        ax.annotate(rf"$T^\star$={t_star:,.0f}", (t_star, s_mb), fontsize=7, xytext=(4, -11), textcoords="offset points", color=color)
        print(f"{name:34s} |KV|/token={sz['kv_per_token']:>7,d}  |S|={sz['state_elems']:>12,d}  T*={t_star:,.0f}")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("context length $T$ (tokens)"); ax.set_ylabel("memory in bf16 (MB)")
    ax.set_title("Two inference memories: KV cache grows, state is fixed", fontsize=9)
    ax.grid(alpha=0.3, which="both", lw=0.4); ax.legend(fontsize=6.2, ncol=1, loc="upper left")
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("pdf", "png"):
        fig.savefig(OUT / f"memory_crossover.{ext}", bbox_inches="tight", dpi=300)
    print("wrote", os.path.relpath(OUT / "memory_crossover.pdf", ROOT), "and .png")


if __name__ == "__main__":
    main()
