"""Paper figures from results/*.jsonl, saved as PDF under paper/figures/.

(a) heatmap_<model>.pdf      kv_bits x state_bits heatmap of mean NLL (averaged over seeds)
(b) curves_<model>.pdf       decode-position NLL curves (nll_curve, averaged over seeds) per cell
(c) budget_<model>.pdf       best NLL under the same total memory budget: KV-only vs state-only vs joint

    python analysis/make_figures.py                       # all results/*.jsonl
    python analysis/make_figures.py results/grid_0.8b.jsonl --T 1024

(c) needs the model config (AutoConfig, small download, cached); if it cannot be
loaded the figure is skipped with a warning. Missing results only produce a warning.
"""
from __future__ import annotations

import argparse
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
sys.path.insert(0, str(ROOT / "src"))

from make_tables import aggregate, load_rows  # noqa: E402

FIG_DIR = ROOT / "paper" / "figures"


def _tag(model: str) -> str:
    return model.split("/")[-1].replace(".", "p")


def fig_heatmap(agg: pd.DataFrame, model: str, plt):
    sub = agg[agg.model == model]
    pv = sub.pivot(index="kv_bits", columns="state_bits", values="nll_mean").sort_index(ascending=False)
    fig, ax = plt.subplots(figsize=(4.2, 3.4))
    im = ax.imshow(pv.values, cmap="viridis_r", aspect="auto")
    ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
    ax.set_xlabel("state bits"); ax.set_ylabel("KV bits"); ax.set_title(f"mean NLL, {model.split('/')[-1]}")
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if not np.isnan(v):
                ax.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=8, color="w" if v > np.nanmean(pv.values) else "k")
    fig.colorbar(im, ax=ax, shrink=0.8)
    out = FIG_DIR / f"heatmap_{_tag(model)}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_curves(df: pd.DataFrame, model: str, plt, window: int = 16):
    sub = df[df.model == model]
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for (kv, st), g in sub.groupby(["kv_bits", "state_bits"]):
        curves = [np.asarray(c, dtype=float) for c in g.nll_curve if c]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], axis=0)
        sm = np.convolve(mean, np.ones(window) / window, mode="valid")
        ax.plot(np.arange(len(sm)) + window, sm, label=f"KV{kv}/S{st}", lw=1.2)
    ax.set_xlabel("decode position"); ax.set_ylabel(f"NLL (moving avg {window})")
    ax.set_title(f"error vs position, {model.split('/')[-1]}"); ax.grid(alpha=0.3); ax.legend(fontsize=6, ncol=2)
    out = FIG_DIR / f"curves_{_tag(model)}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    return out


def fig_budget(agg: pd.DataFrame, model: str, T: int, plt):
    try:
        from transformers import AutoConfig
        from hybridquant.budget import cache_sizes, total_bytes
        sizes = cache_sizes(AutoConfig.from_pretrained(model).get_text_config(decoder=True))
    except Exception as e:  # noqa: BLE001
        print(f"WARNING: budget figure skipped for {model}: {type(e).__name__}: {e}")
        return None
    sub = agg[agg.model == model].copy()
    sub["bytes"] = [total_bytes(sizes, kv, st, T) for kv, st in zip(sub.kv_bits, sub.state_bits)]
    joint = sub[(sub.kv_bits < 16) & (sub.state_bits < 16)]
    budgets = sorted(joint.bytes.unique())
    groups = {
        "KV-only": lambda d: d[(d.state_bits == 16) & (d.kv_bits < 16)],
        "state-only": lambda d: d[(d.kv_bits == 16) & (d.state_bits < 16)],
        "joint": lambda d: d[(d.kv_bits < 16) & (d.state_bits < 16)],
    }
    rows = []
    for b in budgets:
        within = sub[sub.bytes <= b * 1.0001]
        rec = {"budget_MB": b / 1e6}
        for name, sel in groups.items():
            cand = sel(within)
            if len(cand):
                best = cand.loc[cand.nll_mean.idxmin()]
                rec[name] = best.nll_mean; rec[name + "_cfg"] = f"KV{int(best.kv_bits)}/S{int(best.state_bits)}"
            else:
                rec[name] = np.nan; rec[name + "_cfg"] = ""
        rows.append(rec)
    tab = pd.DataFrame(rows)
    fp = sub[(sub.kv_bits == 16) & (sub.state_bits == 16)].nll_mean
    fig, ax = plt.subplots(figsize=(6, 3.6))
    x = np.arange(len(tab)); w = 0.26
    for i, name in enumerate(groups):
        ax.bar(x + (i - 1) * w, tab[name], w, label=name)
        for xi, (v, c) in enumerate(zip(tab[name], tab[name + "_cfg"])):
            if not np.isnan(v):
                ax.text(xi + (i - 1) * w, v, c, ha="center", va="bottom", fontsize=6, rotation=90)
    if len(fp):
        ax.axhline(fp.iloc[0], ls="--", color="gray", lw=1, label="fp16 (16/16)")
    ax.set_xticks(x, [f"{b:.1f} MB" for b in tab.budget_MB], fontsize=7)
    ax.set_xlabel(f"total cache budget at T={T} (KV + state)"); ax.set_ylabel("best mean NLL within budget")
    ax.set_title(f"same budget: KV-only vs state-only vs joint, {model.split('/')[-1]}"); ax.legend(fontsize=7)
    out = FIG_DIR / f"budget_{_tag(model)}.pdf"
    fig.savefig(out, bbox_inches="tight"); plt.close(fig)
    print(tab.round(4).to_string(index=False))
    return out


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("paths", nargs="*", help="jsonl files (default: results/*.jsonl)")
    p.add_argument("--T", type=int, default=None, help="context length for the budget figure (default: prefill+decode of the rows)")
    p.add_argument("--window", type=int, default=16)
    args = p.parse_args(argv)

    paths = args.paths or sorted(glob.glob(str(ROOT / "results" / "*.jsonl")))
    if not paths:
        print("WARNING: no results/*.jsonl found; no figures made")
        return 0
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 9})
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    df = load_rows(paths)
    agg = aggregate(df)
    for model in sorted(df.model.unique()):
        T = args.T or int((df[df.model == model].prefill + df[df.model == model].decode).max())
        for out in (fig_heatmap(agg, model, plt), fig_curves(df, model, plt, args.window), fig_budget(agg, model, T, plt)):
            if out is not None:
                print("saved", out.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
