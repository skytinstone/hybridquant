"""Appendix B model table: layer counts, head geometry, |KV| per token, |S| and the crossover T* for every model
in the study, read from the HF configs (paper eqs. 1 and 2). Writes Markdown and a LaTeX tabular.

    python analysis/model_table.py           # writes paper/tables/models.md and models.tex
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from hybridquant.budget import cache_sizes  # noqa: E402

OUT = ROOT / "paper" / "tables"
MODELS = ["Qwen/Qwen3.5-0.8B", "Qwen/Qwen3.5-2B", "Qwen/Qwen3.5-4B", "Qwen/Qwen3.5-9B", "tiiuae/Falcon-H1-1.5B-Instruct"]


def row(name: str) -> dict:
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(name).get_text_config(decoder=True)
    sz = cache_sizes(cfg)
    return {"model": name.split("/")[-1], "mixer": "Mamba-2" if cfg.model_type == "falcon_h1" else "GDN",
            "L": int(cfg.num_hidden_layers), "L_a": sz["L_a"], "L_g": sz["L_g"], "hidden": int(cfg.hidden_size),
            "H_kv": sz["H_kv"], "d": sz["d"], "H": sz["H"], "d_k": sz["d_k"], "d_v": sz["d_v"],
            "kv_per_token": sz["kv_per_token"], "state_elems": sz["state_elems"],
            "state_mb_bf16": sz["state_elems"] * 2 / 1e6, "t_star": sz["state_elems"] / sz["kv_per_token"]}


def main():
    rows = [row(m) for m in MODELS]
    OUT.mkdir(parents=True, exist_ok=True)
    hdr = ["model", "mixer", "L", "L_a", "L_g", "hidden", "H_kv", "d", "H", "d_k", "d_v", "|KV|/token", "|S| (M)", "|S| bf16 MB", "T*"]
    md = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    tex = ["\\begin{tabular}{llrrrrrrrrrrrrr}", "\\toprule",
           "Model & Mixer & $L$ & $L_a$ & $L_g$ & $d_{\\text{model}}$ & $H_{kv}$ & $d$ & $H$ & $d_k$ & $d_v$ & $|\\mathrm{KV}|/\\text{tok}$ & $|S|$ (M) & $|S|$ bf16 (MB) & $T^\\star$ \\\\",
           "\\midrule"]
    for r in rows:
        vals = [r["model"], r["mixer"], r["L"], r["L_a"], r["L_g"], r["hidden"], r["H_kv"], r["d"], r["H"], r["d_k"], r["d_v"],
                f"{r['kv_per_token']:,}", f"{r['state_elems'] / 1e6:.2f}", f"{r['state_mb_bf16']:.1f}", f"{r['t_star']:,.0f}"]
        md.append("| " + " | ".join(str(v) for v in vals) + " |")
        tex.append(" & ".join(str(v).replace("_", "\\_") for v in vals) + " \\\\")
    tex += ["\\bottomrule", "\\end{tabular}"]
    note = ("\n\n|KV|/token = 2 L_a H_kv d (elements, K and V); |S| = L_g H d_k d_v (elements); T* = |S| / (|KV|/token) is the "
            "context length at which the two memories are equal. Falcon-H1 has attention and Mamba-2 in every layer "
            "(L_a = L_g = L); its state is [H, d_head, d_state] with d_k = d_head and d_v = d_state.\n")
    (OUT / "models.md").write_text("\n".join(md) + note)
    (OUT / "models.tex").write_text("\n".join(tex) + "\n")
    print("\n".join(md))
    print("wrote", os.path.relpath(OUT / "models.md", ROOT), "and models.tex")


if __name__ == "__main__":
    main()
