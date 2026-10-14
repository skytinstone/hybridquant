"""Appendix C tables as LaTeX, one file per stage, from results/*.jsonl (same aggregation as the reports).

Each grid table lists every (KV bits, state bits) cell with the RTN and EF deltas versus bf16 and their paired
bootstrap 95% CIs, and the cache footprint at T = 512. The ablation table lists the state format variants.

    python analysis/make_tex_tables.py            # writes paper/tables/{grid_0.8b,grid_c4_0.8b,grid_2b,grid_4b_core,grid_9b,ablation_0.8b}.tex
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
from common import FP_LABEL, aggregate_grid, borrow_fp_rows, drop_cross_file_duplicates, load_jsonl  # noqa: E402

OUT = ROOT / "paper" / "tables"
# (output name, input jsonl files, caption)
GRIDS = [
    ("grid_0.8b", ["grid_0.8b.jsonl", "superadd_0.8b.jsonl"], "Qwen3.5-0.8B, WikiText-2: full grid and the noise-matched cells."),
    ("grid_c4_0.8b", ["r2_c4_0.8b.jsonl"], "Qwen3.5-0.8B, C4 validation: core cells."),
    ("grid_2b", ["r2_scale_2b.jsonl"], "Qwen3.5-2B, WikiText-2: core cells."),
    ("grid_4b_core", ["grid_4b_core.jsonl", "r3_recipe_4b.jsonl"], "Qwen3.5-4B, WikiText-2: core and recipe cells."),
    ("grid_9b", ["r2_scale_9b.jsonl"], "Qwen3.5-9B, WikiText-2: core cells."),
    ("grid_falcon_1.5b", ["r2_falcon_1.5b.jsonl"], "Falcon-H1-1.5B-Instruct (Mamba-2), WikiText-2: core cells."),
]


def fmt(d, lo, hi):
    if not np.isfinite(d):
        return "--"
    star = "" if (lo <= 0 <= hi) else ""
    return f"${d:+.3f}$ [{lo:+.3f}, {hi:+.3f}]{star}"


def bits_name(b: int) -> str:
    return "bf16" if b >= 16 else str(b)


def grid_table(name: str, files: list[str], caption: str) -> str | None:
    paths = [ROOT / "results" / f for f in files if (ROOT / "results" / f).exists()]
    if not paths:
        return None
    df = drop_cross_file_duplicates(load_jsonl(paths))
    df = borrow_fp_rows(df)
    agg = aggregate_grid(df)
    agg = agg[(~agg.kv_noise_match) & (~agg.state_noise_match)]
    fp_mem = agg[agg.label == FP_LABEL].mem_bytes_T512.iloc[0] if (agg.label == FP_LABEL).any() else np.nan
    rows = {}
    for _, r in agg.iterrows():
        key = (int(r.kv_bits), int(r.state_bits))
        rows.setdefault(key, {})["ef" if r.error_feedback else "rtn"] = r
    lines = ["\\begin{tabular}{rrrllr}", "\\toprule",
             "$b_{kv}$ & $b_s$ & Mem.\\ (MB, \\%) & $\\Delta$NLL RTN [95\\% CI] & $\\Delta$NLL EF [95\\% CI] & $n$ \\\\", "\\midrule"]
    for (kv, st) in sorted(rows, key=lambda k: (-k[0], -k[1])):
        r = rows[(kv, st)].get("rtn"); e = rows[(kv, st)].get("ef")
        any_r = r if r is not None else e
        mem = any_r.mem_bytes_T512 / 1e6
        pct = f"{100 * any_r.mem_bytes_T512 / fp_mem:.0f}" if np.isfinite(fp_mem) else "--"
        rtn = fmt(r.delta, r.ci_lo, r.ci_hi) if r is not None else "--"
        ef = fmt(e.delta, e.ci_lo, e.ci_hi) if e is not None else ("n/a" if st >= 16 else "--")
        n = f"{int(any_r.n_seeds)}$\\times${int(any_r.n_docs)}"
        lines.append(f"{bits_name(kv)} & {bits_name(st)} & {mem:.1f} ({pct}) & {rtn} & {ef} & {n} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              f"% {caption} Memory = bits x elements / 8 at T = 512 without scale/zero overhead; n = seeds x documents per cell."]
    return "\n".join(lines) + "\n"


def ablation_table() -> str | None:
    p = ROOT / "results" / "r2_ablation_0.8b.jsonl"
    if not p.exists():
        return None
    df = borrow_fp_rows(load_jsonl([p]))
    agg = aggregate_grid(df)
    agg = agg[agg.label != FP_LABEL]
    from figures import drop_duplicate_group
    agg = drop_duplicate_group(agg)
    lines = ["\\begin{tabular}{rrrccl}", "\\toprule",
             "$b_{kv}$ & $b_s$ & group & symmetric & stochastic & $\\Delta$NLL [95\\% CI] \\\\", "\\midrule"]
    for _, r in agg.sort_values(["state_bits", "state_group", "state_sym", "state_sr"], ascending=[False, True, True, True]).iterrows():
        g = "row" if int(r.state_group) in (-1, 0) else str(int(r.state_group))
        lines.append(f"{int(r.kv_bits)} & {int(r.state_bits)} & {g} & {'yes' if r.state_sym else 'no'} & "
                     f"{'yes' if r.state_sr else 'no'} & {fmt(r.delta, r.ci_lo, r.ci_hi)} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% Qwen3.5-0.8B, WikiText-2, KV at 4 bits, RTN. 'row' = one scale per state row (group = d_v). fp baseline borrowed from grid_0.8b (same documents and seeds)."]
    return "\n".join(lines) + "\n"


def _task_acc(path: Path, label: str, ef: bool):
    """Accuracy and n for one cell of a GSM8K or NIAH jsonl (mean over context lengths for NIAH)."""
    import json
    if not path.exists():
        return None, 0
    ok = n = 0
    with open(path, encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("label") == label + ("ef" if ef else "") and bool(r.get("error_feedback", False)) == ef:
                ok += bool(r.get("correct")); n += 1
    return (ok / n if n else None), n


GSM8K_CONFIGS = ("r2_gsm8k_4b.yaml", "r3_gsm8k_s8ef_4b.yaml")       # the GSM8K configs the runners execute
NIAH_CONFIGS = ("r2_niah_4b.yaml", "r3_niah_recipe_4b.yaml")
GRID_CONFIGS = ("grid_4b_core.yaml", "r3_recipe_4b.yaml")
PENDING = "\\pending{}"   # LaTeX macro defined in the manuscript preamble


def _configured(names) -> set:
    """(kv_bits, state_bits, error_feedback) cells scheduled by the given configs."""
    import yaml
    out = set()
    for n in names:
        path = ROOT / "configs" / n
        if not path.exists():
            continue
        cfg = yaml.safe_load(open(path))
        efs_default = cfg.get("state_error_feedback", [False])
        for c in cfg.get("cells") or []:
            efs = [bool(c["error_feedback"])] if "error_feedback" in c else [bool(e) for e in efs_default]
            for ef in efs:
                if ef and int(c["state_bits"]) >= 16:
                    continue
                out.add((int(c["kv_bits"]), int(c["state_bits"]), ef))
    return out


def _mem_pct(kv: int, st: int) -> str:
    """Footprint of both caches at T = 512 relative to bf16, from eqs. (1) and (2); never pending."""
    from common import T_BUDGET, _sizes_for
    sz = _sizes_for("Qwen/Qwen3.5-4B")
    if not sz:
        return PENDING
    f = lambda a, b: a * sz["kv_per_token"] * T_BUDGET + b * sz["state_elems"]
    return f"{100 * f(kv, st) / f(16, 16):.0f}"


def table1() -> str:
    """Headline table on Qwen3.5-4B. Mem. is analytic; dNLL/NIAH/GSM8K are measured, \pending{} when scheduled
    but not finished, and -- when the cell is not in any config for that task."""
    grid = [ROOT / "results" / f for f in ("grid_4b_core.jsonl", "r3_recipe_4b.jsonl") if (ROOT / "results" / f).exists()]
    agg = aggregate_grid(drop_cross_file_duplicates(load_jsonl(grid))) if grid else None
    gsm = [ROOT / "results" / f for f in ("r2_gsm8k_4b.jsonl", "r3_gsm8k_s8ef_4b.jsonl")]
    niah = [ROOT / "results" / f for f in ("r2_niah_4b.jsonl", "r3_niah_recipe_4b.jsonl")]
    gsm_cfg, niah_cfg, grid_cfg = _configured(GSM8K_CONFIGS), _configured(NIAH_CONFIGS), _configured(GRID_CONFIGS) | {(16, 16, False)}
    rows = [("bf16", 16, 16, False), ("KV-only (KV4)", 4, 16, False), ("State-only (S8, EF)", 16, 8, True),
            ("Joint KV4 + S6, EF", 4, 6, True), ("Joint KV4 + S8", 4, 8, False), ("Joint KV4 + S8, EF", 4, 8, True)]
    lab = lambda kv, st: f"K{'fp' if kv >= 16 else f'int{kv}g128'}_V{'fp' if kv >= 16 else f'int{kv}g128'}_S{'fp' if st >= 16 else f'int{st}g128'}"
    lines = ["\\begin{tabular}{lrrrr}", "\\toprule", "Setting & Mem.\\ (\\%) & $\\Delta$NLL & GSM8K & NIAH \\\\", "\\midrule"]
    for name, kv, st, ef in rows:
        label = lab(kv, st); key = (kv, st, ef)
        # dNLL
        a = agg[(agg.label == label + ("ef" if ef else "")) & (agg.error_feedback == ef)] if agg is not None else None
        if a is not None and len(a):
            d = "0" if (kv, st) == (16, 16) else f"${a.iloc[0].delta:+.3f}$"
        else:
            d = PENDING if key in grid_cfg else "--"
        # GSM8K: value only when all 200 problems are done
        gres = [_task_acc(p, label, ef) for p in gsm]; gres = [x for x in gres if x[0] is not None]
        if gres and gres[0][1] >= 200:
            g = f"{gres[0][0]:.3f}"
        else:
            g = PENDING if key in gsm_cfg else "--"
        # NIAH: mean over the two context lengths
        vals = [x for x, _ in (_task_acc(p, label, ef) for p in niah) if x is not None]
        nv = f"{sum(vals) / len(vals):.2f}" if vals else (PENDING if key in niah_cfg or (kv, st) == (16, 16) else "--")
        lines.append(f"{name} & {_mem_pct(kv, st)} & {d} & {g} & {nv} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}",
              "% Generated by analysis/make_tex_tables.py. Mem. = both caches at T = 512 relative to bf16 from eqs. (1)(2), "
              "no scale/zero overhead, EF residual excluded. \\pending{} = scheduled, not finished; -- = not in any config for that task."]
    return "\n".join(lines) + "\n"


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    written = []
    (OUT / "table1.tex").write_text(table1()); written.append("table1")
    for name, files, caption in GRIDS:
        t = grid_table(name, files, caption)
        if t:
            (OUT / f"{name}.tex").write_text(t); written.append(name)
    t = ablation_table()
    if t:
        (OUT / "ablation_0.8b.tex").write_text(t); written.append("ablation_0.8b")
    for w in written:
        print("wrote", os.path.relpath(OUT / f"{w}.tex", ROOT))


if __name__ == "__main__":
    main()
