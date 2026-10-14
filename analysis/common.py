"""Shared loading and statistics for reports and figures. Every number comes from results/*.jsonl."""
from __future__ import annotations

import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FP_LABEL = "Kfp_Vfp_Sfp"
N_BOOT = 1000
BOOT_SEED = 0
T_BUDGET = 512
CORE_CELLS = [(16, 16), (4, 16), (16, 8), (16, 4), (4, 4), (2, 8)]


# ---------------------------------------------------------------- loading


def load_jsonl(paths) -> pd.DataFrame:
    rows = []
    for p in paths:
        p = str(p)
        if not os.path.exists(p):
            continue
        with open(p, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    r = json.loads(line)
                    r["source"] = os.path.basename(p)
                    rows.append(r)
    if not rows:
        return pd.DataFrame()
    return ensure_defaults(drop_duplicate_rows(pd.DataFrame(rows)))


# One run = one row. A runner started twice on the same results file appends the same run again;
# the first row is kept. "source" keeps the check inside each file.
DEDUP_KEYS = ("source", "model", "label", "seed", "state_bits", "layer_idx", "context_len", "depth", "idx")


def drop_duplicate_rows(df: pd.DataFrame) -> pd.DataFrame:
    keys = [k for k in DEDUP_KEYS if k in df.columns]
    if not keys:
        return df
    return df.drop_duplicates(subset=keys, keep="first").reset_index(drop=True)


DEFAULTS = (("seed", 0), ("error_feedback", False), ("kv_noise_match", False), ("state_noise_match", False),
            ("dataset", "wikitext2"), ("state_group", 128), ("state_sym", False), ("state_sr", False), ("kv_group", 128))


def ensure_defaults(df: pd.DataFrame) -> pd.DataFrame:
    """Fill columns that older result files lack (per_doc_nll stays None where absent)."""
    df = df.copy()
    for col, default in DEFAULTS:
        if col not in df:
            df[col] = default
        df[col] = df[col].fillna(default)
    if "per_doc_nll" not in df:
        df["per_doc_nll"] = None
    return df


def kind_of(df: pd.DataFrame) -> str:
    cols = set(df.columns)
    if "layer_idx" in cols and "delta_nll" in cols:
        return "layer_sweep"
    if "gold" in cols and "correct" in cols:
        return "gsm8k"
    if "code" in cols and "correct" in cols:
        return "niah"
    if "mean_nll" in cols:
        return "grid"
    return "unknown"


def split_by_kind(df: pd.DataFrame) -> dict[str, pd.DataFrame]:
    out: dict[str, list] = {}
    for src, g in df.groupby("source"):
        g = ensure_defaults(g.dropna(axis=1, how="all"))
        out.setdefault(kind_of(g), []).append(g)
    return {k: ensure_defaults(pd.concat(v, ignore_index=True)) for k, v in out.items()}


def model_size_b(model: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)B", model)
    return float(m.group(1)) if m else None


def short_model(model: str) -> str:
    return model.split("/")[-1]


def cell_name(r) -> str:
    kv = "fp" if int(r["kv_bits"]) >= 16 else f"KV{int(r['kv_bits'])}"
    st = "fp" if int(r["state_bits"]) >= 16 else f"S{int(r['state_bits'])}"
    tags = []
    if r.get("kv_noise_match", False):
        tags.append("KVgn")
    if r.get("state_noise_match", False):
        tags.append("Sgn")
    if r.get("error_feedback", False):
        tags.append("EF")
    base = "fp" if kv == "fp" and st == "fp" else f"{kv}+{st}"
    return base + ("".join(f"[{t}]" for t in tags))


def cell_type(kv_bits, state_bits) -> str:
    if kv_bits < 16 and state_bits < 16:
        return "joint"
    if kv_bits < 16:
        return "KV-only"
    if state_bits < 16:
        return "state-only"
    return "fp"


# ---------------------------------------------------------------- statistics


def bootstrap_ci(x, n_boot: int = N_BOOT, seed: int = BOOT_SEED, alpha: float = 0.05):
    """Percentile bootstrap CI of the mean of ``x``. Returns (lo, hi) or (nan, nan) if too few values."""
    x = np.asarray([v for v in x if v is not None and np.isfinite(v)], dtype=float)
    if x.size < 2:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, x.size, size=(n_boot, x.size))
    means = x[idx].mean(axis=1)
    return (float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2)))


def ci_contains_zero(ci) -> bool:
    return not (np.isnan(ci[0]) or np.isnan(ci[1])) and ci[0] <= 0 <= ci[1]


def fmt_ci(ci) -> str:
    if np.isnan(ci[0]):
        return "N/A"
    return f"{ci[0]:+.3f}~{ci[1]:+.3f}"


def _paired_deltas(df: pd.DataFrame, model: str, dataset: str, label: str) -> list[float]:
    """Per-document NLL differences against the fp cell of the same (model, dataset, seed)."""
    deltas = []
    cell = df[(df.model == model) & (df.dataset == dataset) & (df.label == label)]
    fp = df[(df.model == model) & (df.dataset == dataset) & (df.label == FP_LABEL)]
    for _, r in cell.iterrows():
        f = fp[fp.seed == r.seed]
        if len(f) == 0 or not isinstance(r.get("per_doc_nll"), list) or not isinstance(f.iloc[0].get("per_doc_nll"), list):
            continue
        a, b = r["per_doc_nll"], f.iloc[0]["per_doc_nll"]
        n = min(len(a), len(b))
        deltas += [a[i] - b[i] for i in range(n)]
    return deltas


def aggregate_grid(df: pd.DataFrame) -> pd.DataFrame:
    """One row per (model, dataset, label): seed mean/std, delta vs fp, paired bootstrap CI, memory bytes."""
    rows = []
    sizes_cache: dict[str, dict | None] = {}
    for (model, dataset, label), g in df.groupby(["model", "dataset", "label"]):
        per_seed = g.groupby("seed")["mean_nll"].mean()
        fp = df[(df.model == model) & (df.dataset == dataset) & (df.label == FP_LABEL)]
        fp_per_seed = fp.groupby("seed")["mean_nll"].mean()
        common = per_seed.index.intersection(fp_per_seed.index)
        delta = float((per_seed[common] - fp_per_seed[common]).mean()) if len(common) else float("nan")
        deltas = _paired_deltas(df, model, dataset, label)
        ci = bootstrap_ci(deltas)
        r0 = g.iloc[0]
        if model not in sizes_cache:
            sizes_cache[model] = _sizes_for(model)
        sz = sizes_cache[model]
        mem = (int(r0.kv_bits) * sz["kv_per_token"] * T_BUDGET + int(r0.state_bits) * sz["state_elems"]) / 8 if sz else float("nan")
        rows.append({
            "model": model, "dataset": dataset, "label": label, "cell": cell_name(r0),
            "kv_bits": int(r0.kv_bits), "state_bits": int(r0.state_bits),
            "error_feedback": bool(r0.error_feedback), "kv_noise_match": bool(r0.kv_noise_match),
            "state_noise_match": bool(r0.state_noise_match), "state_group": int(r0.state_group),
            "state_sym": bool(r0.state_sym), "state_sr": bool(r0.state_sr),
            "type": cell_type(int(r0.kv_bits), int(r0.state_bits)),
            "n_seeds": int(per_seed.size), "n_docs": int(g.n_docs.iloc[0]) if "n_docs" in g else 0,
            "nll_mean": float(per_seed.mean()),
            "nll_std": float(per_seed.std(ddof=1)) if per_seed.size > 1 else float(g.std_nll.mean()) if "std_nll" in g else float("nan"),
            "std_kind": "seeds" if per_seed.size > 1 else "docs",
            "delta": delta, "ci_lo": ci[0], "ci_hi": ci[1], "n_paired": len(deltas),
            "mem_bytes_T512": mem,
        })
    return pd.DataFrame(rows)


def _sizes_for(model: str) -> dict | None:
    try:
        from transformers import AutoConfig
        import sys
        sys.path.insert(0, str(ROOT / "src"))
        from hybridquant.budget import cache_sizes
        return cache_sizes(AutoConfig.from_pretrained(model).get_text_config(decoder=True))
    except Exception:  # noqa: BLE001
        return None


def spearman(x, y, n_perm: int = 2000, seed: int = BOOT_SEED):
    """Spearman rho with a permutation p-value (two-sided). Returns (rho, p)."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan")
    rx = pd.Series(x).rank().to_numpy(); ry = pd.Series(y).rank().to_numpy()
    rho = float(np.corrcoef(rx, ry)[0, 1])
    rng = np.random.default_rng(seed)
    count = 0
    for _ in range(n_perm):
        r = float(np.corrcoef(rx, rng.permutation(ry))[0, 1])
        count += abs(r) >= abs(rho) - 1e-12
    return rho, (count + 1) / (n_perm + 1)


def pearson(x, y, n_perm: int = 2000, seed: int = BOOT_SEED):
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return float("nan"), float("nan")
    r = float(np.corrcoef(x, y)[0, 1])
    rng = np.random.default_rng(seed)
    count = sum(abs(float(np.corrcoef(x, rng.permutation(y))[0, 1])) >= abs(r) - 1e-12 for _ in range(n_perm))
    return r, (count + 1) / (n_perm + 1)


def run_conditions(df: pd.DataFrame) -> dict:
    """Metadata summary across rows: models, GPUs, dtypes, commits, dates, seeds, docs."""
    def uniq(series):
        return sorted({str(v) for v in series.dropna().tolist()}) if series is not None else []

    meta = [m for m in df.get("meta", pd.Series([None] * len(df))) if isinstance(m, dict)]
    return {
        "models": uniq(df["model"]) if "model" in df else [],
        "gpus": sorted({str(m.get("device")) for m in meta if m.get("device")}),
        "dtypes": uniq(df["dtype"]) if "dtype" in df else [],
        "commits": sorted({str(m.get("git_commit")) for m in meta if m.get("git_commit")}),
        "dates": sorted({str(m.get("timestamp_utc"))[:10] for m in meta if m.get("timestamp_utc")}),
        "n_seeds": int(df["seed"].dropna().nunique()) if "seed" in df else 0,
        "n_docs": sorted({int(v) for v in df["n_docs"].dropna().tolist()}) if "n_docs" in df else [],
        "n_rows": int(len(df)),
        "sources": uniq(df["source"]) if "source" in df else [],
    }
