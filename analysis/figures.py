"""Figures for every stage, computed only from results/*.jsonl. Saved as PDF and PNG under results/figures/<stage>/.

Style: white background, 10 pt, viridis (sequential) / RdBu (diverging), English text inside figures.
Each figure function returns (path_without_ext, caption_kr, caption_en) or None when the data is missing.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CORE_CELLS, FP_LABEL, ROOT, aggregate_grid, bootstrap_ci, cell_name, model_size_b, pearson, \
    short_model, spearman  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

plt.rcParams.update({"figure.facecolor": "white", "axes.facecolor": "white", "font.size": 10,
                     "axes.grid": True, "grid.alpha": 0.3, "savefig.bbox": "tight", "savefig.dpi": 150})
VIRIDIS, RDBU = "viridis", "RdBu"


def _save(fig, out_dir: Path, name: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    base = out_dir / name
    fig.savefig(str(base) + ".pdf"); fig.savefig(str(base) + ".png")
    plt.close(fig)
    return base


def _core(agg: pd.DataFrame) -> pd.DataFrame:
    sel = agg[(~agg.kv_noise_match) & (~agg.state_noise_match)]
    sel = sel[[(a, b) in CORE_CELLS for a, b in zip(sel.kv_bits, sel.state_bits)]]
    return sel


# ================================================================ grid-like


def grid_heatmap(agg: pd.DataFrame, model: str, out_dir: Path):
    sub = agg[(agg.model == model) & (~agg.kv_noise_match) & (~agg.state_noise_match)]
    panels = [(False, "EF off")] + ([(True, "EF on")] if sub.error_feedback.any() else [])
    fig, axes = plt.subplots(1, len(panels), figsize=(4.4 * len(panels), 3.6), squeeze=False)
    vmax = np.nanmax(np.abs(sub.delta.values)) if len(sub) else 1
    for ax, (ef, title) in zip(axes[0], panels):
        s = sub[sub.error_feedback == ef] if ef else sub[(~sub.error_feedback) | (sub.state_bits >= 16)]
        pv = s.pivot_table(index="kv_bits", columns="state_bits", values="delta").sort_index(ascending=False).astype(float)
        if pv.empty:
            ax.set_visible(False); continue
        im = ax.imshow(pv.values, cmap=RDBU + "_r", vmin=-vmax, vmax=vmax, aspect="auto")
        ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index)
        ax.set_xlabel("state bits"); ax.set_ylabel("KV bits"); ax.set_title(f"{short_model(model)}: dNLL vs fp, {title}")
        ax.grid(False)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                v = pv.values[i, j]
                if np.isfinite(v):
                    ax.text(j, i, f"{v:+.3f}", ha="center", va="center", fontsize=8)
        fig.colorbar(im, ax=ax, shrink=0.8, label="dNLL (nats)")
    base = _save(fig, out_dir, f"heatmap_{short_model(model)}")
    return base, f"{short_model(model)} KV bits x state bits 격자의 fp 대비 NLL 증가(ΔNLL), EF off/on", \
        f"{short_model(model)}: NLL increase over fp (dNLL) on the KV bits x state bits grid, EF off/on"


def grid_lines(agg: pd.DataFrame, model: str, out_dir: Path):
    sub = agg[(agg.model == model) & (~agg.kv_noise_match) & (~agg.state_noise_match) & (~agg.error_feedback)]
    if sub.empty:
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.6))
    kvs = sorted(sub.kv_bits.unique(), reverse=True)
    colors = plt.get_cmap(VIRIDIS)(np.linspace(0.1, 0.9, len(kvs)))
    for c, kv in zip(colors, kvs):
        s = sub[sub.kv_bits == kv].sort_values("state_bits")
        ax.plot(s.state_bits, s.delta, "o-", color=c, label=f"KV {kv if kv < 16 else 'fp'}")
        if s.ci_lo.notna().any():
            ax.fill_between(s.state_bits, s.ci_lo.fillna(s.delta), s.ci_hi.fillna(s.delta), color=c, alpha=0.15)
    ax.set_xscale("log", base=2); ax.set_xticks(sorted(sub.state_bits.unique()), sorted(sub.state_bits.unique()))
    ax.set_xlabel("state bits (16 = fp)"); ax.set_ylabel("dNLL vs fp (nats)")
    ax.set_title(f"{short_model(model)}: state bits vs dNLL, EF off"); ax.legend(fontsize=8)
    base = _save(fig, out_dir, f"lines_{short_model(model)}")
    return base, "state bits별 ΔNLL, KV bits는 색, 음영은 문서 대응 부트스트랩 95% CI", \
        "dNLL versus state bits; colour = KV bits; shading = paired bootstrap 95% CI"


def grid_pareto(agg: pd.DataFrame, model: str, out_dir: Path):
    sub = agg[(agg.model == model) & (~agg.kv_noise_match) & (~agg.state_noise_match) & agg.mem_bytes_T512.notna()]
    if sub.empty or sub.mem_bytes_T512.isna().all():
        return None
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    markers = {"fp": "*", "KV-only": "s", "state-only": "^", "joint": "o"}
    for t, m in markers.items():
        s = sub[sub.type == t]
        if s.empty:
            continue
        ax.scatter(s.mem_bytes_T512 / 1e6, s.delta, marker=m, s=45, label=t,
                   c=["C3" if ef else "C0" for ef in s.error_feedback], edgecolors="k", linewidths=0.4)
        for _, r in s.iterrows():
            ax.annotate(r.cell, (r.mem_bytes_T512 / 1e6, r.delta), fontsize=6, xytext=(3, 2), textcoords="offset points")
    pts = [(r.mem_bytes_T512 / 1e6, r.delta) for _, r in sub.iterrows() if np.isfinite(r.delta)]
    front = sorted(p for p in pts if not any((q[0] <= p[0] and q[1] <= p[1] and q != p) for q in pts))
    if len(front) >= 2:
        ax.plot([p[0] for p in front], [p[1] for p in front], "k--", lw=1, label="Pareto front")
    ax.set_xlabel("total cache memory at T=512 (MB, bits x elements)"); ax.set_ylabel("dNLL vs fp (nats)")
    ax.set_title(f"{short_model(model)}: memory vs accuracy (blue EF off, red EF on)"); ax.legend(fontsize=7)
    base = _save(fig, out_dir, f"pareto_{short_model(model)}")
    return base, "총 메모리(T=512) 대 ΔNLL 파레토 산점도, 마커는 KV-only/state-only/joint", \
        "Memory (T=512) versus dNLL Pareto scatter; marker = KV-only / state-only / joint"


def grid_curves(df: pd.DataFrame, model: str, out_dir: Path, window: int = 16):
    sub = df[(df.model == model) & (~df.kv_noise_match) & (~df.state_noise_match) & (~df.error_feedback)]
    sub = sub[[(a, b) in CORE_CELLS for a, b in zip(sub.kv_bits, sub.state_bits)]]
    if sub.empty or "nll_curve" not in sub:
        return None
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    cmap = plt.get_cmap(VIRIDIS)
    groups = list(sub.groupby("label"))
    for i, (label, g) in enumerate(groups):
        curves = [np.asarray(c, dtype=float) for c in g.nll_curve if isinstance(c, list) and c]
        if not curves:
            continue
        n = min(len(c) for c in curves)
        mean = np.mean([c[:n] for c in curves], axis=0)
        stds = [np.asarray(c, dtype=float)[:n] for c in g.get("nll_curve_std", pd.Series([None] * len(g))) if isinstance(c, list)]
        sm = np.convolve(mean, np.ones(window) / window, mode="valid"); x = np.arange(len(sm)) + window
        color = cmap(i / max(1, len(groups) - 1))
        ax.plot(x, sm, color=color, lw=1.2, label=cell_name(g.iloc[0]))
        if stds:
            sd = np.convolve(np.mean(stds, axis=0), np.ones(window) / window, mode="valid")
            ax.fill_between(x, sm - sd, sm + sd, color=color, alpha=0.12)
    ax.set_xlabel("decode position"); ax.set_ylabel(f"NLL (moving avg {window})")
    ax.set_title(f"{short_model(model)}: NLL vs decode position, core cells"); ax.legend(fontsize=7, ncol=2)
    base = _save(fig, out_dir, f"curves_{short_model(model)}")
    return base, "핵심 셀의 decode 위치별 NLL(이동평균 16), 음영은 문서 간 ±1 std", \
        "NLL by decode position for core cells (moving average 16); shading = ±1 std across documents"


def grid_violin(df: pd.DataFrame, model: str, out_dir: Path):
    sub = df[(df.model == model) & (~df.kv_noise_match) & (~df.state_noise_match) & (~df.error_feedback)]
    sub = sub[[(a, b) in CORE_CELLS and (a, b) != (16, 16) for a, b in zip(sub.kv_bits, sub.state_bits)]]
    data, names = [], []
    for label, g in sub.groupby("label"):
        deltas = []
        for _, r in g.iterrows():
            fp = df[(df.model == model) & (df.dataset == r.dataset) & (df.label == FP_LABEL) & (df.seed == r.seed)]
            if len(fp) and isinstance(r.per_doc_nll, list) and isinstance(fp.iloc[0].per_doc_nll, list):
                deltas += [a - b for a, b in zip(r.per_doc_nll, fp.iloc[0].per_doc_nll)]
        if len(deltas) >= 2:
            data.append(deltas); names.append(cell_name(g.iloc[0]))
    if not data:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.6))
    ax.violinplot(data, showmeans=True)
    ax.set_xticks(range(1, len(names) + 1), names, rotation=20, fontsize=8)
    ax.axhline(0, color="gray", lw=0.8); ax.set_ylabel("per-document dNLL vs fp (nats)")
    ax.set_title(f"{short_model(model)}: per-document dNLL distribution")
    base = _save(fig, out_dir, f"violin_{short_model(model)}")
    return base, "핵심 셀의 문서별 ΔNLL 분포(violin, 평균 표시)", "Per-document dNLL distribution for core cells (violin, mean marked)"


def grid_figures(df: pd.DataFrame, stage: str, out_dir: Path):
    agg = aggregate_grid(df)
    out = []
    for model in sorted(df.model.unique()):
        for fn in (grid_heatmap, grid_lines, grid_pareto):
            try:
                r = fn(agg, model, out_dir)
                if r: out.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: {fn.__name__} failed for {model}: {type(e).__name__}: {e}")
        for fn in (grid_curves, grid_violin):
            try:
                r = fn(df, model, out_dir)
                if r: out.append(r)
            except Exception as e:  # noqa: BLE001
                print(f"WARNING: {fn.__name__} failed for {model}: {type(e).__name__}: {e}")
    return out


# ================================================================ superadditivity


def superadd_figures(df: pd.DataFrame, stage: str, out_dir: Path):
    out = []
    agg = aggregate_grid(df)
    for model in sorted(df.model.unique()):
        a = agg[agg.model == model].set_index("label")
        need = ["Kfp_Vfp_Sint4g128", "Kint4g128_Vint4g128_Sfp", "Kint4g128_Vint4g128_Sint4g128"]
        if not all(n in a.index for n in need):
            print(f"WARNING: superadd cells missing for {model}"); continue
        dS, dK, dJ = (a.loc[n, "delta"] for n in need)
        labels = ["S4 only", "KV4 only", "KV4+S4"] + [cell_name(a.loc[l]) for l in a.index if "gn" in l]
        vals = [dS, dK, dJ] + [a.loc[l, "delta"] for l in a.index if "gn" in l]
        errs = [(a.loc[n, "delta"] - a.loc[n, "ci_lo"], a.loc[n, "ci_hi"] - a.loc[n, "delta"]) for n in need] + \
               [(a.loc[l, "delta"] - a.loc[l, "ci_lo"], a.loc[l, "ci_hi"] - a.loc[l, "delta"]) for l in a.index if "gn" in l]
        errs = np.array([[0 if np.isnan(e[0]) else e[0] for e in errs], [0 if np.isnan(e[1]) else e[1] for e in errs]])
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(range(len(vals)), vals, yerr=errs, capsize=3, color=plt.get_cmap(VIRIDIS)(np.linspace(0.2, 0.8, len(vals))))
        ax.axhline(dS + dK, ls=":", color="k", label=f"predicted fp+dKV+dS = {dS + dK:+.3f}")
        ax.set_xticks(range(len(vals)), labels, rotation=15, fontsize=8); ax.set_ylabel("dNLL vs fp (nats)")
        ax.set_title(f"{short_model(model)}: superadditivity (excess = {dJ - dS - dK:+.3f})"); ax.legend(fontsize=8)
        out.append((_save(fig, out_dir, f"superadd_bars_{short_model(model)}"),
                    "셀별 ΔNLL 막대(95% CI)와 fp+ΔKV+ΔS 예측값(점선), 차이가 초과분",
                    "dNLL per cell (95% CI) with the additive prediction fp+dKV+dS (dotted); the gap is the excess"))
        # (b) per-document excess histogram
        ex = per_doc_excess(df, model)
        if len(ex) >= 2:
            fig, ax = plt.subplots(figsize=(5, 3.3))
            ax.hist(ex, bins=20, color=plt.get_cmap(VIRIDIS)(0.5)); ax.axvline(0, color="k", lw=0.8)
            ci = bootstrap_ci(ex)
            ax.set_xlabel("per-document excess: d(KV4+S4) - dKV4 - dS4 (nats)"); ax.set_ylabel("documents")
            ax.set_title(f"{short_model(model)}: mean {np.mean(ex):+.3f}, 95% CI {ci[0]:+.3f}~{ci[1]:+.3f}")
            out.append((_save(fig, out_dir, f"superadd_hist_{short_model(model)}"),
                        "문서별 초과분 히스토그램", "Histogram of the per-document excess"))
        # (c) noise-match vs real quantization, per document
        pairs = [("Kint4g128gn_Vint4g128gn_Sint4g128", "Kint4g128_Vint4g128_Sint4g128", "KV noise-match vs KV quant (S4 fixed)"),
                 ("Kint4g128_Vint4g128_Sint4g128gn", "Kint4g128_Vint4g128_Sint4g128", "state noise-match vs state quant (KV4 fixed)")]
        fig, axes = plt.subplots(1, 2, figsize=(8, 3.6))
        drawn = False
        for ax, (gn, real, title) in zip(axes, pairs):
            x, y = per_doc_pairs(df, model, gn, real)
            if len(x) == 0:
                ax.set_visible(False); continue
            drawn = True
            ax.scatter(x, y, s=14, color=plt.get_cmap(VIRIDIS)(0.4))
            lo, hi = min(min(x), min(y)), max(max(x), max(y)); ax.plot([lo, hi], [lo, hi], "k--", lw=0.8)
            ax.set_xlabel("NLL with Gaussian noise (gn)"); ax.set_ylabel("NLL with real quantization"); ax.set_title(title, fontsize=9)
        if drawn:
            out.append((_save(fig, out_dir, f"superadd_noise_{short_model(model)}"),
                        "noise_match 조건과 실제 양자화 조건의 문서별 NLL 대응(대각선 = 동일)",
                        "Per-document NLL, noise-match versus real quantization (diagonal = equal)"))
        else:
            plt.close(fig)
    return out


def per_doc_excess(df: pd.DataFrame, model: str) -> list[float]:
    ex = []
    for seed in sorted(df[df.model == model].seed.unique()):
        d = df[(df.model == model) & (df.seed == seed)].set_index("label")
        need = [FP_LABEL, "Kfp_Vfp_Sint4g128", "Kint4g128_Vint4g128_Sfp", "Kint4g128_Vint4g128_Sint4g128"]
        if not all(n in d.index and isinstance(d.loc[n, "per_doc_nll"], list) for n in need):
            continue
        f, s, k, j = (d.loc[n, "per_doc_nll"] for n in need)
        n = min(map(len, (f, s, k, j)))
        ex += [(j[i] - f[i]) - (s[i] - f[i]) - (k[i] - f[i]) for i in range(n)]
    return ex


def per_doc_pairs(df, model, label_x, label_y):
    xs, ys = [], []
    for seed in sorted(df[df.model == model].seed.unique()):
        d = df[(df.model == model) & (df.seed == seed)].set_index("label")
        if label_x in d.index and label_y in d.index and isinstance(d.loc[label_x, "per_doc_nll"], list):
            a, b = d.loc[label_x, "per_doc_nll"], d.loc[label_y, "per_doc_nll"]
            n = min(len(a), len(b)); xs += a[:n]; ys += b[:n]
    return xs, ys


# ================================================================ layer sweep


def layer_sweep_figures(df: pd.DataFrame, stage: str, out_dir: Path):
    out = []
    for model in sorted(df.model.unique()):
        sub = df[(df.model == model) & (df.layer_idx >= 0)].sort_values("layer_idx")
        if sub.empty:
            continue
        fig, ax = plt.subplots(figsize=(6, 3.3))
        ax.bar(sub.layer_idx.astype(str), sub.delta_nll, color=plt.get_cmap(VIRIDIS)(0.5))
        ax.set_xlabel("GDN layer index"); ax.set_ylabel(f"dNLL, state int{int(sub.state_bits.iloc[0])} on this layer only")
        ax.set_title(f"{short_model(model)}: per-layer state sensitivity")
        out.append((_save(fig, out_dir, f"layer_bars_{short_model(model)}"), "레이어 하나씩만 state 양자화했을 때의 ΔNLL",
                    "dNLL when only one layer's state is quantized"))
        if "n_half" in sub:
            x, y = sub.n_half.to_numpy(float), sub.delta_nll.to_numpy(float)
            rho, p_s = spearman(x, y); r, p_p = pearson(x, y)
            fig, ax = plt.subplots(figsize=(5, 3.6))
            ax.scatter(x, y, color=plt.get_cmap(VIRIDIS)(0.3))
            for xi, yi, li in zip(x, y, sub.layer_idx):
                ax.annotate(str(li), (xi, yi), fontsize=6, xytext=(2, 2), textcoords="offset points")
            ok = np.isfinite(x) & np.isfinite(y)
            if ok.sum() >= 2:
                k, b = np.polyfit(x[ok], y[ok], 1); xx = np.linspace(x[ok].min(), x[ok].max(), 50)
                ax.plot(xx, k * xx + b, "k--", lw=0.8)
            ax.set_xlabel("error half-life n_half (tokens)"); ax.set_ylabel("dNLL (nats)")
            ax.set_title(f"{short_model(model)}: Spearman {rho:.2f} (p={p_s:.3f}), Pearson {r:.2f} (p={p_p:.3f})", fontsize=9)
            out.append((_save(fig, out_dir, f"layer_halflife_{short_model(model)}"),
                        "오차 반감 토큰 수 n_half 대 ΔNLL 산점도, 회귀선, Spearman/Pearson(순열 p값)",
                        "Error half-life n_half versus dNLL with regression line; Spearman/Pearson with permutation p-values"))
        if "mean_alpha" in sub and "mean_beta" in sub:
            fig, ax = plt.subplots(figsize=(5, 3.6))
            sc = ax.scatter(sub.mean_alpha, sub.mean_beta, c=sub.delta_nll, cmap=VIRIDIS, s=40, edgecolors="k", linewidths=0.4)
            for _, r in sub.iterrows():
                ax.annotate(str(int(r.layer_idx)), (r.mean_alpha, r.mean_beta), fontsize=6, xytext=(2, 2), textcoords="offset points")
            fig.colorbar(sc, ax=ax, label="dNLL"); ax.set_xlabel("mean alpha = exp(g)"); ax.set_ylabel("mean beta")
            ax.set_title(f"{short_model(model)}: decay vs write gate per layer")
            out.append((_save(fig, out_dir, f"layer_alpha_beta_{short_model(model)}"),
                        "레이어별 평균 α와 평균 β 산점도, 색은 ΔNLL", "Per-layer mean alpha versus mean beta, colour = dNLL"))
    return out


# ================================================================ ablation


def ablation_figures(df: pd.DataFrame, stage: str, out_dir: Path):
    agg = aggregate_grid(df)
    out = []
    for model in sorted(agg.model.unique()):
        sub = agg[(agg.model == model) & (agg.state_bits < 16)]
        cells = sorted({(a, b) for a, b in zip(sub.kv_bits, sub.state_bits)})
        if not cells:
            continue
        fig, axes = plt.subplots(1, len(cells), figsize=(5 * len(cells), 3.6), squeeze=False)
        for ax, (kv, st) in zip(axes[0], cells):
            s = sub[(sub.kv_bits == kv) & (sub.state_bits == st)]
            groups = sorted(s.state_group.unique(), key=lambda g: (g < 0, g))
            combos = [(False, False), (True, False), (False, True), (True, True)]
            w = 0.2
            for i, (sym, sr) in enumerate(combos):
                vals = [s[(s.state_group == g) & (s.state_sym == sym) & (s.state_sr == sr)].delta.mean() for g in groups]
                ax.bar(np.arange(len(groups)) + (i - 1.5) * w, vals, w, label=f"sym={int(sym)} sr={int(sr)}",
                       color=plt.get_cmap(VIRIDIS)(i / 3))
            ax.set_xticks(range(len(groups)), [str(g) if g > 0 else "whole" for g in groups])
            ax.set_xlabel("state group size"); ax.set_ylabel("dNLL vs fp"); ax.set_title(f"{short_model(model)}: KV{kv}+S{st}")
            ax.legend(fontsize=7)
        out.append((_save(fig, out_dir, f"ablation_{short_model(model)}"),
                    "state 양자화 포맷 ablation: group 크기 × (symmetric, stochastic), 셀별 패널",
                    "State-format ablation: group size x (symmetric, stochastic), one panel per cell"))
    return out


# ================================================================ gsm8k


def gsm8k_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label), g in df.groupby(["model", "label"]):
        per_seed = g.groupby("seed")["correct"].mean()
        ci = bootstrap_ci(g.groupby("idx")["correct"].mean().to_numpy(float))
        rows.append({"model": model, "label": label, "cell": cell_name(g.iloc[0]), "error_feedback": bool(g.error_feedback.iloc[0]),
                     "kv_bits": int(g.kv_bits.iloc[0]), "state_bits": int(g.state_bits.iloc[0]),
                     "acc": float(g.correct.mean()), "acc_std_seeds": float(per_seed.std(ddof=1)) if per_seed.size > 1 else float("nan"),
                     "acc_ci_lo": ci[0], "acc_ci_hi": ci[1], "n": int(len(g)), "n_seeds": int(per_seed.size),
                     "mean_tokens": float(g.n_tokens.mean()), "budget_rate": float(g.hit_budget.mean()),
                     "repeat_ratio": float(g.repeat_ratio.mean())})
    return pd.DataFrame(rows)


def gsm8k_figures(df: pd.DataFrame, stage: str, out_dir: Path):
    out = []
    summ = gsm8k_summary(df)
    for model in sorted(summ.model.unique()):
        s = summ[summ.model == model].sort_values(["kv_bits", "state_bits"], ascending=False)
        colors = ["C3" if ef else "C0" for ef in s.error_feedback]
        err = np.array([s.acc - s.acc_ci_lo, s.acc_ci_hi - s.acc]); err = np.nan_to_num(err)
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.bar(range(len(s)), s.acc, yerr=err, capsize=3, color=colors)
        ax.set_xticks(range(len(s)), s.cell, rotation=25, fontsize=8); ax.set_ylabel("GSM8K accuracy")
        ax.set_title(f"{short_model(model)}: accuracy (blue EF off, red EF on; bars = bootstrap 95% CI over problems)", fontsize=9)
        out.append((_save(fig, out_dir, f"gsm8k_acc_{short_model(model)}"), "셀별 GSM8K 정답률(문제 부트스트랩 95% CI), EF 색 구분",
                    "GSM8K accuracy per cell (bootstrap 95% CI over problems), colour = EF"))
        sub = df[df.model == model]
        labels = list(s.label)
        data = [sub[sub.label == l].n_tokens.to_numpy(float) for l in labels]
        fig, ax = plt.subplots(figsize=(6.5, 3.6))
        ax.violinplot([d for d in data if len(d)], showmeans=True)
        ax.set_xticks(range(1, len(labels) + 1), s.cell, rotation=25, fontsize=8); ax.set_ylabel("generated tokens")
        ax.set_title(f"{short_model(model)}: generation length")
        out.append((_save(fig, out_dir, f"gsm8k_tokens_{short_model(model)}"), "셀별 생성 토큰 수 분포", "Generated-token distribution per cell"))
        fig, ax = plt.subplots(figsize=(6.5, 3.6)); w = 0.4
        ax.bar(np.arange(len(s)) - w / 2, s.budget_rate, w, label="budget exhausted rate", color=plt.get_cmap(VIRIDIS)(0.3))
        ax.bar(np.arange(len(s)) + w / 2, s.repeat_ratio, w, label="8-gram repeat ratio", color=plt.get_cmap(VIRIDIS)(0.7))
        ax.set_xticks(range(len(s)), s.cell, rotation=25, fontsize=8); ax.legend(fontsize=8); ax.set_title(f"{short_model(model)}: behaviour")
        out.append((_save(fig, out_dir, f"gsm8k_behaviour_{short_model(model)}"), "예산 소진율과 8-gram 반복률", "Budget-exhaustion rate and 8-gram repeat ratio"))
        fig, ax = plt.subplots(figsize=(5, 3.6))
        ax.scatter(s.mean_tokens, s.acc, c=colors, s=40, edgecolors="k", linewidths=0.4)
        for _, r in s.iterrows():
            ax.annotate(r.cell, (r.mean_tokens, r.acc), fontsize=6, xytext=(3, 2), textcoords="offset points")
        ax.set_xlabel("mean generated tokens"); ax.set_ylabel("accuracy"); ax.set_title(f"{short_model(model)}: accuracy vs length")
        out.append((_save(fig, out_dir, f"gsm8k_acc_vs_tokens_{short_model(model)}"), "정답률 대 평균 생성 토큰 수", "Accuracy versus mean generated tokens"))
    return out


# ================================================================ niah


def niah_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (model, label, L), g in df.groupby(["model", "label", "context_len"]):
        ci = bootstrap_ci(g.correct.to_numpy(float))
        rows.append({"model": model, "label": label, "cell": cell_name(g.iloc[0]), "error_feedback": bool(g.error_feedback.iloc[0]),
                     "kv_bits": int(g.kv_bits.iloc[0]), "state_bits": int(g.state_bits.iloc[0]), "context_len": int(L),
                     "acc": float(g.correct.mean()), "ci_lo": ci[0], "ci_hi": ci[1], "n": int(len(g))})
    return pd.DataFrame(rows)


def niah_figures(df: pd.DataFrame, stage: str, out_dir: Path, grid_df: pd.DataFrame | None = None):
    out = []
    summ = niah_summary(df)
    for model in sorted(summ.model.unique()):
        s = summ[summ.model == model]
        pv = s.pivot_table(index="cell", columns="context_len", values="acc").astype(float)
        fig, ax = plt.subplots(figsize=(4.5, 0.35 * len(pv) + 1.5))
        im = ax.imshow(pv.values, cmap=VIRIDIS, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
        ax.set_xticks(range(len(pv.columns)), pv.columns); ax.set_yticks(range(len(pv.index)), pv.index, fontsize=8)
        for i in range(pv.shape[0]):
            for j in range(pv.shape[1]):
                ax.text(j, i, f"{pv.values[i, j]:.2f}", ha="center", va="center", fontsize=7, color="w" if pv.values[i, j] < 0.5 else "k")
        fig.colorbar(im, ax=ax, shrink=0.8, label="accuracy"); ax.set_xlabel("context length"); ax.set_title(f"{short_model(model)}: NIAH accuracy")
        out.append((_save(fig, out_dir, f"niah_heatmap_{short_model(model)}"), "셀 × 컨텍스트 길이 NIAH 정확도", "NIAH accuracy, cell x context length"))
        sub = df[df.model == model]
        Ls = sorted(sub.context_len.unique())
        fig, axes = plt.subplots(1, len(Ls), figsize=(4.5 * len(Ls), 0.35 * sub.label.nunique() + 1.5), squeeze=False)
        for ax, L in zip(axes[0], Ls):
            pv = sub[sub.context_len == L].assign(cell=lambda d: d.apply(cell_name, axis=1), correct=lambda d: d.correct.astype(float)).pivot_table(index="cell", columns="depth", values="correct").astype(float)
            im = ax.imshow(pv.values, cmap=VIRIDIS, vmin=0, vmax=1, aspect="auto"); ax.grid(False)
            ax.set_xticks(range(len(pv.columns)), [f"{d:.1f}" for d in pv.columns]); ax.set_yticks(range(len(pv.index)), pv.index, fontsize=8)
            ax.set_xlabel("needle depth"); ax.set_title(f"context {L}")
        fig.colorbar(im, ax=axes[0].tolist(), shrink=0.8, label="accuracy")
        out.append((_save(fig, out_dir, f"niah_depth_{short_model(model)}"), "깊이 × 셀 정확도(길이별 패널)", "Accuracy by needle depth x cell, one panel per context length"))
        # (c) KV-only vs state-only, next to NLL deltas if a grid is available
        ko = s[(s.kv_bits < 16) & (s.state_bits >= 16)]; so = s[(s.kv_bits >= 16) & (s.state_bits < 16)]
        have_grid = grid_df is not None and not grid_df[grid_df.model == model].empty
        fig, axes = plt.subplots(1, 2 if have_grid else 1, figsize=(9 if have_grid else 5, 3.6), squeeze=False)
        ax = axes[0][0]
        rows = pd.concat([ko.assign(kind="KV-only"), so.assign(kind="state-only")])
        if not rows.empty:
            for i, (L, g) in enumerate(rows.groupby("context_len")):
                ax.bar(np.arange(len(g)) + i * 0.4, g.acc, 0.4, label=f"context {L}", color=plt.get_cmap(VIRIDIS)(0.3 + 0.4 * i))
                ax.set_xticks(np.arange(len(g)) + 0.2, g.cell, rotation=20, fontsize=8)
            ax.set_ylabel("NIAH accuracy"); ax.legend(fontsize=8); ax.set_title(f"{short_model(model)}: KV-only vs state-only (retrieval)")
        if have_grid:
            agg = aggregate_grid(grid_df[grid_df.model == model])
            a = agg[(agg.type.isin(["KV-only", "state-only"])) & (~agg.error_feedback) & (~agg.kv_noise_match) & (~agg.state_noise_match)]
            a = a[a.cell.isin(set(rows.cell))]
            ax2 = axes[0][1]
            ax2.bar(range(len(a)), a.delta, color=plt.get_cmap(VIRIDIS)(0.5)); ax2.set_xticks(range(len(a)), a.cell, rotation=20, fontsize=8)
            ax2.set_ylabel("dNLL vs fp (nats)"); ax2.set_title("same cells: language modelling (dNLL)")
        out.append((_save(fig, out_dir, f"niah_kv_vs_state_{short_model(model)}"),
                    "KV-only 대 state-only: 회수 정확도(왼쪽)와 같은 셀의 ΔNLL(오른쪽), 태스크별 민감도 비교",
                    "KV-only versus state-only: retrieval accuracy (left) and dNLL for the same cells (right)"))
    return out


# ================================================================ scale and summary


def scale_figure(grid_df: pd.DataFrame, out_dir: Path):
    agg = aggregate_grid(grid_df)
    agg = agg[(~agg.kv_noise_match) & (~agg.state_noise_match) & (~agg.error_feedback)]
    agg["size_b"] = agg.model.map(model_size_b)
    agg = agg[agg.size_b.notna() & (agg.label != FP_LABEL)]
    agg = agg[[(a, b) in CORE_CELLS for a, b in zip(agg.kv_bits, agg.state_bits)]]
    if agg.model.nunique() < 2:
        return None
    fig, ax = plt.subplots(figsize=(6, 3.8))
    cells = sorted(agg.cell.unique()); cmap = plt.get_cmap(VIRIDIS)
    for i, cell in enumerate(cells):
        s = agg[agg.cell == cell]
        q = s[~s.model.str.contains("Falcon")].sort_values("size_b"); f = s[s.model.str.contains("Falcon")]
        ax.plot(q.size_b, q.delta, "o-", color=cmap(i / max(1, len(cells) - 1)), label=cell)
        if len(f):
            ax.scatter(f.size_b, f.delta, marker="^", s=60, color=cmap(i / max(1, len(cells) - 1)), edgecolors="k")
    ax.set_xscale("log"); ax.set_xlabel("model size (B parameters, log)"); ax.set_ylabel("dNLL vs fp (nats)")
    ax.set_title("dNLL across model sizes (circles Qwen3.5, triangles Falcon-H1)"); ax.legend(fontsize=7, ncol=2)
    return _save(fig, out_dir, "scale"), "모델 크기별 ΔNLL, 셀별 선, Falcon-H1은 삼각형", \
        "dNLL versus model size per cell; Falcon-H1 as triangles"


def summary_figure(figure_bases: list, out_dir: Path):
    """2 x 4 overview assembled from the PNGs of up to eight representative figures."""
    import matplotlib.image as mpimg
    bases = [b for b in figure_bases if os.path.exists(str(b) + ".png")][:8]
    if not bases:
        return None
    fig, axes = plt.subplots(2, 4, figsize=(20, 8))
    for ax in axes.flat:
        ax.axis("off")
    for ax, b in zip(axes.flat, bases):
        ax.imshow(mpimg.imread(str(b) + ".png")); ax.set_title(os.path.basename(str(b)), fontsize=8)
    return _save(fig, out_dir, "summary_overview"), "대표 그림 8장 개요", "Overview of eight representative figures"
