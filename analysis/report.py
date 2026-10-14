"""Bilingual stage report from results/*.jsonl (numbers only from the files; nothing is typed in by hand).

    python analysis/report.py --stage grid_0.8b --inputs results/grid_0.8b.jsonl --out results/reports/grid_0.8b.md
    python analysis/report.py --stage all --inputs results/*.jsonl --out results/reports/summary.md

Sections, always in this order: [KR] 핵심 요약, [EN] Key Findings, [KR/EN] 표, [KR/EN] 그림 목록, 조건.
Items without data are written as "N/A (아직 실행 안 됨)". Differences whose paired bootstrap 95% CI
contains zero are marked "구분 불가 / not distinguishable".
"""
from __future__ import annotations

import argparse
import datetime as dt
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FP_LABEL, ROOT, T_BUDGET, aggregate_grid, bootstrap_ci, borrow_fp_rows, ci_contains_zero, \
    drop_cross_file_duplicates, fmt_ci, load_jsonl, \
    run_conditions, short_model, spearman, split_by_kind  # noqa: E402
import figures as F  # noqa: E402

NA = "N/A (아직 실행 안 됨)"
FIG_ROOT = ROOT / "results" / "figures"


def _d(ci) -> str:
    return " (구분 불가 / not distinguishable)" if ci_contains_zero(ci) else ""


# ---------------------------------------------------------------- grid-like


def grid_findings(df: pd.DataFrame):
    agg = aggregate_grid(df)
    kr, en, tables = [], [], []
    for model in sorted(agg.model.unique()):
        for dataset in sorted(agg[agg.model == model].dataset.unique()):
            a = agg[(agg.model == model) & (agg.dataset == dataset)]
            plain = a[(~a.kv_noise_match) & (~a.state_noise_match) & (~a.error_feedback)].set_index("label")
            name = f"{short_model(model)} / {dataset}"
            if FP_LABEL not in plain.index:
                kr.append(f"- {name}: fp 셀이 없어 Δ를 계산할 수 없습니다 ({NA})."); en.append(f"- {name}: no fp cell, deltas unavailable."); continue
            fp = plain.loc[FP_LABEL]
            kr.append(f"- {name}: fp 베이스라인 mean NLL {fp.nll_mean:.3f} (시드 {fp.n_seeds}개, 문서 {fp.n_docs}개/시드).")
            en.append(f"- {name}: fp baseline mean NLL {fp.nll_mean:.3f} ({fp.n_seeds} seeds, {fp.n_docs} docs per seed).")
            for lab, kr_name, en_name in (("Kfp_Vfp_Sint4g128", "state 4bit (KV fp)", "state 4-bit (KV fp)"),
                                          ("Kint4g128_Vint4g128_Sfp", "KV 4bit (state fp)", "KV 4-bit (state fp)"),
                                          ("Kfp_Vfp_Sint8g128", "state 8bit", "state 8-bit"),
                                          ("Kint2g128_Vint2g128_Sfp", "KV 2bit", "KV 2-bit")):
                if lab in plain.index:
                    r = plain.loc[lab]; ci = (r.ci_lo, r.ci_hi)
                    kr.append(f"- {kr_name}: fp 대비 {r.delta:+.3f} nats (95% CI {fmt_ci(ci)}){_d(ci)}.")
                    en.append(f"- {en_name}: {r.delta:+.3f} nats over fp (95% CI {fmt_ci(ci)}){_d(ci)}.")
            # superadditivity for the (4,4) corner
            need = ["Kfp_Vfp_Sint4g128", "Kint4g128_Vint4g128_Sfp", "Kint4g128_Vint4g128_Sint4g128"]
            if all(n in plain.index for n in need):
                ex = F.per_doc_excess(df[(df.model == model) & (df.dataset == dataset)], model)
                ci = bootstrap_ci(ex)
                excess = plain.loc[need[2]].delta - plain.loc[need[0]].delta - plain.loc[need[1]].delta
                kr.append(f"- KV4+S4 초가산: 실측 Δ {plain.loc[need[2]].delta:+.3f} 대 단독합 {plain.loc[need[0]].delta + plain.loc[need[1]].delta:+.3f}, 초과분 {excess:+.3f} nats (문서 대응 95% CI {fmt_ci(ci)}){_d(ci)}.")
                en.append(f"- KV4+S4 superadditivity: measured {plain.loc[need[2]].delta:+.3f} vs sum of single effects {plain.loc[need[0]].delta + plain.loc[need[1]].delta:+.3f}; excess {excess:+.3f} nats (paired 95% CI {fmt_ci(ci)}){_d(ci)}.")
            # error feedback effect
            ef = a[a.error_feedback & (~a.kv_noise_match) & (~a.state_noise_match)]
            if len(ef):
                diffs = []
                for _, r in ef.iterrows():
                    base_label = r.label.replace("ef", "")
                    if base_label in plain.index:
                        diffs.append(r.delta - plain.loc[base_label].delta)
                if diffs:
                    kr.append(f"- error feedback: 같은 비트의 RTN 대비 ΔNLL 변화 평균 {np.mean(diffs):+.3f} nats ({len(diffs)}개 셀, 범위 {min(diffs):+.3f}~{max(diffs):+.3f}).")
                    en.append(f"- error feedback: mean change in dNLL versus RTN at the same bits {np.mean(diffs):+.3f} nats ({len(diffs)} cells, range {min(diffs):+.3f}~{max(diffs):+.3f}).")
            # noise match
            for gn, real, what_kr, what_en in (("Kint4g128gn_Vint4g128gn_Sint4g128", "Kint4g128_Vint4g128_Sint4g128", "KV 노이즈 대조", "KV noise control"),
                                                ("Kint4g128_Vint4g128_Sint4g128gn", "Kint4g128_Vint4g128_Sint4g128", "state 노이즈 대조", "state noise control")):
                b = a.set_index("label")
                if gn in b.index and real in b.index:
                    kr.append(f"- {what_kr}(같은 RMS의 가우시안 노이즈): Δ {b.loc[gn].delta:+.3f} 대 실제 양자화 {b.loc[real].delta:+.3f} nats.")
                    en.append(f"- {what_en} (Gaussian noise of equal RMS): {b.loc[gn].delta:+.3f} vs real quantization {b.loc[real].delta:+.3f} nats.")
            tables.append((name, a))
    return kr, en, tables


def grid_table_md(a: pd.DataFrame) -> str:
    cols = ["cell", "nll_mean", "nll_std", "std_kind", "delta", "ci", "mem_MB", "n_seeds", "n_docs"]
    hdr = "| 셀 (cell) | 평균 NLL (mean NLL) | ± std (시드/문서) | Δ vs fp (delta) | 95% CI (paired bootstrap) | 총 메모리 MB, T=512 (memory) | 시드 (seeds) | 문서/시드 (docs) |"
    lines = [hdr, "|---|---|---|---|---|---|---|---|"]
    for _, r in a.sort_values(["kv_bits", "state_bits", "error_feedback"], ascending=[False, False, True]).iterrows():
        ci = (r.ci_lo, r.ci_hi)
        mem = f"{r.mem_bytes_T512 / 1e6:.2f}" if np.isfinite(r.mem_bytes_T512) else "N/A"
        lines.append(f"| {r.cell} | {r.nll_mean:.4f} | ± {r.nll_std:.4f} ({r.std_kind}) | {r.delta:+.4f} | {fmt_ci(ci)}{' *' if ci_contains_zero(ci) else ''} | {mem} | {r.n_seeds} | {r.n_docs} |")
    lines.append("\n\\* CI가 0을 포함: fp와 구분 불가 / CI contains zero: not distinguishable from fp. 메모리는 비트 × 원소 수이며 scale/zero 오버헤드는 뺀 값입니다 / memory is bits x elements without scale/zero overhead.")
    return "\n".join(lines)


# ---------------------------------------------------------------- layer sweep


def layer_findings(df: pd.DataFrame):
    kr, en, tables = [], [], []
    for model in sorted(df.model.unique()):
        sub = df[(df.model == model) & (df.layer_idx >= 0)].sort_values("layer_idx")
        if sub.empty:
            continue
        top = sub.loc[sub.delta_nll.idxmax()]; low = sub.loc[sub.delta_nll.idxmin()]
        kr.append(f"- {short_model(model)}: 레이어 하나만 state int{int(sub.state_bits.iloc[0])}로 양자화했을 때 ΔNLL 최대는 레이어 {int(top.layer_idx)} ({top.delta_nll:+.4f}), 최소는 레이어 {int(low.layer_idx)} ({low.delta_nll:+.4f}), 전체 합 {sub.delta_nll.sum():+.4f}.")
        en.append(f"- {short_model(model)}: quantizing one layer's state to int{int(sub.state_bits.iloc[0])}, the largest dNLL is layer {int(top.layer_idx)} ({top.delta_nll:+.4f}) and the smallest layer {int(low.layer_idx)} ({low.delta_nll:+.4f}); sum over layers {sub.delta_nll.sum():+.4f}.")
        if "n_half" in sub:
            rho, p = spearman(sub.n_half, sub.delta_nll)
            kr.append(f"- n_half(오차 반감 토큰 수)와 ΔNLL의 Spearman 상관 {rho:+.2f} (순열 p={p:.3f}).")
            en.append(f"- Spearman correlation between n_half (error half-life) and dNLL: {rho:+.2f} (permutation p={p:.3f}).")
        t = sub[["layer_idx", "delta_nll"] + [c for c in ("mean_alpha", "n_half", "mean_beta") if c in sub]].copy()
        tables.append((short_model(model), t))
    return kr, en, tables


# ---------------------------------------------------------------- gsm8k / niah


def gsm8k_findings(df: pd.DataFrame):
    summ = F.gsm8k_summary(df)
    kr, en, tables = [], [], []
    for model in sorted(summ.model.unique()):
        s = summ[summ.model == model].set_index("label")
        if FP_LABEL in s.index:
            fp = s.loc[FP_LABEL]
            kr.append(f"- {short_model(model)} GSM8K fp 정답률 {fp.acc:.3f} (문제 {fp.n}개, 95% CI {fp.acc_ci_lo:.3f}~{fp.acc_ci_hi:.3f}), 평균 생성 토큰 {fp.mean_tokens:.0f}, 예산 소진율 {fp.budget_rate:.2f}.")
            en.append(f"- {short_model(model)} GSM8K fp accuracy {fp.acc:.3f} ({fp.n} problems, 95% CI {fp.acc_ci_lo:.3f}~{fp.acc_ci_hi:.3f}), mean {fp.mean_tokens:.0f} tokens, budget-exhaustion {fp.budget_rate:.2f}.")
            for lab, r in s.iterrows():
                if lab == FP_LABEL:
                    continue
                kr.append(f"- {r.cell}: 정답률 {r.acc:.3f} (fp 대비 {r.acc - fp.acc:+.3f}), 토큰 {r.mean_tokens:.0f}, 반복률 {r.repeat_ratio:.3f}.")
                en.append(f"- {r.cell}: accuracy {r.acc:.3f} ({r.acc - fp.acc:+.3f} vs fp), {r.mean_tokens:.0f} tokens, repeat ratio {r.repeat_ratio:.3f}.")
        tables.append((short_model(model), summ[summ.model == model]))
    return kr, en, tables


def niah_findings(df: pd.DataFrame):
    summ = F.niah_summary(df)
    kr, en, tables = [], [], []
    for model in sorted(summ.model.unique()):
        s = summ[summ.model == model]
        for L in sorted(s.context_len.unique()):
            t = s[s.context_len == L].set_index("label")
            if FP_LABEL in t.index:
                fp = t.loc[FP_LABEL]
                worst = t.loc[t.acc.idxmin()]
                kr.append(f"- {short_model(model)} NIAH 컨텍스트 {L}: fp 정확도 {fp.acc:.2f} (문제 {fp.n}개), 가장 낮은 셀 {worst.cell} {worst.acc:.2f}.")
                en.append(f"- {short_model(model)} NIAH context {L}: fp accuracy {fp.acc:.2f} ({fp.n} problems); lowest cell {worst.cell} at {worst.acc:.2f}.")
        tables.append((short_model(model), s))
    return kr, en, tables


# ---------------------------------------------------------------- assembly


def df_to_md(t: pd.DataFrame, floatfmt: str = "{:.4f}") -> str:
    t = t.copy()
    for c in t.columns:
        if t[c].dtype.kind == "f":
            t[c] = t[c].map(lambda v: "N/A" if pd.isna(v) else floatfmt.format(v))
    lines = ["| " + " | ".join(map(str, t.columns)) + " |", "|" + "---|" * len(t.columns)]
    for _, r in t.iterrows():
        lines.append("| " + " | ".join(str(v) for v in r.values) + " |")
    return "\n".join(lines)


def build_report(stage: str, inputs: list[str], out_path: str) -> str:
    df = load_jsonl(inputs)
    if len(df) and stage == "all":
        # smoke runs (few docs, one seed) must not be pooled with the full grids of the same model
        df = df[~df.source.str.startswith("smoke")].reset_index(drop=True)
        df = drop_cross_file_duplicates(df)
    if len(df):
        df = borrow_fp_rows(df)
    fig_dir = FIG_ROOT / stage
    kr, en, tables, figs = [], [], [], []
    kinds = split_by_kind(df) if len(df) else {}
    grid_df = kinds.get("grid")
    if "grid" in kinds:
        k, e, t = grid_findings(kinds["grid"]); kr += k; en += e; tables += [(n, grid_table_md(a)) for n, a in t]
        srcs = set(kinds["grid"].source)
        if any("superadd" in s for s in srcs):
            figs += F.superadd_figures(kinds["grid"][kinds["grid"].source.str.contains("superadd")], stage, fig_dir)
        if any("ablation" in s for s in srcs):
            figs += F.ablation_figures(kinds["grid"][kinds["grid"].source.str.contains("ablation")], stage, fig_dir)
        rest = kinds["grid"][~kinds["grid"].source.str.contains("superadd|ablation")]
        if len(rest):
            figs += F.grid_figures(rest, stage, fig_dir)
        if stage == "all":
            r = F.scale_figure(rest, fig_dir)
            if r: figs.append(r)
    if "layer_sweep" in kinds:
        k, e, t = layer_findings(kinds["layer_sweep"]); kr += k; en += e; tables += [(n, df_to_md(a)) for n, a in t]
        figs += F.layer_sweep_figures(kinds["layer_sweep"], stage, fig_dir)
    if "gsm8k" in kinds:
        k, e, t = gsm8k_findings(kinds["gsm8k"]); kr += k; en += e
        tables += [(n, df_to_md(a.drop(columns=["label"]))) for n, a in t]
        figs += F.gsm8k_figures(kinds["gsm8k"], stage, fig_dir)
    if "niah" in kinds:
        k, e, t = niah_findings(kinds["niah"]); kr += k; en += e
        tables += [(n, df_to_md(a.drop(columns=["label"]))) for n, a in t]
        figs += F.niah_figures(kinds["niah"], stage, fig_dir, grid_df)
    if stage == "all" and figs:
        r = F.summary_figure([f[0] for f in figs], fig_dir)
        if r: figs.append(r)

    cond = run_conditions(df) if len(df) else {}
    lines = [f"# Report: {stage}", f"generated {dt.datetime.now(dt.timezone.utc).isoformat(timespec='seconds')} from {', '.join(os.path.basename(p) for p in inputs if os.path.exists(p)) or 'no files'}", ""]
    lines += ["## [KR] 핵심 요약"] + (kr or [f"- {NA}"]) + [""]
    lines += ["## [EN] Key Findings"] + (en or ["- N/A (not run yet)"]) + [""]
    lines += ["## [KR/EN] 표"]
    if tables:
        for name, md in tables:
            lines += [f"### {name}", md, ""]
    else:
        lines += [NA, ""]
    lines += ["## [KR/EN] 그림 목록"]
    if figs:
        for base, cap_kr, cap_en in figs:
            rel = os.path.relpath(str(base), ROOT)
            lines.append(f"- `{rel}.pdf` / `.png`: {cap_kr} / {cap_en}")
    else:
        lines.append(NA)
    lines += ["", "## 조건"]
    if cond:
        j = lambda xs: ", ".join(str(x) for x in xs) if xs else "N/A"
        lines += [f"- 모델 (models): {j(cond['models'])}", f"- GPU: {j(cond['gpus'])}", f"- dtype: {j(cond['dtypes'])}",
                  f"- commit: {j(cond['commits'])}", f"- 날짜 (dates): {j(cond['dates'])}",
                  f"- 시드 수 (seeds): {cond['n_seeds']}, 문서 수/시드 (docs): {cond['n_docs']}, 행 (rows): {cond['n_rows']}",
                  f"- 입력 파일 (sources): {', '.join(cond['sources'])}"]
        fails = ROOT / "results" / "push_failures.log"
        lines.append(f"- 실패/건너뜀 기록 (failures): {'see results/push_failures.log' if fails.exists() else 'none logged'}")
    else:
        lines.append(NA)
    text = "\n".join(lines) + "\n"
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(text)
    return text


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--stage", required=True)
    p.add_argument("--inputs", nargs="+", required=True)
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)
    out = args.out or str(ROOT / "results" / "reports" / f"{args.stage}.md")
    text = build_report(args.stage, args.inputs, out)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
