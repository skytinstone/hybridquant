"""report.py and figures.py on synthetic jsonl files: bootstrap CI, tables, section order, every figure saved."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import common  # noqa: E402
import figures as F  # noqa: E402
import report  # noqa: E402

MODEL = "test/Tiny-0.8B"
LABELS = {(16, 16): "Kfp_Vfp_Sfp", (16, 4): "Kfp_Vfp_Sint4g128", (4, 16): "Kint4g128_Vint4g128_Sfp",
          (4, 4): "Kint4g128_Vint4g128_Sint4g128", (16, 8): "Kfp_Vfp_Sint8g128", (2, 8): "Kint2g128_Vint2g128_Sint8g128"}


def _grid_rows(rng, seeds=(0, 1, 2), n_docs=6, ef_cells=((16, 4), (4, 4)), noise=False, source_model=MODEL):
    rows = []
    for seed in seeds:
        base = rng.normal(3.3, 0.4, n_docs)
        for (kv, st), label in LABELS.items():
            shift = (1.3 if st == 4 else 0.02 if st == 8 else 0) + (0.01 if kv == 4 else 0.6 if kv == 2 else 0)
            for ef in ([False, True] if (kv, st) in ef_cells else [False]):
                per_doc = (base + shift - (0.2 if ef else 0) + rng.normal(0, 0.05, n_docs)).tolist()
                rows.append({"model": source_model, "dtype": "bf16", "dataset": "wikitext2", "label": label + ("ef" if ef else ""),
                             "kv_bits": kv, "state_bits": st, "kv_group": 128, "state_group": 128, "state_symmetric": False,
                             "state_stochastic": False, "kv_noise_match": False, "state_noise_match": False, "error_feedback": ef,
                             "state_sym": False, "state_sr": False, "prefill": 256, "decode": 256, "step": 1, "batch_size": 8,
                             "n_docs": n_docs, "seed": seed, "mean_nll": float(np.mean(per_doc)), "std_nll": float(np.std(per_doc)),
                             "per_doc_nll": per_doc, "nll_curve": (np.array(per_doc).mean() + rng.normal(0, 0.3, 64)).tolist(),
                             "nll_curve_std": (0.3 + rng.random(64) * 0.1).tolist(), "sec": 1.0, "config": "x.yaml",
                             "meta": {"git_commit": "abc1234", "device": "TestGPU", "timestamp_utc": "2026-10-14T00:00:00+00:00"}})
        if noise:
            for gn_kv, gn_st, label in ((True, False, "Kint4g128gn_Vint4g128gn_Sint4g128"), (False, True, "Kint4g128_Vint4g128_Sint4g128gn")):
                per_doc = (base + 1.35 + (0.5 if gn_st else 0) + rng.normal(0, 0.05, n_docs)).tolist()
                rows.append({**rows[-1], "label": label, "kv_bits": 4, "state_bits": 4, "kv_noise_match": gn_kv, "state_noise_match": gn_st,
                             "error_feedback": False, "seed": seed, "mean_nll": float(np.mean(per_doc)), "per_doc_nll": per_doc})
    return rows


def _write(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    return str(path)


@pytest.fixture
def synthetic(tmp_path, monkeypatch):
    rng = np.random.default_rng(0)
    monkeypatch.setattr(report, "FIG_ROOT", tmp_path / "figs")
    files = {"grid": _write(tmp_path / "grid_test.jsonl", _grid_rows(rng)),
             "superadd": _write(tmp_path / "superadd_test.jsonl", _grid_rows(rng, ef_cells=(), noise=True)),
             "grid2": _write(tmp_path / "r2_scale_test.jsonl", _grid_rows(rng, source_model="test/Tiny-4B"))}
    ab = []
    for g in (32, 64, 128, -1):
        for sym in (False, True):
            for sr in (False, True):
                for seed in (0, 1):
                    per_doc = (rng.normal(4.4, 0.3, 4) - (0.02 if g == 32 else 0)).tolist()
                    ab.append({**_grid_rows(rng, seeds=(seed,), n_docs=4)[0], "label": f"Kint4g128_Vint4g128_Sint4g{g}{'s' if sym else ''}{'sr' if sr else ''}",
                               "kv_bits": 4, "state_bits": 4, "state_group": g, "state_sym": sym, "state_sr": sr,
                               "state_symmetric": sym, "state_stochastic": sr, "error_feedback": False, "per_doc_nll": per_doc,
                               "mean_nll": float(np.mean(per_doc))})
    ab += _grid_rows(rng, seeds=(0, 1), n_docs=4)[:1] * 1   # fp row for seed 0
    files["ablation"] = _write(tmp_path / "r2_ablation_test.jsonl", ab)
    ls = [{"model": MODEL, "dataset": "wikitext2", "state_bits": 4, "layer_idx": -1, "label": "fp", "mean_nll": 3.3, "delta_nll": 0.0,
           "per_doc_nll": [3.3] * 4, "seed": 0, "n_docs": 4, "meta": {"git_commit": "abc1234", "device": "TestGPU"}}]
    for i in range(6):
        ls.append({**ls[0], "layer_idx": i, "label": f"S4@{i}", "delta_nll": 0.02 * i + rng.normal(0, 0.005), "mean_alpha": 0.5 + 0.08 * i,
                   "n_half": 1.0 + 0.5 * i, "mean_beta": 0.4 + 0.02 * i})
    files["layer"] = _write(tmp_path / "layer_sweep_test.jsonl", ls)
    gs = []
    for (kv, st), label in list(LABELS.items())[:4]:
        for ef in ([False, True] if st < 16 else [False]):
            for idx in range(30):
                n_tok = int(rng.integers(200, 2000))
                gs.append({"model": "test/Tiny-4B", "label": label + ("ef" if ef else ""), "seed": 0, "kv_bits": kv, "state_bits": st,
                           "error_feedback": ef, "kv_noise_match": False, "state_noise_match": False, "idx": idx, "gold": "1",
                           "answer": "1", "correct": bool(rng.random() < (0.8 - 0.3 * (st == 4))), "n_tokens": n_tok,
                           "hit_budget": n_tok > 1900, "repeat_ratio": float(rng.random() * 0.05),
                           "meta": {"git_commit": "abc1234", "device": "TestGPU"}})
    files["gsm8k"] = _write(tmp_path / "r2_gsm8k_test.jsonl", gs)
    ni = []
    for (kv, st), label in list(LABELS.items())[:4]:
        for L in (4096, 16384):
            for d in (0.1, 0.5, 0.9):
                for idx in range(4):
                    ni.append({"model": MODEL, "label": label, "seed": 0, "kv_bits": kv, "state_bits": st, "error_feedback": False,
                               "kv_noise_match": False, "state_noise_match": False, "context_len": L, "depth": d, "idx": idx,
                               "code": "ABC123", "answer": "ABC123", "correct": bool(rng.random() < (0.9 - 0.4 * (kv == 4))),
                               "meta": {"git_commit": "abc1234", "device": "TestGPU"}})
    files["niah"] = _write(tmp_path / "r2_niah_test.jsonl", ni)
    return tmp_path, files


def test_bootstrap_ci_and_helpers():
    rng = np.random.default_rng(1)
    x = rng.normal(1.0, 0.1, 200)
    lo, hi = common.bootstrap_ci(x)
    assert lo < 1.0 < hi and hi - lo < 0.1
    assert common.ci_contains_zero(common.bootstrap_ci(rng.normal(0, 1, 100)))
    assert common.bootstrap_ci([1.0]) == (float("nan"), float("nan")) or np.isnan(common.bootstrap_ci([1.0])[0])
    rho, p = common.spearman([1, 2, 3, 4, 5, 6], [1, 2, 3, 5, 4, 6])
    assert rho > 0.9 and p < 0.1
    assert common.cell_name({"kv_bits": 4, "state_bits": 16, "error_feedback": True}) == "KV4+fp[EF]"


def test_aggregate_grid_deltas_and_ci(synthetic):
    tmp, files = synthetic
    df = common.load_jsonl([files["grid"]])
    agg = common.aggregate_grid(df)
    a = agg.set_index("label")
    assert abs(a.loc["Kfp_Vfp_Sfp", "delta"]) < 1e-9
    s4 = a.loc["Kfp_Vfp_Sint4g128"]
    assert 1.2 < s4.delta < 1.4 and s4.ci_lo < s4.delta < s4.ci_hi and s4.n_seeds == 3 and s4.std_kind == "seeds"
    assert s4.n_paired == 18
    kv4 = a.loc["Kint4g128_Vint4g128_Sfp"]
    assert kv4.ci_lo < 0.05 and kv4.ci_hi > -0.05
    assert np.isnan(s4.mem_bytes_T512)          # unknown model: memory N/A, not a crash


def test_reports_have_sections_in_order_and_figures(synthetic):
    tmp, files = synthetic
    for stage, inputs in [("grid_test", [files["grid"]]), ("superadd_test", [files["superadd"]]),
                          ("layer_sweep_test", [files["layer"]]), ("ablation_test", [files["ablation"]]),
                          ("gsm8k_test", [files["gsm8k"]]), ("niah_test", [files["niah"], files["grid"]]),
                          ("all", list(files.values()))]:
        out = tmp / "reports" / f"{stage}.md"
        text = report.build_report(stage, inputs, str(out))
        assert out.exists()
        heads = [h for h in ("## [KR] 핵심 요약", "## [EN] Key Findings", "## [KR/EN] 표", "## [KR/EN] 그림 목록", "## 조건") if h in text]
        assert heads == ["## [KR] 핵심 요약", "## [EN] Key Findings", "## [KR/EN] 표", "## [KR/EN] 그림 목록", "## 조건"], stage
        assert text.index("## [KR] 핵심 요약") < text.index("## [EN] Key Findings") < text.index("## [KR/EN] 표") \
            < text.index("## [KR/EN] 그림 목록") < text.index("## 조건")
        figs = sorted((tmp / "figs" / stage).glob("*.png")) if (tmp / "figs" / stage).exists() else []
        assert figs, f"no figures for {stage}"
        for png in figs:
            assert png.with_suffix(".pdf").exists()
        assert "TestGPU" in text and "abc1234" in text
    text = (tmp / "reports" / "all.md").read_text()
    assert "summary_overview" in text and "scale" in text
    assert "구분 불가" in text                       # KV4 vs fp is within noise in the synthetic data


def test_report_with_no_data_is_na(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "FIG_ROOT", tmp_path / "figs")
    text = report.build_report("empty", [str(tmp_path / "missing.jsonl")], str(tmp_path / "empty.md"))
    assert text.count("N/A (아직 실행 안 됨)") >= 3
