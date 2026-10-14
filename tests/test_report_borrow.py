"""borrow_fp_rows fills a missing fp cell from a sibling results file; drop_cross_file_duplicates keeps one row per run."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import common  # noqa: E402

MODEL = "test/Tiny"


def _row(label, seed, nll, source, kv=4, st=6, n_docs=6):
    return {"model": MODEL, "dataset": "wikitext2", "label": label, "seed": seed, "kv_bits": kv, "state_bits": st,
            "prefill": 256, "decode": 256, "n_docs": n_docs, "mean_nll": nll, "per_doc_nll": [nll] * n_docs,
            "error_feedback": False, "source": source}


def _write(path, rows):
    path.write_text("".join(json.dumps({k: v for k, v in r.items() if k != "source"}) + "\n" for r in rows))


def test_borrow_fp_rows_from_sibling_file_not_smoke(tmp_path):
    ablation = pd.DataFrame([_row("Kint4g128_Vint4g128_Sint6g32", s, 3.5, "r2_ablation.jsonl") for s in (0, 1)])
    _write(tmp_path / "grid_0.8b.jsonl", [_row(common.FP_LABEL, s, 3.0, "", kv=16, st=16) for s in (0, 1)])
    _write(tmp_path / "superadd_0.8b.jsonl", [_row(common.FP_LABEL, s, 3.0, "", kv=16, st=16) for s in (0, 1)])
    _write(tmp_path / "smoke_0.8b.jsonl", [_row(common.FP_LABEL, 0, 9.9, "", kv=16, st=16, n_docs=6)])
    out = common.borrow_fp_rows(common.ensure_defaults(ablation), results_dir=tmp_path)
    fp = out[out.label == common.FP_LABEL]
    assert sorted(fp.seed.tolist()) == [0, 1]                       # one fp row per seed, not one per file
    assert fp.mean_nll.eq(3.0).all()                                 # smoke's 9.9 was never used
    assert fp.source.str.contains("fp borrowed").all()
    assert len(out) == 4


def test_borrow_fp_rows_is_noop_when_fp_present(tmp_path):
    df = common.ensure_defaults(pd.DataFrame([_row(common.FP_LABEL, 0, 3.0, "grid.jsonl", kv=16, st=16),
                                              _row("Kint4g128_Vint4g128_Sint6g32", 0, 3.5, "grid.jsonl")]))
    _write(tmp_path / "other.jsonl", [_row(common.FP_LABEL, 0, 1.0, "", kv=16, st=16)])
    out = common.borrow_fp_rows(df, results_dir=tmp_path)
    assert len(out) == 2 and out[out.label == common.FP_LABEL].mean_nll.item() == 3.0


def test_borrow_fp_rows_requires_same_documents(tmp_path):
    df = common.ensure_defaults(pd.DataFrame([_row("Kint4g128_Vint4g128_Sint6g32", 0, 3.5, "abl.jsonl", n_docs=20)]))
    _write(tmp_path / "grid.jsonl", [_row(common.FP_LABEL, 0, 3.0, "", kv=16, st=16, n_docs=8)])   # different docs
    assert len(common.borrow_fp_rows(df, results_dir=tmp_path)) == 1


def test_drop_cross_file_duplicates_keeps_first_source():
    df = pd.DataFrame([_row("A", 0, 1.0, "grid.jsonl"), _row("A", 0, 1.0, "superadd.jsonl"), _row("A", 1, 2.0, "superadd.jsonl")])
    out = common.drop_cross_file_duplicates(df)
    assert len(out) == 2 and out[out.seed == 0].source.item() == "grid.jsonl"
