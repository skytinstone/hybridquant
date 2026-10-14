"""Duplicate result rows (a runner started twice) are dropped by the report loader and by scripts/dedupe_results.py."""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "analysis"))
import common  # noqa: E402


def _row(label, seed, nll, **extra):
    return {"model": "test/Tiny", "label": label, "seed": seed, "kv_bits": 4, "state_bits": 4, "mean_nll": nll,
            "per_doc_nll": [nll], **extra}


def test_load_jsonl_keeps_first_row_per_run(tmp_path):
    p = tmp_path / "grid.jsonl"
    rows = [_row("A", 0, 1.0), _row("B", 0, 2.0), _row("A", 0, 9.0), _row("A", 1, 3.0), _row("B", 0, 9.0)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    df = common.load_jsonl([p])
    assert len(df) == 3
    assert df[(df.label == "A") & (df.seed == 0)].mean_nll.item() == 1.0   # first occurrence wins


def test_load_jsonl_dedupes_within_file_only(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    a.write_text(json.dumps(_row("A", 0, 1.0)) + "\n")
    b.write_text(json.dumps(_row("A", 0, 2.0)) + "\n")
    assert len(common.load_jsonl([a, b])) == 2


def test_task_rows_use_problem_index(tmp_path):
    p = tmp_path / "niah.jsonl"
    rows = [_row("A", 0, 0, context_len=4096, depth=0.1, idx=i) for i in range(3)] + \
           [_row("A", 0, 0, context_len=4096, depth=0.1, idx=1)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    assert len(common.load_jsonl([p])) == 3


def test_dedupe_script_rewrites_file(tmp_path):
    p = tmp_path / "grid.jsonl"
    rows = [_row("A", 0, 1.0), _row("A", 0, 9.0), _row("B", 0, 2.0)]
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))
    out = subprocess.run([sys.executable, str(ROOT / "scripts/dedupe_results.py"), str(p), "--dry-run"],
                         capture_output=True, text=True, check=True).stdout
    assert "3 ->      2" in out and len(p.read_text().splitlines()) == 3      # dry run leaves the file alone
    subprocess.run([sys.executable, str(ROOT / "scripts/dedupe_results.py"), str(p)], check=True)
    kept = [json.loads(l) for l in p.read_text().splitlines()]
    assert [(r["label"], r["mean_nll"]) for r in kept] == [("A", 1.0), ("B", 2.0)]
