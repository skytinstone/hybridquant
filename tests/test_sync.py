"""scripts/sync_results.sh dry run against a temporary clone (no network, no files copied)."""
from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.skipif(shutil.which("rsync") is None or shutil.which("git") is None, reason="needs rsync and git")
def test_sync_results_dry_run(tmp_path):
    clone = tmp_path / "clone"
    subprocess.run(["git", "clone", "-q", str(ROOT), str(clone)], check=True)
    (clone / "results").mkdir(exist_ok=True)
    (clone / "results" / "dummy.jsonl").write_text('{"a": 1}\n')
    dest_root = tmp_path / "downloads"
    env = {**os.environ, "HQ_SYNC_DRY_RUN": "1"}
    out = subprocess.run(["bash", str(ROOT / "scripts" / "sync_results.sh"), str(clone), str(dest_root)],
                         capture_output=True, text=True, env=env, timeout=120)
    assert out.returncode == 0, out.stderr
    assert "copied results/" in out.stdout
    last = (dest_root / "last_sync.txt").read_text()
    assert "pull=" in last and "copied results/" in last
    day_dirs = [p for p in dest_root.iterdir() if p.is_dir()]
    assert len(day_dirs) == 1
    assert not (day_dirs[0] / "dummy.jsonl").exists()      # dry run copies nothing
