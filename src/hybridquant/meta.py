"""Provenance for experiment outputs: which code, which libraries, which device, when."""
from __future__ import annotations

import datetime as _dt
import platform
import subprocess
from pathlib import Path

import torch

__all__ = ["run_metadata", "REQUIRED_KEYS"]

REQUIRED_KEYS = ("git_commit", "git_dirty", "transformers", "torch", "device", "python", "timestamp_utc")


def _git(args: list[str], cwd: Path) -> str | None:
    try:
        out = subprocess.run(["git", *args], cwd=cwd, capture_output=True, text=True, timeout=5)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def run_metadata() -> dict:
    """Return a JSON-serialisable dict describing the current run.

    Keys (all always present):
      git_commit     short hash, or "unknown" outside a git checkout
      git_dirty      True if the working tree has uncommitted changes
      transformers   installed version string
      torch          installed version string
      device         CUDA device name, "mps", or "cpu"
      python         interpreter version
      timestamp_utc  ISO 8601, UTC
      patches        work-arounds applied by load_model in this process (see compat.py)
    """
    repo = Path(__file__).resolve().parents[2]
    commit = _git(["rev-parse", "--short", "HEAD"], repo) or "unknown"
    status = _git(["status", "--porcelain"], repo)
    dirty = bool(status) if status is not None else False

    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(0)
    elif getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    from .compat import APPLIED

    try:
        import transformers
        tf_version = transformers.__version__
    except Exception:  # noqa: BLE001
        tf_version = "unknown"

    return {
        "git_commit": commit,
        "git_dirty": dirty,
        "transformers": tf_version,
        "torch": torch.__version__,
        "device": device,
        "python": platform.python_version(),
        "patches": list(APPLIED),   # upstream work-arounds installed by load_model (hybridquant.compat)
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
    }
