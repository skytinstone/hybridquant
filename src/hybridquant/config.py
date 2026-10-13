"""Experiment settings: YAML file, CLI overrides, and the run list.

Precedence, lowest to highest: GRID_DEFAULTS < YAML config < explicit CLI arguments.

Two ways to say which cells to run:
* grid: ``kv_bits`` x ``state_bits`` (every combination), or
* ``cells``: an explicit list of mappings. Each cell needs ``kv_bits`` and
  ``state_bits`` and may override ``kv_group``, ``state_group``, ``state_symmetric``,
  ``state_stochastic``, ``kv_noise_match``, ``state_noise_match``, ``error_feedback``.
``state_error_feedback`` (list of bools) is crossed with every cell that does not
fix ``error_feedback`` itself; EF is skipped for cells with state_bits >= 16 (a no-op).
"""
from __future__ import annotations

from pathlib import Path

from .hooks import HybridQuantConfig
from .quant import QuantSpec

__all__ = ["GRID_DEFAULTS", "ALLOWED_DATASETS", "CELL_KEYS", "load_config", "resolve_settings", "iter_runs",
           "config_from_run", "run_key"]

GRID_DEFAULTS: dict = {
    "model": "Qwen/Qwen3.5-0.8B",
    "dtype": "auto",
    "kv_bits": [16, 8, 4, 2],
    "state_bits": [16, 8, 4],
    "kv_group": 128,
    "state_group": 128,
    "state_symmetric": False,
    "state_stochastic": False,
    "state_error_feedback": [False],
    "cells": None,
    "seeds": [0],
    "n_docs": 8,
    "prefill": 256,
    "decode": 256,
    "step": 1,
    "batch_size": 8,
    "dataset": "wikitext2",
    "out": "results/grid.jsonl",
}
ALLOWED_DATASETS = ("wikitext2", "c4")
CELL_KEYS = ("kv_bits", "state_bits", "kv_group", "state_group", "state_symmetric", "state_stochastic",
             "kv_noise_match", "state_noise_match", "error_feedback")
_LIST_KEYS = ("kv_bits", "state_bits", "seeds")
_INT_KEYS = ("kv_group", "state_group", "n_docs", "prefill", "decode", "step", "batch_size")
_BOOL_KEYS = ("state_symmetric", "state_stochastic")


def load_config(path: str | Path) -> dict:
    """Read a YAML file and reject keys that are not part of the schema."""
    import yaml

    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: top level must be a mapping")
    unknown = set(data) - set(GRID_DEFAULTS)
    if unknown:
        raise ValueError(f"{path}: unknown keys {sorted(unknown)}; allowed: {sorted(GRID_DEFAULTS)}")
    return data


def _as_bool_list(v) -> list[bool]:
    if isinstance(v, bool):
        return [v]
    return [bool(x) for x in v]


def resolve_settings(config: dict | None = None, overrides: dict | None = None) -> dict:
    """Merge defaults, config and overrides; normalise and validate the result."""
    s = dict(GRID_DEFAULTS)
    s.update(config or {})
    s.update({k: v for k, v in (overrides or {}).items() if v is not None})

    unknown = set(s) - set(GRID_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown setting keys {sorted(unknown)}")
    for k in _LIST_KEYS:
        v = s[k]
        if isinstance(v, int):
            v = [v]
        s[k] = [int(x) for x in v]
        if not s[k]:
            raise ValueError(f"{k} must not be empty")
    for k in _INT_KEYS:
        s[k] = int(s[k])
    for k in _BOOL_KEYS:
        s[k] = bool(s[k])
    s["state_error_feedback"] = _as_bool_list(s["state_error_feedback"])
    if not s["state_error_feedback"]:
        raise ValueError("state_error_feedback must not be empty")
    if s["cells"] is not None:
        cells = []
        for i, c in enumerate(s["cells"]):
            if not isinstance(c, dict) or "kv_bits" not in c or "state_bits" not in c:
                raise ValueError(f"cells[{i}] must be a mapping with kv_bits and state_bits")
            bad = set(c) - set(CELL_KEYS)
            if bad:
                raise ValueError(f"cells[{i}]: unknown keys {sorted(bad)}; allowed {CELL_KEYS}")
            cells.append(dict(c))
        if not cells:
            raise ValueError("cells must not be empty")
        s["cells"] = cells
    if s["dataset"] not in ALLOWED_DATASETS:
        raise ValueError(f"dataset must be one of {ALLOWED_DATASETS}, got {s['dataset']!r}")
    if min(s["step"], s["n_docs"], s["prefill"], s["decode"], s["batch_size"]) < 1:
        raise ValueError("step, n_docs, prefill, decode and batch_size must be >= 1")
    return s


def iter_runs(settings: dict) -> list[dict]:
    """Seed-major run list. Each run carries the full quantization recipe for one cell."""
    base = {k: settings[k] for k in ("kv_group", "state_group", "state_symmetric", "state_stochastic")}
    base.update(kv_noise_match=False, state_noise_match=False)
    cells = settings["cells"] or [{"kv_bits": kv, "state_bits": st}
                                  for kv in settings["kv_bits"] for st in settings["state_bits"]]
    runs = []
    for seed in settings["seeds"]:
        for cell in cells:
            c = {**base, **cell}
            efs = [bool(c.pop("error_feedback"))] if "error_feedback" in c else settings["state_error_feedback"]
            for ef in efs:
                if ef and int(c["state_bits"]) >= 16:
                    continue   # error feedback is a no-op without state quantization
                runs.append({"seed": int(seed), **{k: c[k] for k in CELL_KEYS if k != "error_feedback"},
                             "error_feedback": bool(ef)})
    return runs


def config_from_run(run: dict) -> HybridQuantConfig:
    kv = QuantSpec(int(run["kv_bits"]), int(run["kv_group"]), noise_match=bool(run["kv_noise_match"]))
    st = QuantSpec(int(run["state_bits"]), int(run["state_group"]), symmetric=bool(run["state_symmetric"]),
                   stochastic=bool(run["state_stochastic"]), error_feedback=bool(run["error_feedback"]),
                   noise_match=bool(run["state_noise_match"]))
    return HybridQuantConfig(key=kv, value=kv, state=st)


def run_key(model: str, label: str, seed: int) -> tuple:
    """Identity of one jsonl row, used to resume."""
    return (model, label, int(seed))
