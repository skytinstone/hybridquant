"""Experiment settings: YAML file, CLI overrides, and the (seed, kv_bits, state_bits) run list.

Precedence, lowest to highest: GRID_DEFAULTS < YAML config < explicit CLI arguments.
"""
from __future__ import annotations

from pathlib import Path

__all__ = ["GRID_DEFAULTS", "ALLOWED_DATASETS", "load_config", "resolve_settings", "iter_runs"]

GRID_DEFAULTS: dict = {
    "model": "Qwen/Qwen3.5-0.8B",
    "dtype": "auto",
    "kv_bits": [16, 8, 4, 2],
    "state_bits": [16, 8, 4],
    "kv_group": 128,
    "state_group": 128,
    "state_symmetric": False,
    "state_stochastic": False,
    "seeds": [0],
    "n_docs": 8,
    "prefill": 256,
    "decode": 256,
    "step": 1,
    "dataset": "wikitext2",
    "out": "results/grid.jsonl",
}
ALLOWED_DATASETS = ("wikitext2", "c4")
_LIST_KEYS = ("kv_bits", "state_bits", "seeds")
_INT_KEYS = ("kv_group", "state_group", "n_docs", "prefill", "decode", "step")
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
    if s["dataset"] not in ALLOWED_DATASETS:
        raise ValueError(f"dataset must be one of {ALLOWED_DATASETS}, got {s['dataset']!r}")
    if s["step"] < 1 or s["n_docs"] < 1 or s["prefill"] < 1 or s["decode"] < 1:
        raise ValueError("step, n_docs, prefill and decode must be >= 1")
    return s


def iter_runs(settings: dict) -> list[dict]:
    """Seed-major run list: every (kv_bits, state_bits) cell for seed 0, then for seed 1, ..."""
    return [
        {"seed": seed, "kv_bits": kv, "state_bits": st}
        for seed in settings["seeds"]
        for kv in settings["kv_bits"]
        for st in settings["state_bits"]
    ]
