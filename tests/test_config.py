"""YAML config -> CLI override merge -> run list (grid or cells, EF crossing), with a tiny model and no downloads."""
from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch
import yaml

from hybridquant.budget import cache_sizes, total_bits
from hybridquant.config import GRID_DEFAULTS, config_from_run, iter_runs, load_config, resolve_settings

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_grid import parse_args, settings_from_args  # noqa: E402


def test_defaults_are_a_valid_setting():
    s = resolve_settings()
    assert s == resolve_settings(GRID_DEFAULTS)
    runs = iter_runs(s)
    assert runs and runs[0] == {"seed": 0, "kv_bits": 16, "state_bits": 16, "kv_group": 128, "state_group": 128,
                                "state_symmetric": False, "state_stochastic": False, "kv_noise_match": False,
                                "state_noise_match": False, "error_feedback": False}


def test_yaml_load_merge_and_seed_order(tmp_path):
    cfg = {"model": "tiny", "kv_bits": [16, 4], "state_bits": [16, 2], "seeds": [0, 1, 2],
           "n_docs": 2, "prefill": 8, "decode": 4, "out": str(tmp_path / "o.jsonl")}
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump(cfg))
    s = resolve_settings(load_config(path), {})
    assert s["model"] == "tiny" and s["seeds"] == [0, 1, 2] and s["dataset"] == "wikitext2"
    runs = iter_runs(s)
    assert len(runs) == 3 * 2 * 2
    assert [r["seed"] for r in runs] == [0] * 4 + [1] * 4 + [2] * 4          # seed-major
    assert [(r["kv_bits"], r["state_bits"]) for r in runs[:4]] == [(16, 16), (16, 2), (4, 16), (4, 2)]


def test_error_feedback_crossing_skips_fp_state():
    s = resolve_settings({"kv_bits": [16, 4], "state_bits": [16, 4], "state_error_feedback": [False, True]})
    runs = iter_runs(s)
    cells = [(r["kv_bits"], r["state_bits"], r["error_feedback"]) for r in runs]
    assert cells == [(16, 16, False), (16, 4, False), (16, 4, True), (4, 16, False), (4, 4, False), (4, 4, True)]
    labels = [config_from_run(r).label() for r in runs]
    assert labels[2].endswith("Sint4g128ef") and labels[1].endswith("Sint4g128")


def test_cells_mode_with_per_cell_overrides():
    s = resolve_settings({"cells": [
        {"kv_bits": 16, "state_bits": 16},
        {"kv_bits": 4, "state_bits": 4, "kv_noise_match": True},
        {"kv_bits": 4, "state_bits": 4, "state_group": 32, "state_symmetric": True, "error_feedback": False},
    ], "state_error_feedback": [False, True], "seeds": [0, 1]})
    runs = iter_runs(s)
    per_seed = [r for r in runs if r["seed"] == 0]
    assert len(runs) == 2 * len(per_seed)
    labels = [config_from_run(r).label() for r in per_seed]
    assert labels == ["Kfp_Vfp_Sfp",
                      "Kint4g128gn_Vint4g128gn_Sint4g128", "Kint4g128gn_Vint4g128gn_Sint4g128ef",
                      "Kint4g128_Vint4g128_Sint4g32s"]
    with pytest.raises(ValueError):
        resolve_settings({"cells": [{"kv_bits": 4}]})
    with pytest.raises(ValueError):
        resolve_settings({"cells": [{"kv_bits": 4, "state_bits": 4, "bogus": 1}]})


def test_cli_overrides_yaml_but_untouched_keys_survive(tmp_path):
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"model": "tiny", "seeds": [0, 1, 2], "n_docs": 20, "prefill": 512}))
    args = parse_args(["--config", str(path), "--seeds", "7", "--n-docs", "3", "--state-sr", "--state-ef", "0", "1"])
    s, config_path = settings_from_args(args)
    assert config_path == str(path)
    assert s["seeds"] == [7] and s["n_docs"] == 3                 # overridden
    assert s["prefill"] == 512 and s["model"] == "tiny"            # from yaml
    assert s["state_stochastic"] is True and s["state_symmetric"] is False
    assert s["state_error_feedback"] == [False, True]
    assert s["decode"] == GRID_DEFAULTS["decode"]                  # default

    # legacy single --seed still works and the CLI-only path needs no yaml
    s2, cp = settings_from_args(parse_args(["--model", "x", "--seed", "3", "--kv-bits", "16", "4"]))
    assert cp is None and s2["seeds"] == [3] and s2["kv_bits"] == [16, 4] and s2["state_bits"] == GRID_DEFAULTS["state_bits"]


def test_invalid_settings_are_rejected(tmp_path):
    bad = tmp_path / "bad.yaml"
    bad.write_text(yaml.safe_dump({"model": "tiny", "kv_bitz": [4]}))
    with pytest.raises(ValueError):
        load_config(bad)
    with pytest.raises(ValueError):
        resolve_settings({"dataset": "pile"})
    with pytest.raises(ValueError):
        resolve_settings({"seeds": []})
    with pytest.raises(ValueError):
        resolve_settings({"batch_size": 0})
    with pytest.raises(SystemExit):
        parse_args(["--dataset", "pile"])


def test_repo_configs_load_and_match_requested_grids():
    smoke = resolve_settings(load_config(ROOT / "configs" / "smoke_0.8b.yaml"))
    assert smoke["kv_bits"] == [16, 8, 4, 2] and smoke["state_bits"] == [16, 8, 4]
    assert smoke["seeds"] == [0] and smoke["n_docs"] == 8 and smoke["prefill"] == smoke["decode"] == 256

    g = resolve_settings(load_config(ROOT / "configs" / "grid_0.8b.yaml"))
    assert g["kv_bits"] == [16, 8, 4, 3, 2] and g["state_bits"] == [16, 8, 6, 5, 4]
    assert g["state_error_feedback"] == [False, True] and g["seeds"] == [0, 1, 2] and g["n_docs"] == 20
    assert g["prefill"] == g["decode"] == 256
    assert len(iter_runs(g)) == 3 * (25 + 20)      # EF only for the 20 cells with state < 16

    sa = resolve_settings(load_config(ROOT / "configs" / "superadd_0.8b.yaml"))
    labels = {config_from_run(r).label() for r in iter_runs(sa)}
    assert {"Kfp_Vfp_Sint4g128", "Kint4g128_Vint4g128_Sfp", "Kint4g128_Vint4g128_Sint4g128",
            "Kint4g128gn_Vint4g128gn_Sint4g128", "Kint4g128_Vint4g128_Sint4g128gn"} <= labels

    core = resolve_settings(load_config(ROOT / "configs" / "grid_4b_core.yaml"))
    assert core["model"] == "Qwen/Qwen3.5-4B" and len(core["cells"]) == 7
    assert len(iter_runs(core)) == 3 * (7 + 5)     # 5 cells have state < 16


def test_cache_sizes_on_tiny_config():
    from transformers import Qwen3_5TextConfig

    cfg = Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        layer_types=["linear_attention", "linear_attention", "linear_attention", "full_attention"])
    sizes = cache_sizes(cfg)
    assert sizes["L_a"] == 1 and sizes["L_g"] == 3
    assert sizes["kv_per_token"] == 2 * 1 * 1 * 32 and sizes["state_elems"] == 3 * 2 * 16 * 16
    assert total_bits(sizes, 16, 16, 10) == 16 * 640 + 16 * 1536
    assert torch.tensor(total_bits(sizes, 4, 8, 10)) == 4 * 640 + 8 * 1536
