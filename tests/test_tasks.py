"""Scoring and prompt construction for the GSM8K and NIAH runners (no models, no downloads)."""
from __future__ import annotations

import random
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import run_gsm8k  # noqa: E402
import run_niah  # noqa: E402
from hybridquant.config import iter_runs, load_task_config  # noqa: E402
from hybridquant.eval import extract_answer, normalize_answer, repeat_ratio  # noqa: E402


# ---------------------------------------------------------------- GSM8K scoring


def test_gold_answer_and_boxed_scoring():
    assert run_gsm8k.gold_answer("Natalia sold 48/2 = 24 clips.\n#### 72") == "72"
    assert run_gsm8k.gold_answer("... #### 1,250") == "1250"
    ans, ok = run_gsm8k.score("<think>some reasoning</think>\nSo the answer is \\boxed{72}.", "72")
    assert ans == "72" and ok
    ans, ok = run_gsm8k.score("<think>...</think>\nThe total is 1,250 dollars.", "1250")
    assert normalize_answer(ans) == "1250" and ok
    _, ok = run_gsm8k.score("<think>42 appears here</think> no number", "42")
    assert not ok                       # numbers inside the thinking block do not count
    assert extract_answer("nothing here") is None


def test_repeat_ratio_detects_loops():
    assert repeat_ratio(list(range(100))) == 0.0
    assert repeat_ratio([1, 2, 3, 4, 5, 6, 7, 8] * 10) > 0.8


def test_problem_subset_is_saved_and_reused(tmp_path):
    path = tmp_path / "subset.json"
    a = run_gsm8k.problem_subset(20, 0, str(path), 1319)
    b = run_gsm8k.problem_subset(20, 0, str(path), 1319)
    assert a == b and len(a) == 20 and a == sorted(a) and max(a) < 1319 and len(set(a)) == 20


# ---------------------------------------------------------------- NIAH


class FakeTok:
    """Whitespace tokenizer with a minimal chat template, enough for build_prompt."""

    def __init__(self):
        self.vocab = {}
        self.inv = {}

    def __call__(self, text, add_special_tokens=False, **kw):
        ids = []
        for w in text.split():
            if w not in self.vocab:
                self.vocab[w] = len(self.vocab); self.inv[self.vocab[w]] = w
            ids.append(self.vocab[w])
        return {"input_ids": ids}

    def decode(self, ids, **kw):
        return " ".join(self.inv[i] for i in ids)

    def apply_chat_template(self, msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False):
        return "<user>" + msgs[0]["content"] + "<assistant>"


def test_needle_is_unique_and_scored_robustly():
    rng = random.Random(0)
    needle, question, code = run_niah.make_needle(rng)
    assert code in needle and len(code) == 6 and set(code) <= set(run_niah.CODE_ALPHABET)
    assert run_niah.is_correct(f"The code is {code}.", code)
    assert run_niah.is_correct(f"{code[:3]}-{code[3:]}", code)
    assert run_niah.is_correct(code.lower(), code)
    assert not run_niah.is_correct("I do not know.", code)
    assert not run_niah.is_correct(code[:5] + "Z" if code[5] != "Z" else code[:5] + "A", code)


@pytest.mark.parametrize("depth", [0.1, 0.5, 0.9])
def test_build_prompt_places_needle_at_depth(depth):
    tok = FakeTok()
    filler = tok(" ".join(f"w{i}" for i in range(5000)))["input_ids"]
    needle, question, code = run_niah.make_needle(random.Random(1))
    prompt, n_fill = run_niah.build_prompt(tok, filler, offset=100, context_len=1024, depth=depth,
                                          needle=needle, question=question)
    assert needle in prompt and question in prompt and prompt.endswith("<assistant>")
    assert n_fill == 1024 - 96
    body = prompt.split("Read the document and answer the question at the end.\n\n")[1].split("\n\nQuestion:")[0]
    pos = body.index(needle) / len(body)
    assert abs(pos - depth) < 0.08


def test_task_config_splits_grid_and_task_keys(tmp_path):
    import yaml
    path = tmp_path / "t.yaml"
    path.write_text(yaml.safe_dump({"model": "tiny", "cells": [{"kv_bits": 4, "state_bits": 4}, {"kv_bits": 16, "state_bits": 16}],
                                    "state_error_feedback": [False, True], "seeds": [0], "n_problems": 5,
                                    "max_new_tokens": 8}))
    s, task = load_task_config(path, run_gsm8k.TASK_DEFAULTS, {"batch_size": 2})
    assert s["model"] == "tiny" and s["batch_size"] == 2
    assert task["n_problems"] == 5 and task["max_new_tokens"] == 8 and task["temperature"] == 1.0
    assert len(iter_runs(s)) == 3        # (4,4) off, (4,4) on, (16,16) off
    with pytest.raises(ValueError):
        load_task_config(path, {"n_problems": 1}, {})          # max_new_tokens unknown for this task


def test_repo_task_configs_parse():
    s, task = load_task_config(ROOT / "configs" / "r2_gsm8k_4b.yaml", run_gsm8k.TASK_DEFAULTS)
    assert len(iter_runs(s)) == 10 and task["n_problems"] == 200 and task["max_new_tokens"] == 4096
    s, task = load_task_config(ROOT / "configs" / "r2_gsm8k_4b_full.yaml", run_gsm8k.TASK_DEFAULTS)
    assert len(iter_runs(s)) == 3 * (7 + 5)
    for size in ("0.8b", "4b"):
        s, task = load_task_config(ROOT / "configs" / f"r2_niah_{size}.yaml", run_niah.TASK_DEFAULTS)
        assert len(iter_runs(s)) == 7 + 4 and task["context_lens"] == [4096, 16384] and len(task["depths"]) == 5
