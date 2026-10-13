"""Single-needle needle-in-a-haystack under a quantized cache.

A needle sentence with a random 6-character code is inserted into WikiText-2 train filler at a
given depth; the model is asked for the code (greedy, non-thinking). Accuracy = code appears in the
answer. One jsonl row per (seed, cell, context_len, depth, problem). Resumable.

    python scripts/run_niah.py --config configs/r2_niah_0.8b.yaml
"""
from __future__ import annotations

import argparse
import json
import os
import random
import re
import string
import time

import torch

from hybridquant import load_model, run_metadata
from hybridquant.config import config_from_run, iter_runs, load_task_config
from hybridquant.data import join_texts
from hybridquant.eval import generate_batch

TASK_DEFAULTS = {
    "dataset": "niah", "context_lens": [4096, 16384], "depths": [0.1, 0.3, 0.5, 0.7, 0.9], "n_per_depth": 10,
    "needle_seed": 0, "max_new_tokens": 32,
}
PROJECTS = ["Umbra Lantern", "Quartz Falcon", "Velvet Compass", "Cobalt Orchard", "Saffron Meridian",
            "Obsidian Harbor", "Juniper Signal", "Marble Tangent", "Cinder Atlas", "Willow Cipher"]
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O/1/I


def make_needle(rng: random.Random) -> tuple[str, str, str]:
    code = "".join(rng.choice(CODE_ALPHABET) for _ in range(6))
    project = rng.choice(PROJECTS)
    needle = f"The secret access code for project {project} is {code}."
    question = f"What is the secret access code for project {project}? Answer with the code only."
    return needle, question, code


def build_prompt(tok, filler_ids: list[int], offset: int, context_len: int, depth: float, needle: str,
                 question: str, thinking_switch: bool = True) -> tuple[str, int]:
    """Insert the needle at ``depth`` of a filler window of about ``context_len`` tokens; return (prompt, n_filler)."""
    overhead = 96   # chat template, question, needle (approximate)
    n_fill = max(64, context_len - overhead)
    window = filler_ids[offset:offset + n_fill]
    if len(window) < n_fill:
        window = (filler_ids * (n_fill // max(1, len(filler_ids)) + 2))[offset:offset + n_fill]
    pos = int(depth * len(window))
    text = tok.decode(window[:pos]) + "\n" + needle + "\n" + tok.decode(window[pos:])
    content = f"Read the document and answer the question at the end.\n\n{text}\n\nQuestion: {question}"
    msgs = [{"role": "user", "content": content}]
    try:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=False)
    except TypeError:
        prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return prompt, len(window)


def is_correct(answer: str, code: str) -> bool:
    return code.lower() in re.sub(r"[\s\-_]", "", answer).lower()


def filler_tokens(tok) -> list[int]:
    from datasets import load_dataset
    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="train")
    return tok(join_texts(ds["text"]), add_special_tokens=False)["input_ids"]


def parse_args(argv=None):
    S = argparse.SUPPRESS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=S)
    p.add_argument("--dtype", default=S)
    p.add_argument("--seeds", type=int, nargs="+", default=S)
    p.add_argument("--context-lens", dest="context_lens", type=int, nargs="+", default=S)
    p.add_argument("--n-per-depth", dest="n_per_depth", type=int, default=S)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=S)
    p.add_argument("--out", default=S)
    return p.parse_args(argv)


def existing_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                keys.add((r["model"], r["label"], r["seed"], r["context_len"], r["depth"], r["idx"]))
    return keys


def main(argv=None):
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    s, task = load_task_config(args.config, TASK_DEFAULTS, overrides)
    out = s["out"] if "out" in overrides or args.config else "results/niah.jsonl"
    print("settings:", json.dumps({**s, **task}))

    meta = run_metadata(); meta["config"] = args.config
    model, tok = load_model(s["model"], dtype=s["dtype"])
    filler = filler_tokens(tok)
    print(f"filler: {len(filler):,} WikiText-2 train tokens")

    # problems: (context_len, depth, idx) -> needle; the same needles are used for every cell
    problems = []
    for L in task["context_lens"]:
        for d in task["depths"]:
            for j in range(task["n_per_depth"]):
                rng = random.Random(task["needle_seed"] * 7919 + int(L) * 31 + int(d * 100) * 101 + j)
                needle, question, code = make_needle(rng)
                offset = (j * 1_000_003 + int(d * 100) * 4099) % max(1, len(filler) - L)
                problems.append({"context_len": int(L), "depth": float(d), "idx": j, "needle": needle,
                                 "question": question, "code": code, "offset": offset})
    assert len({p["code"] for p in problems}) == len(problems), "codes must be unique"

    runs = iter_runs(s)
    done = existing_keys(out)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a") as f:
        for run in runs:
            cfg = config_from_run(run); seed = run["seed"]
            todo = [p for p in problems if (s["model"], cfg.label(), seed, p["context_len"], p["depth"], p["idx"]) not in done]
            print(f"seed {seed} {cfg.label()}: {len(todo)} to do, {len(problems) - len(todo)} done")
            n_ok = 0
            for b in range(0, len(todo), s["batch_size"]):
                batch = todo[b:b + s["batch_size"]]
                prompts, n_fill = zip(*[build_prompt(tok, filler, p["offset"], p["context_len"], p["depth"],
                                                     p["needle"], p["question"]) for p in batch])
                t0 = time.time()
                outs = generate_batch(model, tok, list(prompts), cfg, task["max_new_tokens"], noise_seed=seed,
                                      do_sample=False)
                for p, ids, nf in zip(batch, outs, n_fill):
                    text = tok.decode(ids, skip_special_tokens=True)
                    ok = is_correct(text, p["code"])
                    n_ok += ok
                    row = {"model": s["model"], "label": cfg.label(), "seed": seed,
                           **{k: run[k] for k in run if k != "seed"},
                           "context_len": p["context_len"], "depth": p["depth"], "idx": p["idx"],
                           "n_filler_tokens": nf, "code": p["code"], "answer": text[:200], "correct": bool(ok),
                           "sec_batch": time.time() - t0, "config": args.config, "meta": meta}
                    f.write(json.dumps(row) + "\n")
                f.flush()
            if todo:
                print(f"seed {seed} {cfg.label():34s} acc={n_ok / len(todo):.3f} on {len(todo)} problems")
    print("done:", out)


if __name__ == "__main__":
    main()
