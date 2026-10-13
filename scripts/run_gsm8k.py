"""GSM8K under a quantized cache: thinking-mode generation, one jsonl row per (seed, cell, problem).

    python scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml
    python scripts/run_gsm8k.py --config configs/r2_gsm8k_4b.yaml --n-problems 20 --batch-size 4   # override

The problem subset is drawn once (subset_seed) from openai/gsm8k test and saved to ``subset_file``
so every cell sees the same 200 problems. Sampling parameters default to the Qwen3.5 model card
(thinking mode, text tasks). Resumable on (model, label, seed, problem index). The generated text is
stored zlib-compressed (base64) in ``text_zlib_b64``.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import time
import zlib

import torch

from hybridquant import load_model, run_metadata
from hybridquant.config import config_from_run, iter_runs, load_task_config
from hybridquant.eval import extract_answer, generate_batch, normalize_answer, repeat_ratio

TASK_DEFAULTS = {
    "dataset": "gsm8k", "n_problems": 200, "subset_seed": 0, "subset_file": "results/gsm8k_subset200.json",
    "thinking": True, "max_new_tokens": 4096, "temperature": 1.0, "top_p": 0.95, "top_k": 20, "min_p": 0.0,
}
PROMPT_SUFFIX = "\nPlease reason step by step, and put your final answer within \\boxed{}."


def parse_args(argv=None):
    S = argparse.SUPPRESS
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", default=None)
    p.add_argument("--model", default=S)
    p.add_argument("--dtype", default=S)
    p.add_argument("--seeds", type=int, nargs="+", default=S)
    p.add_argument("--n-problems", dest="n_problems", type=int, default=S)
    p.add_argument("--batch-size", dest="batch_size", type=int, default=S)
    p.add_argument("--max-new-tokens", dest="max_new_tokens", type=int, default=S)
    p.add_argument("--out", default=S)
    return p.parse_args(argv)


def gold_answer(answer_field: str) -> str:
    return normalize_answer(answer_field.split("####")[-1])


def score(text: str, gold: str) -> tuple[str | None, bool]:
    ans = extract_answer(text)
    return ans, normalize_answer(ans) == gold


def problem_subset(n: int, seed: int, path: str, total: int) -> list[int]:
    if os.path.exists(path):
        with open(path) as f:
            idx = json.load(f)["indices"]
        if len(idx) >= n:
            return idx[:n]
    g = torch.Generator().manual_seed(seed)
    idx = sorted(torch.randperm(total, generator=g)[:n].tolist())
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump({"dataset": "openai/gsm8k", "split": "test", "seed": seed, "n": n, "indices": idx}, f)
    return idx


def existing_keys(path: str) -> set:
    keys = set()
    if os.path.exists(path):
        for line in open(path, encoding="utf-8"):
            if line.strip():
                r = json.loads(line)
                keys.add((r["model"], r["label"], r["seed"], r["idx"]))
    return keys


def main(argv=None):
    args = parse_args(argv)
    overrides = {k: v for k, v in vars(args).items() if k != "config"}
    s, task = load_task_config(args.config, TASK_DEFAULTS, overrides)
    out = s["out"] if "out" in overrides or args.config else "results/gsm8k.jsonl"
    print("settings:", json.dumps({**s, **task}))

    from datasets import load_dataset
    ds = load_dataset("openai/gsm8k", "main", split="test")
    idx = problem_subset(task["n_problems"], task["subset_seed"], task["subset_file"], len(ds))
    problems = [(i, ds[i]["question"], gold_answer(ds[i]["answer"])) for i in idx]
    print(f"{len(problems)} problems from openai/gsm8k test (subset file {task['subset_file']})")

    runs = iter_runs(s)
    done = existing_keys(out)
    meta = run_metadata(); meta["config"] = args.config
    model, tok = load_model(s["model"], dtype=s["dtype"])
    gen_kwargs = dict(do_sample=True, temperature=task["temperature"], top_p=task["top_p"], top_k=task["top_k"],
                      min_p=task["min_p"])
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    with open(out, "a") as f:
        for run in runs:
            cfg = config_from_run(run); seed = run["seed"]
            todo = [pr for pr in problems if (s["model"], cfg.label(), seed, pr[0]) not in done]
            print(f"seed {seed} {cfg.label()}: {len(todo)} to do, {len(problems) - len(todo)} done")
            n_correct = 0
            for b in range(0, len(todo), s["batch_size"]):
                batch = todo[b:b + s["batch_size"]]
                prompts = [tok.apply_chat_template([{"role": "user", "content": q + PROMPT_SUFFIX}], tokenize=False,
                                                   add_generation_prompt=True, enable_thinking=task["thinking"])
                           for _, q, _ in batch]
                torch.manual_seed(seed * 1_000_003 + batch[0][0])
                t0 = time.time()
                outs = generate_batch(model, tok, prompts, cfg, task["max_new_tokens"], noise_seed=seed, **gen_kwargs)
                for (i, q, gold), ids in zip(batch, outs):
                    text = tok.decode(ids, skip_special_tokens=True)
                    ans, ok = score(text, gold)
                    n_correct += ok
                    row = {"model": s["model"], "label": cfg.label(), "seed": seed,
                           **{k: run[k] for k in run if k != "seed"}, "idx": i, "gold": gold, "answer": ans,
                           "correct": bool(ok), "n_tokens": len(ids), "hit_budget": len(ids) >= task["max_new_tokens"],
                           "repeat_ratio": repeat_ratio(ids), "thinking": task["thinking"],
                           "max_new_tokens": task["max_new_tokens"], "sampling": gen_kwargs,
                           "text_zlib_b64": base64.b64encode(zlib.compress(text.encode())).decode(),
                           "sec_batch": time.time() - t0, "config": args.config, "meta": meta}
                    f.write(json.dumps(row) + "\n")
                f.flush()
                print(f"  problems {b + len(batch)}/{len(todo)} ({time.time() - t0:.0f}s/batch)")
            if todo:
                print(f"seed {seed} {cfg.label():34s} acc={n_correct / len(todo):.3f} on {len(todo)} new problems")
    print("done:", out)


if __name__ == "__main__":
    main()
