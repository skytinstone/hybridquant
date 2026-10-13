"""Evaluation documents: fixed-length token windows cut from one long token stream.

The whole split is joined with "\\n\\n", tokenised once, and cut into non-overlapping
windows of ``min_tokens`` tokens (the usual WikiText perplexity protocol). A fixed
permutation (BASE_SEED) orders the windows; ``seed`` picks block number ``seed`` of
``n_docs`` windows, so different seeds use disjoint documents. With round-to-nearest
quantization nothing else is random, so seed-to-seed variance is document-sample variance.
"""
from __future__ import annotations

import torch

__all__ = ["load_wikitext_docs", "load_c4_docs", "load_docs", "DATASETS", "windows_from_tokens", "join_texts"]

BASE_SEED = 1234
C4_FIRST_SHARD = "en/c4-validation.00000-of-00008.json.gz"
C4_MAX_DOCS = 5000   # first documents of the shard; about 2.5M tokens, enough for >1000 windows of 1024
_TOKEN_CACHE: dict[tuple, list[int]] = {}


def join_texts(texts) -> str:
    return "\n\n".join(t.strip() for t in texts if t and t.strip())


def windows_from_tokens(ids: list[int], n_docs: int, min_tokens: int, seed: int = 0, name: str = "stream"):
    """Cut ``ids`` into windows of ``min_tokens`` and return block ``seed`` of ``n_docs`` windows as [1, T] tensors."""
    n_win = len(ids) // min_tokens
    if n_win == 0:
        raise ValueError(f"[data] {name}: only {len(ids)} tokens, fewer than one window of {min_tokens}")
    g = torch.Generator().manual_seed(BASE_SEED)
    order = torch.randperm(n_win, generator=g).tolist()
    start = seed * n_docs
    picked = order[start:start + n_docs]
    if len(picked) < n_docs:
        # block runs past the available windows: wrap around and warn
        wrapped = [order[(start + i) % n_win] for i in range(n_docs)]
        print(f"[data] WARNING {name}: seed {seed} x n_docs {n_docs} exceeds {n_win} windows of {min_tokens}; "
              f"reusing windows (documents are no longer disjoint across seeds)")
        picked = wrapped
    docs = [torch.tensor(ids[i * min_tokens:(i + 1) * min_tokens], dtype=torch.long).unsqueeze(0) for i in picked]
    if len(docs) < n_docs:
        print(f"[data] WARNING {name}: requested {n_docs} docs, got {len(docs)}")
    print(f"[data] {name}: {len(ids):,} tokens -> {n_win} windows of {min_tokens}; loaded {len(docs)} docs "
          f"({len(docs) * min_tokens:,} tokens, seed {seed})")
    return docs


def _tokens(key: tuple, tok, text: str) -> list[int]:
    if key not in _TOKEN_CACHE:
        _TOKEN_CACHE[key] = tok(text, add_special_tokens=False)["input_ids"]
    return _TOKEN_CACHE[key]


def load_wikitext_docs(tok, n_docs: int, min_tokens: int, seed: int = 0):
    """WikiText-2 (raw) test split as one stream."""
    from datasets import load_dataset

    key = ("wikitext2", getattr(tok, "name_or_path", id(tok)))
    if key not in _TOKEN_CACHE:
        ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
        _tokens(key, tok, join_texts(ds["text"]))
    return windows_from_tokens(_TOKEN_CACHE[key], n_docs, min_tokens, seed, "wikitext2/test")


def load_c4_docs(tok, n_docs: int, min_tokens: int, seed: int = 0):
    """allenai/c4 (en) validation, first shard, first C4_MAX_DOCS documents, as one stream."""
    from datasets import load_dataset

    key = ("c4", getattr(tok, "name_or_path", id(tok)))
    if key not in _TOKEN_CACHE:
        ds = load_dataset("allenai/c4", data_files=C4_FIRST_SHARD, split="train")
        texts = ds["text"][:C4_MAX_DOCS]
        _tokens(key, tok, join_texts(texts))
    return windows_from_tokens(_TOKEN_CACHE[key], n_docs, min_tokens, seed, "c4/validation-shard0")


DATASETS = {"wikitext2": load_wikitext_docs, "c4": load_c4_docs}


def load_docs(dataset: str, tok, n_docs: int, min_tokens: int, seed: int = 0):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return DATASETS[dataset](tok, n_docs, min_tokens, seed)
