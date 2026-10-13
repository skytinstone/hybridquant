"""Evaluation documents: fixed-length token windows from real text.

``seed`` controls which documents are drawn. With round-to-nearest quantization
nothing else is random, so seed-to-seed variance is document-sample variance.
"""
from __future__ import annotations

import torch

__all__ = ["load_wikitext_docs", "load_c4_docs", "load_docs", "DATASETS"]


def _collect(texts, tok, n_docs: int, min_tokens: int):
    docs = []
    for text in texts:
        if len(text) < 2000:
            continue
        ids = tok(text, return_tensors="pt")["input_ids"]
        if ids.shape[1] >= min_tokens:
            docs.append(ids[:, :min_tokens])
        if len(docs) >= n_docs:
            break
    return docs


def load_wikitext_docs(tok, n_docs: int, min_tokens: int, seed: int = 0):
    """WikiText-2 (raw) test split: ``n_docs`` windows of ``min_tokens`` tokens, order shuffled by ``seed``."""
    from datasets import load_dataset

    ds = load_dataset("Salesforce/wikitext", "wikitext-2-raw-v1", split="test")
    texts = [t for t in ds["text"] if len(t) > 2000]
    g = torch.Generator().manual_seed(seed)
    order = torch.randperm(len(texts), generator=g).tolist()
    return _collect((texts[i] for i in order), tok, n_docs, min_tokens)


def load_c4_docs(tok, n_docs: int, min_tokens: int, seed: int = 0, split: str = "validation"):
    """allenai/c4 (en) streamed, shuffled with ``seed``; same window rule as WikiText."""
    from datasets import load_dataset

    ds = load_dataset("allenai/c4", "en", split=split, streaming=True).shuffle(seed=seed, buffer_size=2000)
    return _collect((ex["text"] for ex in ds), tok, n_docs, min_tokens)


DATASETS = {"wikitext2": load_wikitext_docs, "c4": load_c4_docs}


def load_docs(dataset: str, tok, n_docs: int, min_tokens: int, seed: int = 0):
    if dataset not in DATASETS:
        raise ValueError(f"unknown dataset {dataset!r}; choose from {sorted(DATASETS)}")
    return DATASETS[dataset](tok, n_docs, min_tokens, seed)
