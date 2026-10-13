"""Evaluation utilities.

Two complementary metrics:

1. ``decode_nll``  (teacher-forced, token-by-token decoding)
   The prompt is prefilled in one pass; the continuation is fed ONE token at a
   time with the cache. This is the only way the recurrent-state quantization
   error actually accumulates, so it is the primary metric for the paper.
   Returns per-position NLL so you can plot error-vs-position curves.

2. ``generate_and_score``  (free-running generation)
   For reasoning benchmarks (GSM8K etc.) in thinking mode. Also records
   behavioural statistics: number of generated tokens, whether the model hit
   the token budget, repetition rate.
"""
from __future__ import annotations

import math
import os
import re
from dataclasses import dataclass, asdict

import torch

from .hooks import HybridQuantConfig, QuantStats, quantized_cache, make_cache

__all__ = ["decode_nll", "decode_nll_batch", "DecodeResult", "generate_and_score", "GenResult", "load_model", "resolve_dtype", "DTYPE_ALIASES"]


DTYPE_ALIASES = {
    "bf16": torch.bfloat16, "bfloat16": torch.bfloat16,
    "fp16": torch.float16, "float16": torch.float16, "half": torch.float16,
    "fp32": torch.float32, "float32": torch.float32,
}


def resolve_dtype(dtype="auto") -> torch.dtype:
    """Pick a compute dtype for the current device.

    * a ``torch.dtype`` is returned as is;
    * a name in ``DTYPE_ALIASES`` ("bf16", "fp16", "fp32", ...) is looked up;
    * ``"auto"`` first honours the ``HQ_DTYPE`` environment variable (set by the
      Colab GPU-selection cell), otherwise: CUDA with compute capability >= 8.0
      (A100/H100/L4) -> bf16; older CUDA (T4, V100) -> fp16 because they have no
      bf16 tensor cores; Apple MPS -> bf16; CPU -> fp32.
    """
    if isinstance(dtype, torch.dtype):
        return dtype
    name = (dtype or "auto").strip().lower()
    if name == "auto":
        name = os.environ.get("HQ_DTYPE", "auto").strip().lower()
    if name == "auto":
        if torch.cuda.is_available():
            major, _ = torch.cuda.get_device_capability(0)
            return torch.bfloat16 if major >= 8 else torch.float16
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return torch.bfloat16
        return torch.float32
    if name not in DTYPE_ALIASES:
        raise ValueError(f"unknown dtype {dtype!r}; use one of {sorted(DTYPE_ALIASES)} or 'auto'")
    return DTYPE_ALIASES[name]


def load_model(name: str, dtype="auto", device_map="auto", text_only: bool = True, **from_pretrained_kwargs):
    """Load a Qwen3.5 checkpoint. Returns (model, tokenizer).

    ``dtype`` is resolved with :func:`resolve_dtype` ("auto" picks bf16 on
    Ampere+ GPUs and MPS, fp16 on T4-class GPUs, fp32 on CPU).

    ``text_only`` loads the language model without the vision tower, which
    saves VRAM on Colab. Falls back to the multimodal class if unavailable.
    Extra kwargs go to ``from_pretrained`` (e.g. ``use_kernels=False``).
    """
    from transformers import AutoTokenizer, AutoConfig

    dtype = resolve_dtype(dtype)
    # Apple MPS: from_pretrained(device_map="auto") crashes (SIGSEGV/SIGABRT) when the dtype is not bf16
    # on torch 2.14 / transformers 5.19. Load on CPU and move afterwards; CUDA paths are untouched.
    move_to = None
    if device_map == "auto" and not torch.cuda.is_available() and torch.backends.mps.is_available() \
            and dtype is not torch.bfloat16:
        device_map, move_to = "cpu", "mps"
    tok = AutoTokenizer.from_pretrained(name)
    model = None
    if text_only:
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map, **from_pretrained_kwargs)
        except Exception as e:  # noqa: BLE001
            print(f"[load_model] text-only load failed ({type(e).__name__}: {e}); falling back to multimodal class")
    if model is None:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(name, dtype=dtype, device_map=device_map, **from_pretrained_kwargs)
    if move_to is not None:
        model = model.to(move_to)
    model.eval()
    return model, tok


@dataclass
class DecodeResult:
    label: str
    prefill_len: int
    n_decode: int
    nll_per_pos: list[float]
    stats: str

    @property
    def mean_nll(self) -> float:
        return sum(self.nll_per_pos) / max(1, len(self.nll_per_pos))

    @property
    def ppl(self) -> float:
        return math.exp(self.mean_nll)

    def to_dict(self):
        d = asdict(self)
        d["mean_nll"] = self.mean_nll
        d["ppl"] = self.ppl
        return d


@torch.no_grad()
def decode_nll_batch(model, input_ids: torch.Tensor, prefill_len: int, cfg: HybridQuantConfig,
                     max_decode: int | None = None, step: int = 1, noise_seed: int | None = None,
                     stats: QuantStats | None = None) -> list[DecodeResult]:
    """Teacher-forced decode NLL for a batch of equal-length documents.

    input_ids: [B, T] token ids (no padding: every row has the same length).
    Positions [0, prefill_len) are prefilled in one forward; positions
    [prefill_len, T) are fed ``step`` tokens at a time. ``step=1`` is the faithful
    decode path (slow but exact). Larger ``step`` uses the chunked GDN kernel between
    quantization points (faster, coarser). Returns one DecodeResult per row.
    """
    assert input_ids.dim() == 2, "input_ids must be [B, T]"
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    B, T = ids.shape
    end = T if max_decode is None else min(T, prefill_len + max_decode)

    stats = stats if stats is not None else QuantStats()
    nll = [[] for _ in range(B)]
    with quantized_cache(cfg, stats, noise_seed=noise_seed):
        cache = make_cache(model)
        out = model(input_ids=ids[:, :prefill_len], past_key_values=cache, use_cache=True, logits_to_keep=1)
        logits = out.logits[:, -1].float()
        pos = prefill_len
        while pos < end:
            nxt = ids[:, pos:pos + step]
            # score the first token of this chunk with the previous logits
            lp = torch.log_softmax(logits, dim=-1)
            first = -lp.gather(1, nxt[:, :1]).squeeze(1)
            for b in range(B):
                nll[b].append(first[b].item())
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True,
                        logits_to_keep=0 if nxt.shape[1] > 1 else 1)
            if nxt.shape[1] > 1:
                # score remaining tokens of the chunk with their teacher-forced logits
                lps = torch.log_softmax(out.logits[:, :-1].float(), dim=-1)
                rest = -lps.gather(2, nxt[:, 1:].unsqueeze(-1)).squeeze(-1)   # [B, step-1]
                for b in range(B):
                    nll[b].extend(rest[b].tolist())
            logits = out.logits[:, -1].float()
            pos += nxt.shape[1]
    summary = stats.summary()
    return [DecodeResult(cfg.label(), prefill_len, len(nll[b]), nll[b], summary) for b in range(B)]


@torch.no_grad()
def decode_nll(model, input_ids: torch.Tensor, prefill_len: int, cfg: HybridQuantConfig,
               max_decode: int | None = None, step: int = 1, noise_seed: int | None = None) -> DecodeResult:
    """Single-document wrapper around :func:`decode_nll_batch` (input_ids [1, T])."""
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "decode_nll takes one document; use decode_nll_batch"
    return decode_nll_batch(model, input_ids, prefill_len, cfg, max_decode=max_decode, step=step,
                            noise_seed=noise_seed)[0]


@dataclass
class GenResult:
    label: str
    text: str
    n_tokens: int
    hit_budget: bool
    repeat_ratio: float  # fraction of 8-gram repeats, a cheap loop detector
    answer: str | None
    correct: bool | None


def _extract_boxed(text: str) -> str | None:
    m = re.findall(r"\\boxed\{([^{}]*)\}", text)
    return m[-1].strip() if m else None


def _repeat_ratio(ids: list[int], n: int = 8) -> float:
    if len(ids) < 2 * n:
        return 0.0
    seen, rep = set(), 0
    for i in range(len(ids) - n + 1):
        g = tuple(ids[i:i + n])
        if g in seen:
            rep += 1
        seen.add(g)
    return rep / (len(ids) - n + 1)


@torch.no_grad()
def generate_and_score(model, tok, question: str, cfg: HybridQuantConfig, gold: str | None = None,
                       max_new_tokens: int = 4096, thinking: bool = True, **gen_kwargs) -> GenResult:
    """Generate an answer in (optionally) thinking mode and score it.

    ``gold`` is compared against the last ``\\boxed{}`` expression after
    stripping whitespace and commas. Keep it simple; refine for the paper.
    """
    msgs = [{"role": "user", "content": question + "\nPlease reason step by step, and put your final answer within \\boxed{}."}]
    prompt = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=thinking)
    inputs = tok(prompt, return_tensors="pt").to(next(model.parameters()).device)
    gen_defaults = dict(do_sample=True, temperature=1.0, top_p=0.95, top_k=20, presence_penalty=1.5)
    gen_defaults.update(gen_kwargs)
    # transformers uses repetition_penalty, not presence_penalty; drop unsupported keys gracefully
    gen_defaults.pop("presence_penalty", None)

    with quantized_cache(cfg):
        cache = make_cache(model)
        out = model.generate(**inputs, past_key_values=cache, max_new_tokens=max_new_tokens, **gen_defaults)
    new_ids = out[0, inputs["input_ids"].shape[1]:].tolist()
    text = tok.decode(new_ids, skip_special_tokens=True)
    ans = _extract_boxed(text)
    correct = None
    if gold is not None:
        norm = lambda s: re.sub(r"[,\s$]", "", s or "")
        correct = norm(ans) == norm(gold)
    return GenResult(cfg.label(), text, len(new_ids), len(new_ids) >= max_new_tokens,
                     _repeat_ratio(new_ids), ans, correct)
