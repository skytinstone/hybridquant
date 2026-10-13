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
import re
from dataclasses import dataclass, asdict

import torch

from .hooks import HybridQuantConfig, QuantStats, quantized_cache, make_cache

__all__ = ["decode_nll", "DecodeResult", "generate_and_score", "GenResult", "load_model"]


def load_model(name: str, dtype=torch.bfloat16, device_map="auto", text_only: bool = True):
    """Load a Qwen3.5 checkpoint. Returns (model, tokenizer).

    ``text_only`` loads the language model without the vision tower, which
    saves VRAM on Colab. Falls back to the multimodal class if unavailable.
    """
    from transformers import AutoTokenizer, AutoConfig

    tok = AutoTokenizer.from_pretrained(name)
    model = None
    if text_only:
        try:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(name, dtype=dtype, device_map=device_map)
        except Exception as e:  # noqa: BLE001
            print(f"[load_model] text-only load failed ({type(e).__name__}: {e}); falling back to multimodal class")
    if model is None:
        from transformers import AutoModelForMultimodalLM

        model = AutoModelForMultimodalLM.from_pretrained(name, dtype=dtype, device_map=device_map)
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
def decode_nll(model, input_ids: torch.Tensor, prefill_len: int, cfg: HybridQuantConfig,
               max_decode: int | None = None, step: int = 1) -> DecodeResult:
    """Teacher-forced decode NLL under a quantized cache.

    input_ids: [1, T] token ids. Positions [0, prefill_len) are prefilled in one
    forward; positions [prefill_len, T) are fed ``step`` tokens at a time.
    ``step=1`` is the faithful decode path (slow but exact). Larger ``step``
    uses the chunked GDN kernel between quantization points (faster, coarser).
    """
    assert input_ids.dim() == 2 and input_ids.shape[0] == 1, "batch size 1 only"
    device = next(model.parameters()).device
    ids = input_ids.to(device)
    T = ids.shape[1]
    end = T if max_decode is None else min(T, prefill_len + max_decode)

    stats = QuantStats()
    nll = []
    with quantized_cache(cfg, stats):
        cache = make_cache(model)
        out = model(input_ids=ids[:, :prefill_len], past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1].float()
        pos = prefill_len
        while pos < end:
            nxt = ids[:, pos:pos + step]
            # score the first token of this chunk with the previous logits
            lp = torch.log_softmax(logits, dim=-1)
            nll.append(-lp[0, nxt[0, 0]].item())
            out = model(input_ids=nxt, past_key_values=cache, use_cache=True)
            if step > 1:
                # score remaining tokens of the chunk with their teacher-forced logits
                lps = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
                for j in range(1, nxt.shape[1]):
                    nll.append(-lps[j - 1, nxt[0, j]].item())
            logits = out.logits[:, -1].float()
            pos += nxt.shape[1]
    return DecodeResult(cfg.label(), prefill_len, len(nll), nll, stats.summary())


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
