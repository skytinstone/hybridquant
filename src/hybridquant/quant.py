"""Fake quantization primitives (quantize -> dequantize in one step).

Everything here is pure PyTorch and runs on any device. No custom kernels.
The goal is to measure the *accuracy* effect of storing a tensor at low
precision, not to save memory or time.

Conventions
-----------
* ``bits >= 16`` means "no quantization" (identity).
* Group quantization is always along the LAST dimension of the tensor.
  Callers reshape so that the last dim is the one they want grouped.
* Asymmetric uniform quantization (min/max per group) is the default because
  it is what KIVI / KVQuant / DAMP style baselines use for their INT formats.
* ``noise_match`` replaces quantization by Gaussian noise whose RMS equals the
  RMS of the round-to-nearest error on the same tensor (a control condition:
  same error energy, no quantization structure).
* ``error_feedback`` is a flag read by the cache hooks (state only): the
  residual x - Q(x) of the previous store is added before the next store.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch

__all__ = ["QuantSpec", "fake_quant", "quant_error"]


@dataclass(frozen=True)
class QuantSpec:
    """How to fake-quantize one tensor family (K, V, or recurrent state).

    bits:           target bit-width. >=16 disables quantization.
    group_size:     number of elements sharing one (scale, zero) pair along the
                    last dim. -1 means the whole last dim is one group.
    symmetric:      symmetric (absmax) vs asymmetric (min/max) uniform grid.
    stochastic:     stochastic rounding instead of round-to-nearest. Useful for
                    recurrent states where RTN bias compounds over time.
    error_feedback: carry the store residual into the next store (hooks only;
                    meaningful for the recurrent state).
    noise_match:    add Gaussian noise with the RTN error RMS instead of quantizing.
    """

    bits: int = 16
    group_size: int = 128
    symmetric: bool = False
    stochastic: bool = False
    error_feedback: bool = False
    noise_match: bool = False

    @property
    def enabled(self) -> bool:
        return self.bits < 16

    def __str__(self) -> str:  # compact label for logs / plots
        if not self.enabled:
            return "fp"
        tag = f"int{self.bits}g{self.group_size}"
        if self.symmetric:
            tag += "s"
        if self.stochastic:
            tag += "sr"
        if self.error_feedback:
            tag += "ef"
        if self.noise_match:
            tag += "gn"
        return tag


def _round(x: torch.Tensor, stochastic: bool, generator=None) -> torch.Tensor:
    if stochastic:
        return torch.floor(x + torch.rand(x.shape, device=x.device, dtype=x.dtype, generator=generator))
    return torch.round(x)


def _rtn(xf: torch.Tensor, spec: QuantSpec, generator=None) -> torch.Tensor:
    """Uniform fake quantization of a [n, g] fp32 tensor (one group per row)."""
    qmax = 2**spec.bits - 1
    if spec.symmetric:
        # levels in [-(2^(b-1)), 2^(b-1)-1], zero point fixed at 0
        amax = xf.abs().amax(dim=-1, keepdim=True).clamp_min(1e-8)
        half = 2 ** (spec.bits - 1)
        scale = amax / (half - 1)
        q = _round(xf / scale, spec.stochastic, generator).clamp_(-half, half - 1)
        return q * scale
    mn = xf.amin(dim=-1, keepdim=True)
    mx = xf.amax(dim=-1, keepdim=True)
    scale = ((mx - mn) / qmax).clamp_min(1e-8)
    zero = torch.round(-mn / scale)
    q = (_round(xf / scale, spec.stochastic, generator) + zero).clamp_(0, qmax)
    return (q - zero) * scale


@torch.no_grad()
def fake_quant(x: torch.Tensor, spec: QuantSpec, generator: torch.Generator | None = None) -> torch.Tensor:
    """Quantize ``x`` along its last dim with ``spec`` and dequantize back.

    Returns a tensor with the same shape and dtype as ``x``.
    Computation is done in fp32 for numerical safety. ``generator`` seeds the
    stochastic-rounding and noise-match randomness (must live on ``x.device``).
    """
    if not spec.enabled:
        return x

    orig_dtype = x.dtype
    orig_shape = x.shape
    last = orig_shape[-1]
    g = last if spec.group_size in (-1, 0) or spec.group_size > last else spec.group_size
    if last % g != 0:
        raise ValueError(f"last dim {last} not divisible by group_size {g}")

    xf = x.to(torch.float32).reshape(-1, g)
    deq = _rtn(xf, spec, generator)
    if spec.noise_match:
        err_rms = (deq - xf).pow(2).mean().sqrt()
        noise = torch.randn(xf.shape, device=xf.device, dtype=xf.dtype, generator=generator) * err_rms
        deq = xf + noise
    return deq.reshape(orig_shape).to(orig_dtype)


@torch.no_grad()
def quant_error(x: torch.Tensor, spec: QuantSpec) -> dict[str, float]:
    """Relative L2 error and max-abs error of fake-quantizing ``x``."""
    xq = fake_quant(x, spec)
    diff = (xq.float() - x.float())
    denom = x.float().norm().clamp_min(1e-12)
    return {
        "rel_l2": (diff.norm() / denom).item(),
        "max_abs": diff.abs().max().item(),
    }
