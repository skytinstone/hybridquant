"""Upstream work-arounds applied at model load time. Each one is documented in docs/decision_log.md."""
from __future__ import annotations

import torch
import torch.nn.functional as F

__all__ = ["patch_falcon_h1_gated_norm", "apply_model_patches"]


def _gated_rmsnorm_forward(self, hidden_states, gate=None):
    """Shape-preserving replacement for FalconH1RMSNormGated.forward.

    The upstream version squeezes a single-token input to [B, dim] and then multiplies by a [B, 1, dim] gate,
    which broadcasts to [B, B, dim] and mixes the batch rows during cached single-token decoding. With B = 1
    the shapes coincide, so single-sequence generation is unaffected and the bug stays hidden.
    """
    input_dtype = hidden_states.dtype
    shape = hidden_states.shape
    if gate is not None:
        gate = gate.reshape(shape)   # same element count ([B, 1, dim] vs [B, dim]); never let the two broadcast
    hs = hidden_states.to(torch.float32)
    if not self.norm_before_gate and gate is not None:
        hs = hs * F.silu(gate.to(torch.float32))
    dim = shape[-1]
    g = hs.reshape(-1, self.n_groups, dim // self.n_groups)
    g = g * torch.rsqrt(g.pow(2).mean(-1, keepdim=True) + self.variance_epsilon)
    g = self.weight.view(self.n_groups, dim // self.n_groups) * g
    hs = g.reshape(shape)
    if self.norm_before_gate and gate is not None:
        hs = hs * F.silu(gate.to(torch.float32))
    return hs.to(input_dtype)


def patch_falcon_h1_gated_norm() -> bool:
    """Install the fix on transformers' FalconH1RMSNormGated. Idempotent; returns True when applied."""
    try:
        from transformers.models.falcon_h1 import modeling_falcon_h1 as fh
    except Exception:  # noqa: BLE001  (transformers without Falcon-H1)
        return False
    cls = fh.FalconH1RMSNormGated
    if getattr(cls, "_hybridquant_patched", False):
        return True
    cls.forward = _gated_rmsnorm_forward
    cls._hybridquant_patched = True
    return True


def apply_model_patches(model) -> list[str]:
    """Apply the patches relevant to ``model``; returns the names of the ones installed."""
    applied = []
    cfg = getattr(model, "config", None)
    text_cfg = cfg.get_text_config(decoder=True) if cfg is not None and hasattr(cfg, "get_text_config") else cfg
    if getattr(text_cfg, "model_type", "") == "falcon_h1" and patch_falcon_h1_gated_norm():
        applied.append("falcon_h1_gated_norm")
    return applied
