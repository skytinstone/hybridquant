"""Store-time fake-quantization hooks for Hugging Face caches.

Two intervention points cover the whole hybrid memory of a Qwen3.5 model:

* ``DynamicLayer.update(key_states, value_states)``
    Called by every softmax-attention layer right before the new K/V are
    appended to the cache. We quantize the *new* tokens' K and V here, so each
    token is quantized exactly once at write time (per-token groups).

* ``LinearAttentionLayer.update_recurrent_state(state)``
    Called by every Gated DeltaNet layer after the delta-rule update with the
    fresh recurrent state ``S`` of shape [B, H, d_k, d_v] (fp32). We quantize it
    here, so the stored state is low-precision and the *next* step reads the
    rounded state. Rounding error therefore propagates through the recurrence,
    which is exactly the effect we want to measure.

Both patches are class-level and guarded by a context manager so they can be
turned on/off per experiment without reloading the model.

Semantics to keep in mind (document these in the paper):
* During prefill the chunked kernel computes with full-precision intermediate
  states and only the *final* state is quantized. State quantization error
  therefore only accumulates during token-by-token decoding. Use
  ``hybridquant.eval.decode_nll`` (teacher-forced decode) to measure it.
* The conv1d state (tiny: kernel_size x channels) is left in full precision.
"""
from __future__ import annotations

import contextlib
from dataclasses import dataclass, field

import torch
from transformers import cache_utils

from .quant import QuantSpec, fake_quant

__all__ = ["HybridQuantConfig", "QuantStats", "quantized_cache", "make_cache", "tag_cache_layers"]


@dataclass(frozen=True)
class HybridQuantConfig:
    """Bit allocation for the three cached tensor families."""

    key: QuantSpec = QuantSpec()
    value: QuantSpec = QuantSpec()
    state: QuantSpec = QuantSpec()
    # Which layer indices to touch; None = all layers of that kind.
    attn_layers: frozenset[int] | None = None
    gdn_layers: frozenset[int] | None = None

    def label(self) -> str:
        return f"K{self.key}_V{self.value}_S{self.state}"

    @classmethod
    def uniform(cls, kv_bits: int, state_bits: int, kv_group: int = 128, state_group: int = 128,
                state_symmetric: bool = False, state_stochastic: bool = False) -> "HybridQuantConfig":
        return cls(
            key=QuantSpec(kv_bits, kv_group),
            value=QuantSpec(kv_bits, kv_group),
            state=QuantSpec(state_bits, state_group, symmetric=state_symmetric, stochastic=state_stochastic),
        )


@dataclass
class QuantStats:
    """Counters so you can verify hooks actually fired."""

    kv_calls: int = 0
    state_calls: int = 0
    kv_tokens: int = 0
    layers_seen_kv: set = field(default_factory=set)
    layers_seen_state: set = field(default_factory=set)
    # optional: per-layer relative error of the last state quantization
    last_state_rel_err: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"kv_calls={self.kv_calls} kv_tokens={self.kv_tokens} "
                f"attn_layers={sorted(self.layers_seen_kv)} | "
                f"state_calls={self.state_calls} gdn_layers={sorted(self.layers_seen_state)}")


def _layer_idx(layer_obj) -> int | None:
    # Cache layers don't know their index; we attach it in ``_tag_layers``.
    return getattr(layer_obj, "_hq_layer_idx", None)


def _quant_state_tensor(state: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    # state: [B, H, d_k, d_v]. Group along d_v (rows of the associative memory).
    return fake_quant(state, spec)


def _quant_kv_tensor(x: torch.Tensor, spec: QuantSpec) -> torch.Tensor:
    # x: [B, H_kv, T, head_dim]. Per-token groups along head_dim.
    return fake_quant(x, spec)


@contextlib.contextmanager
def quantized_cache(cfg: HybridQuantConfig, stats: QuantStats | None = None, track_error: bool = False):
    """Patch transformers cache layers so every write is fake-quantized.

    Usage::

        with quantized_cache(cfg, stats):
            out = model.generate(**inputs, max_new_tokens=256)
    """
    stats = stats if stats is not None else QuantStats()

    DynamicLayer = cache_utils.DynamicLayer
    LinearLayer = cache_utils.LinearAttentionLayer
    orig_update = DynamicLayer.update
    orig_update_state = LinearLayer.update_recurrent_state

    def patched_update(self, key_states, value_states, *args, **kwargs):
        idx = _layer_idx(self)
        if cfg.attn_layers is None or idx in cfg.attn_layers:
            if cfg.key.enabled:
                key_states = _quant_kv_tensor(key_states, cfg.key)
            if cfg.value.enabled:
                value_states = _quant_kv_tensor(value_states, cfg.value)
            stats.kv_calls += 1
            stats.kv_tokens += key_states.shape[-2]
            if idx is not None:
                stats.layers_seen_kv.add(idx)
        return orig_update(self, key_states, value_states, *args, **kwargs)

    def patched_update_state(self, recurrent_states, state_idx: int = 0, **kwargs):
        idx = _layer_idx(self)
        if cfg.gdn_layers is None or idx in cfg.gdn_layers:
            # Count every call (like the KV hook) so an fp baseline still proves the hook fired.
            if cfg.state.enabled:
                q = _quant_state_tensor(recurrent_states, cfg.state)
                if track_error and idx is not None:
                    d = (q.float() - recurrent_states.float()).norm() / recurrent_states.float().norm().clamp_min(1e-12)
                    stats.last_state_rel_err[idx] = d.item()
                recurrent_states = q
            stats.state_calls += 1
            if idx is not None:
                stats.layers_seen_state.add(idx)
        return orig_update_state(self, recurrent_states, state_idx=state_idx, **kwargs)

    DynamicLayer.update = patched_update
    LinearLayer.update_recurrent_state = patched_update_state
    try:
        yield stats
    finally:
        DynamicLayer.update = orig_update
        LinearLayer.update_recurrent_state = orig_update_state


def tag_cache_layers(cache) -> None:
    """Attach layer indices to cache layer objects so hooks can filter by layer."""
    for i, layer in enumerate(cache.layers):
        layer._hq_layer_idx = i


def make_cache(model):
    """Create a DynamicCache for ``model`` with layer indices tagged.

    Pass the result as ``past_key_values=`` to ``generate``/``forward`` when you
    need per-layer control; otherwise the hooks work on the auto-created cache
    too (layer indices will just be None).
    """
    from transformers import DynamicCache

    cache = DynamicCache(config=model.config)
    tag_cache_layers(cache)
    return cache
