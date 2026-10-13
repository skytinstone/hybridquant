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

Error feedback (state only): with ``QuantSpec.error_feedback`` the hook keeps a
per-layer residual r. It stores Q(S + r) and sets r = (S + r) - Q(S + r), so the
stored values track the true state in cumulative sum (bounded residual). The
residual is reset to zero at the prefill store (first write into a layer's state).

Semantics to keep in mind (document these in the paper):
* During prefill the chunked kernel computes with full-precision intermediate
  states and only the *final* state is quantized. State quantization error
  therefore only accumulates during token-by-token decoding. Use
  ``hybridquant.eval.decode_nll`` (teacher-forced decode) to measure it.
* The conv1d state (tiny: kernel_size x channels) is left in full precision.
"""
from __future__ import annotations

import contextlib
import warnings
from dataclasses import dataclass, field

import torch
from transformers import cache_utils

from .quant import QuantSpec, fake_quant

__all__ = ["HybridQuantConfig", "QuantStats", "quantized_cache", "make_cache", "tag_cache_layers",
           "HookCoverage", "check_hook_coverage"]


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
                state_symmetric: bool = False, state_stochastic: bool = False,
                state_error_feedback: bool = False, kv_noise_match: bool = False,
                state_noise_match: bool = False) -> "HybridQuantConfig":
        kv = QuantSpec(kv_bits, kv_group, noise_match=kv_noise_match)
        return cls(
            key=kv,
            value=kv,
            state=QuantSpec(state_bits, state_group, symmetric=state_symmetric, stochastic=state_stochastic,
                            error_feedback=state_error_feedback, noise_match=state_noise_match),
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
    # error-feedback residual buffers, layer_idx -> fp32 tensor (filled only when EF is on)
    ef_residuals: dict = field(default_factory=dict)

    def summary(self) -> str:
        return (f"kv_calls={self.kv_calls} kv_tokens={self.kv_tokens} "
                f"attn_layers={sorted(self.layers_seen_kv)} | "
                f"state_calls={self.state_calls} gdn_layers={sorted(self.layers_seen_state)}")


def _layer_idx(layer_obj) -> int | None:
    # Cache layers don't know their index; we attach it in ``tag_cache_layers``.
    return getattr(layer_obj, "_hq_layer_idx", None)


def _quant_state_tensor(state: torch.Tensor, spec: QuantSpec, generator=None) -> torch.Tensor:
    # state: [B, H, d_k, d_v]. Group along d_v (rows of the associative memory).
    return fake_quant(state, spec, generator)


def _quant_kv_tensor(x: torch.Tensor, spec: QuantSpec, generator=None) -> torch.Tensor:
    # x: [B, H_kv, T, head_dim]. Per-token groups along head_dim.
    return fake_quant(x, spec, generator)


@contextlib.contextmanager
def quantized_cache(cfg: HybridQuantConfig, stats: QuantStats | None = None, track_error: bool = False,
                    noise_seed: int | None = None):
    """Patch transformers cache layers so every write is fake-quantized.

    ``noise_seed`` fixes the randomness of stochastic rounding and noise-match
    (one torch.Generator per device). ``stats.ef_residuals`` holds the error
    feedback buffers while the context is open.

    Usage::

        with quantized_cache(cfg, stats):
            out = model.generate(**inputs, past_key_values=cache, max_new_tokens=256)
    """
    stats = stats if stats is not None else QuantStats()
    stats.ef_residuals.clear()
    residuals = stats.ef_residuals
    generators: dict[str, torch.Generator] = {}
    warned = {"ef_no_idx": False}

    def gen_for(x: torch.Tensor):
        if noise_seed is None:
            return None
        dev = x.device.type
        if dev not in generators:
            generators[dev] = torch.Generator(device=dev).manual_seed(noise_seed)
        return generators[dev]

    DynamicLayer = cache_utils.DynamicLayer
    LinearLayer = cache_utils.LinearAttentionLayer
    orig_update = DynamicLayer.update
    orig_update_state = LinearLayer.update_recurrent_state

    def patched_update(self, key_states, value_states, *args, **kwargs):
        idx = _layer_idx(self)
        if cfg.attn_layers is None or idx in cfg.attn_layers:
            if cfg.key.enabled:
                key_states = _quant_kv_tensor(key_states, cfg.key, gen_for(key_states))
            if cfg.value.enabled:
                value_states = _quant_kv_tensor(value_states, cfg.value, gen_for(value_states))
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
                x = recurrent_states
                use_ef = cfg.state.error_feedback
                if use_ef and idx is None:
                    if not warned["ef_no_idx"]:
                        warnings.warn("error feedback needs tagged cache layers (use make_cache); EF disabled")
                        warned["ef_no_idx"] = True
                    use_ef = False
                if use_ef:
                    is_prefill = not self.is_recurrent_states_initialized[state_idx]
                    r = residuals.get(idx)
                    if is_prefill or r is None or r.shape != x.shape:
                        r = torch.zeros(x.shape, dtype=torch.float32, device=x.device)
                    x = (x.float() + r).to(recurrent_states.dtype)
                q = _quant_state_tensor(x, cfg.state, gen_for(x))
                if use_ef:
                    residuals[idx] = x.float() - q.float()
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


@dataclass
class HookCoverage:
    """Result of comparing QuantStats against a model's layer_types."""

    passed: bool
    expected_kv: set
    expected_state: set
    seen_kv: set
    seen_state: set
    messages: list

    def report(self) -> str:
        head = "PASS" if self.passed else "FAIL"
        return "\n".join([f"hook coverage: {head}"] + [f"  {m}" for m in self.messages])


def check_hook_coverage(layer_types, stats: QuantStats) -> HookCoverage:
    """Judge whether the hooks fired on exactly the layers that ``layer_types`` says they should.

    ``layers_seen_kv`` must equal the set of ``full_attention`` indices and
    ``layers_seen_state`` must equal the set of ``linear_attention`` indices.
    If the state side never fired at all, the most likely cause on GPU is that
    the kernels hub replaced ``Qwen3_5GatedDeltaNet.forward`` wholesale, which
    skips ``cache.update_recurrent_state``; the message says so.
    """
    layer_types = list(layer_types)
    expected_kv = {i for i, t in enumerate(layer_types) if t == "full_attention"}
    expected_state = {i for i, t in enumerate(layer_types) if t == "linear_attention"}
    seen_kv = set(stats.layers_seen_kv)
    seen_state = set(stats.layers_seen_state)
    msgs = []
    kv_ok = seen_kv == expected_kv and stats.kv_calls > 0
    st_ok = seen_state == expected_state and stats.state_calls > 0
    msgs.append(f"kv:    calls={stats.kv_calls} seen={sorted(seen_kv)} expected={sorted(expected_kv)} "
                f"-> {'ok' if kv_ok else 'MISMATCH'}")
    msgs.append(f"state: calls={stats.state_calls} seen={sorted(seen_state)} expected={sorted(expected_state)} "
                f"-> {'ok' if st_ok else 'MISMATCH'}")
    passed = kv_ok and st_ok
    if not passed and stats.state_calls == 0 and expected_state:
        msgs.append("WARNING: state 훅이 한 번도 호출되지 않았습니다. kernels 허브가 GatedDeltaNet forward를 "
                    "교체해 update_recurrent_state를 건너뛰었을 수 있습니다. transformers를 import하기 전에 "
                    "USE_HUB_KERNELS=NO를 설정하거나 from_pretrained(use_kernels=False)로 다시 실행해 보십시오.")
    return HookCoverage(passed, expected_kv, expected_state, seen_kv, seen_state, msgs)
