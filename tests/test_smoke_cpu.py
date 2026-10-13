"""CPU smoke test: tiny randomly initialised Qwen3.5 text model, no checkpoint download.

Checks that the store-time quantization hooks fire on the real transformers
code path (DynamicLayer.update / LinearAttentionLayer.update_recurrent_state),
that bit-width changes the decode NLL, and that patches are restored.
"""
from __future__ import annotations

import torch
import pytest
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig, cache_utils

from hybridquant import (
    HybridQuantConfig,
    QuantSpec,
    QuantStats,
    decode_nll,
    fake_quant,
    make_cache,
    quantized_cache,
)

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]
KV_GROUP = 32     # == head_dim below
STATE_GROUP = 16  # == linear_value_head_dim below


def _tiny_config() -> Qwen3_5TextConfig:
    return Qwen3_5TextConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=2,
        num_key_value_heads=1,
        head_dim=32,
        linear_num_key_heads=2,
        linear_num_value_heads=2,
        linear_key_head_dim=16,
        linear_value_head_dim=16,
        linear_conv_kernel_dim=4,
        max_position_embeddings=512,
        layer_types=list(LAYER_TYPES),
    )


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    m = Qwen3_5ForCausalLM(_tiny_config())
    m.eval()
    return m


@pytest.fixture(scope="module")
def input_ids():
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, 256, (1, 24), generator=g)


def _uniform(kv_bits: int, state_bits: int) -> HybridQuantConfig:
    return HybridQuantConfig.uniform(kv_bits, state_bits, kv_group=KV_GROUP, state_group=STATE_GROUP)


# ---------------------------------------------------------------- fake_quant


def test_fake_quant_16bit_is_identity():
    x = torch.randn(2, 3, 5, 64)
    spec = QuantSpec(bits=16, group_size=32)
    assert not spec.enabled
    y = fake_quant(x, spec)
    assert y is x


def test_fake_quant_2bit_has_error_and_keeps_shape_dtype():
    torch.manual_seed(0)
    x = torch.randn(2, 3, 5, 64)
    for spec in (QuantSpec(bits=2, group_size=32), QuantSpec(bits=2, group_size=32, symmetric=True)):
        y = fake_quant(x, spec)
        assert y.shape == x.shape and y.dtype == x.dtype
        assert not torch.allclose(x, y)
        rel = (y - x).norm() / x.norm()
        assert 0.05 < rel.item() < 1.0
        # 2-bit asymmetric: at most 4 distinct levels per group
        groups = y.reshape(-1, 32)
        assert all(len(torch.unique(g)) <= 4 for g in groups)


def test_fake_quant_error_shrinks_with_more_bits():
    torch.manual_seed(0)
    x = torch.randn(4, 128)
    errs = [(fake_quant(x, QuantSpec(bits=b, group_size=32)) - x).norm().item() for b in (2, 4, 8)]
    assert errs[0] > errs[1] > errs[2]


# ---------------------------------------------------------------- cache structure


def test_make_cache_layer_classes_and_tags(model):
    cache = make_cache(model)
    assert len(cache.layers) == len(LAYER_TYPES)
    for i, (layer, kind) in enumerate(zip(cache.layers, LAYER_TYPES)):
        assert layer._hq_layer_idx == i
        if kind == "linear_attention":
            assert isinstance(layer, cache_utils.LinearAttentionLayer)
            assert not isinstance(layer, cache_utils.DynamicLayer)
        else:
            assert isinstance(layer, cache_utils.DynamicLayer)
            assert not isinstance(layer, cache_utils.LinearAttentionLayer)


# ---------------------------------------------------------------- hooks fire on the real path


def _expected_layers():
    gdn = {i for i, t in enumerate(LAYER_TYPES) if t == "linear_attention"}
    attn = {i for i, t in enumerate(LAYER_TYPES) if t == "full_attention"}
    return gdn, attn


@pytest.mark.parametrize("kv_bits,state_bits", [(16, 16), (2, 2)])
def test_hooks_fire_during_decode(model, input_ids, kv_bits, state_bits):
    gdn, attn = _expected_layers()
    stats = QuantStats()
    prefill, n_decode = 8, input_ids.shape[1] - 8
    with quantized_cache(_uniform(kv_bits, state_bits), stats):
        cache = make_cache(model)
        model(input_ids=input_ids[:, :prefill], past_key_values=cache, use_cache=True)
        for pos in range(prefill, input_ids.shape[1]):
            model(input_ids=input_ids[:, pos:pos + 1], past_key_values=cache, use_cache=True)
    assert stats.kv_calls > 0 and stats.state_calls > 0
    assert stats.layers_seen_kv == attn
    assert stats.layers_seen_state == gdn
    # one KV write per attention layer per forward; one state write per GDN layer per forward
    n_forwards = 1 + n_decode
    assert stats.kv_calls == len(attn) * n_forwards
    assert stats.state_calls == len(gdn) * n_forwards
    assert stats.kv_tokens == len(attn) * input_ids.shape[1]


def test_decode_nll_reports_hooks(model, input_ids):
    gdn, attn = _expected_layers()
    r = decode_nll(model, input_ids, prefill_len=8, cfg=_uniform(2, 2))
    assert r.n_decode == input_ids.shape[1] - 8
    assert len(r.nll_per_pos) == r.n_decode
    assert all(v >= 0 for v in r.nll_per_pos)
    assert f"attn_layers={sorted(attn)}" in r.stats
    assert f"gdn_layers={sorted(gdn)}" in r.stats
    assert "kv_calls=0" not in r.stats and "state_calls=0" not in r.stats


def test_stored_state_is_actually_quantized(model, input_ids):
    """The state read back from the cache must lie on a 2-bit grid, i.e. storage was low precision."""
    spec = QuantSpec(bits=2, group_size=STATE_GROUP)
    cfg = HybridQuantConfig(state=spec)
    with quantized_cache(cfg):
        cache = make_cache(model)
        model(input_ids=input_ids[:, :8], past_key_values=cache, use_cache=True)
    for i, t in enumerate(LAYER_TYPES):
        if t != "linear_attention":
            continue
        s = cache.layers[i].recurrent_states[0]
        assert s.shape[-1] == STATE_GROUP
        groups = s.float().reshape(-1, STATE_GROUP)
        assert all(len(torch.unique(g)) <= 4 for g in groups)


# ---------------------------------------------------------------- bits change the metric


def test_decode_nll_differs_between_fp_and_2bit(model, input_ids):
    r_fp = decode_nll(model, input_ids, prefill_len=8, cfg=_uniform(16, 16))
    r_q = decode_nll(model, input_ids, prefill_len=8, cfg=_uniform(2, 2))
    assert r_fp.label == "Kfp_Vfp_Sfp"
    assert r_q.label == f"Kint2g{KV_GROUP}_Vint2g{KV_GROUP}_Sint2g{STATE_GROUP}"
    assert r_fp.nll_per_pos != r_q.nll_per_pos
    assert abs(r_fp.mean_nll - r_q.mean_nll) > 1e-6


def test_fp_config_matches_unpatched_forward(model, input_ids):
    """bits=16 everywhere must reproduce the plain transformers decode exactly."""
    r_fp = decode_nll(model, input_ids, prefill_len=8, cfg=_uniform(16, 16))
    with torch.no_grad():
        out = model(input_ids=input_ids, use_cache=False)
        lp = torch.log_softmax(out.logits[0].float(), dim=-1)
        ref = [-lp[p - 1, input_ids[0, p]].item() for p in range(8, input_ids.shape[1])]
    assert torch.allclose(torch.tensor(r_fp.nll_per_pos), torch.tensor(ref), atol=1e-4)


# ---------------------------------------------------------------- patch lifecycle


def test_patches_restored_after_context():
    orig_update = cache_utils.DynamicLayer.update
    orig_state = cache_utils.LinearAttentionLayer.update_recurrent_state
    with quantized_cache(_uniform(2, 2)):
        assert cache_utils.DynamicLayer.update is not orig_update
        assert cache_utils.LinearAttentionLayer.update_recurrent_state is not orig_state
    assert cache_utils.DynamicLayer.update is orig_update
    assert cache_utils.LinearAttentionLayer.update_recurrent_state is orig_state


def test_patches_restored_on_exception():
    orig_update = cache_utils.DynamicLayer.update
    orig_state = cache_utils.LinearAttentionLayer.update_recurrent_state
    with pytest.raises(RuntimeError):
        with quantized_cache(_uniform(2, 2)):
            raise RuntimeError("boom")
    assert cache_utils.DynamicLayer.update is orig_update
    assert cache_utils.LinearAttentionLayer.update_recurrent_state is orig_state
