"""Falcon-H1 cached single-token decoding must give the same NLL for a document whether it is decoded alone or
inside a batch (transformers' FalconH1RMSNormGated mixes batch rows; hybridquant.compat fixes it)."""
from __future__ import annotations

import pytest
import torch

pytest.importorskip("transformers")
from transformers import DynamicCache  # noqa: E402

from hybridquant import HybridQuantConfig  # noqa: E402
from hybridquant.compat import patch_falcon_h1_decode_state_write, patch_falcon_h1_gated_norm  # noqa: E402
from hybridquant.hooks import QuantStats  # noqa: E402
from hybridquant.eval import decode_nll_batch  # noqa: E402


def _tiny_falcon():
    from transformers import FalconH1Config, FalconH1ForCausalLM

    cfg = FalconH1Config(vocab_size=128, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         mamba_n_heads=4, mamba_d_head=16, mamba_d_state=16, mamba_d_ssm=64, mamba_n_groups=1,
                         mamba_d_conv=4, mamba_rms_norm=True, mamba_norm_before_gate=True, max_position_embeddings=256)
    torch.manual_seed(0)
    model = FalconH1ForCausalLM(cfg).eval()
    return model


@torch.no_grad()
def _single_token_nll(model, ids, prefill, n):
    cache = DynamicCache(config=model.config)
    logits = model(input_ids=ids[:, :prefill], past_key_values=cache, use_cache=True).logits[:, -1].float()
    acc = [[] for _ in range(ids.shape[0])]
    for pos in range(prefill, prefill + n):
        lp = torch.log_softmax(logits, -1)
        for b in range(ids.shape[0]):
            acc[b].append(float(-lp[b, ids[b, pos]]))
        logits = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True).logits[:, -1].float()
    return [sum(a) / len(a) for a in acc]


def test_patch_makes_batched_single_token_decode_match_single():
    model = _tiny_falcon()
    assert patch_falcon_h1_gated_norm()
    torch.manual_seed(1)
    ids = torch.randint(0, 128, (3, 40))
    single = [_single_token_nll(model, ids[b:b + 1], 24, 12)[0] for b in range(3)]
    batched = _single_token_nll(model, ids, 24, 12)
    assert batched == pytest.approx(single, abs=1e-4)


def test_decode_nll_batch_matches_per_document_for_falcon():
    model = _tiny_falcon()
    patch_falcon_h1_gated_norm()
    torch.manual_seed(2)
    ids = torch.randint(0, 128, (2, 40))
    fp = HybridQuantConfig.uniform(16, 16)
    batched = [r.mean_nll for r in decode_nll_batch(model, ids, 24, fp, max_decode=12)]
    single = [decode_nll_batch(model, ids[b:b + 1], 24, fp, max_decode=12)[0].mean_nll for b in range(2)]
    assert batched == pytest.approx(single, abs=1e-4)


def test_decode_state_writes_reach_the_hook_and_are_quantized():
    """Every decoded token's Mamba-2 state must pass through the store-time quantizer (the random tiny model's
    output barely depends on its state, so the stored values are checked directly instead of the NLL)."""
    from hybridquant.hooks import make_cache, quantized_cache
    model = _tiny_falcon()
    patch_falcon_h1_gated_norm(); patch_falcon_h1_decode_state_write()
    torch.manual_seed(3)
    ids = torch.randint(0, 128, (2, 40))
    stats = QuantStats()
    with torch.no_grad(), quantized_cache(HybridQuantConfig.uniform(16, 2), stats):
        cache = make_cache(model)
        model(input_ids=ids[:, :24], past_key_values=cache, use_cache=True)
        for pos in range(24, 36):
            model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True)
    n_layers = model.config.num_hidden_layers
    assert stats.state_calls == n_layers * (1 + 12)       # one store after prefill + one per decoded token
    st = cache.layers[0].recurrent_states[0].float()
    rows = st.reshape(-1, st.shape[-1])
    assert max(len(torch.unique(r)) for r in rows) <= 4   # 2-bit codes: at most 4 levels per group (one group per row here)
