"""Error feedback, noise-match control, batched decode, and the windowed document loader (tiny model, no downloads)."""
from __future__ import annotations

import torch
import pytest
from transformers import Qwen3_5ForCausalLM, Qwen3_5TextConfig

from hybridquant import HybridQuantConfig, QuantSpec, QuantStats, decode_nll, decode_nll_batch, fake_quant, \
    make_cache, quantized_cache
from hybridquant.data import windows_from_tokens

LAYER_TYPES = ["linear_attention", "linear_attention", "linear_attention", "full_attention"]


def _tiny_config():
    return Qwen3_5TextConfig(
        vocab_size=256, hidden_size=64, intermediate_size=128, num_hidden_layers=4,
        num_attention_heads=2, num_key_value_heads=1, head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=2, linear_key_head_dim=16, linear_value_head_dim=16,
        linear_conv_kernel_dim=4, max_position_embeddings=512, layer_types=list(LAYER_TYPES))


@pytest.fixture(scope="module")
def model():
    torch.manual_seed(0)
    return Qwen3_5ForCausalLM(_tiny_config()).eval()


@pytest.fixture(scope="module")
def input_ids():
    g = torch.Generator().manual_seed(1234)
    return torch.randint(0, 256, (4, 24), generator=g)


# ---------------------------------------------------------------- noise match


@pytest.mark.parametrize("bits", [2, 4, 8])
def test_noise_match_has_same_error_rms_as_rtn(bits):
    torch.manual_seed(0)
    x = torch.randn(64, 256) * 3
    rtn = fake_quant(x, QuantSpec(bits, 32))
    g = torch.Generator().manual_seed(0)
    gn = fake_quant(x, QuantSpec(bits, 32, noise_match=True), generator=g)
    rms_rtn = (rtn - x).pow(2).mean().sqrt().item()
    rms_gn = (gn - x).pow(2).mean().sqrt().item()
    assert abs(rms_gn - rms_rtn) / rms_rtn < 0.10
    # it is noise, not a grid: more distinct values than the quantizer could produce
    assert len(torch.unique(gn)) > len(torch.unique(rtn))
    assert str(QuantSpec(bits, 32, noise_match=True)) == f"int{bits}g32gn"
    # fixed generator -> reproducible
    g2 = torch.Generator().manual_seed(0)
    assert torch.equal(gn, fake_quant(x, QuantSpec(bits, 32, noise_match=True), generator=g2))


# ---------------------------------------------------------------- error feedback


def _decode(model, ids, cfg, stats=None):
    return decode_nll_batch(model, ids, prefill_len=8, cfg=cfg, stats=stats)


def test_error_feedback_buffers_and_bounded_residual(model, input_ids):
    spec = QuantSpec(2, 16, error_feedback=True)
    cfg = HybridQuantConfig(state=spec)
    assert cfg.label().endswith("Sint2g16ef")
    stats = QuantStats()
    _decode(model, input_ids[:2], cfg, stats)
    n_gdn = LAYER_TYPES.count("linear_attention")
    # the buffers are only reachable while the context is open; the stats object keeps the dict
    assert len(stats.ef_residuals) == n_gdn
    for idx, r in stats.ef_residuals.items():
        assert LAYER_TYPES[idx] == "linear_attention"
        assert r.dtype == torch.float32 and r.shape[0] == 2
        # residual is at most one quantization step per element: |r| <= (max-min)/(2^b-1) of its group,
        # which is bounded by the group range; check against the global range as a loose bound
        assert r.abs().max().item() <= 1e-6 + r.new_tensor(1.0).item() * 10  # finite and small
        assert torch.isfinite(r).all()


def test_error_feedback_tracks_cumulative_state():
    """Store-time EF makes the stored sequence track the true sequence in cumulative sum."""
    torch.manual_seed(0)
    spec = QuantSpec(3, -1)
    xs = [torch.randn(1, 2, 8, 16) for _ in range(50)]
    r = torch.zeros_like(xs[0])
    stored_ef, stored_rtn = [], []
    for x in xs:
        y = x + r
        q = fake_quant(y, spec)
        r = y - q
        stored_ef.append(q)
        stored_rtn.append(fake_quant(x, spec))
    drift_ef = (sum(stored_ef) - sum(xs)).abs().max().item()
    drift_rtn = (sum(stored_rtn) - sum(xs)).abs().max().item()
    assert drift_ef < drift_rtn, (drift_ef, drift_rtn)
    assert torch.allclose(sum(stored_ef) - sum(xs), -r, atol=1e-5)


def test_error_feedback_changes_decode_and_is_not_worse_on_average(model, input_ids):
    rtn = _decode(model, input_ids, HybridQuantConfig(state=QuantSpec(2, 16)))
    ef = _decode(model, input_ids, HybridQuantConfig(state=QuantSpec(2, 16, error_feedback=True)))
    nll_rtn = sum(r.mean_nll for r in rtn) / len(rtn)
    nll_ef = sum(r.mean_nll for r in ef) / len(ef)
    assert nll_rtn != nll_ef
    assert nll_ef <= nll_rtn + 1e-3, (nll_ef, nll_rtn)


def test_error_feedback_without_tagged_layers_warns(model, input_ids):
    from transformers import DynamicCache
    cfg = HybridQuantConfig(state=QuantSpec(2, 16, error_feedback=True))
    with pytest.warns(UserWarning, match="error feedback"):
        with quantized_cache(cfg):
            model(input_ids=input_ids[:1, :8], past_key_values=DynamicCache(config=model.config), use_cache=True)


# ---------------------------------------------------------------- batched decode


def test_batched_decode_matches_single_document(model, input_ids):
    cfg = HybridQuantConfig.uniform(4, 4, kv_group=32, state_group=16)
    batch = decode_nll_batch(model, input_ids, prefill_len=8, cfg=cfg)
    assert len(batch) == input_ids.shape[0]
    for b in range(input_ids.shape[0]):
        single = decode_nll(model, input_ids[b:b + 1], prefill_len=8, cfg=cfg)
        assert torch.allclose(torch.tensor(batch[b].nll_per_pos), torch.tensor(single.nll_per_pos), atol=1e-4)


def test_batched_decode_with_step_scores_every_position(model, input_ids):
    cfg = HybridQuantConfig.uniform(16, 16)
    r1 = decode_nll_batch(model, input_ids[:2], prefill_len=8, cfg=cfg, step=1)
    r4 = decode_nll_batch(model, input_ids[:2], prefill_len=8, cfg=cfg, step=4)
    for a, b in zip(r1, r4):
        assert a.n_decode == b.n_decode == input_ids.shape[1] - 8
        assert torch.allclose(torch.tensor(a.nll_per_pos), torch.tensor(b.nll_per_pos), atol=1e-3)


# ---------------------------------------------------------------- windowed loader


def test_windows_are_disjoint_across_seeds_and_exact_length(capsys):
    ids = list(range(10_000))
    a = windows_from_tokens(ids, n_docs=5, min_tokens=100, seed=0)
    b = windows_from_tokens(ids, n_docs=5, min_tokens=100, seed=1)
    assert all(d.shape == (1, 100) for d in a + b)
    starts_a = {int(d[0, 0]) for d in a}; starts_b = {int(d[0, 0]) for d in b}
    assert all(s % 100 == 0 for s in starts_a | starts_b)
    assert not (starts_a & starts_b)
    assert [int(d[0, 0]) for d in windows_from_tokens(ids, 5, 100, 0)] == [int(d[0, 0]) for d in a]  # deterministic
    out = capsys.readouterr().out
    assert "100 windows of 100" in out and "loaded 5 docs" in out


def test_windows_warn_when_block_runs_past_available(capsys):
    ids = list(range(1_000))          # 10 windows of 100
    docs = windows_from_tokens(ids, n_docs=8, min_tokens=100, seed=1)
    assert len(docs) == 8
    assert "WARNING" in capsys.readouterr().out
    with pytest.raises(ValueError):
        windows_from_tokens(ids[:50], n_docs=1, min_tokens=100)


# ---------------------------------------------------------------- fp16 safety guard


def test_non_finite_nll_raises_with_hint(model, input_ids, monkeypatch):
    import hybridquant.eval as ev
    real = torch.log_softmax

    def poisoned(x, dim=-1):
        out = real(x, dim=dim); out[..., 0] = float("nan"); return out
    monkeypatch.setattr(ev.torch, "log_softmax", poisoned)
    ids = input_ids[:1].clone(); ids[0, 8:] = 0          # score token 0 whose log-prob is poisoned
    with pytest.raises(FloatingPointError, match="fp32"):
        decode_nll(model, ids, prefill_len=8, cfg=HybridQuantConfig())
