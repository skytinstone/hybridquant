"""Equation checks shared with notebooks/02_equations_walkthrough.ipynb.

Section 2: the single-head gated delta rule written as a 10-line reference must
match transformers' torch_recurrent_gated_delta_rule (outputs and final state).
Section 4A: with per-step state fake-quantization, the state error obeys
    S~_t - S*_t = sum_i (prod_{j=i+1..t} M_j) delta_i,   M_j = alpha_j (I - beta_j k_j k_j^T)
exactly (up to float64 round-off).
"""
from __future__ import annotations

import math

import pytest
import torch
import torch.nn.functional as F

from hybridquant import QuantSpec, fake_quant

# ---------------------------------------------------------------- reference implementation (same lines as the notebook)


def preprocess_qk(q: torch.Tensor, k: torch.Tensor, l2norm: bool = True):
    """What transformers does before the recurrence: fla-style l2norm (eps 1e-6) and q / sqrt(d_k)."""
    if l2norm:
        q = q * torch.rsqrt((q * q).sum(-1, keepdim=True) + 1e-6)
        k = k * torch.rsqrt((k * k).sum(-1, keepdim=True) + 1e-6)
    return q / math.sqrt(q.shape[-1]), k


def reference_gdn(q, k, v, alpha, beta, S0=None, decay_first: bool = True):
    """Single head. q,k: [T,d_k]  v: [T,d_v]  alpha,beta: [T].
    decay_first=True:  S_t = alpha_t (I - beta_t k_t k_t^T) S_{t-1} + beta_t k_t v_t^T   (transformers / GDN paper)
    decay_first=False: S_t = alpha_t S_{t-1} + beta_t k_t (v_t - S_{t-1}^T k_t)^T           (naive reading, does NOT match)
    """
    S = torch.zeros(k.shape[1], v.shape[1], dtype=q.dtype) if S0 is None else S0.clone()
    out = torch.zeros(v.shape, dtype=q.dtype)
    for t in range(k.shape[0]):
        S_prev = alpha[t] * S if decay_first else S
        S = alpha[t] * S + beta[t] * torch.outer(k[t], v[t] - S_prev.T @ k[t])
        out[t] = S.T @ q[t]
    return out, S


def gdn_trajectory(q, k, v, alpha, beta, spec: QuantSpec | None = None):
    """Decay-first recurrence; if ``spec`` is given the state is fake-quantized after every step.
    Returns the list of stored states S_t and the list of deltas delta_t = Q(S^_t) - S^_t."""
    S = torch.zeros(k.shape[1], v.shape[1], dtype=q.dtype)
    states, deltas = [], []
    for t in range(k.shape[0]):
        S = alpha[t] * S
        S_hat = S + beta[t] * torch.outer(k[t], v[t] - S.T @ k[t])
        S = fake_quant(S_hat, spec) if spec is not None else S_hat
        states.append(S.clone())
        deltas.append(S - S_hat)
    return states, deltas


def propagated_error(k, alpha, beta, deltas, t: int) -> torch.Tensor:
    """Right-hand side of eq. 4: sum_i (prod_{j=i+1..t} M_j) delta_i with M_j = alpha_j (I - beta_j k_j k_j^T)."""
    d_k = k.shape[1]
    M = [alpha[j] * (torch.eye(d_k, dtype=k.dtype) - beta[j] * torch.outer(k[j], k[j])) for j in range(t + 1)]
    e = torch.zeros_like(deltas[0])
    for i in range(t + 1):
        term = deltas[i]
        for j in range(i + 1, t + 1):
            term = M[j] @ term
        e = e + term
    return e


def _random_inputs(T=24, d_k=16, d_v=16, dtype=torch.float64, seed=0):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(T, d_k, generator=g, dtype=dtype)
    k = torch.randn(T, d_k, generator=g, dtype=dtype)
    v = torch.randn(T, d_v, generator=g, dtype=dtype)
    log_alpha = -F.softplus(torch.randn(T, generator=g, dtype=dtype))      # g <= 0  ->  alpha in (0, 1)
    beta = torch.sigmoid(torch.randn(T, generator=g, dtype=dtype))
    return q, k, v, log_alpha, beta


# ---------------------------------------------------------------- section 2: reference == transformers kernel


def test_reference_gdn_matches_transformers_recurrent_kernel():
    from transformers.models.qwen3_5.modeling_qwen3_5 import torch_recurrent_gated_delta_rule

    B, T, H, d_k, d_v = 1, 24, 3, 16, 16
    g = torch.Generator().manual_seed(1)
    q = torch.randn(B, T, H, d_k, generator=g)
    k = torch.randn(B, T, H, d_k, generator=g)
    v = torch.randn(B, T, H, d_v, generator=g)
    log_alpha = -F.softplus(torch.randn(B, T, H, generator=g))
    beta = torch.sigmoid(torch.randn(B, T, H, generator=g))

    out_tf, S_tf = torch_recurrent_gated_delta_rule(
        q, k, v, g=log_alpha, beta=beta, initial_state=None, output_final_state=True, use_qk_l2norm_in_kernel=True
    )
    assert out_tf.shape == (B, T, H, d_v) and S_tf.shape == (B, H, d_k, d_v)

    worst_exact, worst_naive = 0.0, 0.0
    for h in range(H):
        qh, kh = preprocess_qk(q[0, :, h], k[0, :, h])
        out_ref, S_ref = reference_gdn(qh, kh, v[0, :, h], log_alpha[0, :, h].exp(), beta[0, :, h])
        worst_exact = max(worst_exact, (out_ref - out_tf[0, :, h]).abs().max().item(),
                          (S_ref - S_tf[0, h]).abs().max().item())
        out_nv, S_nv = reference_gdn(qh, kh, v[0, :, h], log_alpha[0, :, h].exp(), beta[0, :, h], decay_first=False)
        worst_naive = max(worst_naive, (out_nv - out_tf[0, :, h]).abs().max().item())
    assert worst_exact < 1e-4, worst_exact
    # the form with the undecayed S_{t-1} inside the correction is a different recurrence
    assert worst_naive > 100 * max(worst_exact, 1e-7), (worst_naive, worst_exact)


def test_transition_matrix_norm_equals_alpha_for_unit_keys():
    q, k, v, log_alpha, beta = _random_inputs()
    _, k = preprocess_qk(q, k)
    for j in range(k.shape[0]):
        M = log_alpha[j].exp() * (torch.eye(k.shape[1], dtype=k.dtype) - beta[j] * torch.outer(k[j], k[j]))
        assert abs(torch.linalg.matrix_norm(M, ord=2).item() - log_alpha[j].exp().item()) < 1e-9


# ---------------------------------------------------------------- section 4A: eq. 4 is exact


@pytest.mark.parametrize("bits", [2, 4])
def test_state_error_propagation_identity(bits):
    q, k, v, log_alpha, beta = _random_inputs()
    q, k = preprocess_qk(q, k)
    alpha = log_alpha.exp()
    spec = QuantSpec(bits=bits, group_size=-1)
    clean, _ = gdn_trajectory(q, k, v, alpha, beta)
    noisy, deltas = gdn_trajectory(q, k, v, alpha, beta, spec)
    for t in (0, 5, len(clean) - 1):
        direct = noisy[t] - clean[t]
        formula = propagated_error(k, alpha, beta, deltas, t)
        scale = direct.abs().max().item()
        assert scale > 0, "quantization produced no error; check bits"
        assert (direct - formula).abs().max().item() < 1e-10 * max(1.0, scale)
