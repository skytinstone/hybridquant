"""Cache sizes (paper eq. 1) and bit budgets (eq. 5) from a text config.

Field names follow the installed transformers ``Qwen3_5TextConfig``:
layer_types, num_key_value_heads, head_dim, linear_num_value_heads,
linear_key_head_dim, linear_value_head_dim. The recurrent state tensor is
``[B, linear_num_value_heads, d_k, d_v]``; the small conv1d state is ignored.
"""
from __future__ import annotations

__all__ = ["cache_sizes", "total_bits", "total_bytes"]


def cache_sizes(text_config) -> dict:
    lt = list(text_config.layer_types)
    sizes = {
        "L_a": lt.count("full_attention"),
        "L_g": lt.count("linear_attention"),
        "H_kv": int(text_config.num_key_value_heads),
        "d": int(text_config.head_dim),
        "H": int(text_config.linear_num_value_heads),
        "d_k": int(text_config.linear_key_head_dim),
        "d_v": int(text_config.linear_value_head_dim),
    }
    sizes["kv_per_token"] = 2 * sizes["L_a"] * sizes["H_kv"] * sizes["d"]          # K and V elements per token
    sizes["state_elems"] = sizes["L_g"] * sizes["H"] * sizes["d_k"] * sizes["d_v"]  # fixed
    return sizes


def total_bits(sizes: dict, kv_bits: int, state_bits: int, T: int) -> int:
    """b_kv * |KV|(T) + b_s * |S| in bits (eq. 5 left-hand side)."""
    return kv_bits * sizes["kv_per_token"] * T + state_bits * sizes["state_elems"]


def total_bytes(sizes: dict, kv_bits: int, state_bits: int, T: int) -> float:
    return total_bits(sizes, kv_bits, state_bits, T) / 8
