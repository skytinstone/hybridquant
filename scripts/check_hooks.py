"""Verify that the hybridquant cache hooks fire on a real checkpoint.

Loads the model, runs one short teacher-forced decode under a quantized cache,
and checks QuantStats against config.layer_types. Exit code 0 on PASS, 1 on FAIL.

Example (Colab):
    python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4
    python scripts/check_hooks.py --model Qwen/Qwen3.5-0.8B --kv-bits 2 --state-bits 4 --no-hub-kernels

--no-hub-kernels
    Disables the Hugging Face `kernels` hub integration. Two official switches,
    both read from the installed transformers / kernels source:
    1. env USE_HUB_KERNELS (transformers/integrations/hub_kernels.py): when it is
       not a true value, `use_kernel_forward_from_hub` and friends become identity
       decorators, so `kernelize()` can never replace Qwen3_5GatedDeltaNet.forward.
       It is read at import time, so we set it BEFORE importing transformers.
    2. from_pretrained(use_kernels=False) (transformers/modeling_utils.py): the
       default is already False; kernelize() only runs for use_kernels=True or
       when a kernel_config is passed. We pass it explicitly anyway.
"""
from __future__ import annotations

import argparse
import os
import sys


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="Qwen/Qwen3.5-0.8B")
    p.add_argument("--kv-bits", type=int, default=2)
    p.add_argument("--state-bits", type=int, default=4)
    p.add_argument("--prefill", type=int, default=128)
    p.add_argument("--decode", type=int, default=32)
    p.add_argument("--dtype", default="auto",
                   help="bf16 | fp16 | fp32 | auto (auto: bf16 on Ampere+/MPS, fp16 on T4-class GPUs, fp32 on CPU; "
                        "HQ_DTYPE env overrides)")
    p.add_argument("--no-hub-kernels", action="store_true",
                   help="set USE_HUB_KERNELS=NO before importing transformers and pass use_kernels=False")
    p.add_argument("--text", default="The theory of general relativity describes gravity as a geometric "
                                     "property of spacetime, where mass and energy curve the geometry. ")
    return p.parse_args(argv)


def get_layer_types(config) -> list[str]:
    """layer_types from a text config or from config.text_config (multimodal wrapper)."""
    lt = getattr(config, "layer_types", None)
    if lt is None and getattr(config, "text_config", None) is not None:
        lt = getattr(config.text_config, "layer_types", None)
    if lt is None:
        raise RuntimeError("config has no layer_types; is this a hybrid model?")
    return list(lt)


def replaced_forwards(model) -> list[str]:
    """Module names whose forward was bound on the instance (what kernels.kernelize does)."""
    out = []
    for name, mod in model.named_modules():
        if "forward" in vars(mod):
            out.append(name or type(mod).__name__)
    return out


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.no_hub_kernels:
        if "transformers" in sys.modules:
            print("WARNING: transformers already imported; USE_HUB_KERNELS must be set before import")
        os.environ["USE_HUB_KERNELS"] = "NO"

    import torch  # noqa: E402
    from hybridquant import HybridQuantConfig, QuantStats, check_hook_coverage, load_model, make_cache, quantized_cache  # noqa: E402

    try:
        from transformers.integrations import hub_kernels
        print(f"transformers hub kernels enabled: {hub_kernels._kernels_enabled} "
              f"(USE_HUB_KERNELS={os.environ.get('USE_HUB_KERNELS', '<unset>')})")
    except Exception as e:  # noqa: BLE001
        print(f"could not inspect hub_kernels: {e}")

    load_kwargs = {"use_kernels": False} if args.no_hub_kernels else {}
    if torch.cuda.is_available():
        print(f"cuda device: {torch.cuda.get_device_name(0)} capability={torch.cuda.get_device_capability(0)}")
    model, tok = load_model(args.model, dtype=args.dtype, **load_kwargs)
    print(f"model: {args.model} on {next(model.parameters()).device} dtype={next(model.parameters()).dtype} "
          f"use_kernels={getattr(model, 'use_kernels', None)}")

    layer_types = get_layer_types(model.config)
    print(f"layer_types ({len(layer_types)}): {layer_types}")

    rep = replaced_forwards(model)
    if rep:
        print(f"NOTE: {len(rep)} module(s) have an instance-level forward (kernelized?): {rep[:8]}")

    n_tokens = args.prefill + args.decode
    text = args.text * (n_tokens // 8 + 1)
    ids = tok(text, return_tensors="pt")["input_ids"][:, :n_tokens]
    if ids.shape[1] < n_tokens:
        print(f"WARNING: only {ids.shape[1]} tokens available, wanted {n_tokens}")
    ids = ids.to(next(model.parameters()).device)

    cfg = HybridQuantConfig.uniform(args.kv_bits, args.state_bits)
    stats = QuantStats()
    nll = []
    with torch.no_grad(), quantized_cache(cfg, stats):
        cache = make_cache(model)
        out = model(input_ids=ids[:, :args.prefill], past_key_values=cache, use_cache=True)
        logits = out.logits[:, -1].float()
        for pos in range(args.prefill, ids.shape[1]):
            lp = torch.log_softmax(logits, dim=-1)
            nll.append(-lp[0, ids[0, pos]].item())
            out = model(input_ids=ids[:, pos:pos + 1], past_key_values=cache, use_cache=True)
            logits = out.logits[:, -1].float()
    mean_nll = sum(nll) / max(1, len(nll))
    print(f"config {cfg.label()}: prefill={args.prefill} decode={len(nll)} mean_nll={mean_nll:.4f}")

    print("QuantStats:")
    print(f"  kv_calls={stats.kv_calls} kv_tokens={stats.kv_tokens} layers_seen_kv={sorted(stats.layers_seen_kv)}")
    print(f"  state_calls={stats.state_calls} layers_seen_state={sorted(stats.layers_seen_state)}")

    cov = check_hook_coverage(layer_types, stats)
    print(cov.report())
    if not cov.passed and not args.no_hub_kernels:
        print("hint: rerun with --no-hub-kernels")
    return 0 if cov.passed else 1


if __name__ == "__main__":
    sys.exit(main())
