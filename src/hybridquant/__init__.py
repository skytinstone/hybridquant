"""hybridquant: joint KV-cache + recurrent-state fake quantization for hybrid LLMs."""
from .quant import QuantSpec, fake_quant, quant_error
from .hooks import HybridQuantConfig, QuantStats, quantized_cache, make_cache, HookCoverage, check_hook_coverage
from .eval import decode_nll, generate_and_score, load_model, resolve_dtype, DecodeResult, GenResult
from .meta import run_metadata

__version__ = "0.0.1"
__all__ = [
    "QuantSpec", "fake_quant", "quant_error",
    "HybridQuantConfig", "QuantStats", "quantized_cache", "make_cache", "HookCoverage", "check_hook_coverage",
    "decode_nll", "generate_and_score", "load_model", "resolve_dtype", "DecodeResult", "GenResult",
    "run_metadata",
]
