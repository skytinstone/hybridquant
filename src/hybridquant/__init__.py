"""hybridquant: joint KV-cache + recurrent-state fake quantization for hybrid LLMs."""
from .quant import QuantSpec, fake_quant, quant_error
from .hooks import HybridQuantConfig, QuantStats, quantized_cache, make_cache
from .eval import decode_nll, generate_and_score, load_model, DecodeResult, GenResult

__version__ = "0.0.1"
__all__ = [
    "QuantSpec", "fake_quant", "quant_error",
    "HybridQuantConfig", "QuantStats", "quantized_cache", "make_cache",
    "decode_nll", "generate_and_score", "load_model", "DecodeResult", "GenResult",
]
