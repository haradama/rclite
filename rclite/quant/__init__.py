"""Quantization-aware tooling for rclite.

  - `QuantConfig`         — Q-format configuration (frac bits per quantity)
  - `QuantTarget`         — storage/accumulator type (I32FixedPoint, ...)
  - `TanhLUTSpec`         — LUT geometry; provides table builders
  - `QuantizedModel`      — i32 weights + LUT, ready for execution or export
  - `quantize_model()`    — float trained model → QuantizedModel
  - `QuantizedExecutor`   — bit-exact Python reference
  - `search_quantization()` — QAT search over state_frac
"""
from .config import QuantConfig
from .target import QuantTarget, I32FixedPoint, I16FixedPoint, I8Affine
from .tanh_lut import TanhLUTSpec
from .model import QuantizedModel
from .quantize import quantize_model, quantize_W_out
from .executor import QuantizedExecutor
from .search import search_quantization, derive_frac_bits, SearchResult
from .ir_builder import build_ir_from_quantized
from .online import IntegerLMSLearner

__all__ = [
    "QuantConfig",
    "QuantTarget", "I32FixedPoint", "I16FixedPoint", "I8Affine",
    "TanhLUTSpec",
    "QuantizedModel",
    "quantize_model", "quantize_W_out",
    "QuantizedExecutor",
    "search_quantization", "derive_frac_bits", "SearchResult",
    "build_ir_from_quantized",
    "IntegerLMSLearner",
]
