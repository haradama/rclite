"""Quantization-aware tooling for rclite.

Symmetric Q-format family (`I32FixedPoint` / `I16FixedPoint` / `I8Symmetric`):
  - `QuantConfig`         — Q-format configuration (frac bits per quantity)
  - `QuantTarget`         — storage/accumulator type
  - `TanhLUTSpec`         — LUT geometry; provides table builders
  - `QuantizedModel`      — integer weights + LUT, ready for execution or export
  - `quantize_model()`    — float trained model → QuantizedModel
  - `QuantizedExecutor`   — bit-exact Python reference
  - `search_quantization()` — QAT search over state_frac

Asymmetric per-tensor affine family (TFLM-style, `rclite.quant.affine`):
  - `AffineParams` / `AffineQuantConfig`  — per-tensor (scale, zero_point)
  - `calibrate_from_data()`               — float traces → config
  - `AffineQuantizedModel`                — quantized weights + LUT + precomputed
  - `quantize_model_affine()`             — float exe + config → model
  - `AffineQuantizedExecutor`             — integer Python reference
"""
from .config import QuantConfig
from .target import (
    QuantTarget, I32FixedPoint, I16FixedPoint, I8Symmetric, I8Affine,
)
from .tanh_lut import TanhLUTSpec
from .model import QuantizedModel
from .quantize import quantize_model, quantize_W_out
from .executor import QuantizedExecutor
from .search import search_quantization, derive_frac_bits, SearchResult
from .ir_builder import build_ir_from_quantized
from .online import IntegerLMSLearner
from .affine import (
    AffineParams, AffineQuantConfig,
    calibrate_from_data,
    LUTStrategy, LUTKind, LUTArtifacts, build_lut_artifacts,
    AffineQuantizedModel, quantize_model_affine,
    AffineQuantizedExecutor,
    search_quantization_affine, AffineSearchResult,
    quantize_multiplier,
    build_ir_from_quantized_affine,
)

__all__ = [
    "QuantConfig",
    "QuantTarget", "I32FixedPoint", "I16FixedPoint", "I8Symmetric", "I8Affine",
    "TanhLUTSpec",
    "QuantizedModel",
    "quantize_model", "quantize_W_out",
    "QuantizedExecutor",
    "search_quantization", "derive_frac_bits", "SearchResult",
    "build_ir_from_quantized",
    "IntegerLMSLearner",
    # Affine
    "AffineParams", "AffineQuantConfig",
    "calibrate_from_data",
    "AffineQuantizedModel", "quantize_model_affine",
    "AffineQuantizedExecutor",
    "search_quantization_affine", "AffineSearchResult",
    "quantize_multiplier", "build_ir_from_quantized_affine",
    "LUTStrategy", "LUTKind", "LUTArtifacts", "build_lut_artifacts",
]
