"""Asymmetric per-tensor affine quantization (TFLM-style).

Unlike the symmetric Q-format family (`I32FixedPoint` / `I16FixedPoint` /
`I8Symmetric`), the affine path carries a `(scale: float, zero_point: int)`
pair per tensor. Real-value mapping:

    r = (q - zero_point) * scale

For weights (`W_in` / `W_res` / `W_out`) we use **symmetric per-tensor**
quantization (`zero_point = 0`, scale picked from `max(|W|)`), matching
the TFLM kernel convention. For activations (raw input, pre-activation,
state) we allow asymmetric `zero_point` so the storage range is fully
used even when the distribution isn't centered at 0.

This module currently implements the Python reference path. The LLVM
emit + on-device kernel is deferred (Phase 2b).

Public API:
  - `AffineParams`               — per-tensor (scale, zero_point)
  - `AffineQuantConfig`          — per-tensor params for every quantity
  - `calibrate_from_data()`      — observe float traces → derive params
  - `AffineQuantizedModel`       — quantized weights + LUT + precomputed terms
  - `quantize_model_affine()`    — float exe + config → AffineQuantizedModel
  - `AffineQuantizedExecutor`    — bit-exact integer Python reference
  - `search_quantization_affine()` — QAT refit search (iterative)
  - `AffineSearchResult`         — return type of the QAT search
"""
from .types import AffineParams, AffineQuantConfig
from .calibrate import calibrate_from_data
from .quantize import AffineQuantizedModel, quantize_model_affine
from .executor import AffineQuantizedExecutor
from .search import search_quantization_affine, AffineSearchResult
from .multiplier import quantize_multiplier
from .ir_builder import build_ir_from_quantized_affine

__all__ = [
    "AffineParams",
    "AffineQuantConfig",
    "calibrate_from_data",
    "AffineQuantizedModel",
    "quantize_model_affine",
    "AffineQuantizedExecutor",
    "search_quantization_affine",
    "AffineSearchResult",
    "quantize_multiplier",
    "build_ir_from_quantized_affine",
]
