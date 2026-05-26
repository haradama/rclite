"""Quantization target — encodes the storage type, accumulator type, and the
fixed-point arithmetic rules used by a particular hardware target.

`I32FixedPoint` is the only concrete target shipped today (mirage-style:
i32 storage, i64 accumulator, Q-format with per-quantity fractional bits).
Adding `I16FixedPoint` or `I8Affine` is an exercise in subclassing this.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from .config import QuantConfig


class QuantTarget(ABC):
    """Abstract quantization target."""

    name: ClassVar[str]
    storage_bits: ClassVar[int]      # bit width of the stored values (e.g. 32)
    accum_bits: ClassVar[int]        # bit width of the multiply-accumulate register
    signed: ClassVar[bool] = True

    @property
    def storage_dtype(self) -> np.dtype:
        if not self.signed:
            return np.dtype(f"uint{self.storage_bits}")
        return np.dtype(f"int{self.storage_bits}")

    @property
    def accum_dtype(self) -> np.dtype:
        if not self.signed:
            return np.dtype(f"uint{self.accum_bits}")
        return np.dtype(f"int{self.accum_bits}")

    @property
    def llvm_storage_type(self) -> str:
        return f"i{self.storage_bits}"

    @property
    def llvm_accum_type(self) -> str:
        return f"i{self.accum_bits}"

    def quantize_state(self, x: float, cfg: QuantConfig) -> int:
        return self._saturate(int(x * cfg.state_scale))

    def quantize_input(self, x: float, cfg: QuantConfig) -> int:
        return self._saturate(int(x * cfg.input_scale))

    def quantize_weight(self, w: float, cfg: QuantConfig) -> int:
        return self._saturate(int(w * cfg.weight_scale))

    def quantize_state_array(self, arr: np.ndarray, cfg: QuantConfig) -> np.ndarray:
        return self._saturate_array(np.asarray(arr) * cfg.state_scale)

    def quantize_input_array(self, arr: np.ndarray, cfg: QuantConfig) -> np.ndarray:
        return self._saturate_array(np.asarray(arr) * cfg.input_scale)

    def quantize_weight_array(self, arr: np.ndarray, cfg: QuantConfig) -> np.ndarray:
        return self._saturate_array(np.asarray(arr) * cfg.weight_scale)

    def dequantize_state(self, q: int, cfg: QuantConfig) -> float:
        return q / cfg.state_scale

    def dequantize_state_array(self, q: np.ndarray, cfg: QuantConfig) -> np.ndarray:
        return q.astype(np.float64) / cfg.state_scale

    def _saturate(self, x: int) -> int:
        lo, hi = self._range()
        return int(max(lo, min(hi, x)))

    def _saturate_array(self, arr: np.ndarray) -> np.ndarray:
        lo, hi = self._range()
        return np.clip(np.rint(arr), lo, hi).astype(self.storage_dtype)

    def _range(self) -> tuple[int, int]:
        if not self.signed:
            return 0, (1 << self.storage_bits) - 1
        b = self.storage_bits
        return -(1 << (b - 1)), (1 << (b - 1)) - 1


@dataclass(frozen=True)
class I32FixedPoint(QuantTarget):
    """i32 storage, i64 accumulator. mirage-compatible.

    Multiplication shifts:
      state*weight       → shift by weight_frac          (recurrent path)
      input*weight       → shift by weight_frac+input_frac-state_frac  (input path)
      LUT interpolation  → shift by state_frac
    """
    name: ClassVar[str] = "i32"
    storage_bits: ClassVar[int] = 32
    accum_bits: ClassVar[int] = 64


@dataclass(frozen=True)
class I16FixedPoint(QuantTarget):
    """i16 storage, i32 accumulator. Smaller binary, more saturation risk."""
    name: ClassVar[str] = "i16"
    storage_bits: ClassVar[int] = 16
    accum_bits: ClassVar[int] = 32


@dataclass(frozen=True)
class I8Affine(QuantTarget):
    """TFLM-style affine i8 quantization (skeleton — no LLVM emit yet).

    Real-value mapping per tensor::

        r = (q - zero_point) * scale

    Unlike the symmetric `IxFixedPoint` family (which uses pure Q-format
    scale powers of 2 with zero_point implicitly = 0), the affine target
    carries a per-tensor `(scale: float, zero_point: int)` pair. This is
    what TFLite Micro uses for i8 deployment and what gives the tightest
    range packing.

    Implementation roadmap:

      1. Replace `QuantConfig.{state,input,weight}_frac` with per-tensor
         `AffineParams(scale: float, zero_point: int, scale_M0: int,
         scale_n: int)`, where the multiplier `scale_a * scale_b / scale_c`
         is represented as `M0 * 2^-n` for int32 fixed-point use.
      2. Update `quantize_weights` to produce i8 outputs with the right
         zero_point shifts in the cross terms of `(q_a - z_a)(q_w - z_w)`.
      3. Add an `_I8AffineLowerer` that emits the requantize-and-multiply
         pattern (TFLM kernel-style) instead of pure Q-format mul-shift.

    For now this class exists so user code can register the target name
    and the framework's abstraction surfaces the future i8 path.
    Constructing `quantize_model(..., target=I8Affine())` raises with a
    pointer to this docstring.
    """
    name: ClassVar[str] = "i8-affine"
    storage_bits: ClassVar[int] = 8
    accum_bits: ClassVar[int] = 32

    def quantize_state(self, x, cfg):  # type: ignore[override]
        raise NotImplementedError(
            "I8Affine target is a skeleton — full LLVM emit is not "
            "implemented. See rclite/quant/target.py docstring for the "
            "implementation roadmap. Use I32FixedPoint or I16FixedPoint."
        )

    quantize_input = quantize_state
    quantize_weight = quantize_state
    quantize_state_array = quantize_state
    quantize_input_array = quantize_state
    quantize_weight_array = quantize_state
