"""Per-tensor affine quantization params.

`AffineParams` holds a (scale, zero_point) pair for one tensor; storage
width is also recorded so quantize/dequantize knows how to saturate.
`AffineQuantConfig` bundles params for every quantity in a reservoir
computer — weights symmetric, activations possibly asymmetric.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class AffineParams:
    """Per-tensor affine quantization: r = (q - zero_point) * scale."""

    scale: float
    zero_point: int = 0
    storage_bits: int = 8

    def __post_init__(self):
        if not (self.scale > 0):
            raise ValueError(f"scale must be > 0, got {self.scale}")
        if self.storage_bits not in (8, 16, 32):
            raise ValueError(
                f"storage_bits must be 8/16/32, got {self.storage_bits}"
            )
        qmin, qmax = self._range()
        if not (qmin <= self.zero_point <= qmax):
            raise ValueError(
                f"zero_point {self.zero_point} outside [{qmin}, {qmax}] "
                f"for {self.storage_bits}-bit storage"
            )

    def _range(self) -> tuple[int, int]:
        b = self.storage_bits
        return -(1 << (b - 1)), (1 << (b - 1)) - 1

    @property
    def storage_dtype(self) -> np.dtype:
        return np.dtype(f"int{self.storage_bits}")

    def quantize(self, r: float) -> int:
        qmin, qmax = self._range()
        q = int(round(r / self.scale)) + self.zero_point
        return max(qmin, min(qmax, q))

    def quantize_array(self, arr) -> np.ndarray:
        qmin, qmax = self._range()
        q = (np.rint(np.asarray(arr, dtype=np.float64) / self.scale)
             .astype(np.int64)) + self.zero_point
        return np.clip(q, qmin, qmax).astype(self.storage_dtype)

    def dequantize(self, q: int) -> float:
        return (int(q) - self.zero_point) * self.scale

    def dequantize_array(self, q_arr) -> np.ndarray:
        return (np.asarray(q_arr, dtype=np.float64) - self.zero_point) * self.scale

    @classmethod
    def symmetric_absmax(cls, arr, storage_bits: int = 8,
                          eps: float = 1e-8) -> "AffineParams":
        """Pick scale from max |arr|; zero_point=0. TFLM weight convention."""
        m = float(np.max(np.abs(np.asarray(arr, dtype=np.float64))))
        if m < eps:
            m = eps
        max_q = (1 << (storage_bits - 1)) - 1
        return cls(scale=m / max_q, zero_point=0, storage_bits=storage_bits)

    @staticmethod
    def symmetric_absmax_peraxis(arr, storage_bits: int = 8,
                                 eps: float = 1e-8) -> np.ndarray:
        """Per-row symmetric scales: scale[i] = max|arr[i,:]| / qmax.

        Returns a 1-D array of per-row scales (zero_point=0 throughout), the
        per-channel analogue of `symmetric_absmax` along axis 0 (one scale per
        output row). Used for per-channel weight quantization.
        """
        a = np.asarray(arr, dtype=np.float64)
        m = np.abs(a).max(axis=1)
        m = np.where(m < eps, eps, m)
        max_q = (1 << (storage_bits - 1)) - 1
        return (m / max_q).astype(np.float64)

    @classmethod
    def asymmetric_minmax(cls, arr, storage_bits: int = 8,
                           eps: float = 1e-8) -> "AffineParams":
        """Pick scale + zero_point from observed [min, max] range.

        Follows TFLM convention: the representable range always includes
        real value 0 (extending [min, max] to span 0 if necessary). This
        guarantees zero_point is a representable storage value and that
        the dequantized 0 is exactly representable.
        """
        a = np.asarray(arr, dtype=np.float64)
        lo = float(a.min())
        hi = float(a.max())
        # Always include 0 in the representable range
        lo = min(lo, 0.0)
        hi = max(hi, 0.0)
        if hi - lo < eps:
            # Degenerate (all zeros) — fall back to tiny symmetric range
            half = max(eps, abs(lo))
            lo, hi = -half, half
        qmin = -(1 << (storage_bits - 1))
        qmax = (1 << (storage_bits - 1)) - 1
        scale = (hi - lo) / (qmax - qmin)
        zp = int(round(qmin - lo / scale))
        zp = max(qmin, min(qmax, zp))
        return cls(scale=scale, zero_point=zp, storage_bits=storage_bits)


@dataclass(frozen=True)
class AffineQuantConfig:
    """Per-tensor affine params for every quantity in a reservoir computer.

    Convention:
      - Weights (W_in / W_res / W_out_*) : symmetric (zero_point=0)
      - Activations (input, u_pre, pre)  : asymmetric (zero_point may be != 0)
      - `state` is shared between the stored reservoir state h *and* the
        tanh activation output (both bounded by tanh, both ~symmetric in
        practice). One param keeps the leaky integration scale-coherent.
      - `output` is the scale at which the readout y is reported.

    W_out splits into up to three column blocks (bias / input / state)
    each with its OWN symmetric scale. This is the mirage-style mixed
    encoding adapted to affine quant — strict per-tensor for W_out would
    crush small coefficients (bias coefs are O(0.1), state coefs O(100)),
    so we relax just for W_out. Each block stays symmetric (zp=0) and is
    still "per-tensor" in the TFLM sense within its block.
    """

    input: AffineParams                       # raw X
    u_pre: AffineParams                       # preprocessed input
    state: AffineParams                       # reservoir state h AND tanh activated
    pre: AffineParams                         # pre-activation
    W_in: AffineParams                        # symmetric (zp=0)
    W_res: AffineParams                       # symmetric (zp=0)
    W_out_state: AffineParams                 # symmetric (zp=0), state-col block
    output: AffineParams                      # readout y
    # Optional: present only when the readout has those phi components.
    W_out_bias: AffineParams | None = None    # symmetric (zp=0)
    W_out_input: AffineParams | None = None   # symmetric (zp=0)
    # Optional per-channel (per reservoir-row) scales for W_res. When set
    # (length N), W_res is quantized per output row instead of per-tensor
    # and the reservoir-step requantize uses a per-row multiplier. `W_res`
    # above stays a valid representative scalar but is unused on this path.
    W_res_scales: "np.ndarray | None" = None
    # Optional per-channel (per readout output-row) scales for the W_out
    # column blocks. When `W_out_state_scales` is set (length M), each output
    # channel gets its own block scales and the readout requantize uses a
    # per-row multiplier. The block `W_out_*` params stay valid representatives
    # but are unused on this path. bias/input arrays are None if absent.
    W_out_bias_scales: "np.ndarray | None" = None
    W_out_input_scales: "np.ndarray | None" = None
    W_out_state_scales: "np.ndarray | None" = None

    def __post_init__(self):
        w_out_fields = ["W_out_state"]
        if self.W_out_bias is not None:
            w_out_fields.append("W_out_bias")
        if self.W_out_input is not None:
            w_out_fields.append("W_out_input")
        weight_fields = ["W_in", "W_res"] + w_out_fields
        for name in weight_fields:
            wp = getattr(self, name)
            if wp.zero_point != 0:
                raise ValueError(
                    f"{name}.zero_point must be 0 (symmetric weight convention)"
                )
        # Activations + reservoir weights share the base storage width.
        sb = self.input.storage_bits
        base_fields = ["u_pre", "state", "pre", "output", "W_in", "W_res"]
        for name in base_fields:
            other = getattr(self, name)
            if other.storage_bits != sb:
                raise ValueError(
                    f"storage_bits mismatch: input={sb}, {name}={other.storage_bits}"
                )
        # W_out blocks may use a *wider* width (mixed precision, e.g. i16
        # readout weights with an i8 reservoir). They must all agree with
        # each other and be >= the base width.
        wob = self.W_out_state.storage_bits
        for name in w_out_fields:
            other = getattr(self, name)
            if other.storage_bits != wob:
                raise ValueError(
                    f"W_out block storage_bits mismatch: "
                    f"W_out_state={wob}, {name}={other.storage_bits}"
                )
        if wob < sb:
            raise ValueError(
                f"W_out storage_bits ({wob}) must be >= base storage_bits ({sb})"
            )

    @property
    def storage_bits(self) -> int:
        return self.input.storage_bits

    @property
    def w_out_storage_bits(self) -> int:
        return self.W_out_state.storage_bits

    @property
    def mixed_precision(self) -> bool:
        return self.w_out_storage_bits != self.storage_bits
