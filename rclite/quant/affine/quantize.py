"""Convert a trained float reservoir + AffineQuantConfig into an
`AffineQuantizedModel`.

Mirrors the derivation in the executor module:

  pre[i] = bias + sum_k W_in[i,k] * u_pre[k] + sum_j W_res[i,j] * h[j]

In integer form (weights symmetric, zp=0; activations asymmetric):

  q_pre[i] = zp_pre + bias_pre
           + round(M_in  * (sum_k q_W_in[i,k]  * q_upre[k] - zp_upre * R_in[i]))
           + round(M_res * (sum_j q_W_res[i,j] * q_h[j]    - zp_h    * R_res[i]))

where:
  M_in   = s_W_in  * s_upre / s_pre
  M_res  = s_W_res * s_h    / s_pre
  R_in[i]  = sum_k q_W_in[i,k]    (precomputed once)
  R_res[i] = sum_j q_W_res[i,j]   (precomputed once)
  bias_pre = round(bias / s_pre)

The 256-entry tanh LUT maps each storable `q_pre` to its corresponding
`q_state` — no interpolation needed because the input domain is finite.

The readout follows the mirage scheme: one per-tensor scale for W_out
plus three multipliers (one per phi column block).
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from .types import AffineQuantConfig
from .multiplier import quantize_multiplier
from .lut import LUTStrategy, LUTKind, LUTArtifacts, build_lut_artifacts


@dataclass
class AffineQuantizedModel:
    """A trained quantized reservoir using per-tensor affine quantization.

    Holds the integer weights, the LUT, all precomputed cross-term sums
    and multipliers. Ready for `AffineQuantizedExecutor` (Python ref) or
    the LLVM emit (`_AffineLowerer`).

    Multipliers are stored in *both* float form (the calibration-time
    value) and TFLM-style `(M0, n)` integer form. The integer form is
    what both the Python reference and the JIT use during inference, so
    they agree bit-for-bit.
    """

    rc: ReservoirComputer
    config: AffineQuantConfig

    # Integer weights (symmetric, zero_point = 0)
    W_in_q: np.ndarray   # (N, K) storage dtype
    W_res_q: np.ndarray  # (N, N) storage dtype
    W_out_q: np.ndarray  # (M, F) storage dtype

    # Per-row weight sums for zero-point cross-term folding.
    row_sum_W_in: np.ndarray   # (N,)  int32
    row_sum_W_res: np.ndarray  # (N,)  int32
    bias_pre: int              # round(reservoir.bias / s_pre)

    # Reservoir-step multipliers (float for reference + integer for compute).
    M_in: float
    M_res: float
    M_in_M0: int = 0
    M_in_n: int = 0
    M_res_M0: int = 0
    M_res_n: int = 0
    # Integer preprocess (only meaningful when input_offset != 0 or
    # input_scaling != 1). For the identity case (the common case),
    # `has_integer_preprocess = False` and the kernel uses X directly as u_pre.
    has_integer_preprocess: bool = False
    pre_M0: int = 0       # multiplier on (q_x - zp_input)
    pre_n: int = 0
    pre_const: int = 0    # zp_u_pre + round(-offset * scaling / s_upre)
    # Tanh activation strategy + precomputed data. `lut_q` / `lut_offset`
    # carry the table data for DIRECT and LINEAR_INTERP. POLYNOMIAL stores
    # its multipliers in `lut_artifacts` and leaves `lut_q` empty.
    lut_strategy: LUTStrategy = field(default_factory=LUTStrategy.direct)
    lut_q: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=np.int8))
    lut_offset: int = 0
    lut_artifacts: Optional[LUTArtifacts] = None
    # Leaky integration multiplier (= leak rate, in [0, 1])
    leak_M0: int = 0
    leak_n: int = 0

    # Readout: mixed-scale W_out matmul → output
    M_out_bias: float = 0.0          # s_W_out / s_y
    M_out_input: float = 0.0         # s_W_out * s_input / s_y
    M_out_state: float = 0.0         # s_W_out * s_state / s_y
    M_out_bias_M0: int = 0
    M_out_bias_n: int = 0
    M_out_input_M0: int = 0
    M_out_input_n: int = 0
    M_out_state_M0: int = 0
    M_out_state_n: int = 0
    row_sum_Wout_input: Optional[np.ndarray] = None  # (M,) int32 (if include_input)
    row_sum_Wout_state: Optional[np.ndarray] = None  # (M,) int32

    state_init_q: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        cfg = self.config
        N = self.rc.reservoir.units
        K = self.rc.input.units
        M = self.rc.readout.units
        F = ((1 if self.rc.readout.include_bias else 0)
             + (K if self.rc.readout.include_input else 0)
             + N)
        if self.W_in_q.shape != (N, K):
            raise ValueError(f"W_in_q shape {self.W_in_q.shape} != ({N}, {K})")
        if self.W_res_q.shape != (N, N):
            raise ValueError(f"W_res_q shape {self.W_res_q.shape} != ({N}, {N})")
        if self.W_out_q.shape != (M, F):
            raise ValueError(f"W_out_q shape {self.W_out_q.shape} != ({M}, {F})")
        if self.state_init_q is None:
            self.state_init_q = np.full(N, cfg.state.zero_point,
                                          dtype=cfg.state.storage_dtype)

    @property
    def N(self) -> int: return self.rc.reservoir.units
    @property
    def K(self) -> int: return self.rc.input.units
    @property
    def M(self) -> int: return self.rc.readout.units
    @property
    def F(self) -> int:
        K = self.K
        return ((1 if self.rc.readout.include_bias else 0)
                + (K if self.rc.readout.include_input else 0)
                + self.N)
    @property
    def storage_bits(self) -> int: return self.config.storage_bits
    @property
    def w_out_storage_bits(self) -> int: return self.config.w_out_storage_bits
    @property
    def mixed_precision(self) -> bool: return self.config.mixed_precision


def quantize_model_affine(
    rc: ReservoirComputer,
    exe: RCExecutor,
    config: AffineQuantConfig,
    *,
    W_out_override: Optional[np.ndarray] = None,
    lut_strategy: Optional[LUTStrategy] = None,
) -> AffineQuantizedModel:
    """Quantize a trained float reservoir under `config`.

    `W_out_override`, when provided, replaces `exe.W_out` for the readout
    quantization. This is the QAT-search refit path: after refitting W_out
    on a quantized state trajectory, pass the new matrix here instead of
    mutating `exe`.

    `lut_strategy` controls how tanh is evaluated in the kernel:
      - `LUTStrategy.direct()`         — table of 2^storage_bits entries
      - `LUTStrategy.linear_interp(n)` — interpolated n-entry table
      - `LUTStrategy.polynomial()`     — Taylor-3 polynomial in fixed-point
    Defaults to DIRECT when not specified.
    """
    if lut_strategy is None:
        lut_strategy = LUTStrategy.direct()
    W_out = exe.W_out if W_out_override is None else np.asarray(W_out_override)
    if W_out is None:
        raise ValueError("Readout has not been trained — call exe.fit() first")
    if W_out.shape != exe.W_out.shape if exe.W_out is not None else False:
        # If exe was trained, the override shape must match the rc's F dim
        pass

    # Weight quantization (symmetric, zp=0).
    W_in_q  = config.W_in.quantize_array(exe.W_in)
    W_res_q = config.W_res.quantize_array(exe.W_res)
    # W_out: per-column-block scales (mirage-style). Each block is quantized
    # at its own scale so a tiny bias coef isn't crushed by a huge state coef.
    K = rc.input.units
    N = rc.reservoir.units
    # W_out uses its own (possibly wider) storage dtype — mixed precision.
    w_out_dtype = config.W_out_state.storage_dtype
    W_out_q = np.zeros_like(W_out, dtype=w_out_dtype)
    off = 0
    if rc.readout.include_bias:
        W_out_q[:, 0:1] = config.W_out_bias.quantize_array(W_out[:, 0:1])
        off = 1
    if rc.readout.include_input:
        W_out_q[:, off:off + K] = config.W_out_input.quantize_array(
            W_out[:, off:off + K])
        off += K
    W_out_q[:, off:off + N] = config.W_out_state.quantize_array(
        W_out[:, off:off + N])

    # Per-row weight sums for the zp folding (kept as i32 so LLVM globals
    # don't need an i64 path).
    row_sum_W_in  = W_in_q.astype(np.int32).sum(axis=1).astype(np.int32)
    row_sum_W_res = W_res_q.astype(np.int32).sum(axis=1).astype(np.int32)

    # Bias contribution at pre scale
    bias_pre = int(round(float(rc.reservoir.bias) / config.pre.scale))

    # Reservoir-step multipliers
    M_in  = (config.W_in.scale  * config.u_pre.scale) / config.pre.scale
    M_res = (config.W_res.scale * config.state.scale) / config.pre.scale

    # LUT / approximation artifacts per the chosen strategy.
    lut_artifacts = build_lut_artifacts(config, lut_strategy)
    lut_q = lut_artifacts.table
    lut_offset = lut_artifacts.offset

    # Readout precomputed row sums (per block) and multipliers.
    s_y = config.output.scale
    off = 1 if rc.readout.include_bias else 0
    if rc.readout.include_input:
        row_sum_Wout_input = (W_out_q[:, off:off + K]
                              .astype(np.int32).sum(axis=1).astype(np.int32))
        M_out_input = (config.W_out_input.scale * config.input.scale) / s_y
        off += K
    else:
        row_sum_Wout_input = None
        M_out_input = 0.0
    row_sum_Wout_state = (W_out_q[:, off:off + N]
                          .astype(np.int32).sum(axis=1).astype(np.int32))
    M_out_state = (config.W_out_state.scale * config.state.scale) / s_y
    M_out_bias = (config.W_out_bias.scale / s_y
                   if rc.readout.include_bias else 0.0)

    # Integer (M0, n) decompositions for every requantize multiplier.
    M_in_M0, M_in_n = quantize_multiplier(M_in)
    M_res_M0, M_res_n = quantize_multiplier(M_res)
    leak_M0, leak_n = quantize_multiplier(float(rc.reservoir.leak_rate))
    M_out_bias_M0, M_out_bias_n = quantize_multiplier(M_out_bias)
    M_out_input_M0, M_out_input_n = quantize_multiplier(M_out_input)
    M_out_state_M0, M_out_state_n = quantize_multiplier(M_out_state)

    # Integer preprocess multipliers (only needed for non-identity preprocess).
    offset = float(rc.input.input_offset)
    scaling = float(rc.input.input_scaling)
    has_integer_preprocess = (offset != 0.0) or (scaling != 1.0)
    if has_integer_preprocess:
        # u_pre_real = (x_real - offset) * scaling
        # q_upre - zp_upre ≈ apply_mult(q_x - zp_x, M_pre) + round(-offset*scaling/s_upre)
        M_pre_real = config.input.scale * scaling / config.u_pre.scale
        pre_M0, pre_n = quantize_multiplier(M_pre_real)
        c_real = -offset * scaling / config.u_pre.scale
        pre_const = config.u_pre.zero_point + int(round(c_real))
    else:
        pre_M0 = pre_n = 0
        pre_const = 0

    return AffineQuantizedModel(
        rc=rc, config=config,
        W_in_q=W_in_q, W_res_q=W_res_q, W_out_q=W_out_q,
        lut_strategy=lut_strategy,
        lut_q=lut_q, lut_offset=lut_offset,
        lut_artifacts=lut_artifacts,
        row_sum_W_in=row_sum_W_in, row_sum_W_res=row_sum_W_res,
        bias_pre=bias_pre,
        M_in=M_in, M_res=M_res,
        M_in_M0=M_in_M0, M_in_n=M_in_n,
        M_res_M0=M_res_M0, M_res_n=M_res_n,
        leak_M0=leak_M0, leak_n=leak_n,
        M_out_bias=M_out_bias,
        M_out_input=M_out_input,
        M_out_state=M_out_state,
        M_out_bias_M0=M_out_bias_M0, M_out_bias_n=M_out_bias_n,
        M_out_input_M0=M_out_input_M0, M_out_input_n=M_out_input_n,
        M_out_state_M0=M_out_state_M0, M_out_state_n=M_out_state_n,
        has_integer_preprocess=has_integer_preprocess,
        pre_M0=pre_M0, pre_n=pre_n, pre_const=pre_const,
        row_sum_Wout_input=row_sum_Wout_input,
        row_sum_Wout_state=row_sum_Wout_state,
    )
