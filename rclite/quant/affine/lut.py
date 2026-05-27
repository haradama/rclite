"""Tanh activation strategy for the affine quantized kernel.

The activation step takes `q_pre` (storage_ty) and produces `q_act` at
`(s_state, zp_state)`. Three strategies are supported:

  * `DIRECT`         — table indexed by every storable `q_pre`. Smallest
                       per-step cost (one load), but table size grows with
                       2^storage_bits: 256 B at i8, 128 KB at i16.
  * `LINEAR_INTERP`  — uniformly subsampled table with linear interpolation
                       between adjacent entries. Configurable `n_entries`
                       (default 256). Trades two loads + a multiply for a
                       dramatic table-size reduction. The natural choice
                       for i16 MCU deploy.
  * `POLYNOMIAL`     — no table; tanh evaluated as a low-degree polynomial
                       in fixed-point arithmetic. Smallest .rodata
                       footprint, but accuracy bounded by the polynomial
                       (default: Taylor degree-3 in Q.16, clipped to ±1).

`LUTStrategy` carries the choice; `build_*_lut()` build the per-strategy
data the model needs (table values, normalised multipliers, etc.).
"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

import numpy as np

from .types import AffineQuantConfig
from .multiplier import quantize_multiplier


class LUTKind(Enum):
    DIRECT = "direct"
    LINEAR_INTERP = "linear_interp"
    POLYNOMIAL = "polynomial"


@dataclass(frozen=True)
class LUTStrategy:
    """Tanh approximation choice for the quantized kernel."""

    kind: LUTKind = LUTKind.DIRECT
    # For LINEAR_INTERP only:
    n_entries: int = 256
    interp_frac_bits: int = 8
    # For POLYNOMIAL only:
    poly_qf_bits: int = 16   # Q-format fractional bits for intermediate compute
    poly_clip: float = 2.0   # clip |x| ≤ poly_clip before evaluating

    def __post_init__(self):
        if self.kind == LUTKind.LINEAR_INTERP:
            if self.n_entries < 4:
                raise ValueError(
                    f"LINEAR_INTERP needs n_entries >= 4, got {self.n_entries}"
                )
            if not (0 < self.interp_frac_bits < 16):
                raise ValueError(
                    f"interp_frac_bits must be in (0, 16), got {self.interp_frac_bits}"
                )
        if self.kind == LUTKind.POLYNOMIAL:
            if not (8 <= self.poly_qf_bits <= 24):
                raise ValueError(
                    f"poly_qf_bits must be in [8, 24], got {self.poly_qf_bits}"
                )
            if self.poly_clip <= 0:
                raise ValueError(f"poly_clip must be > 0, got {self.poly_clip}")

    @classmethod
    def direct(cls) -> "LUTStrategy":
        return cls(kind=LUTKind.DIRECT)

    @classmethod
    def linear_interp(cls, n_entries: int = 256,
                       interp_frac_bits: int = 8) -> "LUTStrategy":
        return cls(kind=LUTKind.LINEAR_INTERP,
                    n_entries=n_entries, interp_frac_bits=interp_frac_bits)

    @classmethod
    def polynomial(cls, poly_qf_bits: int = 16,
                    poly_clip: float = 2.0) -> "LUTStrategy":
        return cls(kind=LUTKind.POLYNOMIAL,
                    poly_qf_bits=poly_qf_bits, poly_clip=poly_clip)


# ---------------------------------------------------------------------------
# Per-strategy precomputation


@dataclass(frozen=True)
class LUTArtifacts:
    """Whatever the chosen strategy needs at kernel time.

    For DIRECT:
        table   : (2^storage_bits,) storage dtype
        offset  : -qmin (lookup index = q_pre + offset)

    For LINEAR_INTERP:
        table   : (n_entries,) storage dtype, uniformly subsampled over
                  the full q_pre range
        offset  : -qmin
        idx_M0, idx_n : multiplier converting (q_pre - qmin) into a
                  fixed-point position in Q.interp_frac_bits

    For POLYNOMIAL:
        table   : empty (np.zeros(0, dtype=storage))
        offset  : 0
        x_to_qf_M0, x_to_qf_n   : multiplier converting (q_pre - zp_pre)
                                   to x in Q.poly_qf_bits
        qf_to_state_M0, qf_to_state_n :
                                   multiplier converting Q.poly_qf_bits
                                   tanh value back to state scale (mantissa
                                   only — caller adds zp_state)
        x_clip_qf : |x| clamp threshold, in Q.poly_qf_bits
        one_qf    : value of 1.0 in Q.poly_qf_bits (for clamp ceiling)
    """
    table: np.ndarray
    offset: int = 0
    # LINEAR_INTERP
    idx_M0: int = 0
    idx_n: int = 0
    # POLYNOMIAL
    x_to_qf_M0: int = 0
    x_to_qf_n: int = 0
    qf_to_state_M0: int = 0
    qf_to_state_n: int = 0
    x_clip_qf: int = 0
    one_qf: int = 0


def build_lut_artifacts(config: AffineQuantConfig,
                          strategy: LUTStrategy) -> LUTArtifacts:
    """Dispatch to the right per-strategy builder."""
    if strategy.kind == LUTKind.DIRECT:
        return _build_direct(config)
    if strategy.kind == LUTKind.LINEAR_INTERP:
        return _build_linear_interp(config, strategy)
    if strategy.kind == LUTKind.POLYNOMIAL:
        return _build_polynomial(config, strategy)
    raise ValueError(f"unknown LUTKind: {strategy.kind}")


def _storage_range(sb: int) -> tuple[int, int]:
    return -(1 << (sb - 1)), (1 << (sb - 1)) - 1


def _build_direct(config: AffineQuantConfig) -> LUTArtifacts:
    sb = config.storage_bits
    qmin, qmax = _storage_range(sb)
    q_pres = np.arange(qmin, qmax + 1, dtype=np.int64)
    reals = config.pre.dequantize_array(q_pres)
    state_qs = config.state.quantize_array(np.tanh(reals))
    return LUTArtifacts(table=state_qs, offset=-qmin)


def _build_linear_interp(config: AffineQuantConfig,
                           strategy: LUTStrategy) -> LUTArtifacts:
    sb = config.storage_bits
    qmin, qmax = _storage_range(sb)
    n = strategy.n_entries
    # Sample n_entries uniformly over [qmin, qmax]; each entry stores the
    # quantized tanh at that sample point.
    sample_q_pres = np.linspace(qmin, qmax, n, dtype=np.float64)
    reals = (sample_q_pres - config.pre.zero_point) * config.pre.scale
    state_qs = config.state.quantize_array(np.tanh(reals))

    # We compute t = (q_pre - qmin) * (n-1) / (qmax - qmin) at Q.interp_frac_bits
    # precision. Express the multiplier (n-1)/(qmax-qmin) * 2^interp_frac_bits
    # via the (M0, n) decomposition so Python + JIT use the same integer math.
    f = strategy.interp_frac_bits
    M_real = (n - 1) / float(qmax - qmin) * float(1 << f)
    M0, n_shift = quantize_multiplier(M_real)
    return LUTArtifacts(
        table=state_qs, offset=-qmin,
        idx_M0=M0, idx_n=n_shift,
    )


def _build_polynomial(config: AffineQuantConfig,
                        strategy: LUTStrategy) -> LUTArtifacts:
    sb = config.storage_bits
    qf = strategy.poly_qf_bits
    # 1) Multiplier that turns (q_pre - zp_pre) into x in Q.qf:
    #    x_qf ≈ (q_pre - zp_pre) * s_pre * 2^qf
    M_x_real = config.pre.scale * float(1 << qf)
    M_x_M0, M_x_n = quantize_multiplier(M_x_real)

    # 2) Multiplier that turns Q.qf tanh value into Δq_state (then caller
    #    adds zp_state):  q_state - zp_state = round(tanh_qf / (2^qf * s_state))
    M_back_real = 1.0 / (float(1 << qf) * config.state.scale)
    M_back_M0, M_back_n = quantize_multiplier(M_back_real)

    # Polynomial uses Taylor degree-3 (tanh ≈ x - x³/3) clamped to |x| ≤ clip
    # and the result clamped to ±1. Both clamp constants in Q.qf:
    x_clip_qf = int(round(strategy.poly_clip * (1 << qf)))
    one_qf = 1 << qf

    return LUTArtifacts(
        table=np.zeros(0, dtype=config.state.storage_dtype),
        offset=0,
        x_to_qf_M0=M_x_M0, x_to_qf_n=M_x_n,
        qf_to_state_M0=M_back_M0, qf_to_state_n=M_back_n,
        x_clip_qf=x_clip_qf, one_qf=one_qf,
    )
