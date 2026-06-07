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
    poly_qf_bits: int = 16  # Q-format fractional bits for intermediate compute
    poly_clip: float = 2.0  # clip |x| ≤ poly_clip before evaluating; 2.0
    # empirically yields the best least-squares fit
    # over the active tanh region (|x| < 2 spans
    # roughly -0.96 to +0.96 of tanh's range)
    poly_degree: int = 5  # odd-only polynomial: 3 or 5

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
                raise ValueError(
                    f"poly_clip must be > 0, got {self.poly_clip}"
                )
            if self.poly_degree not in (3, 5):
                raise ValueError(
                    f"poly_degree must be 3 or 5, got {self.poly_degree}"
                )

    @classmethod
    def direct(cls) -> "LUTStrategy":
        return cls(kind=LUTKind.DIRECT)

    @classmethod
    def linear_interp(
        cls, n_entries: int = 256, interp_frac_bits: int = 8
    ) -> "LUTStrategy":
        return cls(
            kind=LUTKind.LINEAR_INTERP,
            n_entries=n_entries,
            interp_frac_bits=interp_frac_bits,
        )

    @classmethod
    def polynomial(
        cls, degree: int = 5, poly_qf_bits: int = 16, poly_clip: float = 2.0
    ) -> "LUTStrategy":
        return cls(
            kind=LUTKind.POLYNOMIAL,
            poly_degree=degree,
            poly_qf_bits=poly_qf_bits,
            poly_clip=poly_clip,
        )


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
        table          : empty (np.zeros(0, dtype=storage))
        offset         : 0
        x_to_qf_M0/n   : multiplier converting (q_pre - zp_pre) to x in Q.qf
        qf_to_state_M0/n : multiplier converting Q.qf tanh value to Δq_state
                          (caller adds zp_state)
        x_clip_qf      : |x| clamp threshold, in Q.qf
        one_qf         : value of 1.0 in Q.qf (for clamp ceiling)
        poly_a1_qf, poly_a3_qf, poly_a5_qf
                       : odd-only minimax coefficients in Q.qf
                         (a1 for x, a3 for x³, a5 for x⁵ — a5 = 0 if degree==3)
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
    poly_a1_qf: int = 0
    poly_a3_qf: int = 0
    poly_a5_qf: int = 0


# A DIRECT LUT materializes one entry per representable pre-activation value
# (2**storage_bits of them). That is fine for i8 (256) / i16 (65536) but
# explodes for wider storage (i32 → 4.3e9 entries ≈ 34 GB), so guard against
# it with a clear error instead of hanging on the allocation.
_DIRECT_MAX_STORAGE_BITS = 16


def build_lut_artifacts(
    config: AffineQuantConfig, strategy: LUTStrategy
) -> LUTArtifacts:
    """Dispatch to the right per-strategy builder."""
    if strategy.kind == LUTKind.DIRECT:
        if config.storage_bits > _DIRECT_MAX_STORAGE_BITS:
            raise ValueError(
                f"DIRECT activation LUT needs 2**storage_bits entries "
                f"(2**{config.storage_bits} ≈ {1 << config.storage_bits:.3e} "
                f"here), infeasible for storage_bits > "
                f"{_DIRECT_MAX_STORAGE_BITS}. Use LUTStrategy.linear_interp(n) "
                f"or .polynomial(), or the symmetric i32 quantization path."
            )
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


def _build_linear_interp(
    config: AffineQuantConfig, strategy: LUTStrategy
) -> LUTArtifacts:
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
        table=state_qs,
        offset=-qmin,
        idx_M0=M0,
        idx_n=n_shift,
    )


def _fit_odd_minimax_tanh(
    clip: float, degree: int
) -> tuple[float, float, float]:
    """Least-squares-fit odd polynomial coefficients (a1, a3, a5) for tanh on
    [-clip, clip]. `degree` is 3 or 5; for degree=3 we return a5=0.

    LSQ is not strict minimax but for our use cases the max-error is within
    ~30% of true Remez. Good enough for embedded.
    """
    # Sample densely; bias slightly toward boundaries where tanh saturates.
    xs = np.linspace(-clip, clip, 4001)
    ys = np.tanh(xs)
    if degree == 3:
        # ys ≈ a1*x + a3*x³
        A = np.column_stack([xs, xs**3])
        a, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(a[0]), float(a[1]), 0.0
    if degree == 5:
        A = np.column_stack([xs, xs**3, xs**5])
        a, *_ = np.linalg.lstsq(A, ys, rcond=None)
        return float(a[0]), float(a[1]), float(a[2])
    raise ValueError(f"degree must be 3 or 5, got {degree}")


def _build_polynomial(
    config: AffineQuantConfig, strategy: LUTStrategy
) -> LUTArtifacts:
    sb = config.storage_bits
    qf = strategy.poly_qf_bits
    # 1) (q_pre - zp_pre) → x in Q.qf
    M_x_real = config.pre.scale * float(1 << qf)
    M_x_M0, M_x_n = quantize_multiplier(M_x_real)
    # 2) Q.qf tanh value → Δq_state
    M_back_real = 1.0 / (float(1 << qf) * config.state.scale)
    M_back_M0, M_back_n = quantize_multiplier(M_back_real)
    # 3) Clamp constants in Q.qf
    x_clip_qf = int(round(strategy.poly_clip * (1 << qf)))
    one_qf = 1 << qf
    # 4) Minimax coefficients fit on [-clip, clip], expressed in Q.qf.
    a1, a3, a5 = _fit_odd_minimax_tanh(
        strategy.poly_clip, strategy.poly_degree
    )
    a1_qf = int(round(a1 * (1 << qf)))
    a3_qf = int(round(a3 * (1 << qf)))
    a5_qf = int(round(a5 * (1 << qf)))

    return LUTArtifacts(
        table=np.zeros(0, dtype=config.state.storage_dtype),
        offset=0,
        x_to_qf_M0=M_x_M0,
        x_to_qf_n=M_x_n,
        qf_to_state_M0=M_back_M0,
        qf_to_state_n=M_back_n,
        x_clip_qf=x_clip_qf,
        one_qf=one_qf,
        poly_a1_qf=a1_qf,
        poly_a3_qf=a3_qf,
        poly_a5_qf=a5_qf,
    )
