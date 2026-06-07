"""Fixed-point softmax via an exp lookup table — shared by both quant families.

Classification probabilities on an MCU: `softmax(z)_m = exp(z_m - max) /
sum_j exp(z_j - max)`. The max-subtraction keeps every exponent <= 0, so a
single exp LUT over a clamped negative range `[d_min, 0]` (with linear
interpolation, like the tanh LUT) covers the whole domain.

The kernel works directly on the *quantized* readout scores `logits_q`,
needing only one scalar to relate the integer score difference to the float
difference fed to exp:

    float(z_m - z_j) = (logits_q[m] - logits_q[j]) * S_diff

where `S_diff` is one quantized-score-difference unit in float:
  - symmetric Q-format : S_diff = 1 / state_scale   (scores live at state scale)
  - affine per-tensor  : S_diff = output_scale       (zero point cancels in the diff)

Probabilities are emitted at `Q.prob_frac` in the storage type, where
`prob_frac = min(storage_bits - 1, 15)` (Q15 for i16/i32, Q7 for i8). The
exact integer algorithm in `softmax_q` is mirrored bit-for-bit by the LLVM
lowering and the generated C, and `build_params` precomputes the LUT.
"""

from __future__ import annotations
from dataclasses import dataclass

import numpy as np

# Interpolation works at a fixed precision regardless of storage width; the
# lerp arithmetic uses i32/i64 intermediates (the position can exceed i16).
SM_IDX_FRAC = 15


@dataclass(frozen=True)
class SoftmaxLUTSpec:
    """Geometry of the exp LUT used for fixed-point softmax.

    `d_min` is the most-negative score difference the table represents;
    `exp(d_min)` underflows to ~0, so anything below clamps to entry 0.
    `n` is the number of (linearly interpolated) table entries.
    """

    d_min: float = -16.0
    n: int = 256

    def __post_init__(self):
        if self.d_min >= 0:
            raise ValueError(
                f"SoftmaxLUTSpec.d_min must be < 0, got {self.d_min}"
            )
        if self.n < 2:
            raise ValueError(f"SoftmaxLUTSpec.n must be >= 2, got {self.n}")


@dataclass(frozen=True)
class SoftmaxParams:
    """Precomputed integer parameters for one model's softmax head."""

    lut_q: np.ndarray  # (n,) storage dtype, exp samples at Q.prob_frac
    dmin_q: int  # d_min in quantized-score-difference units (< 0)
    n: int
    idx_frac: int
    prob_frac: int  # == out_frac; probabilities live at Q.prob_frac
    storage_bits: int


def _prob_frac(storage_bits: int) -> int:
    return min(storage_bits - 1, 15)


def build_params(
    spec: SoftmaxLUTSpec, s_diff: float, storage_bits: int, storage_dtype
) -> SoftmaxParams:
    """Build the integer softmax parameters for the given score scale.

    `s_diff` is the float value of one quantized-score-difference unit (see
    module docstring). `storage_dtype` is the numpy dtype of the LUT global.
    """
    prob_frac = _prob_frac(storage_bits)
    qmax = (1 << (storage_bits - 1)) - 1
    dmin_q = int(round(spec.d_min / s_diff))
    if dmin_q >= 0:
        raise ValueError(
            f"softmax d_min maps to dmin_q={dmin_q} >= 0 (s_diff={s_diff}); "
            "score scale too coarse for the requested d_min"
        )
    ds = np.linspace(spec.d_min, 0.0, spec.n)
    table = np.round(np.exp(ds) * (1 << prob_frac)).astype(np.int64)
    table = np.clip(table, 0, qmax).astype(storage_dtype)
    return SoftmaxParams(
        lut_q=table,
        dmin_q=dmin_q,
        n=spec.n,
        idx_frac=SM_IDX_FRAC,
        prob_frac=prob_frac,
        storage_bits=storage_bits,
    )


def softmax_q(logits_q: np.ndarray, params: SoftmaxParams) -> np.ndarray:
    """Bit-exact integer softmax reference.

    `logits_q` is a length-M integer vector of quantized readout scores.
    Returns length-M integer probabilities at Q.prob_frac (storage range),
    matching the LLVM / C kernels exactly.
    """
    lut = params.lut_q.astype(np.int64)
    n = params.n
    idx_frac = params.idx_frac
    prob_frac = params.prob_frac
    dmin_q = params.dmin_q
    qmax = (1 << (params.storage_bits - 1)) - 1

    z = logits_q.astype(np.int64)
    max_q = int(z.max())
    e = np.zeros(len(z), dtype=np.int64)
    for m in range(len(z)):
        d = int(z[m]) - max_q
        if d < dmin_q:
            d = dmin_q
        num = d - dmin_q  # in [0, -dmin_q]
        pos = (num * (n - 1) << idx_frac) // (-dmin_q)
        i0 = pos >> idx_frac
        if i0 < 0:
            i0 = 0
        if i0 > n - 2:
            i0 = n - 2
        frac = pos - (i0 << idx_frac)
        y0 = int(lut[i0])
        y1 = int(lut[i0 + 1])
        e[m] = y0 + (((y1 - y0) * frac) >> idx_frac)
    s = int(e.sum())
    out = np.zeros(len(z), dtype=np.int64)
    for m in range(len(z)):
        p = (int(e[m]) << prob_frac) // s
        if p > qmax:
            p = qmax
        out[m] = p
    return out
