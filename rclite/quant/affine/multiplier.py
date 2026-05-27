"""Fixed-point multiplier decomposition for affine requantization.

TFLM-style: an arbitrary positive multiplier `M` is decomposed into
`(M0, n)` such that `M ≈ M0 * 2^-n`, with `M0` a signed 32-bit integer
in the range `[2^30, 2^31)` for maximum precision. The integer
requantization is then:

    result = (x * M0 + (1 << (n-1))) >> n      # arithmetic shift

For x in int32 and M0 in int32, the product fits in int64. The rounding
bias `1 << (n-1)` gives round-half-toward-+∞ semantics (combined with
arithmetic shift's floor behavior on negatives — this is what gemmlowp
calls `SaturatingRoundingDoublingHighMul` minus the doubling, simplified).

Python and LLVM both implement this exactly the same way, so the Python
reference and the JIT kernel agree bit-for-bit on the requantize step.
"""
from __future__ import annotations
from typing import Tuple
import math

import numpy as np


_INT31_MAX = (1 << 31) - 1


def quantize_multiplier(M: float) -> Tuple[int, int]:
    """Decompose `M >= 0` into `(M0, n)` with `M ≈ M0 * 2^-n`.

    `M0` is a signed 32-bit integer in `[2^30, 2^31)` (or 0 when M=0).
    `n` is a non-negative shift amount.

    For M < 0, raises ValueError. Negative multipliers don't arise in our
    use cases (all our M values are products of positive scale ratios).
    """
    if M < 0:
        raise ValueError(f"quantize_multiplier expects M >= 0, got {M}")
    if M == 0:
        return 0, 0
    # Find n such that 2^30 <= M * 2^n < 2^31
    # i.e. 30 <= log2(M) + n < 31 → n = 30 - floor(log2(M))
    log2_M = math.log2(M)
    n = 30 - math.floor(log2_M)
    M0 = int(round(M * (1 << n)))
    # Rounding can push M0 up to 2^31; renormalize.
    if M0 >= (1 << 31):
        M0 //= 2
        n -= 1
    if M0 < (1 << 30) and M > 0:
        # Float-precision edge case at the boundary; bump up
        M0 *= 2
        n += 1
    return int(M0), int(n)


def apply_multiplier_scalar(x: int, M0: int, n: int) -> int:
    """Compute round(x * M0 / 2^n) using integer arithmetic.

    Equivalent to the LLVM IR sequence:
        prod   = sext(x, i64) * sext(M0, i64)
        biased = add prod, (1 << (n-1))   if n > 0 else prod
        shr    = ashr biased, n
    Result is returned as a Python int (caller decides truncation).
    """
    prod = int(x) * int(M0)
    if n > 0:
        prod += (1 << (n - 1))
    # Python's >> on negative ints is arithmetic (floors toward -inf), same as ashr
    return prod >> n


def apply_multiplier_array(x_arr: np.ndarray, M0: int, n: int) -> np.ndarray:
    """Vectorized version of `apply_multiplier_scalar` over a NumPy array.

    Operates on int64 throughout. Returns int64 (caller saturates/narrows).
    """
    M0_64 = np.int64(M0)
    prod = x_arr.astype(np.int64) * M0_64
    if n > 0:
        prod = prod + np.int64(1 << (n - 1))
    # NumPy's >> on signed ints is arithmetic
    return prod >> n
