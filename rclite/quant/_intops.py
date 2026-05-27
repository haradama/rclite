"""Bit-exact integer arithmetic helpers matching the LLVM lowering.

These functions reproduce in numpy what the integer LLVM kernel does at
each fixed-point multiply: i32 -> i64 promote, multiply, arithmetic shift
right, truncate to i32 (two's-complement wrap-around).

Using the same helpers in the QuantizedExecutor and the LLVM emitter
keeps QAT parity exact.
"""
from __future__ import annotations
import numpy as np


_INT32_WRAP = np.uint32(0xFFFFFFFF)


def trunc_i32(x_i64: np.ndarray) -> np.ndarray:
    """Truncate i64 → i32 with two's-complement wrap (mod 2^32)."""
    return (x_i64.astype(np.int64) & _INT32_WRAP).astype(np.uint32).view(np.int32)


def fixed_mul_i32(a: np.ndarray, b: np.ndarray, shift: int) -> np.ndarray:
    """`(a * b) >> shift` with i64 product and i32 truncation."""
    prod = a.astype(np.int64) * b.astype(np.int64)
    return trunc_i32(prod >> shift)


def wrap_to_storage(x: np.ndarray, storage_bits: int) -> np.ndarray:
    """Wrap an integer array to signed `storage_bits` two's-complement, return i32.

    Mirrors what the LLVM kernel does at every `trunc(value, storage_ty)`:
    take the low `storage_bits` bits and re-interpret as a signed value.
    The result is returned as int32 so downstream NumPy arithmetic still
    has headroom (matching how the JIT sign-extends the loaded storage
    value back to the accumulator width).
    """
    if storage_bits >= 32:
        return trunc_i32(x.astype(np.int64))
    mask = (1 << storage_bits) - 1
    sign_bit = 1 << (storage_bits - 1)
    u = x.astype(np.int64) & mask
    return ((u ^ sign_bit) - sign_bit).astype(np.int32)


def fixed_mul_scalar_i32(a: int, b: int, shift: int) -> int:
    prod = (int(a) * int(b)) >> shift
    return _wrap_i32_scalar(prod)


def _wrap_i32_scalar(x: int) -> int:
    x &= 0xFFFFFFFF
    if x >= 0x80000000:
        x -= 0x100000000
    return x


def fixed_div_i32(a: int, b: int, frac_bits: int) -> int:
    """`(a << frac_bits) / b` truncated to i32."""
    return _wrap_i32_scalar((int(a) << frac_bits) // int(b))


def tanh_lut_lookup(
    x_q: np.ndarray,
    lut_q: np.ndarray,
    xmin_q: int,
    xmax_q: int,
    state_frac: int,
) -> np.ndarray:
    """Quantized tanh via linear interpolation, vectorized.

    Mirrors mirage's `tanh_lut_q` exactly: clamp → fixed_div → integer
    index + fractional remainder → lerp between neighboring table entries.
    """
    n = lut_q.shape[0]
    x = np.clip(x_q, xmin_q, xmax_q).astype(np.int64)
    denom = np.int64(xmax_q) - np.int64(xmin_q)
    num = x - np.int64(xmin_q)
    # t_q = (num << state_frac) / denom  (i32)
    t_q = trunc_i32((num << state_frac) // denom)
    # pos_q = t_q * (n - 1)  (i32 wrap)
    pos_q = trunc_i32(t_q.astype(np.int64) * np.int64(n - 1))
    # i0 = pos_q >> state_frac, clamped to [0, n-2]
    i0 = np.clip((pos_q >> state_frac).astype(np.int64), 0, n - 2)
    frac_q = pos_q - trunc_i32(i0 << state_frac)
    y0 = lut_q[i0]
    y1 = lut_q[i0 + 1]
    dy = (y1.astype(np.int64) - y0.astype(np.int64))
    interp = trunc_i32(y0.astype(np.int64) + ((dy * frac_q.astype(np.int64)) >> state_frac))
    return interp
