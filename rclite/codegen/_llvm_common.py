"""Shared LLVM IR helpers for the rclite codegen backends.

Lazily initializes llvmlite, defines the scalar type aliases and the
small IR-emission primitives (loads, stores, loops, constant helpers)
shared by the float, symmetric-int and affine lowerers. Kept
import-light so every codegen module can depend on it without cycles.
"""

from __future__ import annotations
import ctypes
from contextlib import contextmanager

import numpy as np
from llvmlite import ir
import llvmlite.binding as llvm

from rclite.core.profile import Activation


_initialized = False
_all_targets_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _initialized = True


def _ensure_all_targets() -> None:
    """Initialize every LLVM target/asmprinter. Required for cross-compile."""
    global _all_targets_initialized
    if _all_targets_initialized:
        return
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    _all_targets_initialized = True


_F64 = ir.DoubleType()
_F32 = ir.FloatType()
_I64 = ir.IntType(64)
_I32 = ir.IntType(32)

# Float activations the LLVM backend can emit (matches the reference runtime).
# tanh/sigmoid import libm (tanh[f]/exp[f]); relu/identity import nothing.
_SUPPORTED_ACTIVATIONS = (
    Activation.TANH,
    Activation.SIGMOID,
    Activation.RELU,
    Activation.IDENTITY,
)


def _dtype_bindings(dtype: str):
    """Return (fty, tanh_name, np_dtype, ctype) for the requested float type."""
    if dtype == "f64":
        return _F64, "tanh", np.float64, ctypes.c_double
    if dtype == "f32":
        return _F32, "tanhf", np.float32, ctypes.c_float
    raise ValueError(f"unknown dtype: {dtype!r}; expected 'f32' or 'f64'")


def _cf(x: float, fty: ir.Type = _F64) -> ir.Constant:
    return ir.Constant(fty, float(x))


def _ci(x: int) -> ir.Constant:
    return ir.Constant(_I64, int(x))


def _ci32(x: int) -> ir.Constant:
    return ir.Constant(_I32, int(x))


def _load1d(b: ir.IRBuilder, ptr, i):
    return b.load(b.gep(ptr, [i]))


def _store1d(b: ir.IRBuilder, ptr, i, val) -> None:
    b.store(val, b.gep(ptr, [i]))


def _load2d_global(b: ir.IRBuilder, g, ncols: int, i, j):
    flat = b.add(b.mul(i, _ci(ncols)), j)
    return b.load(b.gep(g, [_ci32(0), flat]))


def _load1d_global(b: ir.IRBuilder, g, i):
    """Load element i from a global array (pointer-to-[N x ty])."""
    return b.load(b.gep(g, [_ci32(0), i]))


@contextmanager
def _loop(b: ir.IRBuilder, count, name: str = "i"):
    """Emit a 0..count-1 loop. Yields the loop index value (i64)."""
    fn = b.block.function
    hdr = fn.append_basic_block(name + "_hdr")
    body = fn.append_basic_block(name + "_body")
    done = fn.append_basic_block(name + "_done")

    idx = b.alloca(_I64, name=name + "_idx")
    b.store(_ci(0), idx)
    b.branch(hdr)

    b.position_at_end(hdr)
    cond = b.icmp_signed("<", b.load(idx), count)
    b.cbranch(cond, body, done)

    b.position_at_end(body)
    cur = b.load(idx, name=name + "_v")
    try:
        yield cur
    finally:
        b.store(b.add(b.load(idx), _ci(1)), idx)
        b.branch(hdr)
        b.position_at_end(done)


@contextmanager
def _loop_strided(b: ir.IRBuilder, start, end, stride, name: str = "i"):
    """Emit a `for i = start; i < end; i += stride` loop."""
    fn = b.block.function
    hdr = fn.append_basic_block(name + "_hdr")
    body = fn.append_basic_block(name + "_body")
    done = fn.append_basic_block(name + "_done")

    idx = b.alloca(_I64, name=name + "_idx")
    b.store(start, idx)
    b.branch(hdr)

    b.position_at_end(hdr)
    cond = b.icmp_signed("<", b.load(idx), end)
    b.cbranch(cond, body, done)

    b.position_at_end(body)
    cur = b.load(idx, name=name + "_v")
    try:
        yield cur
    finally:
        b.store(b.add(b.load(idx), stride), idx)
        b.branch(hdr)
        b.position_at_end(done)


# ----------------------------------------------------------------------------
# Value specialization for baked unroll weights
#
# In the "unroll" sparse kernel each nonzero W_res weight is a compile-time
# constant baked into the IR. When that constant is +-1 or +-2**k the multiply
# can be replaced by a negate / shift (or, for floats, +-1 by add/sub), which
# removes a multiply per nonzero MAC -- the win the roadmap flags for FPU-less
# / multiplier-light cores. Exact zeros never reach here (SparsifyReservoir
# prunes them), so we only special-case the power-of-two magnitudes.


def _pow2_exp(v: int):
    """Return k if abs(int(v)) == 2**k (k >= 0), else None.

    `+-1` maps to k=0. Callers must pass an integer-valued weight (the
    quantized integer paths do); the float path checks `+-1.0` directly
    because a fractional float like 1.5 would truncate to a spurious k.
    """
    a = abs(int(v))
    if a == 0 or (a & (a - 1)) != 0:
        return None
    return a.bit_length() - 1
