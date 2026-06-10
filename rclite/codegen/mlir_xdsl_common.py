"""Shared xDSL/MLIR building blocks for the quantized reservoir emitters.

`mlir_symmetric_xdsl` and `mlir_affine_xdsl` assemble the same arith / memref /
scf / func IR skeleton: the type constants, a private-constant `memref.global`
factory, and a handful of SSA-construction helpers (integer/index constants,
sign-extend, fixed-point multiply, `scf.for` with iter_args, single-result
call). They live here so both emitters share one definition; each emitter keeps
only its scheme-specific lowering (requantize-multiplier vs fixed-point shift,
activation, readout, ...).

Every helper emits into the currently-active xDSL `ImplicitBuilder` — exactly
as the inline closures they replace did — so the printed IR is unchanged.
"""

from __future__ import annotations

import numpy as np

from xdsl.builder import ImplicitBuilder
from xdsl.ir import Region, Block, SSAValue
from xdsl.dialects import arith, memref, scf, func
from xdsl.dialects.builtin import (
    IntegerType,
    IndexType,
    MemRefType,
    TensorType,
    DenseIntOrFPElementsAttr,
    StringAttr,
    UnitAttr,
    IntegerAttr,
)

from rclite.core.profile import Topology

_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)
_HEADS = (None, "logits", "classify", "proba")
_IDX = IndexType()
_I32 = IntegerType(32)
_I64 = IntegerType(64)


def _np_t(bits):
    return {8: np.int8, 16: np.int16, 32: np.int32}[bits]


def _dense_global(name, arr, bits) -> memref.GlobalOp:
    """A `memref.global "private" constant` from a flat integer array."""
    flat = np.asarray(arr).reshape(-1).astype(_np_t(bits))
    ty = MemRefType(IntegerType(bits), [int(flat.size)])
    # initial_value must be a tensor-typed dense attr (mlir-opt requirement);
    # the global's sym_type stays the memref type.
    init = DenseIntOrFPElementsAttr.from_list(
        TensorType(IntegerType(bits), [int(flat.size)]), [int(v) for v in flat]
    )
    return memref.GlobalOp.get(
        StringAttr(name),
        ty,
        init,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr(),
    )


# ---- SSA helpers: each emits into the current ImplicitBuilder ----
def c_i(v, ty):
    """An `arith.constant` integer of the given type."""
    return arith.ConstantOp.from_int_and_width(int(v), ty).result


def c_idx(v):
    """An `arith.constant` of `index` type."""
    return arith.ConstantOp(IntegerAttr.from_index_int_value(int(v))).result


def ext(v, ty=_I64):
    """Sign-extend `v` to `ty` (i64 by default)."""
    return arith.ExtSIOp(v, ty).result


def fmul(av, bv, shift):
    """(sext(a,i64)*sext(b,i64))>>shift, truncated to i32 (wrapping)."""
    p = arith.MuliOp(ext(av), ext(bv)).result
    s = arith.ShRSIOp(p, c_i(shift, _I64)).result
    return arith.TruncIOp(s, _I32).result


def call(name, args, ret):
    """A single-result `func.call`."""
    return func.CallOp(name, args, [ret]).res[0]


def for_(lb, ub, step, inits, body):
    """scf.for with iter_args; `body(iv, args)->yields`. Returns results."""
    arg_tys = [_IDX] + [SSAValue.get(x).type for x in inits]
    region = Region([Block(arg_types=arg_tys)])
    with ImplicitBuilder(region.block) as bargs:
        ys = body(bargs[0], list(bargs[1:]))
        scf.YieldOp(*ys)
    return scf.ForOp(lb, ub, step, inits, region).results
