"""Shared first-class `quant` layer for the MLIR reservoir emitters.

`mlir_symmetric_xdsl` (Q-format / shift requantize) and `mlir_affine_xdsl`
(TFLM-style `(x*M0+round)>>n` requantize + zero-point cross-terms) used to carry
their own private copies of the quantized-kernel building blocks. This module
gives both one definition of the quantization primitives:

  * `QuantUniform` / `uniform_type` — the `!quant.uniform<...>` *type* that
    declares scale/zero-point (per-tensor or per-axis). Re-exported by
    `mlir_quant_types_xdsl`; this is the single canonical definition.
  * `emit_sat_func` / `emit_clip32_func` — the saturation helpers
    (`i32 -> storage`, `i64 -> i32`).
  * `emit_requantize_func` — the affine `(x*M0 + (1<<(n-1)))>>n` round-shift
    requantize as a private `func` (`M0==0` folds to a constant 0).
  * `emit_requantize_axis_func` — its per-axis (per-channel) counterpart:
    `@name(x, idx)` loads `(M0, n)` per row from i32 globals (dynamic shift).
  * `zp_cross` — the affine zero-point cross-term `clip32(acc - zp*sext(rs))`.
  * `emit_argmax_head` / `emit_softmax_head` — the classify/proba kernel tails.

Every emitter helper builds the *same* arith/memref/scf/func ops the inline code
did (verified byte-identical against captured golden MLIR), so the printed IR —
and therefore the host<->device bit-exactness — is unchanged. The heads emit
into the currently-active `ImplicitBuilder`; the `func`-returning helpers build
their own self-contained region and the caller appends the result.
"""

from __future__ import annotations

import numpy as np

from xdsl.builder import ImplicitBuilder
from xdsl.ir import Region, Block, ParametrizedAttribute, TypeAttribute
from xdsl.irdl import irdl_attr_definition
from xdsl.dialects import arith, memref, func
from xdsl.dialects.builtin import StringAttr, MemRefType
from xdsl.printer import Printer

from .mlir_xdsl_common import _I32, _I64, _IDX, c_i, ext, call, for_


# ---------------------------------------------------------------------------
# First-class quant.uniform type (xDSL has no quant dialect; this reproduces
# mlir-opt's surface syntax verbatim so the emitted type round-trips through it)
# ---------------------------------------------------------------------------
@irdl_attr_definition
class QuantUniform(ParametrizedAttribute, TypeAttribute):
    """`!quant.uniform<...>` — body holds the verbatim inner syntax."""

    name = "quant.uniform"
    body: StringAttr

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<" + self.body.data + ">")


def _f(x) -> str:
    return f"{float(x):.8e}"


def uniform_type(sb: int, scale, zero_point: int = 0) -> QuantUniform:
    """`!quant.uniform` type. `scale` may be scalar (per-tensor) or 1-D array
    (per-axis along output axis 0). zero_point applies per-tensor."""
    arr = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if arr.size == 1:
        zp = f":{int(zero_point)}" if zero_point else ""
        inner = f"i{sb}:f32, {_f(arr[0])}{zp}"
    else:
        scales = ",".join(_f(s) for s in arr)
        inner = f"i{sb}:f32:0, {{{scales}}}"
    return QuantUniform(StringAttr(inner))


# ---------------------------------------------------------------------------
# Saturation helpers (return a private `func` for the caller to append)
# ---------------------------------------------------------------------------
def emit_sat_func(qmin, qmax, isb) -> func.FuncOp:
    """`@sat(i32) -> isb`: clamp to [qmin, qmax] then truncate to storage."""
    r = Region([Block(arg_types=[_I32])])
    with ImplicitBuilder(r.block) as (x,):
        b = arith.MinSIOp(
            arith.MaxSIOp(x, c_i(qmin, _I32)).result, c_i(qmax, _I32)
        ).result
        func.ReturnOp(arith.TruncIOp(b, isb).result)
    return func.FuncOp("sat", ([_I32], [isb]), r, visibility="private")


def emit_clip32_func() -> func.FuncOp:
    """`@clip32(i64) -> i32`: saturate an i64 accumulator into i32."""
    r = Region([Block(arg_types=[_I64])])
    with ImplicitBuilder(r.block) as (x,):
        b = arith.MinSIOp(
            arith.MaxSIOp(x, c_i(-2147483648, _I64)).result,
            c_i(2147483647, _I64),
        ).result
        func.ReturnOp(arith.TruncIOp(b, _I32).result)
    return func.FuncOp("clip32", ([_I64], [_I32]), r, visibility="private")


# ---------------------------------------------------------------------------
# Affine requantize multiplier: (x*M0 + (1<<(n-1))) >> n, truncated to i32
# ---------------------------------------------------------------------------
def emit_requantize_func(name, M0, n) -> func.FuncOp:
    """`@name(i32) -> i32`: round-half-up fixed-point requantize. `M0==0`
    folds the whole channel to a constant 0 (a dead scale)."""
    r = Region([Block(arg_types=[_I32])])
    with ImplicitBuilder(r.block) as (x,):
        if M0 == 0:
            func.ReturnOp(c_i(0, _I32))
        else:
            p = arith.MuliOp(ext(x), c_i(M0, _I64)).result
            if n > 0:
                p = arith.AddiOp(p, c_i(1 << (n - 1), _I64)).result
            s = arith.ShRSIOp(p, c_i(n, _I64)).result
            func.ReturnOp(arith.TruncIOp(s, _I32).result)
    return func.FuncOp(name, ([_I32], [_I32]), r, visibility="private")


def emit_requantize_axis_func(
    name, m0_global, n_global, length
) -> func.FuncOp:
    """Per-axis requantize: `@name(x:i32, idx:index) -> i32`.

    Loads `(M0, n) = (m0_global[idx], n_global[idx])` from two i32 constant
    globals (length `length`) and applies the *dynamic-shift* round-half-up
    `(x*M0 + (n>0 ? 1<<(n-1) : 0)) >> n`. Mirrors `apply_multiplier_perrow` /
    `_AffineLowerer._emit_requantize_i32_dynamic` bit-for-bit: the shift amount
    is data-dependent, and `n==0` (a dead `M0==0` channel) takes no rounding
    bias and an identity `ashr 0`. This is the per-channel counterpart of
    `emit_requantize_func` — same maths, multiplier/shift fetched per row."""
    r = Region([Block(arg_types=[_I32, _IDX])])
    with ImplicitBuilder(r.block) as (x, idx):
        m0mr = memref.GetGlobalOp(m0_global, MemRefType(_I32, [length])).memref
        nmr = memref.GetGlobalOp(n_global, MemRefType(_I32, [length])).memref
        m0 = ext(memref.LoadOp.get(m0mr, [idx]).res)
        nn = ext(memref.LoadOp.get(nmr, [idx]).res)
        prod = arith.MuliOp(ext(x), m0).result
        nz = arith.CmpiOp(nn, c_i(0, _I64), "sgt").result
        safe_sh = arith.SelectOp(
            nz, arith.SubiOp(nn, c_i(1, _I64)).result, c_i(0, _I64)
        ).result
        half = arith.SelectOp(
            nz, arith.ShLIOp(c_i(1, _I64), safe_sh).result, c_i(0, _I64)
        ).result
        biased = arith.AddiOp(prod, half).result
        shr = arith.ShRSIOp(biased, nn).result
        func.ReturnOp(arith.TruncIOp(shr, _I32).result)
    return func.FuncOp(name, ([_I32, _IDX], [_I32]), r, visibility="private")


# ---------------------------------------------------------------------------
# Affine zero-point cross-term: clip32(acc_i64 - zp * sext(row_sum))
# ---------------------------------------------------------------------------
def zp_cross(acc, zp, rs_val):
    """`clip32(acc - zp*sext(rs_val))`. `acc` is i64, `rs_val` the (already
    loaded) i32 row-sum SSA value, `zp` a Python int. Requires `@clip32` in the
    module. Emits into the active builder; returns the i32 result."""
    rs64 = ext(rs_val)
    zr = arith.MuliOp(c_i(zp, _I64), rs64).result
    return call("clip32", [arith.SubiOp(acc, zr).result], _I32)


# ---------------------------------------------------------------------------
# Classification heads (emit into the active builder; no value returned)
# ---------------------------------------------------------------------------
def emit_argmax_head(logits, Y, t, c0, c1, cM):
    """argmax over `logits[0:M]` -> class id (i32) stored at `Y[t]`."""
    bv0 = memref.LoadOp.get(logits, [c0]).res

    def amax(m, args):
        bv, bi = args
        v = memref.LoadOp.get(logits, [m]).res
        gt = arith.CmpiOp(v, bv, "sgt").result
        return [
            arith.SelectOp(gt, v, bv).result,
            arith.SelectOp(gt, m, bi).result,
        ]

    best = for_(c1, cM, c1, [bv0, c0], amax)
    memref.StoreOp.get(arith.IndexCastOp(best[1], _I32).result, Y, [t])


def emit_softmax_head(
    logits,
    exps,
    Y,
    SM,
    tM,
    isb,
    qmax,
    sm_n,
    sm_dmin,
    sm_idxf,
    sm_pf,
    c0,
    c1,
    cM,
    z64,
):
    """softmax over `logits[0:M]` via the shared softmax LUT (`SM`); writes the
    per-class quantized probabilities to `Y[t*M + m]`. `exps` is the i32 scratch
    buffer for the unnormalized exponentials."""
    mx0 = arith.ExtSIOp(memref.LoadOp.get(logits, [c0]).res, _I32).result

    def fmax(m, args):
        (mxa,) = args
        v = arith.ExtSIOp(memref.LoadOp.get(logits, [m]).res, _I32).result
        gt = arith.CmpiOp(v, mxa, "sgt").result
        return [arith.SelectOp(gt, v, mxa).result]

    mx = for_(c1, cM, c1, [mx0], fmax)[0]
    dmin = c_i(sm_dmin, _I32)
    ndmin64 = c_i(-sm_dmin, _I64)
    smnm1 = c_i(sm_n - 1, _I64)
    idxf64 = c_i(sm_idxf, _I64)
    smnm2 = c_i(sm_n - 2, _I64)

    def sbody(m, args):
        (sa,) = args
        v = arith.ExtSIOp(memref.LoadOp.get(logits, [m]).res, _I32).result
        d0 = arith.SubiOp(v, mx).result
        d = arith.SelectOp(
            arith.CmpiOp(d0, dmin, "slt").result, dmin, d0
        ).result
        num = arith.SubiOp(d, dmin).result
        nn = arith.MuliOp(ext(num), smnm1).result
        pos = arith.DivSIOp(arith.ShLIOp(nn, idxf64).result, ndmin64).result
        i0r = arith.ShRSIOp(pos, idxf64).result
        i0 = arith.MinSIOp(arith.MaxSIOp(i0r, z64).result, smnm2).result
        frac = arith.SubiOp(pos, arith.ShLIOp(i0, idxf64).result).result
        i0idx = arith.IndexCastOp(i0, _IDX).result
        i1idx = arith.AddiOp(i0idx, c1).result
        y0 = ext(memref.LoadOp.get(SM, [i0idx]).res)
        y1 = ext(memref.LoadOp.get(SM, [i1idx]).res)
        dy = arith.SubiOp(y1, y0).result
        sh = arith.ShRSIOp(arith.MuliOp(dy, frac).result, idxf64).result
        e = arith.AddiOp(y0, sh).result
        memref.StoreOp.get(arith.TruncIOp(e, _I32).result, exps, [m])
        return [arith.AddiOp(sa, e).result]

    total = for_(c0, cM, c1, [z64], sbody)[0]
    pfc = c_i(sm_pf, _I64)
    qmaxc = c_i(qmax, _I64)

    def pbody(m, _):
        e = memref.LoadOp.get(exps, [m]).res
        p = arith.DivSIOp(arith.ShLIOp(ext(e), pfc).result, total).result
        pq = arith.TruncIOp(arith.MinSIOp(p, qmaxc).result, isb).result
        memref.StoreOp.get(pq, Y, [arith.AddiOp(tM, m).result])
        return []

    for_(c0, cM, c1, [], pbody)
