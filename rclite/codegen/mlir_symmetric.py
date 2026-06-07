"""MLIR codegen for the SYMMETRIC (Q-format) quantized reservoir (phase-2).

Mirrors `_IntLowerer` / `QuantizedExecutor` in MLIR (arith/memref/scf), lowers
with mlir-opt -> mlir-translate -> llc, links a host .so with gcc, runs via
ctypes. Bit-exact with the executor (same integer ops).

Symmetric path specifics vs the affine emitter:
  - no zero points; requantize is a fixed arithmetic shift (shift_in /
    shift_res / state_frac), not a (M0,n) multiplier
  - WRAPPING two's-complement arithmetic throughout (the executor wraps; we
    use plain arith.addi/muli + trunci, never saturating adds)
  - tanh via an interpolating LUT (clip -> normalize -> lerp)
  - readout accumulates in i64 then >> state_frac and saturates to storage

Scope: dense (RANDOM/ESN_STANDARD) + structured (DLR/SCR/DLRB) + CSR-sparse,
identity preprocess, i8/i16/i32, logits/argmax/softmax heads.
Link with gcc (PATH clang is the llvm-mos cross compiler).
"""

from __future__ import annotations
import ctypes
import shutil
import subprocess
import tempfile
import pathlib
from typing import List, Optional

import numpy as np

from rclite.core.profile import Topology
from rclite.quant.model import QuantizedModel

_TOOLS = ("mlir-opt", "mlir-translate", "llc")
_LOWER_PASSES = [
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]
_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)
_HEADS = (None, "logits", "classify", "proba")


def _flat_i(arr, bits):
    np_t = {8: np.int8, 16: np.int16, 32: np.int32}[bits]
    return ", ".join(
        str(int(v)) for v in np.asarray(arr).reshape(-1).astype(np_t)
    )


def _global(name, arr, bits):
    n = int(np.asarray(arr).size)
    return (
        f'memref.global "private" constant @{name} : memref<{n}xi{bits}> '
        f"= dense<[{_flat_i(arr, bits)}]>"
    )


def tools_available() -> bool:
    return (
        all(shutil.which(t) for t in _TOOLS)
        and shutil.which("gcc") is not None
    )


def emit_symmetric_mlir(
    qmodel: QuantizedModel,
    *,
    head: Optional[str] = None,
    sparse: Optional[str] = None,
) -> str:
    if head not in _HEADS:
        raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
    rc = qmodel.rc
    cfg = qmodel.config
    topo = rc.reservoir.topology
    structured = topo in _STRUCTURED
    if rc.input.input_offset != 0.0 or rc.input.input_scaling != 1.0:
        raise NotImplementedError("symmetric MLIR: identity preprocess only")
    if qmodel.lut_table_q is None:
        raise NotImplementedError("symmetric MLIR: tanh LUT required")
    use_sparse = bool(sparse) and not structured

    N, K, M, F = qmodel.N, qmodel.K, qmodel.M, qmodel.F
    sb = qmodel.target.storage_bits
    if sb not in (8, 16):
        # i32 storage needs i32<->i32 cast special-casing throughout; the
        # embedded-relevant widths are i8/i16 (matches the affine emitter).
        raise NotImplementedError(
            "symmetric MLIR: i8/i16 storage only (i32 TODO)"
        )
    sf = cfg.state_frac
    shift_in = cfg.weight_frac + cfg.input_frac - cfg.state_frac
    shift_res = cfg.weight_frac
    state_scale = 1 << sf
    bias_q = qmodel.target.quantize_state(float(rc.reservoir.bias), cfg)
    leak_q = qmodel.target.quantize_state(float(rc.reservoir.leak_rate), cfg)
    one_ml_q = state_scale - leak_q
    lut_n = int(np.asarray(qmodel.lut_table_q).size)
    xmin_q = int(qmodel.lut.xmin * state_scale)
    xmax_q = int(qmodel.lut.xmax * state_scale)
    denom = xmax_q - xmin_q
    inc_b = bool(rc.readout.include_bias)
    inc_i = bool(rc.readout.include_input)
    off_i = 1 if inc_b else 0
    off_s = off_i + (K if inc_i else 0)
    qmin, qmax = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    classify = head == "classify"
    proba = head == "proba"
    has_logits_buf = classify or proba
    out_bits = 32 if classify else sb
    if shift_in < 0:
        raise NotImplementedError(
            f"symmetric MLIR needs shift_in>=0 ({shift_in})"
        )
    if structured:
        wsc = 1 << cfg.weight_frac
        cw_q = int(round(float(rc.reservoir.chain_weight) * wsc))
        cf_q = int(round(float(rc.reservoir.chain_feedback) * wsc))

    L: List[str] = []
    a = L.append

    a(_global("W_in", qmodel.W_in_q, sb))
    a(_global("W_out", qmodel.W_out_q, sb))
    a(_global("lut", qmodel.lut_table_q, sb))
    if not structured:
        if use_sparse:
            from rclite.ir.passes.sparsify import build_csr

            val, col, rptr = build_csr(np.asarray(qmodel.W_res_q))
            a(_global("Wres_val", val, sb))
            a(_global("Wres_col", col, 32))
            a(_global("Wres_rptr", rptr, 32))
        else:
            a(_global("W_res", qmodel.W_res_q, sb))
    if proba:
        from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params

        sm = build_params(
            SoftmaxLUTSpec(),
            s_diff=1.0 / state_scale,
            storage_bits=sb,
            storage_dtype=np.dtype(f"int{sb}"),
        )
        a(_global("sm_lut", sm.lut_q, sb))
        sm_n, sm_dmin, sm_idxf, sm_pf = (
            sm.n,
            sm.dmin_q,
            sm.idx_frac,
            sm.prob_frac,
        )
        sm_size = int(np.asarray(sm.lut_q).size)

    # fixed-point multiply -> i32 (wrapping): (sext(a,i64)*sext(b,i64))>>shift
    def fmul_i32(av, bv, shift, dst):
        a(f"        %{dst}a = arith.extsi {av} : i{sb} to i64")
        a(f"        %{dst}b = arith.extsi {bv} : i{sb} to i64")
        a(f"        %{dst}p = arith.muli %{dst}a, %{dst}b : i64")
        a(f"        %{dst}sc = arith.constant {shift} : i64")
        a(f"        %{dst}s = arith.shrsi %{dst}p, %{dst}sc : i64")
        a(f"        %{dst} = arith.trunci %{dst}s : i64 to i32")

    # ---- saturate i32 -> storage (readout / final) ----
    a(f"func.func private @sat(%x: i32) -> i{sb} {{")
    a(f"  %lo = arith.constant {qmin} : i32")
    a(f"  %hi = arith.constant {qmax} : i32")
    a("  %a = arith.maxsi %x, %lo : i32")
    a("  %b = arith.minsi %a, %hi : i32")
    a(f"  %r = arith.trunci %b : i32 to i{sb}")
    a(f"  return %r : i{sb}")
    a("}")

    # ---- tanh LUT (interpolating), storage -> storage ----
    a(f"func.func private @activate(%p: i{sb}) -> i{sb} {{")
    a(f"  %lut = memref.get_global @lut : memref<{lut_n}xi{sb}>")
    a(f"  %x0 = arith.extsi %p : i{sb} to i32")
    a(f"  %xmin = arith.constant {xmin_q} : i32")
    a(f"  %xmax = arith.constant {xmax_q} : i32")
    a("  %xa = arith.maxsi %x0, %xmin : i32")
    a("  %x = arith.minsi %xa, %xmax : i32")
    a("  %num = arith.subi %x, %xmin : i32")
    a("  %num64 = arith.extsi %num : i32 to i64")
    a(f"  %sf64 = arith.constant {sf} : i64")
    a("  %nsh = arith.shli %num64, %sf64 : i64")
    a(f"  %den = arith.constant {denom} : i64")
    a("  %tq64 = arith.divsi %nsh, %den : i64")
    a("  %tq = arith.trunci %tq64 : i64 to i32")
    a(f"  %nm1 = arith.constant {lut_n - 1} : i32")
    a("  %tq_64 = arith.extsi %tq : i32 to i64")
    a("  %nm1_64 = arith.extsi %nm1 : i32 to i64")
    a("  %pos64 = arith.muli %tq_64, %nm1_64 : i64")
    a("  %posq = arith.trunci %pos64 : i64 to i32")
    a(f"  %sf32 = arith.constant {sf} : i32")
    a("  %i0r = arith.shrsi %posq, %sf32 : i32")
    a("  %z32 = arith.constant 0 : i32")
    a(f"  %nm2 = arith.constant {lut_n - 2} : i32")
    a("  %i0a = arith.maxsi %i0r, %z32 : i32")
    a("  %i0 = arith.minsi %i0a, %nm2 : i32")
    a("  %i0sh = arith.shli %i0, %sf32 : i32")
    a("  %frac = arith.subi %posq, %i0sh : i32")
    a("  %i0i = arith.index_cast %i0 : i32 to index")
    a("  %c1i = arith.constant 1 : index")
    a("  %i1i = arith.addi %i0i, %c1i : index")
    a(f"  %y0 = memref.load %lut[%i0i] : memref<{lut_n}xi{sb}>")
    a(f"  %y1 = memref.load %lut[%i1i] : memref<{lut_n}xi{sb}>")
    a(f"  %y032 = arith.extsi %y0 : i{sb} to i32")
    a(f"  %y132 = arith.extsi %y1 : i{sb} to i32")
    a("  %dy = arith.subi %y132, %y032 : i32")
    a("  %dy64 = arith.extsi %dy : i32 to i64")
    a("  %fr64 = arith.extsi %frac : i32 to i64")
    a("  %dfp = arith.muli %dy64, %fr64 : i64")
    a("  %dfs = arith.shrsi %dfp, %sf64 : i64")
    a("  %dfs32 = arith.trunci %dfs : i64 to i32")
    a("  %res = arith.addi %y032, %dfs32 : i32")
    a(f"  %r = arith.trunci %res : i32 to i{sb}")
    a(f"  return %r : i{sb}")
    a("}")

    # ---- main ----
    a(
        f"func.func @rc_predict(%T: i64, %X: memref<?xi{sb}>, "
        f"%Y: memref<?xi{out_bits}>) attributes {{llvm.emit_c_interface}} {{"
    )
    a("  %c0 = arith.constant 0 : index")
    a("  %c1 = arith.constant 1 : index")
    a(f"  %cN = arith.constant {N} : index")
    a(f"  %cK = arith.constant {K} : index")
    a(f"  %cM = arith.constant {M} : index")
    a("  %z32 = arith.constant 0 : i32")
    a("  %z64 = arith.constant 0 : i64")
    a("  %Ti = arith.index_cast %T : i64 to index")
    a(f"  %Win = memref.get_global @W_in : memref<{N * K}xi{sb}>")
    a(f"  %Wout = memref.get_global @W_out : memref<{M * F}xi{sb}>")
    if not structured:
        if use_sparse:
            a(
                f"  %WrV = memref.get_global @Wres_val : memref<{val.size}xi{sb}>"
            )
            a(f"  %WrC = memref.get_global @Wres_col : memref<{col.size}xi32>")
            a(
                f"  %WrP = memref.get_global @Wres_rptr : memref<{rptr.size}xi32>"
            )
        else:
            a(f"  %Wres = memref.get_global @W_res : memref<{N * N}xi{sb}>")
    if proba:
        a(f"  %SM = memref.get_global @sm_lut : memref<{sm_size}xi{sb}>")
    a(f"  %h = memref.alloca() : memref<{N}xi{sb}>")
    a(f"  %pre = memref.alloca() : memref<{N}xi{sb}>")
    if has_logits_buf:
        a(f"  %logits = memref.alloca() : memref<{M}xi{sb}>")
    if proba:
        a(f"  %exps = memref.alloca() : memref<{M}xi32>")
    a(f"  %zsb = arith.constant 0 : i{sb}")
    a("  scf.for %i = %c0 to %cN step %c1 {")
    a(f"    memref.store %zsb, %h[%i] : memref<{N}xi{sb}>")
    a("  }")

    a("  scf.for %t = %c0 to %Ti step %c1 {")
    a("    %tK = arith.muli %t, %cK : index")
    a("    %tM = arith.muli %t, %cM : index")

    # pre-activation
    a("    scf.for %i = %c0 to %cN step %c1 {")
    a(f"      %biasc = arith.constant {bias_q} : i32")
    # acc_in: wrapping i32 over k
    a(
        "      %accin = scf.for %k = %c0 to %cK step %c1 "
        "iter_args(%ai = %biasc) -> (i32) {"
    )
    a("        %iKin = arith.muli %i, %cK : index")
    a("        %widx = arith.addi %iKin, %k : index")
    a(f"        %w = memref.load %Win[%widx] : memref<{N * K}xi{sb}>")
    a("        %xidx = arith.addi %tK, %k : index")
    a(f"        %x = memref.load %X[%xidx] : memref<?xi{sb}>")
    fmul_i32("%w", "%x", shift_in, "tin")
    a("        %na = arith.addi %ai, %tin : i32")
    a("        scf.yield %na : i32")
    a("      }")
    # acc_res
    if structured:
        if topo == Topology.SCR:
            a(f"      %cw = arith.constant {cw_q} : i{sb}")
            a("      %iz = arith.cmpi eq, %i, %c0 : index")
            a(f"      %nm1 = arith.constant {N - 1} : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %iprev = arith.select %iz, %nm1, %im1 : index")
            a(f"      %hv = memref.load %h[%iprev] : memref<{N}xi{sb}>")
            fmul_i32("%cw", "%hv", shift_res, "tr")
            a("      %accres = arith.addi %accin, %tr : i32")
        elif topo == Topology.DLR:
            a(f"      %cw = arith.constant {cw_q} : i{sb}")
            a("      %ipos = arith.cmpi sgt, %i, %c0 : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %isafe = arith.select %ipos, %im1, %c0 : index")
            a(f"      %hv = memref.load %h[%isafe] : memref<{N}xi{sb}>")
            fmul_i32("%cw", "%hv", shift_res, "tr")
            a("      %trsel = arith.select %ipos, %tr, %z32 : i32")
            a("      %accres = arith.addi %accin, %trsel : i32")
        else:  # DLRB
            a(f"      %cw = arith.constant {cw_q} : i{sb}")
            a(f"      %cfk = arith.constant {cf_q} : i{sb}")
            a(f"      %nm1 = arith.constant {N - 1} : index")
            a("      %ipos = arith.cmpi sgt, %i, %c0 : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %ib = arith.select %ipos, %im1, %c0 : index")
            a(f"      %hb = memref.load %h[%ib] : memref<{N}xi{sb}>")
            fmul_i32("%cw", "%hb", shift_res, "tb")
            a("      %tbsel = arith.select %ipos, %tb, %z32 : i32")
            a("      %ilt = arith.cmpi slt, %i, %nm1 : index")
            a("      %ip1 = arith.addi %i, %c1 : index")
            a("      %iff = arith.select %ilt, %ip1, %nm1 : index")
            a(f"      %hf = memref.load %h[%iff] : memref<{N}xi{sb}>")
            fmul_i32("%cfk", "%hf", shift_res, "tf")
            a("      %tfsel = arith.select %ilt, %tf, %z32 : i32")
            a("      %accr2 = arith.addi %accin, %tbsel : i32")
            a("      %accres = arith.addi %accr2, %tfsel : i32")
    elif use_sparse:
        a("      %rp0 = memref.load %WrP[%i] : memref<" + f"{rptr.size}xi32>")
        a("      %ip1 = arith.addi %i, %c1 : index")
        a(
            "      %rp1 = memref.load %WrP[%ip1] : memref<"
            + f"{rptr.size}xi32>"
        )
        a("      %rp0i = arith.index_cast %rp0 : i32 to index")
        a("      %rp1i = arith.index_cast %rp1 : i32 to index")
        a(
            "      %accres = scf.for %p = %rp0i to %rp1i step %c1 "
            "iter_args(%ar = %accin) -> (i32) {"
        )
        a(f"        %w = memref.load %WrV[%p] : memref<{val.size}xi{sb}>")
        a(f"        %cj = memref.load %WrC[%p] : memref<{col.size}xi32>")
        a("        %cji = arith.index_cast %cj : i32 to index")
        a(f"        %hv = memref.load %h[%cji] : memref<{N}xi{sb}>")
        fmul_i32("%w", "%hv", shift_res, "tr")
        a("        %na = arith.addi %ar, %tr : i32")
        a("        scf.yield %na : i32")
        a("      }")
    else:
        a("      %iN = arith.muli %i, %cN : index")
        a(
            "      %accres = scf.for %j = %c0 to %cN step %c1 "
            "iter_args(%ar = %accin) -> (i32) {"
        )
        a("        %widx = arith.addi %iN, %j : index")
        a(f"        %w = memref.load %Wres[%widx] : memref<{N * N}xi{sb}>")
        a(f"        %hv = memref.load %h[%j] : memref<{N}xi{sb}>")
        fmul_i32("%w", "%hv", shift_res, "tr")
        a("        %na = arith.addi %ar, %tr : i32")
        a("        scf.yield %na : i32")
        a("      }")
    a(f"      %preq = arith.trunci %accres : i32 to i{sb}")
    a(f"      memref.store %preq, %pre[%i] : memref<{N}xi{sb}>")
    a("    }")

    # activation + leaky integration
    a("    scf.for %i = %c0 to %cN step %c1 {")
    a(f"      %p = memref.load %pre[%i] : memref<{N}xi{sb}>")
    a(
        "      %act = func.call @activate(%p) : (i"
        + str(sb)
        + ") -> i"
        + str(sb)
    )
    a(f"      %hold = memref.load %h[%i] : memref<{N}xi{sb}>")
    a(f"      %omlc = arith.constant {one_ml_q} : i{sb}")
    a(f"      %leakc = arith.constant {leak_q} : i{sb}")
    fmul_i32("%hold", "%omlc", sf, "t1x")
    a(f"      %t1 = arith.trunci %t1x : i32 to i{sb}")
    fmul_i32("%act", "%leakc", sf, "t2x")
    a(f"      %t2 = arith.trunci %t2x : i32 to i{sb}")
    a(f"      %nh = arith.addi %t1, %t2 : i{sb}")
    a(f"      memref.store %nh, %h[%i] : memref<{N}xi{sb}>")
    a("    }")

    # readout: i64 accumulate -> >> sf -> saturate
    a("    scf.for %m = %c0 to %cM step %c1 {")
    a(f"      %cF = arith.constant {F} : index")
    a("      %mF = arith.muli %m, %cF : index")
    if inc_b:
        a(f"      %ssc = arith.constant {state_scale} : i64")
        a(f"      %obidx = arith.addi %mF, %c0 : index")
        a(f"      %w0 = memref.load %Wout[%obidx] : memref<{M * F}xi{sb}>")
        a("      %w064 = arith.extsi %w0 : i" + str(sb) + " to i64")
        a("      %yb = arith.muli %ssc, %w064 : i64")
    yb = "%yb" if inc_b else "%z64"
    if inc_i:
        a(f"      %ci = arith.constant {off_i} : index")
        a(
            "      %accoi = scf.for %k = %c0 to %cK step %c1 "
            f"iter_args(%ao = {yb}) -> (i64) {{"
        )
        a("        %coff = arith.addi %ci, %k : index")
        a("        %widx = arith.addi %mF, %coff : index")
        a(f"        %w = memref.load %Wout[%widx] : memref<{M * F}xi{sb}>")
        a("        %xidx = arith.addi %tK, %k : index")
        a(f"        %x = memref.load %X[%xidx] : memref<?xi{sb}>")
        a(f"        %w64 = arith.extsi %w : i{sb} to i64")
        a(f"        %x64 = arith.extsi %x : i{sb} to i64")
        a("        %pr = arith.muli %w64, %x64 : i64")
        a("        %na = arith.addi %ao, %pr : i64")
        a("        scf.yield %na : i64")
        a("      }")
    yi = "%accoi" if inc_i else yb
    a(f"      %cs = arith.constant {off_s} : index")
    a(
        "      %accos = scf.for %j = %c0 to %cN step %c1 "
        f"iter_args(%ao = {yi}) -> (i64) {{"
    )
    a("        %coff = arith.addi %cs, %j : index")
    a("        %widx = arith.addi %mF, %coff : index")
    a(f"        %w = memref.load %Wout[%widx] : memref<{M * F}xi{sb}>")
    a(f"        %hv = memref.load %h[%j] : memref<{N}xi{sb}>")
    a(f"        %w64 = arith.extsi %w : i{sb} to i64")
    a(f"        %h64 = arith.extsi %hv : i{sb} to i64")
    a("        %pr = arith.muli %w64, %h64 : i64")
    a("        %na = arith.addi %ao, %pr : i64")
    a("        scf.yield %na : i64")
    a("      }")
    a(f"      %sfc = arith.constant {sf} : i64")
    a("      %shifted = arith.shrsi %accos, %sfc : i64")
    a(f"      %lo = arith.constant {qmin} : i64")
    a(f"      %hi = arith.constant {qmax} : i64")
    a("      %cla = arith.maxsi %shifted, %lo : i64")
    a("      %clb = arith.minsi %cla, %hi : i64")
    a(f"      %yq = arith.trunci %clb : i64 to i{sb}")
    if has_logits_buf:
        a(f"      memref.store %yq, %logits[%m] : memref<{M}xi{sb}>")
    else:
        a("      %yidx = arith.addi %tM, %m : index")
        a(f"      memref.store %yq, %Y[%yidx] : memref<?xi{sb}>")
    a("    }")

    # head
    if classify:
        a(f"      %bv0 = memref.load %logits[%c0] : memref<{M}xi{sb}>")
        a(
            "      %best:2 = scf.for %m = %c1 to %cM step %c1 "
            f"iter_args(%bv = %bv0, %bi = %c0) -> (i{sb}, index) {{"
        )
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %gt = arith.cmpi sgt, %v, %bv : i{sb}")
        a(f"        %nv = arith.select %gt, %v, %bv : i{sb}")
        a("        %ni = arith.select %gt, %m, %bi : index")
        a(f"        scf.yield %nv, %ni : i{sb}, index")
        a("      }")
        a("      %cls = arith.index_cast %best#1 : index to i32")
        a("      memref.store %cls, %Y[%t] : memref<?xi32>")
    elif proba:
        a(f"      %mx0 = memref.load %logits[%c0] : memref<{M}xi{sb}>")
        a(f"      %mx032 = arith.extsi %mx0 : i{sb} to i32")
        a(
            "      %mx = scf.for %m = %c1 to %cM step %c1 "
            "iter_args(%mxa = %mx032) -> (i32) {"
        )
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %v32 = arith.extsi %v : i{sb} to i32")
        a("        %gt = arith.cmpi sgt, %v32, %mxa : i32")
        a("        %nm = arith.select %gt, %v32, %mxa : i32")
        a("        scf.yield %nm : i32")
        a("      }")
        a(f"      %dmin = arith.constant {sm_dmin} : i32")
        a(f"      %ndmin64 = arith.constant {-sm_dmin} : i64")
        a(f"      %smnm1 = arith.constant {sm_n - 1} : i64")
        a(f"      %idxf64 = arith.constant {sm_idxf} : i64")
        a(f"      %smnm2 = arith.constant {sm_n - 2} : i64")
        a(
            "      %sum = scf.for %m = %c0 to %cM step %c1 "
            "iter_args(%sa = %z64) -> (i64) {"
        )
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %v32 = arith.extsi %v : i{sb} to i32")
        a("        %d0 = arith.subi %v32, %mx : i32")
        a("        %dlt = arith.cmpi slt, %d0, %dmin : i32")
        a("        %d = arith.select %dlt, %dmin, %d0 : i32")
        a("        %num = arith.subi %d, %dmin : i32")
        a("        %num64 = arith.extsi %num : i32 to i64")
        a("        %nn = arith.muli %num64, %smnm1 : i64")
        a("        %posn = arith.shli %nn, %idxf64 : i64")
        a("        %pos = arith.divsi %posn, %ndmin64 : i64")
        a("        %i0r = arith.shrsi %pos, %idxf64 : i64")
        a("        %i0a = arith.maxsi %i0r, %z64 : i64")
        a("        %i0 = arith.minsi %i0a, %smnm2 : i64")
        a("        %i0sh = arith.shli %i0, %idxf64 : i64")
        a("        %frac = arith.subi %pos, %i0sh : i64")
        a("        %i0idx = arith.index_cast %i0 : i64 to index")
        a("        %i1idx = arith.addi %i0idx, %c1 : index")
        a(f"        %y0 = memref.load %SM[%i0idx] : memref<{sm_size}xi{sb}>")
        a(f"        %y1 = memref.load %SM[%i1idx] : memref<{sm_size}xi{sb}>")
        a(f"        %y064 = arith.extsi %y0 : i{sb} to i64")
        a(f"        %y164 = arith.extsi %y1 : i{sb} to i64")
        a("        %dy = arith.subi %y164, %y064 : i64")
        a("        %mdf = arith.muli %dy, %frac : i64")
        a("        %sh = arith.shrsi %mdf, %idxf64 : i64")
        a("        %e = arith.addi %y064, %sh : i64")
        a("        %e32 = arith.trunci %e : i64 to i32")
        a(f"        memref.store %e32, %exps[%m] : memref<{M}xi32>")
        a("        %ns = arith.addi %sa, %e : i64")
        a("        scf.yield %ns : i64")
        a("      }")
        a(f"      %pfc = arith.constant {sm_pf} : i64")
        a(f"      %qmaxc = arith.constant {qmax} : i64")
        a("      scf.for %m = %c0 to %cM step %c1 {")
        a(f"        %e = memref.load %exps[%m] : memref<{M}xi32>")
        a("        %e64 = arith.extsi %e : i32 to i64")
        a("        %esh = arith.shli %e64, %pfc : i64")
        a("        %p = arith.divsi %esh, %sum : i64")
        a("        %pc = arith.minsi %p, %qmaxc : i64")
        a(f"        %pq = arith.trunci %pc : i64 to i{sb}")
        a("        %yidx = arith.addi %tM, %m : index")
        a(f"        memref.store %pq, %Y[%yidx] : memref<?xi{sb}>")
        a("      }")

    a("  }")
    a("  return")
    a("}")
    return "\n".join(L) + "\n"


def build_shared_lib(mlir_text: str, out_dir: pathlib.Path) -> pathlib.Path:
    missing = [t for t in (*_TOOLS, "gcc") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"MLIR pipeline needs {missing} on PATH")
    out_dir = pathlib.Path(out_dir)
    (out_dir / "rc.mlir").write_text(mlir_text)

    def run(cmd):
        r = subprocess.run(cmd, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[:4000]}")
        return r.stdout

    (out_dir / "rc.llvm.mlir").write_text(
        run(["mlir-opt", str(out_dir / "rc.mlir"), *_LOWER_PASSES])
    )
    (out_dir / "rc.ll").write_text(
        run(
            [
                "mlir-translate",
                "--mlir-to-llvmir",
                str(out_dir / "rc.llvm.mlir"),
            ]
        )
    )
    run(
        [
            "llc",
            "-O3",
            "-relocation-model=pic",
            "-filetype=obj",
            str(out_dir / "rc.ll"),
            "-o",
            str(out_dir / "rc.o"),
        ]
    )
    so = out_dir / "rc.so"
    run(["gcc", "-shared", "-o", str(so), str(out_dir / "rc.o")])
    return so


class _MemRef1D(ctypes.Structure):
    _fields_ = [
        ("alloc", ctypes.c_void_p),
        ("align", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


def _desc(arr):
    p = arr.ctypes.data_as(ctypes.c_void_p)
    return _MemRef1D(p, p, 0, arr.shape[0], 1)


class CompiledSymmetricMLIR:
    def __init__(
        self,
        qmodel: QuantizedModel,
        *,
        head: Optional[str] = None,
        sparse: Optional[str] = None,
    ):
        self.qmodel = qmodel
        self.head = head
        self.sb = qmodel.target.storage_bits
        self._classify = head == "classify"
        self._np_in = {8: np.int8, 16: np.int16, 32: np.int32}[self.sb]
        self._np_out = np.int32 if self._classify else self._np_in
        self._tmp = tempfile.TemporaryDirectory(prefix="rc_mlir_sym_")
        so = build_shared_lib(
            emit_symmetric_mlir(qmodel, head=head, sparse=sparse),
            pathlib.Path(self._tmp.name),
        )
        self._lib = ctypes.CDLL(str(so))
        self._fn = self._lib._mlir_ciface_rc_predict
        self._fn.argtypes = [
            ctypes.c_int64,
            ctypes.POINTER(_MemRef1D),
            ctypes.POINTER(_MemRef1D),
        ]
        self._fn.restype = None

    def predict_q(self, X_q: np.ndarray) -> np.ndarray:
        q = self.qmodel
        X_q = np.ascontiguousarray(X_q, dtype=self._np_in).reshape(-1)
        T = X_q.size // q.K
        Y = np.zeros(T if self._classify else T * q.M, dtype=self._np_out)
        dx, dy = _desc(X_q), _desc(Y)
        self._fn(ctypes.c_int64(T), ctypes.byref(dx), ctypes.byref(dy))
        return Y if self._classify else Y.reshape(T, q.M)
