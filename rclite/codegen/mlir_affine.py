"""MLIR codegen for the affine quantized reservoir (phase-2).

Emits textual MLIR (`func`/`arith`/`memref`/`scf`) that reproduces the exact
integer arithmetic of `_AffineLowerer` / `AffineQuantizedExecutor`, lowers it
with `mlir-opt -> mlir-translate -> llc`, links a host shared library with
gcc, and exposes `predict()` via ctypes. Bit-exactness is guaranteed because
the emitted ops mirror the hand-rolled kernel one-for-one (validated against
the executor in tests).

Feature parity with `_AffineLowerer`:
  - dense (RANDOM/ESN_STANDARD) + structured (DLR/SCR/DLRB) reservoirs
  - optional CSR-sparse dense W_res (`sparse="csr"|"auto"`)
  - identity or integer preprocess
  - DIRECT / LINEAR_INTERP / POLYNOMIAL tanh LUT
  - logits / argmax (classify) / softmax (proba) heads
  - affine i8 / i16, mixed-precision W_out

Toolchain note: link with `gcc` (the `clang` that may be on PATH is the
llvm-mos cross compiler and cannot link host objects).
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
from rclite.quant.affine.quantize import AffineQuantizedModel
from rclite.quant.affine.lut import LUTKind

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


def _flat_i(arr, bits) -> str:
    np_t = {8: np.int8, 16: np.int16, 32: np.int32}[bits]
    flat = np.asarray(arr).reshape(-1).astype(np_t)
    return ", ".join(str(int(v)) for v in flat)


def _global(name, arr, bits) -> str:
    n = int(np.asarray(arr).size)
    return (f'memref.global "private" constant @{name} : memref<{n}xi{bits}> '
            f"= dense<[{_flat_i(arr, bits)}]>")


def emit_affine_mlir(qmodel: AffineQuantizedModel, *,
                     head: Optional[str] = None,
                     sparse: Optional[str] = None) -> str:
    """Emit textual MLIR for an affine quantized reservoir (full feature set)."""
    if head not in _HEADS:
        raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
    rc = qmodel.rc
    cfg = qmodel.config
    art = qmodel.lut_artifacts
    strat = qmodel.lut_strategy
    topo = rc.reservoir.topology
    structured = topo in _STRUCTURED
    if qmodel.M_res_M0_arr is not None or qmodel.M_out_state_M0_arr is not None:
        raise NotImplementedError("MLIR affine: per-channel not yet supported")
    use_sparse = bool(sparse) and not structured

    N, K, M, F = qmodel.N, qmodel.K, qmodel.M, qmodel.F
    sb = qmodel.storage_bits
    wob = qmodel.w_out_storage_bits
    qmin, qmax = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    zp_u = cfg.u_pre.zero_point
    zp_state = cfg.state.zero_point
    zp_pre = cfg.pre.zero_point
    zp_out = cfg.output.zero_point
    zp_input = cfg.input.zero_point
    inc_b = bool(rc.readout.include_bias)
    inc_i = bool(rc.readout.include_input)
    off_b, off_i = 0, (1 if inc_b else 0)
    off_s = off_i + (K if inc_i else 0)
    lut_kind = strat.kind
    lut_size = int(np.asarray(qmodel.lut_q).size)
    int_pre = qmodel.has_integer_preprocess
    classify = head == "classify"
    proba = head == "proba"
    has_logits_buf = classify or proba
    out_bits = 32 if classify else sb

    L: List[str] = []
    a = L.append

    # ---- globals ----
    a(_global("W_in", qmodel.W_in_q, sb))
    a(_global("W_out", qmodel.W_out_q, wob))
    a(_global("rs_in", qmodel.row_sum_W_in, 32))
    a(_global("rs_out_s", qmodel.row_sum_Wout_state, 32))
    if inc_i:
        a(_global("rs_out_i", qmodel.row_sum_Wout_input, 32))
    if not structured:
        if use_sparse:
            from rclite.ir.passes.sparsify import build_csr
            val, col, rptr = build_csr(np.asarray(qmodel.W_res_q))
            a(_global("Wres_val", val, sb))
            a(_global("Wres_col", col, 32))
            a(_global("Wres_rptr", rptr, 32))
        else:
            a(_global("W_res", qmodel.W_res_q, sb))
        a(_global("rs_res", qmodel.row_sum_W_res, 32))
    if lut_kind in (LUTKind.DIRECT, LUTKind.LINEAR_INTERP):
        a(_global("lut", qmodel.lut_q, sb))
    if proba:
        from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params
        sm = build_params(SoftmaxLUTSpec(), s_diff=cfg.output.scale,
                          storage_bits=sb, storage_dtype=np.dtype(f"int{sb}"))
        a(_global("sm_lut", sm.lut_q, sb))
        sm_n, sm_dmin, sm_idxf, sm_pf = sm.n, sm.dmin_q, sm.idx_frac, sm.prob_frac
        sm_size = int(np.asarray(sm.lut_q).size)

    # chain weights (structured), quantized at W_res scale like ir_builder
    if structured:
        def _qchain(v):
            return max(qmin, min(qmax, int(round(float(v) / cfg.W_res.scale))))
        cw_q = _qchain(rc.reservoir.chain_weight)
        cf_q = _qchain(rc.reservoir.chain_feedback)

    # ---- requantize helpers ----
    def rq_fn(name, M0, n):
        a(f"func.func private @{name}(%x: i32) -> i32 {{")
        if M0 == 0:
            a("  %z = arith.constant 0 : i32"); a("  return %z : i32"); a("}")
            return
        a("  %x64 = arith.extsi %x : i32 to i64")
        a(f"  %m0 = arith.constant {M0} : i64")
        a("  %p = arith.muli %x64, %m0 : i64")
        if n > 0:
            a(f"  %hf = arith.constant {1 << (n - 1)} : i64")
            a("  %pb = arith.addi %p, %hf : i64")
            src = "%pb"
        else:
            src = "%p"
        a(f"  %nn = arith.constant {n} : i64")
        a(f"  %s = arith.shrsi {src}, %nn : i64")
        a("  %r = arith.trunci %s : i64 to i32")
        a("  return %r : i32")
        a("}")

    rq_fn("rq_in", qmodel.M_in_M0, qmodel.M_in_n)
    rq_fn("rq_res", qmodel.M_res_M0, qmodel.M_res_n)
    rq_fn("rq_leak", qmodel.leak_M0, qmodel.leak_n)
    rq_fn("rq_ob", qmodel.M_out_bias_M0, qmodel.M_out_bias_n)
    rq_fn("rq_oi", qmodel.M_out_input_M0, qmodel.M_out_input_n)
    rq_fn("rq_os", qmodel.M_out_state_M0, qmodel.M_out_state_n)
    if int_pre:
        rq_fn("rq_pre", qmodel.pre_M0, qmodel.pre_n)
    if lut_kind == LUTKind.LINEAR_INTERP:
        rq_fn("rq_lidx", art.idx_M0, art.idx_n)
    if lut_kind == LUTKind.POLYNOMIAL:
        rq_fn("rq_polyx", art.x_to_qf_M0, art.x_to_qf_n)
        rq_fn("rq_polyb", art.qf_to_state_M0, art.qf_to_state_n)

    # ---- clip i64 -> i32 (saturate) ----
    a("func.func private @clip32(%x: i64) -> i32 {")
    a("  %lo = arith.constant -2147483648 : i64")
    a("  %hi = arith.constant 2147483647 : i64")
    a("  %a = arith.maxsi %x, %lo : i64")
    a("  %b = arith.minsi %a, %hi : i64")
    a("  %r = arith.trunci %b : i64 to i32")
    a("  return %r : i32")
    a("}")

    # ---- saturate i32 -> storage ----
    a(f"func.func private @sat(%x: i32) -> i{sb} {{")
    a(f"  %lo = arith.constant {qmin} : i32")
    a(f"  %hi = arith.constant {qmax} : i32")
    a("  %a = arith.maxsi %x, %lo : i32")
    a("  %b = arith.minsi %a, %hi : i32")
    a(f"  %r = arith.trunci %b : i32 to i{sb}")
    a(f"  return %r : i{sb}")
    a("}")

    # ---- activation helper: storage -> storage ----
    a(f"func.func private @activate(%p: i{sb}) -> i{sb} {{")
    if lut_kind == LUTKind.DIRECT:
        a(f"  %lut = memref.get_global @lut : memref<{lut_size}xi{sb}>")
        a(f"  %p32 = arith.extsi %p : i{sb} to i32")
        a(f"  %off = arith.constant {qmodel.lut_offset} : i32")
        a("  %idx32 = arith.addi %p32, %off : i32")
        a("  %idx = arith.index_cast %idx32 : i32 to index")
        a(f"  %v = memref.load %lut[%idx] : memref<{lut_size}xi{sb}>")
        a(f"  return %v : i{sb}")
    elif lut_kind == LUTKind.LINEAR_INTERP:
        f = strat.interp_frac_bits
        n = strat.n_entries
        a(f"  %lut = memref.get_global @lut : memref<{lut_size}xi{sb}>")
        a(f"  %p32 = arith.extsi %p : i{sb} to i32")
        a(f"  %off = arith.constant {art.offset} : i32")
        a("  %norm = arith.addi %p32, %off : i32")
        a("  %t = func.call @rq_lidx(%norm) : (i32) -> i32")
        a(f"  %fc = arith.constant {f} : i32")
        a("  %idxr = arith.shrsi %t, %fc : i32")
        a("  %z0 = arith.constant 0 : i32")
        a(f"  %nm2 = arith.constant {n - 2} : i32")
        a("  %ge0 = arith.maxsi %idxr, %z0 : i32")
        a("  %idxc = arith.minsi %ge0, %nm2 : i32")
        a("  %ish = arith.shli %idxc, %fc : i32")
        a("  %frac = arith.subi %t, %ish : i32")
        a("  %idx = arith.index_cast %idxc : i32 to index")
        a("  %c1i = arith.constant 1 : index")
        a("  %idx1 = arith.addi %idx, %c1i : index")
        a(f"  %y0 = memref.load %lut[%idx] : memref<{lut_size}xi{sb}>")
        a(f"  %y1 = memref.load %lut[%idx1] : memref<{lut_size}xi{sb}>")
        a(f"  %y032 = arith.extsi %y0 : i{sb} to i32")
        a(f"  %y132 = arith.extsi %y1 : i{sb} to i32")
        a("  %dy = arith.subi %y132, %y032 : i32")
        a("  %dy64 = arith.extsi %dy : i32 to i64")
        a("  %fr64 = arith.extsi %frac : i32 to i64")
        a("  %mul = arith.muli %dy64, %fr64 : i64")
        a(f"  %fc64 = arith.constant {f} : i64")
        a("  %sc = arith.shrsi %mul, %fc64 : i64")
        a("  %sc32 = arith.trunci %sc : i64 to i32")
        a("  %interp = arith.addi %y032, %sc32 : i32")
        a("  %r = func.call @sat(%interp) : (i32) -> i" + str(sb))
        a(f"  return %r : i{sb}")
    else:  # POLYNOMIAL
        qf = strat.poly_qf_bits
        a(f"  %p32 = arith.extsi %p : i{sb} to i32")
        a(f"  %zpp = arith.constant {zp_pre} : i32")
        a("  %cent = arith.subi %p32, %zpp : i32")
        a("  %xq32 = func.call @rq_polyx(%cent) : (i32) -> i32")
        a("  %xq0 = arith.extsi %xq32 : i32 to i64")
        a(f"  %clp = arith.constant {art.x_clip_qf} : i64")
        a(f"  %cln = arith.constant {-art.x_clip_qf} : i64")
        a("  %xqa = arith.maxsi %xq0, %cln : i64")
        a("  %xq = arith.minsi %xqa, %clp : i64")
        a(f"  %qfc = arith.constant {qf} : i64")
        a("  %xx = arith.muli %xq, %xq : i64")
        a("  %x2 = arith.shrsi %xx, %qfc : i64")
        a(f"  %a5 = arith.constant {art.poly_a5_qf} : i64")
        a(f"  %a3 = arith.constant {art.poly_a3_qf} : i64")
        a(f"  %a1 = arith.constant {art.poly_a1_qf} : i64")
        a("  %m5 = arith.muli %x2, %a5 : i64")
        a("  %s5 = arith.shrsi %m5, %qfc : i64")
        a("  %inner = arith.addi %s5, %a3 : i64")
        a("  %mi = arith.muli %x2, %inner : i64")
        a("  %si = arith.shrsi %mi, %qfc : i64")
        a("  %outer = arith.addi %si, %a1 : i64")
        a("  %my = arith.muli %xq, %outer : i64")
        a("  %yq0 = arith.shrsi %my, %qfc : i64")
        a(f"  %onep = arith.constant {art.one_qf} : i64")
        a(f"  %onen = arith.constant {-art.one_qf} : i64")
        a("  %yqa = arith.maxsi %yq0, %onen : i64")
        a("  %yq = arith.minsi %yqa, %onep : i64")
        a("  %yq32 = arith.trunci %yq : i64 to i32")
        a("  %delta = func.call @rq_polyb(%yq32) : (i32) -> i32")
        a(f"  %zps = arith.constant {zp_state} : i32")
        a("  %tot = arith.addi %delta, %zps : i32")
        a("  %r = func.call @sat(%tot) : (i32) -> i" + str(sb))
        a(f"  return %r : i{sb}")
    a("}")

    # ---- main ----
    a(f"func.func @rc_predict(%T: i64, %X: memref<?xi{sb}>, "
      f"%Y: memref<?xi{out_bits}>) attributes {{llvm.emit_c_interface}} {{")
    a("  %c0 = arith.constant 0 : index")
    a("  %c1 = arith.constant 1 : index")
    a(f"  %cN = arith.constant {N} : index")
    a(f"  %cK = arith.constant {K} : index")
    a(f"  %cM = arith.constant {M} : index")
    a("  %z64 = arith.constant 0 : i64")
    a(f"  %zps32 = arith.constant {zp_state} : i32")
    a(f"  %zps = arith.constant {zp_state} : i{sb}")
    a("  %Ti = arith.index_cast %T : i64 to index")
    a(f"  %Win = memref.get_global @W_in : memref<{N*K}xi{sb}>")
    a(f"  %Wout = memref.get_global @W_out : memref<{M*F}xi{wob}>")
    a(f"  %rsIn = memref.get_global @rs_in : memref<{N}xi32>")
    a(f"  %rsOS = memref.get_global @rs_out_s : memref<{M}xi32>")
    if inc_i:
        a(f"  %rsOI = memref.get_global @rs_out_i : memref<{M}xi32>")
    if not structured:
        a(f"  %rsRes = memref.get_global @rs_res : memref<{N}xi32>")
        if use_sparse:
            a(f"  %WrV = memref.get_global @Wres_val : memref<{val.size}xi{sb}>")
            a(f"  %WrC = memref.get_global @Wres_col : memref<{col.size}xi32>")
            a(f"  %WrP = memref.get_global @Wres_rptr : memref<{rptr.size}xi32>")
        else:
            a(f"  %Wres = memref.get_global @W_res : memref<{N*N}xi{sb}>")
    a(f"  %h = memref.alloca() : memref<{N}xi{sb}>")
    a(f"  %pre = memref.alloca() : memref<{N}xi{sb}>")
    if int_pre:
        a(f"  %upre = memref.alloca() : memref<{K}xi{sb}>")
    if has_logits_buf:
        a(f"  %logits = memref.alloca() : memref<{M}xi{sb}>")
    if proba:
        a(f"  %exps = memref.alloca() : memref<{M}xi32>")
    a("  scf.for %i = %c0 to %cN step %c1 {")
    a(f"    memref.store %zps, %h[%i] : memref<{N}xi{sb}>")
    a("  }")

    # time loop
    a("  scf.for %t = %c0 to %Ti step %c1 {")
    a("    %tK = arith.muli %t, %cK : index")
    a("    %tM = arith.muli %t, %cM : index")

    # --- integer preprocess: u_pre[k] ---
    if int_pre:
        a("    scf.for %k = %c0 to %cK step %c1 {")
        a("      %xidx = arith.addi %tK, %k : index")
        a(f"      %xq = memref.load %X[%xidx] : memref<?xi{sb}>")
        a(f"      %xq32 = arith.extsi %xq : i{sb} to i32")
        a(f"      %zpi = arith.constant {zp_input} : i32")
        a("      %cent = arith.subi %xq32, %zpi : i32")
        a("      %d = func.call @rq_pre(%cent) : (i32) -> i32")
        a(f"      %pc = arith.constant {qmodel.pre_const} : i32")
        a("      %tot = arith.addi %d, %pc : i32")
        a("      %uq = func.call @sat(%tot) : (i32) -> i" + str(sb))
        a(f"      memref.store %uq, %upre[%k] : memref<{K}xi{sb}>")
        a("    }")

    # --- pre-activation: for i in 0..N ---
    a("    scf.for %i = %c0 to %cN step %c1 {")
    # acc_in
    a("      %accin = scf.for %k = %c0 to %cK step %c1 "
      "iter_args(%ai = %z64) -> (i64) {")
    a("        %iKin = arith.muli %i, %cK : index")
    a("        %widx = arith.addi %iKin, %k : index")
    a(f"        %w = memref.load %Win[%widx] : memref<{N*K}xi{sb}>")
    if int_pre:
        a(f"        %x = memref.load %upre[%k] : memref<{K}xi{sb}>")
    else:
        a("        %xidx = arith.addi %tK, %k : index")
        a(f"        %x = memref.load %X[%xidx] : memref<?xi{sb}>")
    a(f"        %w64 = arith.extsi %w : i{sb} to i64")
    a(f"        %x64 = arith.extsi %x : i{sb} to i64")
    a("        %pr = arith.muli %w64, %x64 : i64")
    a("        %na = arith.addi %ai, %pr : i64")
    a("        scf.yield %na : i64")
    a("      }")
    a(f"      %rsi = memref.load %rsIn[%i] : memref<{N}xi32>")
    a("      %rsi64 = arith.extsi %rsi : i32 to i64")
    a(f"      %zpu = arith.constant {zp_u} : i64")
    a("      %zrin = arith.muli %zpu, %rsi64 : i64")
    a("      %adjin = arith.subi %accin, %zrin : i64")
    a("      %cin = func.call @clip32(%adjin) : (i64) -> i32")
    a("      %rqin = func.call @rq_in(%cin) : (i32) -> i32")

    # acc_res (dense / csr / structured)
    if structured:
        # acc_res_i32 = chain contribution (already i32)
        if topo == Topology.SCR:
            a(f"      %cw = arith.constant {cw_q} : i32")
            a("      %iz = arith.cmpi eq, %i, %c0 : index")
            a(f"      %nm1 = arith.constant {N - 1} : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %iprev = arith.select %iz, %nm1, %im1 : index")
            a(f"      %hv = memref.load %h[%iprev] : memref<{N}xi{sb}>")
            a(f"      %hv32 = arith.extsi %hv : i{sb} to i32")
            a("      %hc = arith.subi %hv32, %zps32 : i32")
            a("      %accres = arith.muli %cw, %hc : i32")
        elif topo == Topology.DLR:
            a(f"      %cw = arith.constant {cw_q} : i32")
            a("      %ipos = arith.cmpi sgt, %i, %c0 : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %isafe = arith.select %ipos, %im1, %c0 : index")
            a(f"      %hv = memref.load %h[%isafe] : memref<{N}xi{sb}>")
            a(f"      %hv32 = arith.extsi %hv : i{sb} to i32")
            a("      %hc = arith.subi %hv32, %zps32 : i32")
            a("      %prod = arith.muli %cw, %hc : i32")
            a("      %z32 = arith.constant 0 : i32")
            a("      %accres = arith.select %ipos, %prod, %z32 : i32")
        else:  # DLRB
            a(f"      %cw = arith.constant {cw_q} : i32")
            a(f"      %cfk = arith.constant {cf_q} : i32")
            a("      %z32 = arith.constant 0 : i32")
            a(f"      %nm1 = arith.constant {N - 1} : index")
            a("      %ipos = arith.cmpi sgt, %i, %c0 : index")
            a("      %im1 = arith.subi %i, %c1 : index")
            a("      %ib = arith.select %ipos, %im1, %c0 : index")
            a(f"      %hb = memref.load %h[%ib] : memref<{N}xi{sb}>")
            a(f"      %hb32 = arith.extsi %hb : i{sb} to i32")
            a("      %hbc = arith.subi %hb32, %zps32 : i32")
            a("      %pb = arith.muli %cw, %hbc : i32")
            a("      %cb = arith.select %ipos, %pb, %z32 : i32")
            a("      %ilt = arith.cmpi slt, %i, %nm1 : index")
            a("      %ip1 = arith.addi %i, %c1 : index")
            a("      %iff = arith.select %ilt, %ip1, %nm1 : index")
            a(f"      %hf = memref.load %h[%iff] : memref<{N}xi{sb}>")
            a(f"      %hf32 = arith.extsi %hf : i{sb} to i32")
            a("      %hfc = arith.subi %hf32, %zps32 : i32")
            a("      %pf = arith.muli %cfk, %hfc : i32")
            a("      %cfwd = arith.select %ilt, %pf, %z32 : i32")
            a("      %accres = arith.addi %cb, %cfwd : i32")
        a("      %rqres = func.call @rq_res(%accres) : (i32) -> i32")
    else:
        if use_sparse:
            a("      %rp0 = memref.load %WrP[%i] : memref<" + f"{rptr.size}xi32>")
            a("      %ip1 = arith.addi %i, %c1 : index")
            a("      %rp1 = memref.load %WrP[%ip1] : memref<" + f"{rptr.size}xi32>")
            a("      %rp0i = arith.index_cast %rp0 : i32 to index")
            a("      %rp1i = arith.index_cast %rp1 : i32 to index")
            a("      %accr = scf.for %p = %rp0i to %rp1i step %c1 "
              "iter_args(%ar = %z64) -> (i64) {")
            a(f"        %w = memref.load %WrV[%p] : memref<{val.size}xi{sb}>")
            a(f"        %cj = memref.load %WrC[%p] : memref<{col.size}xi32>")
            a("        %cji = arith.index_cast %cj : i32 to index")
            a(f"        %hv = memref.load %h[%cji] : memref<{N}xi{sb}>")
            a(f"        %w64 = arith.extsi %w : i{sb} to i64")
            a(f"        %h64 = arith.extsi %hv : i{sb} to i64")
            a("        %pr = arith.muli %w64, %h64 : i64")
            a("        %na = arith.addi %ar, %pr : i64")
            a("        scf.yield %na : i64")
            a("      }")
        else:
            a("      %iN = arith.muli %i, %cN : index")
            a("      %accr = scf.for %j = %c0 to %cN step %c1 "
              "iter_args(%ar = %z64) -> (i64) {")
            a("        %widx = arith.addi %iN, %j : index")
            a(f"        %w = memref.load %Wres[%widx] : memref<{N*N}xi{sb}>")
            a(f"        %hv = memref.load %h[%j] : memref<{N}xi{sb}>")
            a(f"        %w64 = arith.extsi %w : i{sb} to i64")
            a(f"        %h64 = arith.extsi %hv : i{sb} to i64")
            a("        %pr = arith.muli %w64, %h64 : i64")
            a("        %na = arith.addi %ar, %pr : i64")
            a("        scf.yield %na : i64")
            a("      }")
        a(f"      %rsr = memref.load %rsRes[%i] : memref<{N}xi32>")
        a("      %rsr64 = arith.extsi %rsr : i32 to i64")
        a(f"      %zpst = arith.constant {zp_state} : i64")
        a("      %zrres = arith.muli %zpst, %rsr64 : i64")
        a("      %adjres = arith.subi %accr, %zrres : i64")
        a("      %cres = func.call @clip32(%adjres) : (i64) -> i32")
        a("      %rqres = func.call @rq_res(%cres) : (i32) -> i32")

    a(f"      %zpp = arith.constant {zp_pre + qmodel.bias_pre} : i32")
    a("      %s1 = arith.addi %zpp, %rqin : i32")
    a("      %s2 = arith.addi %s1, %rqres : i32")
    a("      %preq = func.call @sat(%s2) : (i32) -> i" + str(sb))
    a(f"      memref.store %preq, %pre[%i] : memref<{N}xi{sb}>")
    a("    }")

    # --- activation + leaky integration ---
    a("    scf.for %i = %c0 to %cN step %c1 {")
    a(f"      %p = memref.load %pre[%i] : memref<{N}xi{sb}>")
    a("      %act = func.call @activate(%p) : (i" + str(sb) + ") -> i" + str(sb))
    a("      %act32 = arith.extsi %act : i" + str(sb) + " to i32")
    a(f"      %hold = memref.load %h[%i] : memref<{N}xi{sb}>")
    a("      %hold32 = arith.extsi %hold : i" + str(sb) + " to i32")
    a("      %hc = arith.subi %hold32, %zps32 : i32")
    a("      %ac = arith.subi %act32, %zps32 : i32")
    a("      %diff = arith.subi %ac, %hc : i32")
    a("      %delta = func.call @rq_leak(%diff) : (i32) -> i32")
    a("      %nhc = arith.addi %hc, %delta : i32")
    a("      %nh = arith.addi %nhc, %zps32 : i32")
    a("      %nhq = func.call @sat(%nh) : (i32) -> i" + str(sb))
    a(f"      memref.store %nhq, %h[%i] : memref<{N}xi{sb}>")
    a("    }")

    # --- readout: for m in 0..M ---
    a("    scf.for %m = %c0 to %cM step %c1 {")
    a(f"      %cF = arith.constant {F} : index")
    a("      %mF = arith.muli %m, %cF : index")
    a(f"      %zpo = arith.constant {zp_out} : i32")
    if inc_b:
        a(f"      %obcol = arith.constant {off_b} : index")
        a("      %obidx = arith.addi %mF, %obcol : index")
        a(f"      %w0 = memref.load %Wout[%obidx] : memref<{M*F}xi{wob}>")
        a(f"      %w064 = arith.extsi %w0 : i{wob} to i64")
        a("      %cb = func.call @clip32(%w064) : (i64) -> i32")
        a("      %rqb = func.call @rq_ob(%cb) : (i32) -> i32")
        a("      %yb = arith.addi %zpo, %rqb : i32")
    yb = "%yb" if inc_b else "%zpo"
    if inc_i:
        a(f"      %ci = arith.constant {off_i} : index")
        a("      %accoi = scf.for %k = %c0 to %cK step %c1 "
          "iter_args(%ao = %z64) -> (i64) {")
        a("        %coff = arith.addi %ci, %k : index")
        a("        %widx = arith.addi %mF, %coff : index")
        a(f"        %w = memref.load %Wout[%widx] : memref<{M*F}xi{wob}>")
        a("        %xidx = arith.addi %tK, %k : index")
        a(f"        %x = memref.load %X[%xidx] : memref<?xi{sb}>")
        a(f"        %w64 = arith.extsi %w : i{wob} to i64")
        a(f"        %x64 = arith.extsi %x : i{sb} to i64")
        a("        %pr = arith.muli %w64, %x64 : i64")
        a("        %na = arith.addi %ao, %pr : i64")
        a("        scf.yield %na : i64")
        a("      }")
        a(f"      %rsoi = memref.load %rsOI[%m] : memref<{M}xi32>")
        a("      %rsoi64 = arith.extsi %rsoi : i32 to i64")
        a(f"      %zpin = arith.constant {zp_input} : i64")
        a("      %zroi = arith.muli %zpin, %rsoi64 : i64")
        a("      %adjoi = arith.subi %accoi, %zroi : i64")
        a("      %coi = func.call @clip32(%adjoi) : (i64) -> i32")
        a("      %rqoi = func.call @rq_oi(%coi) : (i32) -> i32")
        a(f"      %yi = arith.addi {yb}, %rqoi : i32")
    yi = "%yi" if inc_i else yb
    a(f"      %cs = arith.constant {off_s} : index")
    a("      %accos = scf.for %j = %c0 to %cN step %c1 "
      "iter_args(%ao = %z64) -> (i64) {")
    a("        %coff = arith.addi %cs, %j : index")
    a("        %widx = arith.addi %mF, %coff : index")
    a(f"        %w = memref.load %Wout[%widx] : memref<{M*F}xi{wob}>")
    a(f"        %hv = memref.load %h[%j] : memref<{N}xi{sb}>")
    a(f"        %w64 = arith.extsi %w : i{wob} to i64")
    a(f"        %h64 = arith.extsi %hv : i{sb} to i64")
    a("        %pr = arith.muli %w64, %h64 : i64")
    a("        %na = arith.addi %ao, %pr : i64")
    a("        scf.yield %na : i64")
    a("      }")
    a(f"      %rsos = memref.load %rsOS[%m] : memref<{M}xi32>")
    a("      %rsos64 = arith.extsi %rsos : i32 to i64")
    a(f"      %zpst2 = arith.constant {zp_state} : i64")
    a("      %zros = arith.muli %zpst2, %rsos64 : i64")
    a("      %adjos = arith.subi %accos, %zros : i64")
    a("      %cos = func.call @clip32(%adjos) : (i64) -> i32")
    a("      %rqos = func.call @rq_os(%cos) : (i32) -> i32")
    a(f"      %ys = arith.addi {yi}, %rqos : i32")
    a("      %yq = func.call @sat(%ys) : (i32) -> i" + str(sb))
    if has_logits_buf:
        a(f"      memref.store %yq, %logits[%m] : memref<{M}xi{sb}>")
    else:
        a("      %yidx = arith.addi %tM, %m : index")
        a(f"      memref.store %yq, %Y[%yidx] : memref<?xi{sb}>")
    a("    }")

    # --- head ---
    if classify:
        a(f"      %bv0 = memref.load %logits[%c0] : memref<{M}xi{sb}>")
        a("      %best:2 = scf.for %m = %c1 to %cM step %c1 "
          f"iter_args(%bv = %bv0, %bi = %c0) -> (i{sb}, index) {{")
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %gt = arith.cmpi sgt, %v, %bv : i{sb}")
        a(f"        %nv = arith.select %gt, %v, %bv : i{sb}")
        a("        %ni = arith.select %gt, %m, %bi : index")
        a(f"        scf.yield %nv, %ni : i{sb}, index")
        a("      }")
        a("      %cls = arith.index_cast %best#1 : index to i32")
        a("      memref.store %cls, %Y[%t] : memref<?xi32>")
    elif proba:
        # max over logits (i32)
        a(f"      %mx0 = memref.load %logits[%c0] : memref<{M}xi{sb}>")
        a(f"      %mx032 = arith.extsi %mx0 : i{sb} to i32")
        a("      %mx = scf.for %m = %c1 to %cM step %c1 "
          "iter_args(%mxa = %mx032) -> (i32) {")
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %v32 = arith.extsi %v : i{sb} to i32")
        a("        %gt = arith.cmpi sgt, %v32, %mxa : i32")
        a("        %nm = arith.select %gt, %v32, %mxa : i32")
        a("        scf.yield %nm : i32")
        a("      }")
        a(f"      %dmin = arith.constant {sm_dmin} : i32")
        a(f"      %ndmin64 = arith.constant {-sm_dmin} : i64")
        a(f"      %nm1c = arith.constant {sm_n - 1} : i64")
        a(f"      %idxf64 = arith.constant {sm_idxf} : i64")
        a(f"      %nm2i = arith.constant {sm_n - 2} : i64")
        a("      %sum = scf.for %m = %c0 to %cM step %c1 "
          "iter_args(%sa = %z64) -> (i64) {")
        a(f"        %v = memref.load %logits[%m] : memref<{M}xi{sb}>")
        a(f"        %v32 = arith.extsi %v : i{sb} to i32")
        a("        %d0 = arith.subi %v32, %mx : i32")
        a("        %dlt = arith.cmpi slt, %d0, %dmin : i32")
        a("        %d = arith.select %dlt, %dmin, %d0 : i32")
        a("        %num = arith.subi %d, %dmin : i32")
        a("        %num64 = arith.extsi %num : i32 to i64")
        a("        %nn = arith.muli %num64, %nm1c : i64")
        a("        %posn = arith.shli %nn, %idxf64 : i64")
        a("        %pos = arith.divsi %posn, %ndmin64 : i64")
        a("        %i0r = arith.shrsi %pos, %idxf64 : i64")
        a("        %i0a = arith.maxsi %i0r, %z64 : i64")
        a("        %i0 = arith.minsi %i0a, %nm2i : i64")
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

    a("  }")  # end time loop
    a("  return")
    a("}")
    out = "\n".join(L) + "\n"
    if proba:
        # %SM get_global is referenced in the proba head; declare it once near top
        out = out.replace(
            "  %Ti = arith.index_cast %T : i64 to index",
            f"  %SM = memref.get_global @sm_lut : memref<{sm_size}xi{sb}>\n"
            "  %Ti = arith.index_cast %T : i64 to index", 1)
    return out


# ---------------------------------------------------------------------------
# Toolchain driver

def tools_available() -> bool:
    return all(shutil.which(t) for t in _TOOLS) and shutil.which("gcc") is not None


def build_shared_lib(mlir_text: str, out_dir: pathlib.Path) -> pathlib.Path:
    missing = [t for t in (*_TOOLS, "gcc") if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"MLIR pipeline needs {missing} on PATH")
    out_dir = pathlib.Path(out_dir)
    (out_dir / "rc.mlir").write_text(mlir_text)

    def run(cmd, **kw):
        r = subprocess.run(cmd, capture_output=True, text=True, **kw)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[:4000]}")
        return r.stdout

    lowered = run(["mlir-opt", str(out_dir / "rc.mlir"), *_LOWER_PASSES])
    (out_dir / "rc.llvm.mlir").write_text(lowered)
    llvm_ir = run(["mlir-translate", "--mlir-to-llvmir",
                   str(out_dir / "rc.llvm.mlir")])
    (out_dir / "rc.ll").write_text(llvm_ir)
    run(["llc", "-O3", "-relocation-model=pic", "-filetype=obj",
         str(out_dir / "rc.ll"), "-o", str(out_dir / "rc.o")])
    so = out_dir / "rc.so"
    run(["gcc", "-shared", "-o", str(so), str(out_dir / "rc.o")])
    return so


def cross_compile_object(mlir_text: str, *, triple: str, cpu: str = "",
                         features: str = "") -> bytes:
    """Cross-compile MLIR to a target object for `triple` (no host link).

    Connects the MLIR path to the embedded targets — emits a relocatable
    object for e.g. thumbv6m (Cortex-M0), thumbv4t (GBA), wasm32 (WASM),
    the same triples the llvmlite path serves. Uses the same (working) host
    lowering and only retargets `llc`; this proves the MLIR path reaches the
    target backends (object emission), not a packaged firmware.

    NOTE: the integer kernel is emitted scalar — saturating/wrapping
    quantized arithmetic is non-associative, so SIMD vectorization would
    break the host↔device bit-exactness (the existing WASM quantized target
    disables vectorization for the same reason). `features` (e.g. "+simd128")
    only selects the target ISA; the kernel stays scalar.
    """
    missing = [t for t in _TOOLS if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"MLIR cross-compile needs {missing} on PATH")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "rc.mlir").write_text(mlir_text)

        def run(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[:3000]}")
            return r.stdout

        (td / "rc.ll.mlir").write_text(
            run(["mlir-opt", str(td / "rc.mlir"), *_LOWER_PASSES]))
        (td / "rc.ll").write_text(
            run(["mlir-translate", "--mlir-to-llvmir", str(td / "rc.ll.mlir")]))
        llc = ["llc", "-O2", f"-mtriple={triple}", "-filetype=obj",
               str(td / "rc.ll"), "-o", str(td / "rc.o")]
        if cpu:
            llc.append(f"-mcpu={cpu}")
        if features:
            llc.append(f"-mattr={features}")
        run(llc)
        return (td / "rc.o").read_bytes()


class _MemRef1D(ctypes.Structure):
    _fields_ = [("alloc", ctypes.c_void_p), ("align", ctypes.c_void_p),
                ("offset", ctypes.c_int64),
                ("size", ctypes.c_int64), ("stride", ctypes.c_int64)]


def _desc(arr: np.ndarray) -> _MemRef1D:
    p = arr.ctypes.data_as(ctypes.c_void_p)
    return _MemRef1D(p, p, 0, arr.shape[0], 1)


class CompiledAffineMLIR:
    """Compile a (constrained) affine qmodel via MLIR and run it via ctypes."""

    def __init__(self, qmodel: AffineQuantizedModel, *,
                 head: Optional[str] = None, sparse: Optional[str] = None):
        self.qmodel = qmodel
        self.head = head
        self.sb = qmodel.storage_bits
        self._classify = head == "classify"
        self._np_in = {8: np.int8, 16: np.int16}[self.sb]
        self._np_out = np.int32 if self._classify else self._np_in
        self._tmp = tempfile.TemporaryDirectory(prefix="rc_mlir_")
        so = build_shared_lib(
            emit_affine_mlir(qmodel, head=head, sparse=sparse),
            pathlib.Path(self._tmp.name))
        self._lib = ctypes.CDLL(str(so))
        self._fn = self._lib._mlir_ciface_rc_predict
        self._fn.argtypes = [ctypes.c_int64,
                             ctypes.POINTER(_MemRef1D), ctypes.POINTER(_MemRef1D)]
        self._fn.restype = None

    def predict_q(self, X_q: np.ndarray) -> np.ndarray:
        q = self.qmodel
        X_q = np.ascontiguousarray(X_q, dtype=self._np_in).reshape(-1)
        T = X_q.size // q.K
        out_len = T if self._classify else T * q.M
        Y = np.zeros(out_len, dtype=self._np_out)
        dx, dy = _desc(X_q), _desc(Y)
        self._fn(ctypes.c_int64(T), ctypes.byref(dx), ctypes.byref(dy))
        return Y if self._classify else Y.reshape(T, q.M)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        return self.predict_q(self.qmodel.config.input.quantize_array(X))
