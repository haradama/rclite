"""xDSL-built MLIR for the affine quantized reservoir.

Stage-(1) "IR construction" migration of `mlir_affine.py`: the same
func/arith/memref/scf IR that mirrors `_AffineLowerer` / `AffineQuantizedExecutor`,
assembled with the xDSL Python API instead of f-string text emission. The
printed MLIR is fed to the unchanged mlir-opt -> mlir-translate -> llc -> gcc
pipeline (`mlir_affine.build_shared_lib`), so it is bit-exact by construction.

Feature parity with the text emitter: dense + structured (DLR/SCR/DLRB) + CSR
sparse, identity or integer preprocess, DIRECT / LINEAR_INTERP / POLYNOMIAL tanh
LUT, logits / classify / proba heads, affine i8/i16 with mixed-precision W_out.
Per-channel (per-axis) requantize is supported via `emit_requantize_axis_func`:
per-row (M0[i], n[i]) for the W_res step and per-output (M0[m], n[m]) for the
W_out bias/input/state readout blocks. Lower with an LLVM-20 mlir-opt (the nix
devShell) — the IR is printed in generic op form.
"""

from __future__ import annotations
import io
from typing import Optional

import numpy as np

from xdsl.builder import ImplicitBuilder
from xdsl.ir import Region, Block
from xdsl.dialects import arith, memref, func
from xdsl.dialects.builtin import (
    ModuleOp,
    IntegerType,
    MemRefType,
    UnitAttr,
)
from xdsl.printer import Printer

from rclite.core.profile import Topology
from rclite.quant.affine.quantize import AffineQuantizedModel
from rclite.quant.affine.lut import LUTKind

from .mlir_xdsl_common import (
    _STRUCTURED,
    _HEADS,
    _IDX,
    _I32,
    _I64,
    _dense_global,
    c_i,
    c_idx,
    ext,
    call,
    for_,
)
from .mlir_quant_xdsl import (
    emit_sat_func,
    emit_clip32_func,
    emit_requantize_func,
    emit_requantize_axis_func,
    int_matvec_acc,
    zp_cross,
    emit_argmax_head,
    emit_softmax_head,
)


def emit_affine_mlir_xdsl(
    qmodel: AffineQuantizedModel,
    *,
    head: Optional[str] = None,
    sparse: Optional[str] = None,
    vlen: int = 1,
) -> str:
    if head not in _HEADS:
        raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
    rc, cfg = qmodel.rc, qmodel.config
    art, strat = qmodel.lut_artifacts, qmodel.lut_strategy
    topo = rc.reservoir.topology
    structured = topo in _STRUCTURED
    # Per-channel (per-axis) requantize: the step (W_res) and readout (W_out
    # state block) carry per-row (M0[i], n[i]) multipliers instead of a scalar.
    # Mirrors `_AffineLowerer`: per-row W_res only applies to dense/sparse, not
    # the structured chain; per-row readout applies whenever per_channel_W_out.
    pc_res = qmodel.M_res_M0_arr is not None and not structured
    pc_out = qmodel.M_out_state_M0_arr is not None
    use_sparse = bool(sparse) and not structured

    N, K, M, F = qmodel.N, qmodel.K, qmodel.M, qmodel.F
    sb = qmodel.storage_bits
    wob = qmodel.w_out_storage_bits
    isb, iwob = IntegerType(sb), IntegerType(wob)
    qmin, qmax = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    # Vectorized W_res matvec accumulator width (vlen>1). i8 W*h products are
    # <= 2^14, so an i32 sum can't overflow for N up to ~131072 -> i32 (native
    # vpmulld/vpaddd, 8 lanes, FAST). i16 multiplies in i32 (product <= 2^30,
    # exact) and accumulates in i64; LLVM fuses this to vpmaddwd whose i32
    # pair-sum stays exact iff |W_res_q| <= 32767 (then |W*h| <= 32767*32768 <
    # 2^30, pair-sum < 2^31, even if the state saturates). The affine symmetric
    # quant keeps W_res_q in [-32767, 32767]; guard it so the bit-exactness is
    # provable, not just empirical, and fall back to scalar otherwise.
    matvec_acc_bits = 32 if (sb == 8 and N <= (1 << 16)) else 64
    matvec_vec_safe = sb == 8 or (
        qmodel.W_res_q is not None
        and int(np.asarray(qmodel.W_res_q).min()) >= -32767
    )
    zp_u = cfg.u_pre.zero_point
    zp_state = cfg.state.zero_point
    zp_pre = cfg.pre.zero_point
    zp_out = cfg.output.zero_point
    zp_input = cfg.input.zero_point
    inc_b, inc_i = (
        bool(rc.readout.include_bias),
        bool(rc.readout.include_input),
    )
    off_b, off_i = 0, (1 if inc_b else 0)
    off_s = off_i + (K if inc_i else 0)
    # Readout W_out_state·h matvec vectorization (vlen>1). W_out is iwob, h is
    # isb (mixed precision). i32 accumulate only when both are i8; else i64
    # (i32-multiply -> i64-accumulate). vpmaddwd is safe when W_out is i8
    # (products tiny) or |W_out_state_q| <= 32767.
    ro_acc_bits = 32 if (wob == 8 and sb == 8) else 64
    ro_vec_safe = wob == 8 or (
        int(np.asarray(qmodel.W_out_q)[:, off_s : off_s + N].min()) >= -32767
    )
    lut_kind = strat.kind
    lut_size = int(np.asarray(qmodel.lut_q).size)
    int_pre = qmodel.has_integer_preprocess
    classify = head == "classify"
    proba = head == "proba"
    has_logits_buf = classify or proba
    out_bits = 32 if classify else sb
    if structured:

        def _qchain(v):
            return max(qmin, min(qmax, int(round(float(v) / cfg.W_res.scale))))

        cw_q, cf_q = (
            _qchain(rc.reservoir.chain_weight),
            _qchain(rc.reservoir.chain_feedback),
        )
    if use_sparse:
        from rclite.ir.passes.sparsify import build_csr

        wr_val, wr_col, wr_rptr = build_csr(np.asarray(qmodel.W_res_q))
    if proba:
        from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params

        sm = build_params(
            SoftmaxLUTSpec(),
            s_diff=cfg.output.scale,
            storage_bits=sb,
            storage_dtype=np.dtype(f"int{sb}"),
        )
        sm_n, sm_dmin, sm_idxf, sm_pf = (
            sm.n,
            sm.dmin_q,
            sm.idx_frac,
            sm.prob_frac,
        )
        sm_size = int(np.asarray(sm.lut_q).size)

    # ---- globals ----
    globals_ = [
        _dense_global("W_in", qmodel.W_in_q, sb),
        _dense_global("W_out", qmodel.W_out_q, wob),
        _dense_global("rs_in", qmodel.row_sum_W_in, 32),
        _dense_global("rs_out_s", qmodel.row_sum_Wout_state, 32),
    ]
    if inc_i:
        globals_.append(
            _dense_global("rs_out_i", qmodel.row_sum_Wout_input, 32)
        )
    if not structured:
        if use_sparse:
            globals_ += [
                _dense_global("Wres_val", wr_val, sb),
                _dense_global("Wres_col", wr_col, 32),
                _dense_global("Wres_rptr", wr_rptr, 32),
            ]
        else:
            globals_.append(_dense_global("W_res", qmodel.W_res_q, sb))
        globals_.append(_dense_global("rs_res", qmodel.row_sum_W_res, 32))
    if lut_kind in (LUTKind.DIRECT, LUTKind.LINEAR_INTERP):
        globals_.append(_dense_global("lut", qmodel.lut_q, sb))
    if proba:
        globals_.append(_dense_global("sm_lut", sm.lut_q, sb))
    if pc_res:
        globals_ += [
            _dense_global("rq_res_M0", qmodel.M_res_M0_arr, 32),
            _dense_global("rq_res_n", qmodel.M_res_n_arr, 32),
        ]
    if pc_out:
        # per_channel_W_out makes ALL readout blocks (bias/input/state) per-axis
        # over the M output rows (mirrors AffineQuantizedExecutor.predict_one_q).
        globals_ += [
            _dense_global("rq_os_M0", qmodel.M_out_state_M0_arr, 32),
            _dense_global("rq_os_n", qmodel.M_out_state_n_arr, 32),
        ]
        if inc_b:
            globals_ += [
                _dense_global("rq_ob_M0", qmodel.M_out_bias_M0_arr, 32),
                _dense_global("rq_ob_n", qmodel.M_out_bias_n_arr, 32),
            ]
        if inc_i:
            globals_ += [
                _dense_global("rq_oi_M0", qmodel.M_out_input_M0_arr, 32),
                _dense_global("rq_oi_n", qmodel.M_out_input_n_arr, 32),
            ]

    funcs = []

    # ---- requantize funcs: (x:i32) -> i32 via (M0,n) round-shift ----
    def rq_func(name, M0, n):
        funcs.append(emit_requantize_func(name, M0, n))

    rq_func("rq_in", qmodel.M_in_M0, qmodel.M_in_n)
    if pc_res:
        funcs.append(
            emit_requantize_axis_func("rq_res", "rq_res_M0", "rq_res_n", N)
        )
    else:
        rq_func("rq_res", qmodel.M_res_M0, qmodel.M_res_n)
    rq_func("rq_leak", qmodel.leak_M0, qmodel.leak_n)
    # readout requantize: per-axis (one (M0,n) per output row m) when
    # per_channel_W_out, scalar otherwise. bias/input only when their block is on.
    if pc_out and inc_b:
        funcs.append(
            emit_requantize_axis_func("rq_ob", "rq_ob_M0", "rq_ob_n", M)
        )
    else:
        rq_func("rq_ob", qmodel.M_out_bias_M0, qmodel.M_out_bias_n)
    if pc_out and inc_i:
        funcs.append(
            emit_requantize_axis_func("rq_oi", "rq_oi_M0", "rq_oi_n", M)
        )
    else:
        rq_func("rq_oi", qmodel.M_out_input_M0, qmodel.M_out_input_n)
    if pc_out:
        funcs.append(
            emit_requantize_axis_func("rq_os", "rq_os_M0", "rq_os_n", M)
        )
    else:
        rq_func("rq_os", qmodel.M_out_state_M0, qmodel.M_out_state_n)
    if int_pre:
        rq_func("rq_pre", qmodel.pre_M0, qmodel.pre_n)
    if lut_kind == LUTKind.LINEAR_INTERP:
        rq_func("rq_lidx", art.idx_M0, art.idx_n)
    if lut_kind == LUTKind.POLYNOMIAL:
        rq_func("rq_polyx", art.x_to_qf_M0, art.x_to_qf_n)
        rq_func("rq_polyb", art.qf_to_state_M0, art.qf_to_state_n)

    # ---- clip32: i64 -> i32 (saturate) ----
    funcs.append(emit_clip32_func())

    # ---- sat: i32 -> storage (saturate) ----
    funcs.append(emit_sat_func(qmin, qmax, isb))

    # ---- activate: storage -> storage (DIRECT / LINEAR_INTERP / POLYNOMIAL) ----
    r = Region([Block(arg_types=[isb])])
    with ImplicitBuilder(r.block) as (p,):
        if lut_kind == LUTKind.DIRECT:
            lut = memref.GetGlobalOp("lut", MemRefType(isb, [lut_size])).memref
            idx32 = arith.AddiOp(
                arith.ExtSIOp(p, _I32).result, c_i(qmodel.lut_offset, _I32)
            ).result
            idx = arith.IndexCastOp(idx32, _IDX).result
            func.ReturnOp(memref.LoadOp.get(lut, [idx]).res)
        elif lut_kind == LUTKind.LINEAR_INTERP:
            f, ne = strat.interp_frac_bits, strat.n_entries
            lut = memref.GetGlobalOp("lut", MemRefType(isb, [lut_size])).memref
            norm = arith.AddiOp(
                arith.ExtSIOp(p, _I32).result, c_i(art.offset, _I32)
            ).result
            t = call("rq_lidx", [norm], _I32)
            idxr = arith.ShRSIOp(t, c_i(f, _I32)).result
            idxc = arith.MinSIOp(
                arith.MaxSIOp(idxr, c_i(0, _I32)).result, c_i(ne - 2, _I32)
            ).result
            frac = arith.SubiOp(
                t, arith.ShLIOp(idxc, c_i(f, _I32)).result
            ).result
            idx = arith.IndexCastOp(idxc, _IDX).result
            idx1 = arith.AddiOp(idx, c_idx(1)).result
            y0 = arith.ExtSIOp(memref.LoadOp.get(lut, [idx]).res, _I32).result
            y1 = arith.ExtSIOp(memref.LoadOp.get(lut, [idx1]).res, _I32).result
            dy = arith.SubiOp(y1, y0).result
            mul = arith.MuliOp(ext(dy), ext(frac)).result
            sc = arith.TruncIOp(
                arith.ShRSIOp(mul, c_i(f, _I64)).result, _I32
            ).result
            interp = arith.AddiOp(y0, sc).result
            func.ReturnOp(call("sat", [interp], isb))
        else:  # POLYNOMIAL
            qf = strat.poly_qf_bits
            cent = arith.SubiOp(
                arith.ExtSIOp(p, _I32).result, c_i(zp_pre, _I32)
            ).result
            xq32 = call("rq_polyx", [cent], _I32)
            xq = arith.MinSIOp(
                arith.MaxSIOp(ext(xq32), c_i(-art.x_clip_qf, _I64)).result,
                c_i(art.x_clip_qf, _I64),
            ).result
            qfc = c_i(qf, _I64)
            x2 = arith.ShRSIOp(arith.MuliOp(xq, xq).result, qfc).result
            m5 = arith.ShRSIOp(
                arith.MuliOp(x2, c_i(art.poly_a5_qf, _I64)).result, qfc
            ).result
            inner = arith.AddiOp(m5, c_i(art.poly_a3_qf, _I64)).result
            si = arith.ShRSIOp(arith.MuliOp(x2, inner).result, qfc).result
            outer = arith.AddiOp(si, c_i(art.poly_a1_qf, _I64)).result
            yq0 = arith.ShRSIOp(arith.MuliOp(xq, outer).result, qfc).result
            yq = arith.MinSIOp(
                arith.MaxSIOp(yq0, c_i(-art.one_qf, _I64)).result,
                c_i(art.one_qf, _I64),
            ).result
            delta = call("rq_polyb", [arith.TruncIOp(yq, _I32).result], _I32)
            tot = arith.AddiOp(delta, c_i(zp_state, _I32)).result
            func.ReturnOp(call("sat", [tot], isb))
    funcs.append(
        func.FuncOp("activate", ([isb], [isb]), r, visibility="private")
    )

    # ---- main ----
    dyn_x = MemRefType(isb, [memref.DYNAMIC_INDEX])
    dyn_y = MemRefType(IntegerType(out_bits), [memref.DYNAMIC_INDEX])
    main_r = Region([Block(arg_types=[_I64, dyn_x, dyn_y])])
    with ImplicitBuilder(main_r.block) as (T, X, Y):
        cN, cK, cM, cF = c_idx(N), c_idx(K), c_idx(M), c_idx(F)
        c0, c1 = c_idx(0), c_idx(1)
        z64 = c_i(0, _I64)
        zps32 = c_i(zp_state, _I32)
        zps = c_i(zp_state, isb)
        Ti = arith.IndexCastOp(T, _IDX).result
        Win = memref.GetGlobalOp("W_in", MemRefType(isb, [N * K])).memref
        Wout = memref.GetGlobalOp("W_out", MemRefType(iwob, [M * F])).memref
        rsIn = memref.GetGlobalOp("rs_in", MemRefType(_I32, [N])).memref
        rsOS = memref.GetGlobalOp("rs_out_s", MemRefType(_I32, [M])).memref
        rsOI = (
            memref.GetGlobalOp("rs_out_i", MemRefType(_I32, [M])).memref
            if inc_i
            else None
        )
        if not structured:
            rsRes = memref.GetGlobalOp("rs_res", MemRefType(_I32, [N])).memref
            if use_sparse:
                WrV = memref.GetGlobalOp(
                    "Wres_val", MemRefType(isb, [wr_val.size])
                ).memref
                WrC = memref.GetGlobalOp(
                    "Wres_col", MemRefType(_I32, [wr_col.size])
                ).memref
                WrP = memref.GetGlobalOp(
                    "Wres_rptr", MemRefType(_I32, [wr_rptr.size])
                ).memref
            else:
                Wres = memref.GetGlobalOp(
                    "W_res", MemRefType(isb, [N * N])
                ).memref
        SM = (
            memref.GetGlobalOp("sm_lut", MemRefType(isb, [sm_size])).memref
            if proba
            else None
        )
        h = memref.AllocaOp.get(isb, shape=[N]).memref
        pre = memref.AllocaOp.get(isb, shape=[N]).memref
        upre = memref.AllocaOp.get(isb, shape=[K]).memref if int_pre else None
        logits = (
            memref.AllocaOp.get(isb, shape=[M]).memref
            if has_logits_buf
            else None
        )
        exps = memref.AllocaOp.get(_I32, shape=[M]).memref if proba else None

        def init_body(i, _):
            memref.StoreOp.get(zps, h, [i])
            return []

        for_(c0, cN, c1, [], init_body)

        # acc_res -> rqres (i32), given pre-activation index i
        def acc_res(i):
            if structured and topo == Topology.SCR:
                cw = c_i(cw_q, _I32)
                iz = arith.CmpiOp(i, c0, "eq").result
                iprev = arith.SelectOp(
                    iz, c_idx(N - 1), arith.SubiOp(i, c1).result
                ).result
                hv32 = arith.ExtSIOp(
                    memref.LoadOp.get(h, [iprev]).res, _I32
                ).result
                hc = arith.SubiOp(hv32, zps32).result
                accres = arith.MuliOp(cw, hc).result
            elif structured and topo == Topology.DLR:
                cw = c_i(cw_q, _I32)
                ipos = arith.CmpiOp(i, c0, "sgt").result
                isafe = arith.SelectOp(
                    ipos, arith.SubiOp(i, c1).result, c0
                ).result
                hv32 = arith.ExtSIOp(
                    memref.LoadOp.get(h, [isafe]).res, _I32
                ).result
                hc = arith.SubiOp(hv32, zps32).result
                prod = arith.MuliOp(cw, hc).result
                accres = arith.SelectOp(ipos, prod, c_i(0, _I32)).result
            elif structured:  # DLRB
                cw, cfk = c_i(cw_q, _I32), c_i(cf_q, _I32)
                z32 = c_i(0, _I32)
                nm1 = c_idx(N - 1)
                ipos = arith.CmpiOp(i, c0, "sgt").result
                ib = arith.SelectOp(
                    ipos, arith.SubiOp(i, c1).result, c0
                ).result
                hb32 = arith.ExtSIOp(
                    memref.LoadOp.get(h, [ib]).res, _I32
                ).result
                pb = arith.MuliOp(cw, arith.SubiOp(hb32, zps32).result).result
                cb = arith.SelectOp(ipos, pb, z32).result
                ilt = arith.CmpiOp(i, nm1, "slt").result
                iff = arith.SelectOp(
                    ilt, arith.AddiOp(i, c1).result, nm1
                ).result
                hf32 = arith.ExtSIOp(
                    memref.LoadOp.get(h, [iff]).res, _I32
                ).result
                pf = arith.MuliOp(cfk, arith.SubiOp(hf32, zps32).result).result
                cfwd = arith.SelectOp(ilt, pf, z32).result
                accres = arith.AddiOp(cb, cfwd).result
            else:
                if use_sparse:
                    rp0i = arith.IndexCastOp(
                        memref.LoadOp.get(WrP, [i]).res, _IDX
                    ).result
                    rp1i = arith.IndexCastOp(
                        memref.LoadOp.get(
                            WrP, [arith.AddiOp(i, c1).result]
                        ).res,
                        _IDX,
                    ).result

                    def pbody(p, args):
                        (ar,) = args
                        w = memref.LoadOp.get(WrV, [p]).res
                        cj = arith.IndexCastOp(
                            memref.LoadOp.get(WrC, [p]).res, _IDX
                        ).result
                        hv = memref.LoadOp.get(h, [cj]).res
                        pr = arith.MuliOp(ext(w), ext(hv)).result
                        return [arith.AddiOp(ar, pr).result]

                    accr = for_(rp0i, rp1i, c1, [z64], pbody)[0]
                else:
                    iN = arith.MuliOp(i, cN).result
                    if vlen > 1 and matvec_vec_safe:
                        # Vectorise the i64 W_res matvec reduction. The i64 sum
                        # is associative (no mid saturation/overflow), so this is
                        # BIT-EXACT with the scalar order — quantized SIMD that
                        # keeps host<->device equality. The non-associative
                        # zp_cross/requantize below stay scalar (they are per-row,
                        # after the reduction).
                        accr = int_matvec_acc(
                            Wres, iN, h, N, vlen, z64, isb, matvec_acc_bits
                        )
                    else:

                        def jbody(j, args):
                            (ar,) = args
                            w = memref.LoadOp.get(
                                Wres, [arith.AddiOp(iN, j).result]
                            ).res
                            hv = memref.LoadOp.get(h, [j]).res
                            pr = arith.MuliOp(ext(w), ext(hv)).result
                            return [arith.AddiOp(ar, pr).result]

                        accr = for_(c0, cN, c1, [z64], jbody)[0]
                accres = zp_cross(
                    accr, zp_state, memref.LoadOp.get(rsRes, [i]).res
                )
            if pc_res:
                return call("rq_res", [accres, i], _I32)
            return call("rq_res", [accres], _I32)

        def time_body(t, _):
            tK = arith.MuliOp(t, cK).result
            tM = arith.MuliOp(t, cM).result

            # integer preprocess -> upre[k]
            if int_pre:

                def pre_in(k, _):
                    xq = memref.LoadOp.get(X, [arith.AddiOp(tK, k).result]).res
                    cent = arith.SubiOp(
                        arith.ExtSIOp(xq, _I32).result, c_i(zp_input, _I32)
                    ).result
                    d = call("rq_pre", [cent], _I32)
                    tot = arith.AddiOp(d, c_i(qmodel.pre_const, _I32)).result
                    memref.StoreOp.get(call("sat", [tot], isb), upre, [k])
                    return []

                for_(c0, cK, c1, [], pre_in)

            # pre-activation
            def pre_body(i, _):
                def kin(k, args):
                    (ai,) = args
                    widx = arith.AddiOp(arith.MuliOp(i, cK).result, k).result
                    w = memref.LoadOp.get(Win, [widx]).res
                    if int_pre:
                        xv = memref.LoadOp.get(upre, [k]).res
                    else:
                        xv = memref.LoadOp.get(
                            X, [arith.AddiOp(tK, k).result]
                        ).res
                    pr = arith.MuliOp(ext(w), ext(xv)).result
                    return [arith.AddiOp(ai, pr).result]

                accin = for_(c0, cK, c1, [z64], kin)[0]
                cin = zp_cross(accin, zp_u, memref.LoadOp.get(rsIn, [i]).res)
                rqin = call("rq_in", [cin], _I32)
                rqres = acc_res(i)
                zpp = c_i(zp_pre + qmodel.bias_pre, _I32)
                s2 = arith.AddiOp(arith.AddiOp(zpp, rqin).result, rqres).result
                memref.StoreOp.get(call("sat", [s2], isb), pre, [i])
                return []

            for_(c0, cN, c1, [], pre_body)

            # activation + leaky integration
            def act_body(i, _):
                p = memref.LoadOp.get(pre, [i]).res
                act32 = arith.ExtSIOp(call("activate", [p], isb), _I32).result
                hold32 = arith.ExtSIOp(
                    memref.LoadOp.get(h, [i]).res, _I32
                ).result
                hc = arith.SubiOp(hold32, zps32).result
                ac = arith.SubiOp(act32, zps32).result
                delta = call("rq_leak", [arith.SubiOp(ac, hc).result], _I32)
                nh = arith.AddiOp(arith.AddiOp(hc, delta).result, zps32).result
                memref.StoreOp.get(call("sat", [nh], isb), h, [i])
                return []

            for_(c0, cN, c1, [], act_body)

            # readout
            def ro_body(m, _):
                mF = arith.MuliOp(m, cF).result
                zpo = c_i(zp_out, _I32)
                if inc_b:
                    w0 = memref.LoadOp.get(
                        Wout, [arith.AddiOp(mF, c_idx(off_b)).result]
                    ).res
                    cb = call("clip32", [ext(w0)], _I32)
                    rq_ob = (
                        call("rq_ob", [cb, m], _I32)
                        if pc_out
                        else call("rq_ob", [cb], _I32)
                    )
                    yb = arith.AddiOp(zpo, rq_ob).result
                else:
                    yb = zpo
                if inc_i:
                    ci = c_idx(off_i)

                    def kbody(k, args):
                        (ao,) = args
                        widx = arith.AddiOp(
                            mF, arith.AddiOp(ci, k).result
                        ).result
                        w = memref.LoadOp.get(Wout, [widx]).res
                        xv = memref.LoadOp.get(
                            X, [arith.AddiOp(tK, k).result]
                        ).res
                        return [
                            arith.AddiOp(
                                ao, arith.MuliOp(ext(w), ext(xv)).result
                            ).result
                        ]

                    accoi = for_(c0, cK, c1, [z64], kbody)[0]
                    coi = zp_cross(
                        accoi, zp_input, memref.LoadOp.get(rsOI, [m]).res
                    )
                    rq_oi = (
                        call("rq_oi", [coi, m], _I32)
                        if pc_out
                        else call("rq_oi", [coi], _I32)
                    )
                    yi = arith.AddiOp(yb, rq_oi).result
                else:
                    yi = yb
                cs = c_idx(off_s)
                if vlen > 1 and ro_vec_safe:
                    base_s = arith.AddiOp(mF, cs).result
                    accos = int_matvec_acc(
                        Wout,
                        base_s,
                        h,
                        N,
                        vlen,
                        z64,
                        iwob,
                        ro_acc_bits,
                        h_storage_ty=isb,
                    )
                else:

                    def jbody(j, args):
                        (ao,) = args
                        widx = arith.AddiOp(
                            mF, arith.AddiOp(cs, j).result
                        ).result
                        w = memref.LoadOp.get(Wout, [widx]).res
                        hv = memref.LoadOp.get(h, [j]).res
                        return [
                            arith.AddiOp(
                                ao, arith.MuliOp(ext(w), ext(hv)).result
                            ).result
                        ]

                    accos = for_(c0, cN, c1, [z64], jbody)[0]
                cos = zp_cross(
                    accos, zp_state, memref.LoadOp.get(rsOS, [m]).res
                )
                if pc_out:
                    rq_os = call("rq_os", [cos, m], _I32)
                else:
                    rq_os = call("rq_os", [cos], _I32)
                ys = arith.AddiOp(yi, rq_os).result
                yq = call("sat", [ys], isb)
                if has_logits_buf:
                    memref.StoreOp.get(yq, logits, [m])
                else:
                    memref.StoreOp.get(yq, Y, [arith.AddiOp(tM, m).result])
                return []

            for_(c0, cM, c1, [], ro_body)

            # head
            if classify:
                emit_argmax_head(logits, Y, t, c0, c1, cM)
            elif proba:
                emit_softmax_head(
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
                )
            return []

        for_(c0, Ti, c1, [], time_body)
        func.ReturnOp()
    main_fn = func.FuncOp("rc_predict", ([_I64, dyn_x, dyn_y], []), main_r)
    main_fn.attributes["llvm.emit_c_interface"] = UnitAttr()

    mod = ModuleOp([*globals_, *funcs, main_fn])
    if vlen <= 1:
        # xDSL 0.66's arith.extsi IRDL rejects vector operands (stricter than
        # MLIR, which allows elementwise extsi on vectors). The vlen>1 path emits
        # vector<vlen x i*> extsi, which is valid MLIR — verified downstream by
        # mlir-opt — so skip the over-strict in-process verifier for it.
        mod.verify()
    buf = io.StringIO()
    Printer(stream=buf).print_op(mod)
    return buf.getvalue() + "\n"
