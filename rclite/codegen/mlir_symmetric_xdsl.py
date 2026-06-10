"""xDSL-built MLIR for the symmetric (Q-format) quantized reservoir.

This is the stage-(1) "IR construction" migration of `mlir_symmetric.py`: the
exact same arith/memref/scf/func IR, but assembled **programmatically with the
xDSL Python API** instead of f-string text emission. The printed MLIR is fed to
the unchanged `mlir-opt -> mlir-translate -> llc -> gcc` pipeline
(`mlir_symmetric.build_shared_lib`), so it stays bit-exact with the executor by
construction (identical ops -> identical lowering).

Coverage matches the text emitter: dense (RANDOM/ESN_STANDARD) + structured
(DLR/SCR/DLRB) + CSR-sparse, identity preprocess, i8/i16 storage,
logits/argmax(classify)/softmax(proba) heads.

NB: the printed IR uses generic op form; lower it with an LLVM-20 mlir-opt (the
nix devShell) — some other builds choke on it.
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
from rclite.quant.model import QuantizedModel

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
    fmul,
    for_,
)


def emit_symmetric_mlir_xdsl(
    qmodel: QuantizedModel,
    *,
    head: Optional[str] = None,
    sparse: Optional[str] = None,
) -> str:
    if head not in _HEADS:
        raise ValueError(f"head must be one of {_HEADS}, got {head!r}")
    rc, cfg = qmodel.rc, qmodel.config
    topo = rc.reservoir.topology
    structured = topo in _STRUCTURED
    if rc.input.input_offset != 0.0 or rc.input.input_scaling != 1.0:
        raise NotImplementedError("xDSL: identity preprocess only")
    if qmodel.lut_table_q is None:
        raise NotImplementedError("xDSL: tanh LUT required")
    use_sparse = bool(sparse) and not structured

    N, K, M, F = qmodel.N, qmodel.K, qmodel.M, qmodel.F
    sb = qmodel.target.storage_bits
    if sb not in (8, 16):
        raise NotImplementedError("xDSL: i8/i16 storage only (i32 TODO)")
    isb = IntegerType(sb)
    sf = cfg.state_frac
    shift_in = cfg.weight_frac + cfg.input_frac - cfg.state_frac
    shift_res = cfg.weight_frac
    if shift_in < 0:
        raise NotImplementedError(f"xDSL needs shift_in>=0 ({shift_in})")
    state_scale = 1 << sf
    bias_q = int(qmodel.target.quantize_state(float(rc.reservoir.bias), cfg))
    leak_q = int(
        qmodel.target.quantize_state(float(rc.reservoir.leak_rate), cfg)
    )
    one_ml_q = state_scale - leak_q
    lut_n = int(np.asarray(qmodel.lut_table_q).size)
    xmin_q = int(qmodel.lut.xmin * state_scale)
    xmax_q = int(qmodel.lut.xmax * state_scale)
    denom = xmax_q - xmin_q
    inc_b, inc_i = (
        bool(rc.readout.include_bias),
        bool(rc.readout.include_input),
    )
    off_i = 1 if inc_b else 0
    off_s = off_i + (K if inc_i else 0)
    qmin, qmax = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    classify = head == "classify"
    proba = head == "proba"
    has_logits_buf = classify or proba
    out_bits = 32 if classify else sb
    if structured:
        wsc = 1 << cfg.weight_frac
        cw_q = int(round(float(rc.reservoir.chain_weight) * wsc))
        cf_q = int(round(float(rc.reservoir.chain_feedback) * wsc))
    if use_sparse:
        from rclite.ir.passes.sparsify import build_csr

        wr_val, wr_col, wr_rptr = build_csr(np.asarray(qmodel.W_res_q))
    if proba:
        from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params

        sm = build_params(
            SoftmaxLUTSpec(),
            s_diff=1.0 / state_scale,
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
        _dense_global("W_out", qmodel.W_out_q, sb),
        _dense_global("lut", qmodel.lut_table_q, sb),
    ]
    if not structured:
        if use_sparse:
            globals_ += [
                _dense_global("Wres_val", wr_val, sb),
                _dense_global("Wres_col", wr_col, 32),
                _dense_global("Wres_rptr", wr_rptr, 32),
            ]
        else:
            globals_.append(_dense_global("W_res", qmodel.W_res_q, sb))
    if proba:
        globals_.append(_dense_global("sm_lut", sm.lut_q, sb))

    # ---- @sat(i32) -> isb ----
    sat_r = Region([Block(arg_types=[_I32])])
    with ImplicitBuilder(sat_r.block) as (x,):
        b = arith.MinSIOp(
            arith.MaxSIOp(x, c_i(qmin, _I32)), c_i(qmax, _I32)
        ).result
        func.ReturnOp(arith.TruncIOp(b, isb).result)
    sat_fn = func.FuncOp("sat", ([_I32], [isb]), sat_r, visibility="private")

    # ---- @activate(isb) -> isb : interpolating tanh LUT ----
    act_r = Region([Block(arg_types=[isb])])
    with ImplicitBuilder(act_r.block) as (p,):
        lut = memref.GetGlobalOp("lut", MemRefType(isb, [lut_n])).memref
        x0 = arith.ExtSIOp(p, _I32).result
        x = arith.MinSIOp(
            arith.MaxSIOp(x0, c_i(xmin_q, _I32)), c_i(xmax_q, _I32)
        ).result
        num = arith.SubiOp(x, c_i(xmin_q, _I32)).result
        nsh = arith.ShLIOp(ext(num), c_i(sf, _I64)).result
        tq64 = arith.DivSIOp(nsh, c_i(denom, _I64)).result
        tq = arith.TruncIOp(tq64, _I32).result
        pos64 = arith.MuliOp(ext(tq), ext(c_i(lut_n - 1, _I32))).result
        posq = arith.TruncIOp(pos64, _I32).result
        i0r = arith.ShRSIOp(posq, c_i(sf, _I32)).result
        i0 = arith.MinSIOp(
            arith.MaxSIOp(i0r, c_i(0, _I32)), c_i(lut_n - 2, _I32)
        ).result
        i0sh = arith.ShLIOp(i0, c_i(sf, _I32)).result
        frac = arith.SubiOp(posq, i0sh).result
        i0i = arith.IndexCastOp(i0, _IDX).result
        i1i = arith.AddiOp(i0i, c_idx(1)).result
        y0 = arith.ExtSIOp(memref.LoadOp.get(lut, [i0i]).res, _I32).result
        y1 = arith.ExtSIOp(memref.LoadOp.get(lut, [i1i]).res, _I32).result
        dy = arith.SubiOp(y1, y0).result
        dfp = arith.MuliOp(ext(dy), ext(frac)).result
        dfs = arith.ShRSIOp(dfp, c_i(sf, _I64)).result
        res = arith.AddiOp(y0, arith.TruncIOp(dfs, _I32).result).result
        func.ReturnOp(arith.TruncIOp(res, isb).result)
    act_fn = func.FuncOp(
        "activate", ([isb], [isb]), act_r, visibility="private"
    )

    # ---- @rc_predict(T:i64, X:memref<?xisb>, Y:memref<?x i{out_bits}>) ----
    dyn_x = MemRefType(isb, [memref.DYNAMIC_INDEX])
    dyn_y = MemRefType(IntegerType(out_bits), [memref.DYNAMIC_INDEX])
    main_r = Region([Block(arg_types=[_I64, dyn_x, dyn_y])])
    with ImplicitBuilder(main_r.block) as (T, X, Y):
        cN, cK, cM, cF = c_idx(N), c_idx(K), c_idx(M), c_idx(F)
        c0, c1 = c_idx(0), c_idx(1)
        z32, z64 = c_i(0, _I32), c_i(0, _I64)
        Ti = arith.IndexCastOp(T, _IDX).result
        Win = memref.GetGlobalOp("W_in", MemRefType(isb, [N * K])).memref
        Wout = memref.GetGlobalOp("W_out", MemRefType(isb, [M * F])).memref
        if not structured:
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
        if proba:
            SM = memref.GetGlobalOp(
                "sm_lut", MemRefType(isb, [sm_size])
            ).memref
        h = memref.AllocaOp.get(isb, shape=[N]).memref
        pre = memref.AllocaOp.get(isb, shape=[N]).memref
        logits = (
            memref.AllocaOp.get(isb, shape=[M]).memref
            if has_logits_buf
            else None
        )
        exps = memref.AllocaOp.get(_I32, shape=[M]).memref if proba else None
        zsb = c_i(0, isb)

        # acc_res: add the W_res contribution to acc_in, returning the i32 acc.
        def acc_res(i, accin):
            if structured and topo == Topology.SCR:
                cw = c_i(cw_q, isb)
                iz = arith.CmpiOp(i, c0, "eq").result
                iprev = arith.SelectOp(
                    iz, c_idx(N - 1), arith.SubiOp(i, c1).result
                ).result
                hv = memref.LoadOp.get(h, [iprev]).res
                return arith.AddiOp(accin, fmul(cw, hv, shift_res)).result
            if structured and topo == Topology.DLR:
                cw = c_i(cw_q, isb)
                ipos = arith.CmpiOp(i, c0, "sgt").result
                isafe = arith.SelectOp(
                    ipos, arith.SubiOp(i, c1).result, c0
                ).result
                hv = memref.LoadOp.get(h, [isafe]).res
                trsel = arith.SelectOp(
                    ipos, fmul(cw, hv, shift_res), z32
                ).result
                return arith.AddiOp(accin, trsel).result
            if structured:  # DLRB
                cw, cfk = c_i(cw_q, isb), c_i(cf_q, isb)
                nm1 = c_idx(N - 1)
                ipos = arith.CmpiOp(i, c0, "sgt").result
                ib = arith.SelectOp(
                    ipos, arith.SubiOp(i, c1).result, c0
                ).result
                hb = memref.LoadOp.get(h, [ib]).res
                tbsel = arith.SelectOp(
                    ipos, fmul(cw, hb, shift_res), z32
                ).result
                ilt = arith.CmpiOp(i, nm1, "slt").result
                iff = arith.SelectOp(
                    ilt, arith.AddiOp(i, c1).result, nm1
                ).result
                hf = memref.LoadOp.get(h, [iff]).res
                tfsel = arith.SelectOp(
                    ilt, fmul(cfk, hf, shift_res), z32
                ).result
                return arith.AddiOp(
                    arith.AddiOp(accin, tbsel).result, tfsel
                ).result
            if use_sparse:
                rp0 = memref.LoadOp.get(WrP, [i]).res
                rp1 = memref.LoadOp.get(WrP, [arith.AddiOp(i, c1).result]).res
                rp0i = arith.IndexCastOp(rp0, _IDX).result
                rp1i = arith.IndexCastOp(rp1, _IDX).result

                def pbody(p, args):
                    (ar,) = args
                    w = memref.LoadOp.get(WrV, [p]).res
                    cj = arith.IndexCastOp(
                        memref.LoadOp.get(WrC, [p]).res, _IDX
                    ).result
                    hv = memref.LoadOp.get(h, [cj]).res
                    return [arith.AddiOp(ar, fmul(w, hv, shift_res)).result]

                return for_(rp0i, rp1i, c1, [accin], pbody)[0]
            # dense
            iN = arith.MuliOp(i, cN).result

            def jbody(j, args):
                (ar,) = args
                w = memref.LoadOp.get(Wres, [arith.AddiOp(iN, j).result]).res
                hv = memref.LoadOp.get(h, [j]).res
                return [arith.AddiOp(ar, fmul(w, hv, shift_res)).result]

            return for_(c0, cN, c1, [accin], jbody)[0]

        def init_body(i, _):
            memref.StoreOp.get(zsb, h, [i])
            return []

        for_(c0, cN, c1, [], init_body)

        def time_body(t, _):
            tK = arith.MuliOp(t, cK).result
            tM = arith.MuliOp(t, cM).result

            # pre-activation
            def pre_body(i, _):
                def kin(k, args):
                    (ai,) = args
                    widx = arith.AddiOp(arith.MuliOp(i, cK).result, k).result
                    w = memref.LoadOp.get(Win, [widx]).res
                    xv = memref.LoadOp.get(X, [arith.AddiOp(tK, k).result]).res
                    return [arith.AddiOp(ai, fmul(w, xv, shift_in)).result]

                accin = for_(c0, cK, c1, [c_i(bias_q, _I32)], kin)[0]
                accres = acc_res(i, accin)
                memref.StoreOp.get(
                    arith.TruncIOp(accres, isb).result, pre, [i]
                )
                return []

            for_(c0, cN, c1, [], pre_body)

            # activation + leaky integration
            def act_body(i, _):
                p = memref.LoadOp.get(pre, [i]).res
                act = func.CallOp("activate", [p], [isb]).res[0]
                hold = memref.LoadOp.get(h, [i]).res
                t1 = arith.TruncIOp(
                    fmul(hold, c_i(one_ml_q, isb), sf), isb
                ).result
                t2 = arith.TruncIOp(
                    fmul(act, c_i(leak_q, isb), sf), isb
                ).result
                memref.StoreOp.get(arith.AddiOp(t1, t2).result, h, [i])
                return []

            for_(c0, cN, c1, [], act_body)

            # readout: i64 accumulate -> >> sf -> saturate
            def ro_body(m, _):
                mF = arith.MuliOp(m, cF).result
                if inc_b:
                    w0 = memref.LoadOp.get(
                        Wout, [arith.AddiOp(mF, c0).result]
                    ).res
                    init = arith.MuliOp(c_i(state_scale, _I64), ext(w0)).result
                else:
                    init = z64
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
                        pr = arith.MuliOp(ext(w), ext(xv)).result
                        return [arith.AddiOp(ao, pr).result]

                    init = for_(c0, cK, c1, [init], kbody)[0]
                cs = c_idx(off_s)

                def jbody(j, args):
                    (ao,) = args
                    widx = arith.AddiOp(mF, arith.AddiOp(cs, j).result).result
                    w = memref.LoadOp.get(Wout, [widx]).res
                    hv = memref.LoadOp.get(h, [j]).res
                    pr = arith.MuliOp(ext(w), ext(hv)).result
                    return [arith.AddiOp(ao, pr).result]

                accos = for_(c0, cN, c1, [init], jbody)[0]
                shifted = arith.ShRSIOp(accos, c_i(sf, _I64)).result
                cl = arith.MinSIOp(
                    arith.MaxSIOp(shifted, c_i(qmin, _I64)), c_i(qmax, _I64)
                ).result
                yq = arith.TruncIOp(cl, isb).result
                if has_logits_buf:
                    memref.StoreOp.get(yq, logits, [m])
                else:
                    memref.StoreOp.get(yq, Y, [arith.AddiOp(tM, m).result])
                return []

            for_(c0, cM, c1, [], ro_body)

            # head
            if classify:
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
                cls = arith.IndexCastOp(best[1], _I32).result
                memref.StoreOp.get(cls, Y, [t])
            elif proba:
                mx0 = arith.ExtSIOp(
                    memref.LoadOp.get(logits, [c0]).res, _I32
                ).result

                def fmax(m, args):
                    (mxa,) = args
                    v = arith.ExtSIOp(
                        memref.LoadOp.get(logits, [m]).res, _I32
                    ).result
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
                    v = arith.ExtSIOp(
                        memref.LoadOp.get(logits, [m]).res, _I32
                    ).result
                    d0 = arith.SubiOp(v, mx).result
                    dlt = arith.CmpiOp(d0, dmin, "slt").result
                    d = arith.SelectOp(dlt, dmin, d0).result
                    num = arith.SubiOp(d, dmin).result
                    nn = arith.MuliOp(ext(num), smnm1).result
                    posn = arith.ShLIOp(nn, idxf64).result
                    pos = arith.DivSIOp(posn, ndmin64).result
                    i0r = arith.ShRSIOp(pos, idxf64).result
                    i0 = arith.MinSIOp(
                        arith.MaxSIOp(i0r, z64).result, smnm2
                    ).result
                    i0sh = arith.ShLIOp(i0, idxf64).result
                    frac = arith.SubiOp(pos, i0sh).result
                    i0idx = arith.IndexCastOp(i0, _IDX).result
                    i1idx = arith.AddiOp(i0idx, c1).result
                    y0 = ext(memref.LoadOp.get(SM, [i0idx]).res)
                    y1 = ext(memref.LoadOp.get(SM, [i1idx]).res)
                    dy = arith.SubiOp(y1, y0).result
                    mdf = arith.MuliOp(dy, frac).result
                    sh = arith.ShRSIOp(mdf, idxf64).result
                    e = arith.AddiOp(y0, sh).result
                    memref.StoreOp.get(
                        arith.TruncIOp(e, _I32).result, exps, [m]
                    )
                    return [arith.AddiOp(sa, e).result]

                total = for_(c0, cM, c1, [z64], sbody)[0]
                pfc = c_i(sm_pf, _I64)
                qmaxc = c_i(qmax, _I64)

                def pbody(m, _):
                    e = memref.LoadOp.get(exps, [m]).res
                    esh = arith.ShLIOp(ext(e), pfc).result
                    p = arith.DivSIOp(esh, total).result
                    pc = arith.MinSIOp(p, qmaxc).result
                    pq = arith.TruncIOp(pc, isb).result
                    memref.StoreOp.get(pq, Y, [arith.AddiOp(tM, m).result])
                    return []

                for_(c0, cM, c1, [], pbody)
            return []

        for_(c0, Ti, c1, [], time_body)
        func.ReturnOp()
    main_fn = func.FuncOp("rc_predict", ([_I64, dyn_x, dyn_y], []), main_r)
    main_fn.attributes["llvm.emit_c_interface"] = UnitAttr()

    mod = ModuleOp([*globals_, sat_fn, act_fn, main_fn])
    mod.verify()
    buf = io.StringIO()
    Printer(stream=buf).print_op(mod)
    return buf.getvalue() + "\n"
