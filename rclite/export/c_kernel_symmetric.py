"""Portable-C emitter for the **symmetric Q-format** quantized kernel.

Bit-exact with `rclite.quant.QuantizedExecutor` (the Python reference for
the I32/I16/I8 mirage-style path). Mirrors `_IntLowerer` in C:

  * pre-activation: per-row accumulator with two's-complement i32 wrap and
    `(W * x) >> shift` fixed-point multiplies
  * tanh via linear-interpolated LUT (mirage `tanh_lut_q`)
  * leaky integration at the state scale
  * mirage mixed-scale readout (bias / input passthrough / state blocks),
    accumulated in i64 and shifted back by `state_frac`
  * structured topologies (SCR/DLR/DLRB) use a scalar chain (the quantized
    chain weight equals the dense W_res entry, so this matches the
    executor's sparse-over-dense matmul exactly); dense otherwise

Weights/LUT live in Flash via PROGMEM on AVR. The truncation/wrap uses
unsigned arithmetic so it is well-defined C (no signed-overflow UB) while
reproducing numpy's `trunc_i32` / `astype` wrap behaviour.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

from rclite.core.profile import Topology
from rclite.quant.model import QuantizedModel


_C_TYPE = {8: "int8_t", 16: "int16_t", 32: "int32_t"}
_RD = {8: "RC_RD8", 16: "RC_RD16", 32: "RC_RD32"}


def _arr(name: str, ctype: str, values) -> str:
    body = ", ".join(str(int(v)) for v in np.asarray(values).reshape(-1))
    return f"static const {ctype} {name}[] RC_PROGMEM = {{ {body} }};"


def _sat_storage(v: int, sb: int) -> int:
    lo, hi = -(1 << (sb - 1)), (1 << (sb - 1)) - 1
    return max(lo, min(hi, int(v)))


@dataclass(frozen=True)
class _SymBase:
    """Shared scalar setup for both symmetric kernels.

    Bundles the storage widths, shapes, fixed-point shifts and quantized
    bias / leak / tanh-LUT bounds so the bit-exact quantization math
    (truncate-not-round `bias_q` / `leak_q`, the `shift_in` invariant, the
    LUT domain) is computed once for the inference and online emitters.
    """

    sb: int
    storage_t: str
    rd: str
    N: int
    K: int
    M: int
    F: int
    qmin: int
    qmax: int
    state_frac: int
    input_frac: int
    weight_frac: int
    shift_in: int
    state_scale: int
    bias_q: int
    leak_q: int
    one_minus_leak_q: int
    xmin_q: int
    xmax_q: int
    denom: int
    lut_n: int


def _symmetric_base(qmodel: QuantizedModel, *, what: str = "") -> _SymBase:
    """Compute the shared scalar parameters for a symmetric kernel emit.

    `what` is interpolated into the `shift_in` error message ("" for the
    inference kernel, "online " for the LMS kernel) so the diagnostics match
    the original per-emitter messages exactly.
    """
    rc, cfg = qmodel.rc, qmodel.config
    sb = qmodel.target.storage_bits
    state_frac, input_frac, weight_frac = (
        cfg.state_frac,
        cfg.input_frac,
        cfg.weight_frac,
    )
    shift_in = weight_frac + input_frac - state_frac
    if shift_in < 0:
        raise NotImplementedError(
            f"symmetric {what}C emit needs weight_frac+input_frac >= "
            f"state_frac (shift_in={shift_in})"
        )
    state_scale = 1 << state_frac
    # bias/leak follow `QuantTarget.quantize_state`: int(x * scale) (truncate
    # toward zero), then saturate to storage. NOT round() — the executor uses
    # plain int() here, and rounding would diverge by one LSB.
    bias_q = _sat_storage(int(float(rc.reservoir.bias) * state_scale), sb)
    leak_q = _sat_storage(int(float(rc.reservoir.leak_rate) * state_scale), sb)
    xmin_q = int(qmodel.lut.xmin * state_scale)
    xmax_q = int(qmodel.lut.xmax * state_scale)
    return _SymBase(
        sb=sb,
        storage_t=_C_TYPE[sb],
        rd=_RD[sb],
        N=qmodel.N,
        K=qmodel.K,
        M=qmodel.M,
        F=qmodel.F,
        qmin=-(1 << (sb - 1)),
        qmax=(1 << (sb - 1)) - 1,
        state_frac=state_frac,
        input_frac=input_frac,
        weight_frac=weight_frac,
        shift_in=shift_in,
        state_scale=state_scale,
        bias_q=bias_q,
        leak_q=leak_q,
        one_minus_leak_q=state_scale - leak_q,
        xmin_q=xmin_q,
        xmax_q=xmax_q,
        denom=xmax_q - xmin_q,
        lut_n=int(np.asarray(qmodel.lut_table_q).shape[0]),
    )


@dataclass(frozen=True)
class _ReservoirLayout:
    """Topology / chain-weight / CSR-sparse layout shared by both emitters."""

    is_structured: bool
    topo: str
    chain_weight_q: int
    chain_feedback_q: int
    use_sparse: bool
    rd_col: Optional[str]
    col_t: Optional[str]
    csr: Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]


def _reservoir_layout(qmodel: QuantizedModel, sparse) -> _ReservoirLayout:
    """Resolve the W_res representation (structured chain / CSR / dense).

    Chain weights are read straight from the quantized dense W_res so they are
    byte-identical to what the executor's CSR-over-dense matmul reads.
    Structured topologies place the forward chain on the lower sub-diagonal
    (`W_res_q[i, i-1]`) and the DLRB feedback on the upper one
    (`W_res_q[i, i+1]`); all forward (resp. backward) entries share one scalar,
    so a single representative entry is exact. Any requested sparse strategy on
    a dense topology maps to CSR (bounded code size, bit-exact).
    """
    N = qmodel.N
    is_structured = qmodel.rc.reservoir.topology in (
        Topology.DLR,
        Topology.DLRB,
        Topology.SCR,
    )
    topo = qmodel.rc.reservoir.topology.name
    Wr = np.asarray(qmodel.W_res_q)
    chain_weight_q = int(Wr[1, 0]) if (is_structured and N > 1) else 0
    chain_feedback_q = int(Wr[0, 1]) if (is_structured and N > 1) else 0
    use_sparse = bool(sparse) and not is_structured
    rd_col = col_t = csr = None
    if use_sparse:
        from rclite.ir.passes.sparsify import build_csr

        csr = build_csr(Wr)
        col_t = "int16_t" if N <= 32767 else "int32_t"
        rd_col = "RC_RD16" if N <= 32767 else "RC_RD32"
    return _ReservoirLayout(
        is_structured=is_structured,
        topo=topo,
        chain_weight_q=chain_weight_q,
        chain_feedback_q=chain_feedback_q,
        use_sparse=use_sparse,
        rd_col=rd_col,
        col_t=col_t,
        csr=csr,
    )


def emit_symmetric_kernel_c(
    qmodel: QuantizedModel,
    fn_name: str = "rc_predict",
    *,
    head: str = None,
    sparse=None,
) -> str:
    if head not in (None, "logits", "classify", "proba"):
        raise NotImplementedError(
            f"symmetric C kernel supports head in (None, 'logits', "
            f"'classify', 'proba'); got {head!r}"
        )
    classify = head == "classify"
    proba = head == "proba"
    rc = qmodel.rc
    cfg = qmodel.config
    base = _symmetric_base(qmodel)
    sb, storage_t, rd = base.sb, base.storage_t, base.rd
    N, K, M, F = base.N, base.K, base.M, base.F
    qmin, qmax = base.qmin, base.qmax
    state_frac, input_frac, weight_frac = (
        base.state_frac,
        base.input_frac,
        base.weight_frac,
    )
    shift_in = base.shift_in
    shift_res = weight_frac
    bias_q, leak_q = base.bias_q, base.leak_q
    one_minus_leak_q = base.one_minus_leak_q
    xmin_q, xmax_q, denom, lut_n = (
        base.xmin_q,
        base.xmax_q,
        base.denom,
        base.lut_n,
    )

    input_scale = 1 << input_frac
    weight_scale = 1 << weight_frac
    offset_q = int(round(float(rc.input.input_offset) * input_scale))
    scaling_q = int(round(float(rc.input.input_scaling) * weight_scale))

    layout = _reservoir_layout(qmodel, sparse)

    off_bias = 0
    off_input = 1 if rc.readout.include_bias else 0
    off_state = off_input + (K if rc.readout.include_input else 0)

    L: List[str] = []
    a = L.append

    a("/* Auto-generated by rclite — symmetric Q-format reservoir kernel. */")
    a("/* Portable C99. On AVR, tables live in Flash (PROGMEM). */")
    _emit_progmem_prologue(a)
    a("")
    a(f"#define RC_N {N}")
    a(f"#define RC_K {K}")
    a(f"#define RC_M {M}")
    a(f"#define RC_F {F}")
    a(f"#define RC_QMIN ({qmin})")
    a(f"#define RC_QMAX ({qmax})")
    a(f"#define RC_STATE_FRAC {state_frac}")
    a(f"#define RC_SHIFT_IN {shift_in}")
    a(f"#define RC_SHIFT_RES {shift_res}")
    a(f"typedef {storage_t} rc_storage_t;")
    a("")

    # tables
    a(_arr("rc_W_in", storage_t, qmodel.W_in_q))
    a(_arr("rc_W_out", storage_t, qmodel.W_out_q))
    a(_arr("rc_lut", storage_t, qmodel.lut_table_q))
    _emit_wres_tables(a, qmodel, layout, storage_t)
    if proba:
        from rclite.quant.softmax_lut import (
            SoftmaxLUTSpec,
            build_params as _build_sm,
        )

        sm = _build_sm(
            SoftmaxLUTSpec(),
            1.0 / cfg.state_scale,
            sb,
            qmodel.target.storage_dtype,
        )
        a(_arr("rc_sm_lut", storage_t, sm.lut_q))
        a(f"#define RC_SM_N {sm.n}")
        a(f"#define RC_SM_DMIN ({sm.dmin_q})")
        a(f"#define RC_SM_IDXF {sm.idx_frac}")
        a(f"#define RC_SM_PF {sm.prob_frac}")
    a("")
    a("static rc_storage_t rc_h[RC_N];")
    a("static rc_storage_t rc_pre[RC_N];")
    a("static rc_storage_t rc_u[RC_K];")
    a("")

    # helpers: two's-complement wrap (well-defined via unsigned)
    _emit_fixed_helpers(a, storage_t)
    a("")

    # tanh LUT (linear interp), mirage tanh_lut_q
    _emit_tanh_lut(
        a, xmin_q=xmin_q, xmax_q=xmax_q, denom=denom, lut_n=lut_n, rd=rd
    )
    a("")

    # kernel
    y_type = "int32_t" if classify else "rc_storage_t"
    a(f"void {fn_name}(int32_t T, const rc_storage_t *X, {y_type} *Y){{")
    a("    int32_t t, i, j, k, m;")
    if classify:
        a("    int32_t best_m; int64_t best_v;")
    if proba:
        a("    int32_t sm_max; int64_t sm_sum;")
        a("    rc_storage_t rc_lg[RC_M];")
        a("    int32_t rc_eq[RC_M];")
    a("    for (i = 0; i < RC_N; i++) rc_h[i] = 0;")
    a("    for (t = 0; t < T; t++) {")
    # preprocess: u_pre[k] = wrap_s( ((X - offset_q) * scaling_q) >> weight_frac )
    a("        for (k = 0; k < RC_K; k++) {")
    a(f"            int64_t d = (int64_t)X[t*RC_K + k] - ({offset_q});")
    a(
        f"            rc_u[k] = rc_ws(rc_w32((d * ({scaling_q})) >> {weight_frac}));"
    )
    a("        }")
    # pre-activation
    _emit_preactivation(
        a, ind=8, N=N, bias_q=bias_q, rd=rd, input_var="rc_u", layout=layout
    )
    # activation + leaky integration
    _emit_activation_leaky(
        a, ind=8, one_minus_leak_q=one_minus_leak_q, leak_q=leak_q
    )
    # readout (mirage mixed-scale, i64 accumulate)
    if classify:
        a("        best_m = 0; best_v = 0;")
    a("        for (m = 0; m < RC_M; m++) {")
    a("            int64_t out = 0;")
    if rc.readout.include_bias:
        a(
            f"            out += ((int64_t)1 << RC_STATE_FRAC) * (int64_t){rd}(&rc_W_out[m*RC_F + {off_bias}]);"
        )
    if rc.readout.include_input:
        a("            for (k = 0; k < RC_K; k++)")
        a(
            f"                out += (int64_t){rd}(&rc_W_out[m*RC_F + {off_input} + k]) * (int64_t)X[t*RC_K + k];"
        )
    a("            for (j = 0; j < RC_N; j++)")
    a(
        f"                out += (int64_t){rd}(&rc_W_out[m*RC_F + {off_state} + j]) * (int64_t)rc_h[j];"
    )
    a("            int64_t sh = out >> RC_STATE_FRAC;")
    a("            if (sh < RC_QMIN) sh = RC_QMIN;")
    a("            if (sh > RC_QMAX) sh = RC_QMAX;")
    if classify:
        # argmax over the (saturated) quantized scores — monotone, so the
        # class id matches the float readout's argmax.
        a(
            "            if (m == 0 || sh > best_v) { best_v = sh; best_m = m; }"
        )
    elif proba:
        a("            rc_lg[m] = (rc_storage_t)sh;")
    else:
        a("            Y[t*RC_M + m] = (rc_storage_t)sh;")
    a("        }")
    if classify:
        a("        Y[t] = best_m;")
    elif proba:
        # fixed-point softmax over rc_lg[] (exp LUT), bit-exact with softmax_q
        a("        sm_max = rc_lg[0];")
        a(
            "        for (m = 1; m < RC_M; m++) if (rc_lg[m] > sm_max) sm_max = rc_lg[m];"
        )
        a("        sm_sum = 0;")
        a("        for (m = 0; m < RC_M; m++) {")
        a("            int32_t d = (int32_t)rc_lg[m] - sm_max;")
        a("            if (d < RC_SM_DMIN) d = RC_SM_DMIN;")
        a(
            "            int64_t pos = ((int64_t)(d - (RC_SM_DMIN)) * (RC_SM_N - 1) << RC_SM_IDXF) / (-(RC_SM_DMIN));"
        )
        a("            int32_t i0 = (int32_t)(pos >> RC_SM_IDXF);")
        a(
            "            if (i0 < 0) i0 = 0; if (i0 > RC_SM_N - 2) i0 = RC_SM_N - 2;"
        )
        a("            int64_t frac = pos - ((int64_t)i0 << RC_SM_IDXF);")
        a(
            f"            int32_t y0 = {rd}(&rc_sm_lut[i0]); int32_t y1 = {rd}(&rc_sm_lut[i0 + 1]);"
        )
        a(
            "            int64_t ev = (int64_t)y0 + (((int64_t)(y1 - y0) * frac) >> RC_SM_IDXF);"
        )
        a("            rc_eq[m] = (int32_t)ev; sm_sum += ev;")
        a("        }")
        a("        for (m = 0; m < RC_M; m++) {")
        a("            int64_t p = ((int64_t)rc_eq[m] << RC_SM_PF) / sm_sum;")
        a("            if (p > RC_QMAX) p = RC_QMAX;")
        a("            Y[t*RC_M + m] = (rc_storage_t)p;")
        a("        }")
    a("    }")
    a("}")
    a("")
    return "\n".join(L)


# --------------------------------------------------------------------------
# Shared C-emission helpers — used by BOTH the batch inference kernel
# (`emit_symmetric_kernel_c`) and the online learning kernel
# (`emit_symmetric_online_kernel_c`). Keeping the prologue / fixed-point
# helpers / tanh LUT / W_res accumulation in one place is what keeps the two
# emitters bit-exact with each other (and with the executor) by construction.
# --------------------------------------------------------------------------


def _emit_progmem_prologue(a) -> None:
    """`#include`s and the AVR-vs-hosted PROGMEM / table-read macros."""
    a("#include <stdint.h>")
    a("#ifdef __AVR__")
    a("#include <avr/pgmspace.h>")
    a("#define RC_PROGMEM PROGMEM")
    a("#define RC_RD8(p)  ((int8_t)pgm_read_byte(p))")
    a("#define RC_RD16(p) ((int16_t)pgm_read_word(p))")
    a("#define RC_RD32(p) ((int32_t)pgm_read_dword(p))")
    a("#else")
    a("#define RC_PROGMEM")
    a("#define RC_RD8(p)  (*(p))")
    a("#define RC_RD16(p) (*(p))")
    a("#define RC_RD32(p) (*(p))")
    a("#endif")


def _emit_fixed_helpers(a, storage_t: str) -> None:
    """Two's-complement i32 wrap (`rc_w32`/`rc_ws`) and `(a*b)>>s` (`rc_fmul`)."""
    a(
        "static int32_t rc_w32(int64_t v){ return (int32_t)(uint32_t)(uint64_t)v; }"
    )
    a(f"static rc_storage_t rc_ws(int32_t v){{ return ({storage_t})v; }}")
    a("static int32_t rc_fmul(int32_t a, int32_t b, int s){")
    a("    return rc_w32(((int64_t)a * (int64_t)b) >> s);")
    a("}")


def _emit_tanh_lut(a, *, xmin_q, xmax_q, denom, lut_n, rd) -> None:
    """Linear-interpolated tanh LUT (`rc_tanh`), mirage `tanh_lut_q`."""
    a("static rc_storage_t rc_tanh(int32_t x){")
    a(f"    if (x < {xmin_q}) x = {xmin_q};")
    a(f"    if (x > {xmax_q}) x = {xmax_q};")
    a(f"    int64_t num = (int64_t)x - ({xmin_q});")
    a(f"    int32_t tq = rc_w32((num << RC_STATE_FRAC) / ({denom}));")
    a(f"    int32_t pos = rc_w32((int64_t)tq * ({lut_n - 1}));")
    a("    int32_t i0 = pos >> RC_STATE_FRAC;")
    a(f"    if (i0 < 0) i0 = 0; if (i0 > {lut_n - 2}) i0 = {lut_n - 2};")
    a("    int32_t frac = pos - rc_w32((int64_t)i0 << RC_STATE_FRAC);")
    a(f"    int32_t y0 = {rd}(&rc_lut[i0]);")
    a(f"    int32_t y1 = {rd}(&rc_lut[i0 + 1]);")
    a(
        "    int32_t interp = rc_w32((int64_t)y0 + (((int64_t)(y1 - y0) * frac) >> RC_STATE_FRAC));"
    )
    a("    return rc_ws(interp);")
    a("}")


def _emit_wres_accum(
    a,
    *,
    ind,
    is_structured,
    use_sparse,
    topo,
    N,
    chain_weight_q,
    chain_feedback_q,
    rd,
    rd_col=None,
) -> None:
    """Emit the W_res contribution into `acc`, indentation-parametrized.

    `ind` is the base indent (spaces) of the statement: the inference kernel
    emits its reservoir loop body at 12, the online forward step at 8.
    Continuation lines sit at ind+2, loop/branch bodies at ind+4. The three
    paths (structured chain / CSR sparse / dense matvec) match the executor's
    sparse-over-dense matmul and are bit-exact across both callers.
    """
    b = " " * ind  # statement: for / if / opening brace
    c = " " * (ind + 2)  # continuation + nested for/if
    d = " " * (ind + 4)  # innermost MAC body
    if is_structured:
        if topo == "SCR":
            a(f"{b}{{ int32_t ip = (i == 0) ? {N - 1} : (i - 1);")
            a(
                f"{c}acc = rc_w32((int64_t)acc + rc_fmul({chain_weight_q}, rc_h[ip], RC_SHIFT_RES)); }}"
            )
        elif topo == "DLR":
            a(f"{b}if (i > 0)")
            a(
                f"{d}acc = rc_w32((int64_t)acc + rc_fmul({chain_weight_q}, rc_h[i-1], RC_SHIFT_RES));"
            )
        elif topo == "DLRB":
            a(f"{b}if (i > 0)")
            a(
                f"{d}acc = rc_w32((int64_t)acc + rc_fmul({chain_weight_q}, rc_h[i-1], RC_SHIFT_RES));"
            )
            a(f"{b}if (i < RC_N - 1)")
            a(
                f"{d}acc = rc_w32((int64_t)acc + rc_fmul({chain_feedback_q}, rc_h[i+1], RC_SHIFT_RES));"
            )
        else:
            raise ValueError(f"_emit_wres_accum: unexpected topology {topo}")
    elif use_sparse:
        a(f"{b}{{ int32_t rp = RC_RD32(&rc_W_res_rowptr[i]);")
        a(f"{c}int32_t rpe = RC_RD32(&rc_W_res_rowptr[i+1]);")
        a(f"{c}for (j = rp; j < rpe; j++)")
        a(
            f"{d}acc = rc_w32((int64_t)acc + rc_fmul({rd}(&rc_W_res_val[j]), "
            f"rc_h[{rd_col}(&rc_W_res_col[j])], RC_SHIFT_RES)); }}"
        )
    else:
        a(f"{b}for (j = 0; j < RC_N; j++)")
        a(
            f"{d}acc = rc_w32((int64_t)acc + rc_fmul({rd}(&rc_W_res[i*RC_N + j]), rc_h[j], RC_SHIFT_RES));"
        )


def _emit_wres_tables(a, qmodel, layout: _ReservoirLayout, storage_t) -> None:
    """Emit the W_res constant tables: CSR triple (sparse) or dense matrix.

    Structured topologies carry the chain as a baked-in scalar, so they emit
    no W_res table at all.
    """
    if layout.is_structured:
        return
    if layout.use_sparse:
        val, col, rowptr = layout.csr
        a(_arr("rc_W_res_val", storage_t, val))
        a(_arr("rc_W_res_col", layout.col_t, col))
        a(_arr("rc_W_res_rowptr", "int32_t", rowptr))
    else:
        a(_arr("rc_W_res", storage_t, qmodel.W_res_q))


def _emit_preactivation(
    a, *, ind, N, bias_q, rd, input_var, layout: _ReservoirLayout
) -> None:
    """Per-row pre-activation accumulator (W_in MAC + W_res contribution).

    `ind` is the indent (spaces) of the `for (i ...)` row — the inference
    kernel emits its reservoir loop at 8, the online forward step at 4.
    `input_var` is the C name of the input vector: `rc_u` (preprocessed) in
    the inference kernel, `u_q` (caller-supplied) in the online kernel.
    """
    b = " " * ind
    c = " " * (ind + 4)
    d = " " * (ind + 8)
    a(f"{b}for (i = 0; i < RC_N; i++) {{")
    a(f"{c}int32_t acc = {bias_q};")
    a(f"{c}for (k = 0; k < RC_K; k++)")
    a(
        f"{d}acc = rc_w32((int64_t)acc + rc_fmul({rd}(&rc_W_in[i*RC_K + k]), "
        f"{input_var}[k], RC_SHIFT_IN));"
    )
    _emit_wres_accum(
        a,
        ind=ind + 4,
        is_structured=layout.is_structured,
        use_sparse=layout.use_sparse,
        topo=layout.topo,
        N=N,
        chain_weight_q=layout.chain_weight_q,
        chain_feedback_q=layout.chain_feedback_q,
        rd=rd,
        rd_col=layout.rd_col,
    )
    a(f"{c}rc_pre[i] = rc_ws(acc);")
    a(f"{b}}}")


def _emit_activation_leaky(a, *, ind, one_minus_leak_q, leak_q) -> None:
    """tanh activation + leaky integration into rc_h[] (indent-parametrized)."""
    b = " " * ind
    c = " " * (ind + 4)
    a(f"{b}for (i = 0; i < RC_N; i++) {{")
    a(f"{c}rc_storage_t act = rc_tanh((int32_t)rc_pre[i]);")
    a(
        f"{c}rc_storage_t t1 = rc_ws(rc_fmul(rc_h[i], {one_minus_leak_q}, "
        f"RC_STATE_FRAC));"
    )
    a(f"{c}rc_storage_t t2 = rc_ws(rc_fmul(act, {leak_q}, RC_STATE_FRAC));")
    a(f"{c}rc_h[i] = rc_ws((int32_t)t1 + (int32_t)t2);")
    a(f"{b}}}")


def emit_symmetric_online_kernel_c(
    qmodel: QuantizedModel,
    learning_rate: float,
    *,
    normalized: bool = False,
    delta: float = 1.0,
    sparse=None,
) -> str:
    """Portable-C emitter for **on-device integer LMS / NLMS** readout training.

    Bit-exact with `rclite.quant.online.IntegerLMSLearner` (the Python
    reference). With ``normalized=True`` the per-step update is divided by the
    squared norm of φ = [1, u, h] (normalized LMS) — one integer division per
    step — making the effective rate scale-invariant. Emits three entry points
    over a mutable RAM `rc_W_out`:

      * ``rc_train_reset()`` — zero the reservoir state (mirrors
        ``IntegerLMSLearner.reset``).
      * ``rc_infer_step(u_q, y_pred_q)`` — forward one step and read out the
        prediction without learning (mirrors ``step_no_update``; use for the
        warmup window).
      * ``rc_train_step(u_q, y_target_q, y_pred_q)`` — forward, read out, then
        update ``rc_W_out`` in place by one LMS step (mirrors ``step``).

    Inputs are taken **pre-quantized**: ``u_q`` is the K-vector at input scale
    (the caller's ``_quantize_input`` output) and ``y_target_q`` is the
    M-vector at state scale (``quantize_state`` of the float target). The host
    owns those scalar quantizations; the device kernel is pure integer. The
    prediction ``y_pred_q`` is returned at state scale (clipped to storage).

    The forward step and readout reuse the symmetric inference arithmetic, so
    ``rc_infer_step`` matches ``QuantizedExecutor.predict_one_q`` fed the same
    ``u_q``. Unlike the batch inference kernel, the readout's input-passthrough
    block multiplies ``u_q`` (not the raw stimulus), matching the online
    reference which drives the reservoir and the readout with one vector.
    """
    rc = qmodel.rc
    cfg = qmodel.config
    ro = rc.readout

    if ro.units < 1:
        raise ValueError("online kernel needs readout.units >= 1")
    # Online LMS is a regression learner; classification uses RIDGE/PINV.
    base = _symmetric_base(qmodel, what="online ")
    storage_t, rd = base.storage_t, base.rd
    N, K, M, F = base.N, base.K, base.M, base.F
    qmin, qmax = base.qmin, base.qmax
    state_frac, input_frac, weight_frac = (
        base.state_frac,
        base.input_frac,
        base.weight_frac,
    )
    shift_in = base.shift_in
    state_scale = base.state_scale
    bias_q, leak_q = base.bias_q, base.leak_q
    one_minus_leak_q = base.one_minus_leak_q
    xmin_q, xmax_q, denom, lut_n = (
        base.xmin_q,
        base.xmax_q,
        base.denom,
        base.lut_n,
    )

    # Learning rate quantized at state scale — a compile-time constant, baked in.
    lr_q = int(qmodel.target.quantize_state(float(learning_rate), cfg))

    # NLMS squared-norm fixed point: Q*||phi||^2 accumulated at state scale.
    shift_u = 2 * input_frac - state_frac
    if normalized and shift_u < 0:
        raise NotImplementedError(
            "NLMS needs 2*input_frac >= state_frac for the squared-norm fixed "
            f"point (input_frac={input_frac}, state_frac={state_frac})"
        )
    delta_q = int(float(delta) * state_scale)

    layout = _reservoir_layout(qmodel, sparse)

    include_bias = ro.include_bias
    include_input = ro.include_input
    off_bias = 0
    off_input = 1 if include_bias else 0
    off_state = off_input + (K if include_input else 0)

    L: List[str] = []
    a = L.append

    a(
        "/* Auto-generated by rclite — symmetric Q-format ONLINE (integer LMS) kernel. */"
    )
    a(
        "/* Portable C99. Bit-exact with rclite.quant.online.IntegerLMSLearner. */"
    )
    a(
        "/* rc_W_out lives in RAM (mutated by rc_train_step); other tables are const. */"
    )
    _emit_progmem_prologue(a)
    a("")
    a(f"#define RC_N {N}")
    a(f"#define RC_K {K}")
    a(f"#define RC_M {M}")
    a(f"#define RC_F {F}")
    a(f"#define RC_QMIN ({qmin})")
    a(f"#define RC_QMAX ({qmax})")
    a(f"#define RC_STATE_FRAC {state_frac}")
    a(f"#define RC_INPUT_FRAC {input_frac}")
    a(f"#define RC_SHIFT_IN {shift_in}")
    a(f"#define RC_SHIFT_RES {weight_frac}")
    a(f"#define RC_LR_Q ({lr_q})")
    if normalized:
        a(f"#define RC_NLMS 1")
        a(f"#define RC_DELTA_Q ({delta_q})")
        a(f"#define RC_SHIFT_U {shift_u}")
    a(f"typedef {storage_t} rc_storage_t;")
    a("")

    # tables — W_in / W_res / LUT are const (Flash); W_out is mutable RAM.
    a(_arr("rc_W_in", storage_t, qmodel.W_in_q))
    a(_arr("rc_lut", storage_t, qmodel.lut_table_q))
    _emit_wres_tables(a, qmodel, layout, storage_t)
    wout_body = ", ".join(
        str(int(v)) for v in np.asarray(qmodel.W_out_q).reshape(-1)
    )
    a(f"static rc_storage_t rc_W_out[] = {{ {wout_body} }};")
    a("")
    a("static rc_storage_t rc_h[RC_N];")
    a("static rc_storage_t rc_pre[RC_N];")
    a("")

    # helpers (shared with the inference kernel)
    _emit_fixed_helpers(a, storage_t)
    # saturating add into the storage range [RC_QMIN, RC_QMAX] (mirrors
    # IntegerLMSLearner._sadd_sat). Saturating to the storage width — not
    # int32 — is what actually prevents wrap-around: W_out is the storage
    # dtype, so an int32 saturate would still wrap on store for narrow types.
    a("static rc_storage_t rc_sadd_sat(int32_t cur, int64_t dw){")
    a("    int64_t s = (int64_t)cur + dw;")
    a("    if (s > RC_QMAX) s = RC_QMAX;")
    a("    else if (s < RC_QMIN) s = RC_QMIN;")
    a("    return (rc_storage_t)s;")
    a("}")
    a("")

    # tanh LUT (linear interp), identical to the inference kernel
    _emit_tanh_lut(
        a, xmin_q=xmin_q, xmax_q=xmax_q, denom=denom, lut_n=lut_n, rd=rd
    )
    a("")

    # forward one step (advance rc_h) + readout into y_pred_q (state scale).
    # The readout mirrors predict_one_q fed `u_q`.
    a(
        "static void rc_forward_predict(const rc_storage_t *u_q, int32_t *y_pred_q){"
    )
    a("    int32_t i, j, k, m;")
    _emit_preactivation(
        a, ind=4, N=N, bias_q=bias_q, rd=rd, input_var="u_q", layout=layout
    )
    _emit_activation_leaky(
        a, ind=4, one_minus_leak_q=one_minus_leak_q, leak_q=leak_q
    )
    a("    for (m = 0; m < RC_M; m++) {")
    a("        int64_t out = 0;")
    if include_bias:
        a(
            f"        out += ((int64_t)1 << RC_STATE_FRAC) * (int64_t)rc_W_out[m*RC_F + {off_bias}];"
        )
    if include_input:
        a("        for (k = 0; k < RC_K; k++)")
        a(
            f"            out += (int64_t)rc_W_out[m*RC_F + {off_input} + k] * (int64_t)u_q[k];"
        )
    a("        for (j = 0; j < RC_N; j++)")
    a(
        f"            out += (int64_t)rc_W_out[m*RC_F + {off_state} + j] * (int64_t)rc_h[j];"
    )
    a("        int64_t sh = out >> RC_STATE_FRAC;")
    a("        if (sh < RC_QMIN) sh = RC_QMIN;")
    a("        if (sh > RC_QMAX) sh = RC_QMAX;")
    a("        y_pred_q[m] = (int32_t)sh;")
    a("    }")
    a("}")
    a("")

    a(
        "void rc_train_reset(void){ int32_t i; for (i = 0; i < RC_N; i++) rc_h[i] = 0; }"
    )
    a("")
    a("void rc_infer_step(const rc_storage_t *u_q, int32_t *y_pred_q){")
    a("    rc_forward_predict(u_q, y_pred_q);")
    a("}")
    a("")

    # train: forward + readout, then one LMS / NLMS update of rc_W_out with
    # column-specific shifts (bias >> state_frac, input >> 2*input_frac,
    # state >> 2*state_frac). For NLMS the raw product is first divided by the
    # squared norm (truncate toward zero, like Python's _tdiv) and the shift
    # drops by state_frac. dw is full-width (int64); the sum saturates.
    # Matches IntegerLMSLearner._apply_lms_update.
    if normalized:
        bias_dw = "((int64_t)RC_LR_Q * err) / norm"
        input_dw = "(prod / norm) >> RC_SHIFT_U"
        state_dw = "(prod / norm) >> RC_STATE_FRAC"
    else:
        bias_dw = "((int64_t)RC_LR_Q * err) >> RC_STATE_FRAC"
        input_dw = "prod >> (2 * RC_INPUT_FRAC)"
        state_dw = "prod >> (2 * RC_STATE_FRAC)"
    a("void rc_train_step(const rc_storage_t *u_q, const int32_t *y_target_q,")
    a("                   int32_t *y_pred_q){")
    a("    int32_t j, k, m;")
    a("    rc_forward_predict(u_q, y_pred_q);")
    if normalized:
        # squared norm of phi = [1, u, h] at state-scale fixed point (shared
        # across all m; depends only on u_q and the post-step state rc_h).
        a("    int64_t norm = RC_DELTA_Q;")
        if include_bias:
            a("    norm += ((int64_t)1 << RC_STATE_FRAC);")
        if include_input:
            a("    for (k = 0; k < RC_K; k++)")
            a(
                "        norm += ((int64_t)u_q[k] * (int64_t)u_q[k]) >> RC_SHIFT_U;"
            )
        a("    for (j = 0; j < RC_N; j++)")
        a(
            "        norm += ((int64_t)rc_h[j] * (int64_t)rc_h[j]) >> RC_STATE_FRAC;"
        )
        a("    if (norm < 1) norm = 1;")
    a("    for (m = 0; m < RC_M; m++) {")
    a("        int64_t err = (int64_t)y_target_q[m] - (int64_t)y_pred_q[m];")
    if include_bias:
        a(f"        {{ int64_t dw = {bias_dw};")
        a(
            f"          rc_W_out[m*RC_F + {off_bias}] = "
            f"rc_sadd_sat(rc_W_out[m*RC_F + {off_bias}], dw); }}"
        )
    if include_input:
        a("        for (k = 0; k < RC_K; k++) {")
        a(
            "            int64_t prod = (int64_t)RC_LR_Q * err * (int64_t)u_q[k];"
        )
        a(f"            int64_t dw = {input_dw};")
        a(
            f"            rc_W_out[m*RC_F + {off_input} + k] = "
            f"rc_sadd_sat(rc_W_out[m*RC_F + {off_input} + k], dw);"
        )
        a("        }")
    a("        for (j = 0; j < RC_N; j++) {")
    a("            int64_t prod = (int64_t)RC_LR_Q * err * (int64_t)rc_h[j];")
    a(f"            int64_t dw = {state_dw};")
    a(
        f"            rc_W_out[m*RC_F + {off_state} + j] = "
        f"rc_sadd_sat(rc_W_out[m*RC_F + {off_state} + j], dw);"
    )
    a("        }")
    a("    }")
    a("}")
    a("")
    # checkpoint accessor: copy the (possibly learned) readout out as int32.
    a("void rc_export_W_out(int32_t *dst){")
    a(
        "    int32_t i; for (i = 0; i < RC_M * RC_F; i++) dst[i] = (int32_t)rc_W_out[i];"
    )
    a("}")
    a("")
    return "\n".join(L)
