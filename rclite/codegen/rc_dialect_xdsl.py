"""Stage-1 spike: a real `rc` MLIR dialect (xDSL) + a structural-optimization
pass (`FuseStepReadout`) expressed as an xDSL rewrite pattern.

Today the RC structural optimizations live as Python passes over the
`rclite.ir` dataclass IR ([rclite/ir/passes/]); the MLIR emitters bake the
structural choices in as emitter branches. This spike lifts the high-level
reservoir program into an actual MLIR dialect so the optimizations become
declarative, verifiable, composable MLIR passes:

    build_rc_module(ir_module)        rclite IR Module -> `rc` dialect ModuleOp
    fuse_step_readout(mod)            xDSL RewritePattern: step;build_phi;readout
                                      -> rc.fused_step_readout  (3 ops -> 1)
    lower_fused_float(...)            rc.fused_step_readout (float; any activation;
                                      dense/structured/CSR) -> arith/memref/scf

The dialect ops are *marker* ops carrying their parameters in a single
`DictionaryAttr` and sequenced inside an `rc.time_loop` region — enough to make
the structural rewrite a genuine MLIR pattern match while keeping the spike
small. SSA-threaded operands/results are deferred to a later stage.
"""

from __future__ import annotations

import numpy as np

from xdsl.ir import Region, Block, Dialect
from xdsl.irdl import (
    irdl_op_definition,
    IRDLOperation,
    attr_def,
    region_def,
    traits_def,
)
from xdsl.traits import NoTerminator
from xdsl.dialects.builtin import (
    DictionaryAttr,
    StringAttr,
    IntegerAttr,
    FloatAttr,
    Float32Type,
    Float64Type,
    IntegerType,
    ModuleOp,
)
from xdsl.builder import ImplicitBuilder
from xdsl.pattern_rewriter import (
    RewritePattern,
    PatternRewriter,
    PatternRewriteWalker,
    op_type_rewrite_pattern,
)

from rclite.ir.ops import (
    ReservoirStep,
    BuildPhi,
    ReadoutLinear,
    FusedStepReadout,
    TimeLoop,
)

_F64 = Float64Type()
_F32 = Float32Type()
_I64 = IntegerType(64)


# ---------------------------------------------------------------------------
# Attribute helpers (params carried as a single DictionaryAttr per op)
# ---------------------------------------------------------------------------
def _val(v):
    if isinstance(v, bool):
        return IntegerAttr(1 if v else 0, IntegerType(1))
    if isinstance(v, int):
        return IntegerAttr(v, _I64)
    if isinstance(v, float):
        return FloatAttr(float(v), _F64)
    return StringAttr(str(v))


def _params(**kw) -> DictionaryAttr:
    return DictionaryAttr({k: _val(v) for k, v in kw.items()})


def _get(op, key):
    a = op.params.data[key]
    if isinstance(a, StringAttr):
        return a.data
    return a.value.data


# ---------------------------------------------------------------------------
# `rc` dialect ops (marker ops; params in `params` DictionaryAttr)
# ---------------------------------------------------------------------------
@irdl_op_definition
class ReservoirStepOp(IRDLOperation):
    name = "rc.reservoir_step"
    params = attr_def(DictionaryAttr)


@irdl_op_definition
class BuildPhiOp(IRDLOperation):
    name = "rc.build_phi"
    params = attr_def(DictionaryAttr)


@irdl_op_definition
class ReadoutLinearOp(IRDLOperation):
    name = "rc.readout_linear"
    params = attr_def(DictionaryAttr)


@irdl_op_definition
class FusedStepReadoutOp(IRDLOperation):
    name = "rc.fused_step_readout"
    params = attr_def(DictionaryAttr)


@irdl_op_definition
class TimeLoopOp(IRDLOperation):
    name = "rc.time_loop"
    params = attr_def(DictionaryAttr)
    body = region_def()
    # marker-op body: a sequence of rc.* ops, no SSA terminator in this spike.
    traits = traits_def(NoTerminator())


RC = Dialect(
    "rc",
    [
        ReservoirStepOp,
        BuildPhiOp,
        ReadoutLinearOp,
        FusedStepReadoutOp,
        TimeLoopOp,
    ],
)


# ---------------------------------------------------------------------------
# rclite IR Module -> `rc` dialect ModuleOp
# ---------------------------------------------------------------------------
def build_rc_module(ir_module) -> ModuleOp:
    """Lift an rclite IR `Module` into an `rc` dialect MLIR module.

    The time loop becomes an `rc.time_loop` whose region holds the per-step
    body ops (`rc.reservoir_step`, `rc.build_phi`, `rc.readout_linear`, ...).
    """
    mod = ModuleOp([])
    with ImplicitBuilder(mod.body):
        for op in ir_module.ops:
            if isinstance(op, TimeLoop):
                _emit_time_loop(op)
    return mod


def _emit_time_loop(loop: TimeLoop) -> None:
    region = Region([Block()])
    with ImplicitBuilder(region.block):
        for op in loop.body:
            _emit_body_op(op)
    TimeLoopOp(
        attributes={"params": _params(unroll=loop.unroll)}, regions=[region]
    )


def _sparse_kw(op) -> dict:
    """`res_*` param fields carrying an rclite op's optional `SparseSpec`.

    The CSR plan (val/col/rowptr global names) rides along in the marker op so
    the lowering can pick the sparse W_res kernel; dense/structured carry empty
    strings. `unroll` is carried verbatim so the lowering guard rejects it
    explicitly rather than mis-routing to the (absent) dense W_res global.
    """
    s = op.res_sparse
    if s is None:
        return dict(res_kind="", res_val="", res_col="", res_rowptr="")
    return dict(
        res_kind=s.kind,
        res_val=s.val_name,
        res_col=s.col_name,
        res_rowptr=s.rowptr_name,
    )


def _emit_body_op(op) -> None:
    if isinstance(op, ReservoirStep):
        ReservoirStepOp(
            attributes={
                "params": _params(
                    leak=op.leak,
                    bias=op.bias,
                    N=op.N,
                    K=op.K,
                    topology=op.topology.name,
                    activation=op.activation.name,
                    chain_weight=op.chain_weight,
                    chain_feedback=op.chain_feedback,
                    W_in=op.W_in_name,
                    W_res=op.W_res_name or "",
                    **_sparse_kw(op),
                )
            }
        )
    elif isinstance(op, BuildPhi):
        BuildPhiOp(
            attributes={
                "params": _params(
                    include_bias=op.include_bias,
                    include_input=op.include_input,
                    K=op.K,
                    N=op.N,
                )
            }
        )
    elif isinstance(op, ReadoutLinear):
        ReadoutLinearOp(
            attributes={"params": _params(M=op.M, F=op.F, W_out=op.W_out_name)}
        )
    elif isinstance(op, FusedStepReadout):
        FusedStepReadoutOp(attributes={"params": _fused_params_from(op)})
    # other body ops (preprocess/accumulate/finalize/argmax/softmax) are not
    # part of this spike's fuse pattern and are dropped from the rc view.


def _fused_params_from(op: FusedStepReadout) -> DictionaryAttr:
    return _params(
        leak=op.leak,
        bias=op.bias,
        N=op.N,
        K=op.K,
        M=op.M,
        F=op.F,
        topology=op.topology.name,
        activation=op.activation.name,
        chain_weight=op.chain_weight,
        chain_feedback=op.chain_feedback,
        include_bias_phi=op.include_bias_phi,
        include_input_phi=op.include_input_phi,
        W_in=op.W_in_name,
        W_res=op.W_res_name or "",
        W_out=op.W_out_name,
        **_sparse_kw(op),
    )


# ---------------------------------------------------------------------------
# Structural optimization: FuseStepReadout as an xDSL rewrite pattern
# ---------------------------------------------------------------------------
class FuseStepReadoutPattern(RewritePattern):
    """`rc.reservoir_step` ; `rc.build_phi` ; `rc.readout_linear`
    -> `rc.fused_step_readout`  (eliminates the phi buffer).

    The MLIR-dialect counterpart of `rclite.ir.passes.FuseStepReadout`. Matches
    the step op, checks its two following siblings, and replaces the triple with
    one fused op carrying the merged parameters.
    """

    @op_type_rewrite_pattern
    def match_and_rewrite(self, step: ReservoirStepOp, rw: PatternRewriter):
        phi = step.next_op
        if not isinstance(phi, BuildPhiOp):
            return
        ro = phi.next_op
        if not isinstance(ro, ReadoutLinearOp):
            return

        fused_params = _params(
            leak=_get(step, "leak"),
            bias=_get(step, "bias"),
            N=_get(step, "N"),
            K=_get(step, "K"),
            M=_get(ro, "M"),
            F=_get(ro, "F"),
            topology=_get(step, "topology"),
            activation=_get(step, "activation"),
            chain_weight=_get(step, "chain_weight"),
            chain_feedback=_get(step, "chain_feedback"),
            include_bias_phi=bool(_get(phi, "include_bias")),
            include_input_phi=bool(_get(phi, "include_input")),
            W_in=_get(step, "W_in"),
            W_res=_get(step, "W_res"),
            W_out=_get(ro, "W_out"),
            res_kind=_get(step, "res_kind"),
            res_val=_get(step, "res_val"),
            res_col=_get(step, "res_col"),
            res_rowptr=_get(step, "res_rowptr"),
        )
        rw.erase_op(ro)
        rw.erase_op(phi)
        rw.replace_op(
            step, FusedStepReadoutOp(attributes={"params": fused_params})
        )


def fuse_step_readout(mod: ModuleOp) -> ModuleOp:
    """Apply the FuseStepReadout structural optimization in place; returns mod."""
    PatternRewriteWalker(FuseStepReadoutPattern()).rewrite_module(mod)
    return mod


def count_ops(mod: ModuleOp) -> dict:
    """Histogram of rc op names in the module (for spike assertions/inspection)."""
    hist: dict = {}
    for op in mod.walk():
        nm = op.name
        if nm.startswith("rc."):
            hist[nm] = hist.get(nm, 0) + 1
    return hist


# ---------------------------------------------------------------------------
# Lowering: rc.fused_step_readout (float; any activation; dense/CSR/structured)
# ---------------------------------------------------------------------------
# This is the genuine dialect -> MLIR -> executable leg of the spike. It lowers
# the *fused* op to an arith/memref/scf `rc_predict` kernel (no phi buffer; the
# readout reads `h` directly), runnable via the existing mlir_jit pipeline. All
# four activations are supported: relu/identity stay libm-free (import-free on
# WASM / FPU-less cores), tanh/sigmoid declare an external libm scalar. The
# N-wide reductions are the substrate for the Stage-3 `vector` lowering where
# MLIR auto-vectorisation pays off (the scalar activation does not block it).
import io  # noqa: E402

from xdsl.dialects import arith, memref, func, vector  # noqa: E402
from xdsl.dialects.builtin import (  # noqa: E402
    MemRefType,
    TensorType,
    VectorType,
    DenseIntOrFPElementsAttr,
    UnitAttr,
    IndexType,
)
from xdsl.printer import Printer  # noqa: E402

from .mlir_xdsl_common import c_idx, call, for_, _dense_global  # noqa: E402

_IDX = IndexType()
_I32 = IntegerType(32)
_ADD_KIND = vector.CombiningKindAttr(vector.CombiningKindFlag.ADD)


def _matvec_acc(Wmem, base, hmem, N, vlen, acc0, ety):
    """`acc0 + sum_{j<N} Wmem[base+j] * hmem[j]`, returns the scalar acc (`ety`).

    `vlen <= 1`: a plain scalar reduction (IEEE accumulation order — the order
    LLVM is forced to keep, so it leaves the reduction scalar even at -O3).
    `vlen > 1`: the inner product is accumulated in a `vector<vlen x ety>` via
    `vector.fma`, then collapsed with `vector.reduction <add>` plus a scalar tail
    for `N % vlen`. The vectorised partial sums reassociate the float reduction —
    exactly what LLVM's auto-vectoriser won't do without fast-math — so this is
    the Stage-3 structural win over the scalar baseline."""
    if vlen <= 1:

        def jb(j, args):
            (a,) = args
            w = memref.LoadOp.get(Wmem, [arith.AddiOp(base, j).result]).res
            hv = memref.LoadOp.get(hmem, [j]).res
            return [arith.AddfOp(a, arith.MulfOp(w, hv).result).result]

        return for_(c_idx(0), c_idx(N), c_idx(1), [acc0], jb)[0]

    Nv = (N // vlen) * vlen
    vt = VectorType(ety, [vlen])
    acc = acc0
    if Nv > 0:
        vz = arith.ConstantOp(
            DenseIntOrFPElementsAttr.from_list(vt, [0.0] * vlen)
        ).result

        def vbody(jb, args):
            (va,) = args
            wv = vector.LoadOp(
                Wmem, [arith.AddiOp(base, jb).result], vt
            ).result
            hv = vector.LoadOp(hmem, [jb], vt).result
            return [vector.FMAOp(wv, hv, va).res]

        vacc = for_(c_idx(0), c_idx(Nv), c_idx(vlen), [vz], vbody)[0]
        # NB: xDSL 0.66 vector.ReductionOp.__init__ mis-types the result as the
        # vector type; build explicitly with the scalar element result and `acc`
        # as the reduction's start value (= acc + lane-sum).
        acc = vector.ReductionOp.build(
            operands=[vacc, acc],
            result_types=[ety],
            properties={"kind": _ADD_KIND},
        ).results[0]
    if Nv < N:

        def tb(j, args):
            (a,) = args
            w = memref.LoadOp.get(Wmem, [arith.AddiOp(base, j).result]).res
            hv = memref.LoadOp.get(hmem, [j]).res
            return [arith.AddfOp(a, arith.MulfOp(w, hv).result).result]

        acc = for_(c_idx(Nv), c_idx(N), c_idx(1), [acc], tb)[0]
    return acc


def _res_contrib_dlr(acc, i, hmem, chain_weight, ety):
    """DLR: `+ chain_weight*h[i-1]` for i>0 (else +0). Mirrors the float lowerer."""
    z, one = c_idx(0), c_idx(1)
    is_pos = arith.CmpiOp(i, z, "sgt").result
    i_safe = arith.SelectOp(is_pos, arith.SubiOp(i, one).result, z).result
    val = memref.LoadOp.get(hmem, [i_safe]).res
    prod = arith.MulfOp(_fconst(chain_weight, ety), val).result
    contrib = arith.SelectOp(is_pos, prod, _fconst(0.0, ety)).result
    return arith.AddfOp(acc, contrib).result


def _res_contrib_scr(acc, i, N, hmem, chain_weight, ety):
    """SCR (ring): `+ chain_weight*h[(i-1) mod N]`."""
    z, one, nm1 = c_idx(0), c_idx(1), c_idx(N - 1)
    is_zero = arith.CmpiOp(i, z, "eq").result
    i_prev = arith.SelectOp(is_zero, nm1, arith.SubiOp(i, one).result).result
    val = memref.LoadOp.get(hmem, [i_prev]).res
    return arith.AddfOp(
        acc, arith.MulfOp(_fconst(chain_weight, ety), val).result
    ).result


def _res_contrib_dlrb(acc, i, N, hmem, chain_weight, chain_feedback, ety):
    """DLRB: `+ chain_weight*h[i-1] (i>0) + chain_feedback*h[i+1] (i<N-1)`."""
    z, one, nm1 = c_idx(0), c_idx(1), c_idx(N - 1)
    zf = _fconst(0.0, ety)
    is_pos = arith.CmpiOp(i, z, "sgt").result
    i_back = arith.SelectOp(is_pos, arith.SubiOp(i, one).result, z).result
    vb = memref.LoadOp.get(hmem, [i_back]).res
    cb = arith.SelectOp(
        is_pos, arith.MulfOp(_fconst(chain_weight, ety), vb).result, zf
    ).result
    is_lt = arith.CmpiOp(i, nm1, "slt").result
    i_fwd = arith.SelectOp(is_lt, arith.AddiOp(i, one).result, nm1).result
    vf = memref.LoadOp.get(hmem, [i_fwd]).res
    cf = arith.SelectOp(
        is_lt, arith.MulfOp(_fconst(chain_feedback, ety), vf).result, zf
    ).result
    return arith.AddfOp(arith.AddfOp(acc, cb).result, cf).result


def _res_contrib_csr(acc, i, vmem, cmem, rpmem, hmem, ety):
    """CSR: `+ sum_{p in [rowptr[i], rowptr[i+1])} val[p]*h[col[p]]`.

    col/rowptr are i32 globals (index-cast per access); the ascending-p loop
    keeps the float accumulation order the executor uses (ascending column)."""
    start = arith.IndexCastOp(memref.LoadOp.get(rpmem, [i]).res, _IDX).result
    ip1 = arith.AddiOp(i, c_idx(1)).result
    end = arith.IndexCastOp(memref.LoadOp.get(rpmem, [ip1]).res, _IDX).result

    def pbody(p, args):
        (a,) = args
        j = arith.IndexCastOp(memref.LoadOp.get(cmem, [p]).res, _IDX).result
        w = memref.LoadOp.get(vmem, [p]).res
        hv = memref.LoadOp.get(hmem, [j]).res
        return [arith.AddfOp(a, arith.MulfOp(w, hv).result).result]

    return for_(start, end, c_idx(1), [acc], pbody)[0]


def _find_fused(mod: ModuleOp):
    loop = next(o for o in mod.walk() if isinstance(o, TimeLoopOp))
    fused = next(
        o for o in loop.body.block.ops if isinstance(o, FusedStepReadoutOp)
    )
    return fused


def _fconst(v, ety):
    return arith.ConstantOp(FloatAttr(float(v), ety)).result


def _fglobal(name, arr, ety):
    npty = np.float32 if ety is _F32 else np.float64
    flat = np.asarray(arr, dtype=npty).reshape(-1)
    ty = MemRefType(ety, [int(flat.size)])
    init = DenseIntOrFPElementsAttr.from_list(
        TensorType(ety, [int(flat.size)]), [float(v) for v in flat]
    )
    return memref.GlobalOp.get(
        StringAttr(name),
        ty,
        init,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr(),
    )


def _libm_name(base: str, ety) -> str:
    """libm scalar name for the element type: ``tanh`` -> ``tanhf`` in f32."""
    return base + ("f" if ety is _F32 else "")


def _libm_for(activation: str) -> tuple:
    """libm symbols an activation pulls in (declared once at module scope).

    relu/identity are emitted inline (libm-free, so they stay import-free on
    WASM / FPU-less cores); tanh/sigmoid need a scalar libm import.
    """
    if activation == "TANH":
        return ("tanh",)
    if activation == "SIGMOID":
        return ("exp",)
    return ()


def _emit_activation(pre_val, activation: str, ety):
    """Apply the reservoir activation f to a pre-activation scalar ``pre_val``.

    Mirrors `rclite.runtime.reference._ACTIVATIONS` and the LLVM float lowerer
    (`rclite.codegen._llvm_float`):
      identity -> x ; relu -> max(0, x) (cmpf+select, libm-free) ;
      tanh -> @tanh[f](x) ; sigmoid -> 1 / (1 + @exp[f](-x)).
    Emits into the active `ImplicitBuilder`; the libm calls resolve against the
    external declarations `lower_fused_float` adds at module scope.
    """
    if activation == "IDENTITY":
        return pre_val
    if activation == "RELU":
        z = _fconst(0.0, ety)
        gt = arith.CmpfOp(pre_val, z, "ogt").result
        return arith.SelectOp(gt, pre_val, z).result
    if activation == "TANH":
        return call(_libm_name("tanh", ety), [pre_val], ety)
    if activation == "SIGMOID":
        one = _fconst(1.0, ety)
        zero = _fconst(0.0, ety)
        neg = arith.SubfOp(zero, pre_val).result
        e = call(_libm_name("exp", ety), [neg], ety)
        return arith.DivfOp(one, arith.AddfOp(one, e).result).result
    raise NotImplementedError(
        f"rc dialect lowering: unsupported activation {activation!r}"
    )


def lower_fused_float(
    mod: ModuleOp,
    weights: dict,
    vlen: int = 1,
    func_name: str = "rc_predict",
    dtype: str = "f64",
) -> str:
    """Lower a fused `rc` module (float; any activation; dense / structured /
    CSR-sparse reservoir) to an arith/memref/scf `rc_predict` kernel and return
    the printed MLIR.

    Mirrors the fused float reference: per step, `pre = W_in@u + W_res@h + bias`,
    `h = (1-leak)*h + leak*act(pre)` where `act` is tanh / sigmoid / relu /
    identity, then the readout reads `h` directly — `y[m] = [bias] +
    [W_out_in@u] + W_out_state@h`.

    The recurrent `W_res@h` term is lowered per the op's topology: a dense N×N
    matvec (RANDOM/ESN_STANDARD, vectorisable via `vlen`), a CSR gather when the
    op carries a `csr` sparse plan, or an O(N) chain for the structured
    topologies (DLR/SCR/DLRB) — the same three kernels the hand-written LLVM
    lowerer emits, consolidated here as one declarative dialect lowering.

    `vlen` sets the SIMD width of the N-wide reductions: the readout
    `W_out_state@h` always, and the dense `W_res@h` matvec (the structured/CSR
    reservoir kernels stay scalar — they have no dense reduction to widen).
    `vlen=1` is the scalar baseline, `vlen>1` emits the `vector`-dialect
    reduction (Stage 3). Lower the vector form with the extended pipeline
    (`mlir_jit` + `--convert-vector-to-llvm`).

    `dtype` is `f64` or `f32`. f32 packs 4 lanes into a 128-bit SIMD register, so
    `dtype="f32", vlen=4` is what gets armv7 NEON (no f64 SIMD) and WASM SIMD128
    to vectorise; the X/Y c-interface arrays are then f32."""
    ety = _F32 if dtype == "f32" else _F64
    op = _find_fused(mod)
    activation = _get(op, "activation")
    topology = _get(op, "topology")
    res_kind = _get(op, "res_kind")
    chain_weight = _get(op, "chain_weight")
    chain_feedback = _get(op, "chain_feedback")
    if res_kind not in ("", "csr"):
        raise NotImplementedError(
            f"rc lowering: res_sparse kind {res_kind!r} (csr / dense only)"
        )
    if res_kind == "" and topology not in (
        "ESN_STANDARD",
        "RANDOM",
        "DLR",
        "SCR",
        "DLRB",
    ):
        raise NotImplementedError(f"rc lowering: topology {topology!r}")
    N, K, M, F = (_get(op, k) for k in ("N", "K", "M", "F"))
    leak, bias = _get(op, "leak"), _get(op, "bias")
    inc_b = bool(_get(op, "include_bias_phi"))
    inc_i = bool(_get(op, "include_input_phi"))
    off_i = 1 if inc_b else 0
    off_s = off_i + (K if inc_i else 0)
    W_in, W_res, W_out = (
        _get(op, "W_in"),
        _get(op, "W_res"),
        _get(op, "W_out"),
    )

    # W_in / W_out are always dense globals; the reservoir term's global(s)
    # depend on the topology: a dense W_res, the CSR (val/col/rowptr) triple,
    # or nothing for the structured chains (they read h via chain_weight).
    dense = res_kind == "" and topology in ("ESN_STANDARD", "RANDOM")
    globals_ = [
        _fglobal("W_in", weights[W_in], ety),
        _fglobal("W_out", weights[W_out], ety),
    ]
    if dense:
        globals_.append(_fglobal("W_res", weights[W_res], ety))
    elif res_kind == "csr":
        v = _get(op, "res_val")
        c = _get(op, "res_col")
        rp = _get(op, "res_rowptr")
        nnz = int(np.asarray(weights[v]).size)
        nrp = int(np.asarray(weights[rp]).size)
        globals_ += [
            _fglobal(v, weights[v], ety),
            _dense_global(c, weights[c], 32),
            _dense_global(rp, weights[rp], 32),
        ]

    dyn = MemRefType(ety, [memref.DYNAMIC_INDEX])
    region = Region([Block(arg_types=[_I64, dyn, dyn])])
    with ImplicitBuilder(region.block) as (T, X, Y):
        cN, cK, cM, cF = c_idx(N), c_idx(K), c_idx(M), c_idx(F)
        c0, c1 = c_idx(0), c_idx(1)
        Ti = arith.IndexCastOp(T, _IDX).result
        Win = memref.GetGlobalOp("W_in", MemRefType(ety, [N * K])).memref
        Wout = memref.GetGlobalOp("W_out", MemRefType(ety, [M * F])).memref
        Wres = Wval = Wcol = Wrowptr = None
        if dense:
            Wres = memref.GetGlobalOp("W_res", MemRefType(ety, [N * N])).memref
        elif res_kind == "csr":
            Wval = memref.GetGlobalOp(v, MemRefType(ety, [nnz])).memref
            Wcol = memref.GetGlobalOp(c, MemRefType(_I32, [nnz])).memref
            Wrowptr = memref.GetGlobalOp(rp, MemRefType(_I32, [nrp])).memref
        h = memref.AllocaOp.get(ety, shape=[N]).memref
        pre = memref.AllocaOp.get(ety, shape=[N]).memref
        z, one_ml, leakc, biasc = (
            _fconst(0.0, ety),
            _fconst(1.0 - leak, ety),
            _fconst(leak, ety),
            _fconst(bias, ety),
        )

        def init_body(i, _):
            memref.StoreOp.get(z, h, [i])
            return []

        for_(c0, cN, c1, [], init_body)

        def time_body(t, _):
            tK = arith.MuliOp(t, cK).result
            tM = arith.MuliOp(t, cM).result

            def pre_body(i, _):
                iK = arith.MuliOp(i, cK).result

                def kin(k, args):
                    (a,) = args
                    w = memref.LoadOp.get(
                        Win, [arith.AddiOp(iK, k).result]
                    ).res
                    x = memref.LoadOp.get(X, [arith.AddiOp(tK, k).result]).res
                    return [arith.AddfOp(a, arith.MulfOp(w, x).result).result]

                acc = for_(c0, cK, c1, [biasc], kin)[0]
                # Recurrent W_res@h term: dense matvec (vectorisable via vlen),
                # CSR gather, or an O(N) structured chain (DLR/SCR/DLRB).
                if res_kind == "csr":
                    acc = _res_contrib_csr(acc, i, Wval, Wcol, Wrowptr, h, ety)
                elif topology == "DLR":
                    acc = _res_contrib_dlr(acc, i, h, chain_weight, ety)
                elif topology == "SCR":
                    acc = _res_contrib_scr(acc, i, N, h, chain_weight, ety)
                elif topology == "DLRB":
                    acc = _res_contrib_dlrb(
                        acc, i, N, h, chain_weight, chain_feedback, ety
                    )
                else:
                    iN = arith.MuliOp(i, cN).result
                    acc = _matvec_acc(Wres, iN, h, N, vlen, acc, ety)
                memref.StoreOp.get(acc, pre, [i])
                return []

            for_(c0, cN, c1, [], pre_body)

            def act_body(i, _):
                hv = memref.LoadOp.get(h, [i]).res
                pv = memref.LoadOp.get(pre, [i]).res
                # h <- (1-leak)*h + leak*act(pre)  (act applied to the
                # pre-activation, then the leaky-integrator blend; matches
                # reference._preprocess + _llvm_float._emit_activation).
                av = _emit_activation(pv, activation, ety)
                nh = arith.AddfOp(
                    arith.MulfOp(one_ml, hv).result,
                    arith.MulfOp(leakc, av).result,
                ).result
                memref.StoreOp.get(nh, h, [i])
                return []

            for_(c0, cN, c1, [], act_body)

            def ro_body(m, _):
                mF = arith.MuliOp(m, cF).result
                if inc_b:
                    init = memref.LoadOp.get(Wout, [mF]).res
                else:
                    init = z
                if inc_i:
                    ci = c_idx(off_i)

                    def kb(k, args):
                        (a,) = args
                        widx = arith.AddiOp(
                            mF, arith.AddiOp(ci, k).result
                        ).result
                        w = memref.LoadOp.get(Wout, [widx]).res
                        x = memref.LoadOp.get(
                            X, [arith.AddiOp(tK, k).result]
                        ).res
                        return [
                            arith.AddfOp(a, arith.MulfOp(w, x).result).result
                        ]

                    init = for_(c0, cK, c1, [init], kb)[0]
                base_s = arith.AddiOp(mF, c_idx(off_s)).result
                y = _matvec_acc(Wout, base_s, h, N, vlen, init, ety)
                memref.StoreOp.get(y, Y, [arith.AddiOp(tM, m).result])
                return []

            for_(c0, cM, c1, [], ro_body)
            return []

        for_(c0, Ti, c1, [], time_body)
        func.ReturnOp()
    main = func.FuncOp(func_name, ([_I64, dyn, dyn], []), region)
    main.attributes["llvm.emit_c_interface"] = UnitAttr()

    # External libm scalar declarations for tanh/sigmoid (none for relu/identity).
    externs = [
        func.FuncOp.external(_libm_name(base, ety), [ety], [ety])
        for base in _libm_for(activation)
    ]
    out = ModuleOp([*externs, *globals_, main])
    out.verify()
    buf = io.StringIO()
    Printer(stream=buf).print_op(out)
    return buf.getvalue() + "\n"
