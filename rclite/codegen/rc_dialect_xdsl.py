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
    lower_fused_float(...)            rc.fused_step_readout (float dense, identity
                                      activation) -> arith/memref/scf rc_predict

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
# Lowering: rc.fused_step_readout (float dense, identity activation) -> arith
# ---------------------------------------------------------------------------
# This is the genuine dialect -> MLIR -> executable leg of the spike. It lowers
# the *fused* op to an arith/memref/scf `rc_predict` kernel (no phi buffer; the
# readout reads `h` directly), runnable via the existing mlir_jit pipeline. The
# float/identity case keeps the spike libm-free and is the substrate for the
# Stage-3 `vector`/`affine` lowering where MLIR auto-vectorisation pays off.
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

from .mlir_xdsl_common import c_idx, for_  # noqa: E402

_IDX = IndexType()
_ADD_KIND = vector.CombiningKindAttr(vector.CombiningKindFlag.ADD)


def _matvec_acc(Wmem, base, hmem, N, vlen, acc0):
    """`acc0 + sum_{j<N} Wmem[base+j] * hmem[j]`, returns the scalar f64 acc.

    `vlen <= 1`: a plain scalar reduction (IEEE accumulation order — the order
    LLVM is forced to keep, so it leaves the reduction scalar even at -O3).
    `vlen > 1`: the inner product is accumulated in a `vector<vlen x f64>` via
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
    vt = VectorType(_F64, [vlen])
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
            result_types=[_F64],
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


def _find_fused(mod: ModuleOp):
    loop = next(o for o in mod.walk() if isinstance(o, TimeLoopOp))
    fused = next(
        o for o in loop.body.block.ops if isinstance(o, FusedStepReadoutOp)
    )
    return fused


def _fconst(v):
    return arith.ConstantOp(FloatAttr(float(v), _F64)).result


def _fglobal(name, arr):
    flat = np.asarray(arr, dtype=np.float64).reshape(-1)
    ty = MemRefType(_F64, [int(flat.size)])
    init = DenseIntOrFPElementsAttr.from_list(
        TensorType(_F64, [int(flat.size)]), [float(v) for v in flat]
    )
    return memref.GlobalOp.get(
        StringAttr(name),
        ty,
        init,
        sym_visibility=StringAttr("private"),
        constant=UnitAttr(),
    )


def lower_fused_float(
    mod: ModuleOp, weights: dict, vlen: int = 1, func_name: str = "rc_predict"
) -> str:
    """Lower a fused `rc` module (float dense, identity activation) to an
    arith/memref/scf `rc_predict` kernel and return the printed MLIR.

    Mirrors the fused float reference: per step, `pre = W_in@u + W_res@h + bias`,
    `h = (1-leak)*h + leak*pre` (identity activation), then the readout reads `h`
    directly — `y[m] = [bias] + [W_out_in@u] + W_out_state@h`.

    `vlen` sets the SIMD width of the two N-wide reductions (`W_res@h` and the
    readout `W_out_state@h`): `vlen=1` is the scalar baseline, `vlen>1` emits the
    `vector`-dialect reduction (Stage 3). Lower the vector form with the extended
    pipeline (`mlir_jit` + `--convert-vector-to-llvm`)."""
    op = _find_fused(mod)
    if _get(op, "activation") != "IDENTITY":
        raise NotImplementedError("spike lowering: identity activation only")
    if _get(op, "topology") not in ("ESN_STANDARD", "RANDOM"):
        raise NotImplementedError("spike lowering: dense topology only")
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

    globals_ = [
        _fglobal("W_in", weights[W_in]),
        _fglobal("W_res", weights[W_res]),
        _fglobal("W_out", weights[W_out]),
    ]

    dyn = MemRefType(_F64, [memref.DYNAMIC_INDEX])
    region = Region([Block(arg_types=[_I64, dyn, dyn])])
    with ImplicitBuilder(region.block) as (T, X, Y):
        cN, cK, cM, cF = c_idx(N), c_idx(K), c_idx(M), c_idx(F)
        c0, c1 = c_idx(0), c_idx(1)
        Ti = arith.IndexCastOp(T, _IDX).result
        Win = memref.GetGlobalOp("W_in", MemRefType(_F64, [N * K])).memref
        Wres = memref.GetGlobalOp("W_res", MemRefType(_F64, [N * N])).memref
        Wout = memref.GetGlobalOp("W_out", MemRefType(_F64, [M * F])).memref
        h = memref.AllocaOp.get(_F64, shape=[N]).memref
        pre = memref.AllocaOp.get(_F64, shape=[N]).memref
        z, one_ml, leakc, biasc = (
            _fconst(0.0),
            _fconst(1.0 - leak),
            _fconst(leak),
            _fconst(bias),
        )

        def init_body(i, _):
            memref.StoreOp.get(z, h, [i])
            return []

        for_(c0, cN, c1, [], init_body)

        def time_body(t, _):
            tK = arith.MuliOp(t, cK).result
            tM = arith.MuliOp(t, cM).result

            def pre_body(i, _):
                iK, iN = (
                    arith.MuliOp(i, cK).result,
                    arith.MuliOp(i, cN).result,
                )

                def kin(k, args):
                    (a,) = args
                    w = memref.LoadOp.get(
                        Win, [arith.AddiOp(iK, k).result]
                    ).res
                    x = memref.LoadOp.get(X, [arith.AddiOp(tK, k).result]).res
                    return [arith.AddfOp(a, arith.MulfOp(w, x).result).result]

                acc = for_(c0, cK, c1, [biasc], kin)[0]
                acc = _matvec_acc(Wres, iN, h, N, vlen, acc)
                memref.StoreOp.get(acc, pre, [i])
                return []

            for_(c0, cN, c1, [], pre_body)

            def act_body(i, _):
                hv = memref.LoadOp.get(h, [i]).res
                pv = memref.LoadOp.get(pre, [i]).res
                nh = arith.AddfOp(
                    arith.MulfOp(one_ml, hv).result,
                    arith.MulfOp(leakc, pv).result,
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
                y = _matvec_acc(Wout, base_s, h, N, vlen, init)
                memref.StoreOp.get(y, Y, [arith.AddiOp(tM, m).result])
                return []

            for_(c0, cM, c1, [], ro_body)
            return []

        for_(c0, Ti, c1, [], time_body)
        func.ReturnOp()
    main = func.FuncOp(func_name, ([_I64, dyn, dyn], []), region)
    main.attributes["llvm.emit_c_interface"] = UnitAttr()

    out = ModuleOp([*globals_, main])
    out.verify()
    buf = io.StringIO()
    Printer(stream=buf).print_op(out)
    return buf.getvalue() + "\n"
