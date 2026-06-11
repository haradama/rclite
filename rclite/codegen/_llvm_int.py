"""Symmetric fixed-point (i8/i16/i32) LLVM lowering for the RC IDL.

`_IntLowerer` emits an integer ``rc_predict`` kernel from a quantized
model; `emit_quantized_module` is the public entry point. Split out of
the former monolithic ``llvm.py``.
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
import llvmlite.binding as llvm

from rclite.core.profile import Topology
from ._llvm_common import (
    _I32,
    _I64,
    _ci,
    _load1d,
    _load1d_global,
    _load2d_global,
    _loop,
    _loop_strided,
    _pow2_exp,
    _store1d,
)


def emit_quantized_module(
    qmodel, *, passes=None, saturating: bool = True, head=None
) -> ir.Module:
    """Build LLVM IR for the integer quantized path (i32, i16, or i8).

    Function signature:
        void rc_predict(int64_t T, storage_t* X, storage_t* Y);
    where storage_t is int32_t / int16_t / int8_t for the corresponding
    `I32FixedPoint` / `I16FixedPoint` / `I8Symmetric` target. With
    `head="classify"`, Y is `int32_t*` (one class id per step).

    `saturating=True` wraps inner-loop accumulations and the final
    truncation with `@llvm.sadd.sat.*` and clamping selects, so overflow
    saturates instead of wrapping. Strongly recommended for i16 / i8
    (narrow range); cheap to leave on for i32 as well.
    """
    from rclite.quant.ir_builder import build_ir_from_quantized

    ir_module = build_ir_from_quantized(qmodel, head=head)
    if passes is None:
        passes = []
    for p in passes:
        ir_module = p(ir_module)
    return _IntLowerer(ir_module, saturating=saturating).lower()


# ----------------------------------------------------------------------------
# Integer (quantized) lowering


class _IntLowerer:
    """Lower an rclite IR module under `dtype` in {'i32', 'i16', 'i8'} to LLVM IR.

    Parameterized over storage_ty / accum_ty:
        i32 storage : accumulator i64    (mirage default)
        i16 storage : accumulator i32    (-OS / size-constrained)
        i8  storage : accumulator i32    (smallest footprint; symmetric Q-format)

    Fixed-point multiply pattern:
        a:storage * b:storage -> sext to accum -> mul -> ashr -> trunc storage
    Shift amounts depend on operand provenance:
        W_in * input_q   : shift = weight_frac + input_frac - state_frac
        W_res * state_q  : shift = weight_frac
        state * leak_q   : shift = state_frac
        readout accum    : accum_ty accumulator, final shift by state_frac
    Tanh is realized by linear-interpolated LUT lookup; the lookup itself
    uses i32 intermediates regardless of storage width (LUT index/position
    can exceed i16 range).

    `saturating=True` swaps plain integer add for `@llvm.sadd.sat.*` in the
    matmul accumulators (essential for i8/i16 where overflow is realistic).
    """

    def __init__(self, ir_module, *, saturating: bool = True):
        from rclite.ir.ops import (
            BuildPhi,
            ReadoutLinear,
            FusedStepReadout,
            Argmax,
            Softmax,
        )

        self.ir_module = ir_module
        md = ir_module.metadata
        dtype = md.get("dtype")
        if dtype == "i32":
            # Mirage-compatible: storage and per-row accumulator both i32;
            # full i64 product only as an intermediate inside fixed_mul.
            self.storage_ty = _I32
            self.accum_ty = _I32
            self.product_ty = _I64
            self.storage_bits = 32
            self.accum_bits = 32
        elif dtype == "i16":
            # i16 stores narrowly, but per-row accumulation must widen to
            # i32 to survive sums over N terms.
            self.storage_ty = ir.IntType(16)
            self.accum_ty = _I32
            self.product_ty = _I32
            self.storage_bits = 16
            self.accum_bits = 32
        elif dtype == "i8":
            # i8 storage with i32 accumulator. The product i8*i8 fits in
            # i16, but we widen to i32 so post-shift accumulation has
            # plenty of headroom — the per-row matmul sums N terms.
            self.storage_ty = ir.IntType(8)
            self.accum_ty = _I32
            self.product_ty = _I32
            self.storage_bits = 8
            self.accum_bits = 32
        else:
            raise ValueError(
                f"_IntLowerer supports dtype in {{'i32', 'i16', 'i8'}}, got {dtype!r}"
            )
        self.saturating = saturating

        self.state_frac = int(md["state_frac"])
        self.input_frac = int(md["input_frac"])
        self.weight_frac = int(md["weight_frac"])
        self.lut_n = int(md["lut_n"])
        self.lut_xmin_q = int(md["lut_xmin_q"])
        self.lut_xmax_q = int(md["lut_xmax_q"])
        self.leak_q = int(md["leak_q"])
        self.bias_q = int(md["bias_q"])

        self.K, self.N, self.M = ir_module.K, ir_module.N, ir_module.M
        self.shift_in = self.weight_frac + self.input_frac - self.state_frac
        self.shift_res = self.weight_frac
        self.one_minus_leak_q = (1 << self.state_frac) - self.leak_q

        self.module = ir.Module(name=f"rc_jit_{dtype}_{id(ir_module)}")
        self.module.triple = llvm.get_default_triple()

        # Declare saturating add intrinsic for the accumulator type
        # (used in the recurrent matmul where overflow risk is highest).
        if saturating:
            sat_name = f"llvm.sadd.sat.i{self.accum_bits}"
            self.sadd_sat_fn = ir.Function(
                self.module,
                ir.FunctionType(self.accum_ty, [self.accum_ty, self.accum_ty]),
                name=sat_name,
            )
        else:
            self.sadd_sat_fn = None

        # Classification head: argmax produces an int32 class id per step;
        # softmax produces M probabilities (storage type, Q.sm_prob_frac).
        # Both route the readout through a logits scratch.
        flat = list(self._flatten_ops())
        self.out_int = any(isinstance(op, Argmax) for op in flat)
        self.has_softmax = any(isinstance(op, Softmax) for op in flat)
        self.has_head = self.out_int or self.has_softmax
        out_ty = _I32 if self.out_int else self.storage_ty
        if self.has_softmax:
            self.sm_dmin_q = int(md["sm_dmin_q"])
            self.sm_n = int(md["sm_n"])
            self.sm_idx_frac = int(md["sm_idx_frac"])
            self.sm_prob_frac = int(md["sm_prob_frac"])

        # Weight / LUT globals at storage_ty (i32 or i16)
        self.globals = {}
        for name, arr in ir_module.weights.items():
            self.globals[name] = self._emit_int_global(name, arr)

        fnty = ir.FunctionType(
            ir.VoidType(),
            [_I64, self.storage_ty.as_pointer(), out_ty.as_pointer()],
        )
        self.fn = ir.Function(self.module, fnty, name="rc_predict")
        self.T_arg, self.X_arg, self.Y_arg = self.fn.args
        self.T_arg.name = "T"
        self.X_arg.name = "X"
        self.Y_arg.name = "Y"

        entry = self.fn.append_basic_block("entry")
        self.b = ir.IRBuilder(entry)

        needs_phi = any(
            isinstance(op, (ReadoutLinear, BuildPhi)) for op in flat
        )
        max_F = max(
            (
                op.F
                for op in flat
                if isinstance(op, (ReadoutLinear, FusedStepReadout))
            ),
            default=self.N + self.K + 1,
        )

        self.h = self.b.alloca(self.storage_ty, size=_ci(self.N), name="h")
        self.pre_arr = self.b.alloca(
            self.storage_ty, size=_ci(self.N), name="pre"
        )
        self.u_pre = self.b.alloca(
            self.storage_ty,
            size=_ci(max(self.K, 1)),
            name="u_pre",
        )
        self.phi_arr = (
            self.b.alloca(self.storage_ty, size=_ci(max(max_F, 1)), name="phi")
            if needs_phi
            else None
        )
        # Logits scratch (storage_ty) when a classification head consumes the
        # readout; argmax compares these monotone-quantized scores.
        self.logits = (
            self.b.alloca(
                self.storage_ty, size=_ci(max(self.M, 1)), name="logits"
            )
            if self.has_head
            else None
        )
        # exp() scratch (i32, Q.sm_prob_frac) for the softmax head.
        self.exp_scratch = (
            self.b.alloca(_I32, size=_ci(max(self.M, 1)), name="exp_q")
            if self.has_softmax
            else None
        )
        # Accumulator is in accum_ty (wider) — protects against per-row overflow
        # in the matmul over N terms.
        self.acc = self.b.alloca(self.accum_ty, name="acc")
        self.acc64 = self.b.alloca(_I64, name="acc64")  # readout always i64

        with _loop(self.b, _ci(self.N), "init") as i:
            _store1d(self.b, self.h, i, self._cs(0))

        self.t = None

    # ------------------------------------------------------------------
    # helpers

    def _cs(self, v: int) -> ir.Constant:
        """Constant in storage type (i16 or i32)."""
        return ir.Constant(self.storage_ty, int(v))

    def _ca(self, v: int) -> ir.Constant:
        """Constant in accumulator type (i32 or i64)."""
        return ir.Constant(self.accum_ty, int(v))

    def _ci32(self, v: int) -> ir.Constant:
        return ir.Constant(_I32, int(v))

    def _ci64(self, v: int) -> ir.Constant:
        return ir.Constant(_I64, int(v))

    def _emit_int_global(self, name, arr):
        # CSR index arrays (col / rowptr) are always i32 regardless of the
        # storage width; only quantized weight/val arrays use storage_ty.
        if name.endswith(("_col", "_rowptr")):
            flat = np.asarray(arr).reshape(-1).astype(np.int32)
            ty = ir.ArrayType(_I32, flat.size)
            g = ir.GlobalVariable(self.module, ty, name=name)
            g.linkage = "internal"
            g.global_constant = True
            g.initializer = ir.Constant(ty, [self._ci32(int(v)) for v in flat])
            return g
        if self.storage_bits == 8:
            np_dtype = np.int8
        elif self.storage_bits == 16:
            np_dtype = np.int16
        else:
            np_dtype = np.int32
        flat = np.asarray(arr).reshape(-1).astype(np_dtype)
        ty = ir.ArrayType(self.storage_ty, flat.size)
        g = ir.GlobalVariable(self.module, ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(ty, [self._cs(int(v)) for v in flat])
        return g

    def _accum_add(self, a, b_val):
        """Add two accum_ty values. Optionally use saturating intrinsic."""
        if self.saturating and self.sadd_sat_fn is not None:
            return self.b.call(self.sadd_sat_fn, [a, b_val])
        return self.b.add(a, b_val)

    def _flatten_ops(self):
        from rclite.ir.ops import TimeLoop

        for op in self.ir_module.ops:
            yield op
            if isinstance(op, TimeLoop):
                yield from op.body

    def _fixed_mul_to_storage(self, a, b_val, shift: int):
        """(a * b_val) >> shift, storage→product promote, ashr, trunc back to storage."""
        a_p = self.b.sext(a, self.product_ty)
        b_p = self.b.sext(b_val, self.product_ty)
        prod = self.b.mul(a_p, b_p)
        shifted = self.b.ashr(prod, ir.Constant(self.product_ty, shift))
        return self.b.trunc(shifted, self.storage_ty)

    def _fixed_mul_to_accum(self, a, b_val, shift: int):
        """Same operation but result returned in accum_ty.

        For i32 (accum_ty == storage_ty == i32), identical to to_storage.
        For i16 (accum_ty == i32 > storage_ty == i16), keeps the wider
        product/shift result so per-row accumulation has headroom.
        """
        a_p = self.b.sext(a, self.product_ty)
        b_p = self.b.sext(b_val, self.product_ty)
        prod = self.b.mul(a_p, b_p)
        shifted = self.b.ashr(prod, ir.Constant(self.product_ty, shift))
        if self.product_ty == self.accum_ty:
            return shifted
        if self.product_ty.width > self.accum_ty.width:
            return self.b.trunc(shifted, self.accum_ty)
        return self.b.sext(shifted, self.accum_ty)

    def _fixed_const_mul_to_accum(self, wv: int, s, shift: int):
        """(wv * s) >> shift in accum_ty, folding the multiply when wv==+-2**k.

        For wv==+-2**k the product `mul(2**k, sext(s))` equals
        `shl(sext(s), k)` bit-for-bit in the wide product_ty (no overflow:
        product_ty holds storage*storage), and a negative power negates the
        shifted value -- so the subsequent ashr/convert is bit-identical to
        `_fixed_mul_to_accum`. Falls back to the multiply otherwise.
        """
        k = _pow2_exp(wv)
        if k is None:
            return self._fixed_mul_to_accum(self._cs(int(wv)), s, shift)
        b = self.b
        s_p = b.sext(s, self.product_ty)
        if k > 0:
            s_p = b.shl(s_p, ir.Constant(self.product_ty, k))
        if wv < 0:
            s_p = b.sub(ir.Constant(self.product_ty, 0), s_p)
        shifted = b.ashr(s_p, ir.Constant(self.product_ty, shift))
        if self.product_ty == self.accum_ty:
            return shifted
        if self.product_ty.width > self.accum_ty.width:
            return b.trunc(shifted, self.accum_ty)
        return b.sext(shifted, self.accum_ty)

    # ------------------------------------------------------------------
    # dispatcher

    def lower(self) -> ir.Module:
        for op in self.ir_module.ops:
            self._lower(op)
        self.b.ret_void()
        return self.module

    def _lower(self, op):
        from rclite.ir.ops import (
            TimeLoop,
            PreprocessInput,
            ReservoirStep,
            BuildPhi,
            ReadoutLinear,
            Argmax,
            Softmax,
        )

        if isinstance(op, TimeLoop):
            return self._lower_time_loop(op)
        if isinstance(op, PreprocessInput):
            return self._lower_preprocess(op)
        if isinstance(op, ReservoirStep):
            return self._lower_reservoir_step(op)
        if isinstance(op, BuildPhi):
            return self._lower_build_phi(op)
        if isinstance(op, ReadoutLinear):
            return self._lower_readout_linear(op)
        if isinstance(op, Argmax):
            return self._lower_argmax(op)
        if isinstance(op, Softmax):
            return self._lower_softmax(op)
        raise NotImplementedError(
            f"{type(op).__name__} not supported in the integer path"
        )

    def _lower_time_loop(self, op):
        with _loop(self.b, self.T_arg, "t") as t:
            self.t = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None

    def _lower_argmax(self, op):
        """class_id = argmax_m logits[m] over the monotone quantized scores."""
        b = self.b
        best_v = b.alloca(self.storage_ty, name="best_v")
        best_i = b.alloca(_I64, name="best_i")
        b.store(_load1d(b, self.logits, _ci(0)), best_v)
        b.store(_ci(0), best_i)
        with _loop(b, _ci(op.M), "am") as m:
            v = _load1d(b, self.logits, m)
            is_gt = b.icmp_signed(">", v, b.load(best_v))
            b.store(b.select(is_gt, v, b.load(best_v)), best_v)
            b.store(b.select(is_gt, m, b.load(best_i)), best_i)
        _store1d(b, self.Y_arg, self.t, b.trunc(b.load(best_i), _I32))

    def _lower_softmax(self, op):
        """Fixed-point softmax (exp LUT), bit-exact with softmax_q.

        Writes M probabilities at Q.sm_prob_frac into Y (storage type).
        """
        b = self.b
        g_lut = self.globals["sm_lut"]
        n = self.sm_n
        idxf = self.sm_idx_frac
        dmin = self.sm_dmin_q
        pf = self.sm_prob_frac
        M = op.M
        qmax = (1 << (self.storage_bits - 1)) - 1

        # max over logits (i32)
        mx = b.alloca(_I32, name="sm_max")
        b.store(b.sext(_load1d(b, self.logits, _ci(0)), _I32), mx)
        with _loop(b, _ci(M), "smx") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            b.store(
                b.select(b.icmp_signed(">", v, b.load(mx)), v, b.load(mx)), mx
            )

        # exp(d) via clamped, linearly-interpolated LUT; accumulate sum (i64)
        sum_acc = b.alloca(_I64, name="sm_sum")
        b.store(self._ci64(0), sum_acc)
        with _loop(b, _ci(M), "sme") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            d = b.sub(v, b.load(mx))  # <= 0
            d = b.select(
                b.icmp_signed("<", d, self._ci32(dmin)), self._ci32(dmin), d
            )
            num = b.sub(d, self._ci32(dmin))  # [0, -dmin]
            # pos = (num * (n-1) << idxf) / (-dmin)   in i64
            num64 = b.sext(num, _I64)
            posn = b.shl(b.mul(num64, self._ci64(n - 1)), self._ci64(idxf))
            pos = b.sdiv(posn, self._ci64(-dmin))
            i0 = b.ashr(pos, self._ci64(idxf))  # i64 index
            i0 = b.select(
                b.icmp_signed("<", i0, self._ci64(0)), self._ci64(0), i0
            )
            i0 = b.select(
                b.icmp_signed(">", i0, self._ci64(n - 2)),
                self._ci64(n - 2),
                i0,
            )
            frac = b.sub(pos, b.shl(i0, self._ci64(idxf)))
            y0 = b.sext(_load1d_global(b, g_lut, i0), _I64)
            y1 = b.sext(
                _load1d_global(b, g_lut, b.add(i0, self._ci64(1))), _I64
            )
            e = b.add(y0, b.ashr(b.mul(b.sub(y1, y0), frac), self._ci64(idxf)))
            _store1d(b, self.exp_scratch, m, b.trunc(e, _I32))
            b.store(b.add(b.load(sum_acc), e), sum_acc)

        # normalize: p = (e << prob_frac) / sum, clamp to qmax, store
        s = b.load(sum_acc)
        with _loop(b, _ci(M), "smn") as m:
            e = b.sext(_load1d(b, self.exp_scratch, m), _I64)
            p = b.sdiv(b.shl(e, self._ci64(pf)), s)
            p = b.select(
                b.icmp_signed(">", p, self._ci64(qmax)), self._ci64(qmax), p
            )
            tM = b.mul(self.t, _ci(M))
            _store1d(b, self.Y_arg, b.add(tM, m), b.trunc(p, self.storage_ty))

    def _lower_preprocess(self, op):
        """u_pre_q[k] := ((X_raw_q[k] - offset_q) * scaling_q) >> weight_frac

        Both `X_raw_q` and `offset_q` live at input_scale (Q.input_frac).
        `scaling_q` is quantized at weight_scale, so the multiply lands at
        input_scale * weight_scale; shifting right by weight_frac brings the
        result back to input_scale — the scale `ReservoirStep` expects for
        its `u_pre` operand.
        """
        if op.K == 0:
            return
        input_scale = 1 << self.input_frac
        weight_scale = 1 << self.weight_frac
        offset_q = int(round(op.offset * input_scale))
        scaling_q = int(round(op.scale * weight_scale))
        offset_const = self._cs(offset_q)
        scale_const = self._cs(scaling_q)

        tK = self.b.mul(self.t, _ci(op.K))
        with _loop(self.b, _ci(op.K), "kpre") as k:
            x_raw_q = _load1d(self.b, self.X_arg, self.b.add(tK, k))
            diff = self.b.sub(x_raw_q, offset_const)
            u_pre_val = self._fixed_mul_to_storage(
                diff,
                scale_const,
                self.weight_frac,
            )
            _store1d(self.b, self.u_pre, k, u_pre_val)

    def _lower_reservoir_step(self, op):
        g_Win = self.globals["W_in"]
        g_Wres = self.globals.get(op.W_res_name) if op.W_res_name else None
        g_lut = self.globals["lut_table"]
        K, N = op.K, op.N

        spec = op.res_sparse
        if spec is not None and spec.kind == "unroll":
            # Per-row nonzero sets differ → unroll the outer i-loop.
            for i in range(N):
                self._emit_int_row(op, g_Win, g_Wres, _ci(i), spec, i_py=i)
        else:
            with _loop(self.b, _ci(N), "ipre") as i:
                self._emit_int_row(op, g_Win, g_Wres, i, spec, i_py=None)

        with _loop(self.b, _ci(N), "iupd") as i:
            pre_i = _load1d(self.b, self.pre_arr, i)
            activated = self._emit_lut_lookup(pre_i, g_lut)
            h_old = _load1d(self.b, self.h, i)
            t1 = self._fixed_mul_to_storage(
                h_old, self._cs(self.one_minus_leak_q), self.state_frac
            )
            t2 = self._fixed_mul_to_storage(
                activated, self._cs(self.leak_q), self.state_frac
            )
            new_h = self.b.add(t1, t2)
            _store1d(self.b, self.h, i, new_h)

    def _emit_int_row(self, op, g_Win, g_Wres, i, spec, i_py):
        """Compute pre[row i] = trunc(bias + W_in·u + W_res·h) into pre_arr.

        `i` is an SSA index (constant when unrolling). For the unrolled
        kernel (`i_py` is the Python row index) the recurrent term is the
        baked nonzeros in `spec.rows[i_py]`; otherwise the topology kernel
        (dense / CSR / structured chain) runs inside the runtime i-loop.
        """
        b, K = self.b, op.K
        b.store(self._ca(self.bias_q), self.acc)
        with _loop(b, _ci(K), "kin") as k:
            w = _load2d_global(b, g_Win, K, i, k)
            u = _load1d(b, self.u_pre, k)
            prod = self._fixed_mul_to_accum(w, u, self.shift_in)
            b.store(self._accum_add(b.load(self.acc), prod), self.acc)
        if i_py is not None:  # unrolled sparse
            for j, wv in spec.rows[i_py]:
                s = _load1d(b, self.h, _ci(j))
                prod = self._fixed_const_mul_to_accum(
                    int(wv), s, self.shift_res
                )
                b.store(self._accum_add(b.load(self.acc), prod), self.acc)
        elif spec is not None:  # CSR
            self._emit_res_contrib_int_csr(spec, i)
        else:  # dense / structured chain
            self._emit_res_contrib_int(op, g_Wres, i)
        pre_val = b.trunc(b.load(self.acc), self.storage_ty)
        _store1d(b, self.pre_arr, i, pre_val)

    def _emit_res_contrib_int_csr(self, spec, i):
        """W_res·h over row i's nonzeros (CSR), fixed-point, ascending col."""
        b = self.b
        g_val = self.globals[spec.val_name]
        g_col = self.globals[spec.col_name]
        g_rowptr = self.globals[spec.rowptr_name]
        start = b.sext(_load1d_global(b, g_rowptr, i), _I64)
        end = b.sext(_load1d_global(b, g_rowptr, b.add(i, _ci(1))), _I64)
        with _loop_strided(b, start, end, _ci(1), "csr") as p:
            j = b.sext(_load1d_global(b, g_col, p), _I64)
            w = _load1d_global(b, g_val, p)
            s = _load1d(b, self.h, j)
            prod = self._fixed_mul_to_accum(w, s, self.shift_res)
            b.store(self._accum_add(b.load(self.acc), prod), self.acc)

    def _emit_res_contrib_int(self, op, g_Wres, i):
        """Add the W_res @ h contribution to `self.acc`, branching on topology.

        Mirrors `_Lowerer._emit_res_contrib` but with fixed-point arithmetic.
        For DLR/SCR/DLRB this emits O(1) work per row using the scalar
        `chain_weight` (and `chain_feedback` for DLRB) — quantized at
        weight_scale to match the dense quantized matrix's representation
        at the nonzero positions. Dense matmul is the fallback for RANDOM /
        ESN_STANDARD topologies.
        """
        b = self.b
        N = op.N
        weight_scale = 1 << self.weight_frac

        if op.topology == Topology.DLR:
            # h[i-1] contribution for i > 0; mask via select
            cw_q = int(round(op.chain_weight * weight_scale))
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_safe = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val = _load1d(b, self.h, i_safe)
            prod = self._fixed_mul_to_accum(
                self._cs(cw_q), val, self.shift_res
            )
            contrib = b.select(is_pos, prod, self._ca(0))
            b.store(self._accum_add(b.load(self.acc), contrib), self.acc)
        elif op.topology == Topology.SCR:
            # Cyclic chain: prev = (i - 1) mod N
            cw_q = int(round(op.chain_weight * weight_scale))
            is_zero = b.icmp_signed("==", i, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i, _ci(1)))
            val = _load1d(b, self.h, i_prev)
            prod = self._fixed_mul_to_accum(
                self._cs(cw_q), val, self.shift_res
            )
            b.store(self._accum_add(b.load(self.acc), prod), self.acc)
        elif op.topology == Topology.DLRB:
            cw_q = int(round(op.chain_weight * weight_scale))
            cb_q = int(round(op.chain_feedback * weight_scale))
            # Backward chain: chain_weight * h[i-1] for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val_back = _load1d(b, self.h, i_back)
            prod_back = self._fixed_mul_to_accum(
                self._cs(cw_q), val_back, self.shift_res
            )
            contrib_back = b.select(is_pos, prod_back, self._ca(0))
            # Forward chain: chain_feedback * h[i+1] for i < N-1
            is_lt_last = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt_last, b.add(i, _ci(1)), _ci(N - 1))
            val_fwd = _load1d(b, self.h, i_fwd)
            prod_fwd = self._fixed_mul_to_accum(
                self._cs(cb_q), val_fwd, self.shift_res
            )
            contrib_fwd = b.select(is_lt_last, prod_fwd, self._ca(0))
            acc_val = b.load(self.acc)
            b.store(
                self._accum_add(
                    self._accum_add(acc_val, contrib_back), contrib_fwd
                ),
                self.acc,
            )
        else:
            # Dense matmul fallback (RANDOM / ESN_STANDARD)
            if g_Wres is None:
                raise RuntimeError(
                    f"dense matmul requested but W_res not in globals "
                    f"(topology={op.topology.name})"
                )
            with _loop(b, _ci(N), "jres") as j:
                w = _load2d_global(b, g_Wres, N, i, j)
                s = _load1d(b, self.h, j)
                prod = self._fixed_mul_to_accum(w, s, self.shift_res)
                b.store(self._accum_add(b.load(self.acc), prod), self.acc)

    def _emit_lut_lookup(self, x_q, g_lut):
        """Quantized tanh LUT with linear interpolation.

        Internal arithmetic uses i32 (pos_q and t_q may exceed i16 range).
        Input/output are storage_ty (i16 or i32) — sign-extension and
        truncation happen at the boundaries.
        """
        b = self.b
        sf = self.state_frac
        n = self.lut_n

        # Widen input to i32 if needed
        x32 = x_q if self.storage_bits >= 32 else b.sext(x_q, _I32)
        xmin = self._ci32(self.lut_xmin_q)
        xmax = self._ci32(self.lut_xmax_q)

        is_lo = b.icmp_signed("<", x32, xmin)
        x1 = b.select(is_lo, xmin, x32)
        is_hi = b.icmp_signed(">", x1, xmax)
        x = b.select(is_hi, xmax, x1)

        num64 = b.sext(b.sub(x, xmin), _I64)
        denom64 = self._ci64(self.lut_xmax_q - self.lut_xmin_q)
        shl = b.shl(num64, self._ci64(sf))
        div = b.sdiv(shl, denom64)
        t_q = b.trunc(div, _I32)

        n_minus1 = self._ci32(n - 1)
        pos_q = b.mul(t_q, n_minus1)

        i0_raw = b.ashr(pos_q, self._ci32(sf))
        n_minus2 = self._ci32(n - 2)
        too_big = b.icmp_signed(">", i0_raw, n_minus2)
        i0_c1 = b.select(too_big, n_minus2, i0_raw)
        zero32 = self._ci32(0)
        too_neg = b.icmp_signed("<", i0_c1, zero32)
        i0 = b.select(too_neg, zero32, i0_c1)
        i1 = b.add(i0, self._ci32(1))

        i0_shl = b.shl(i0, self._ci32(sf))
        frac_q = b.sub(pos_q, i0_shl)

        i0_idx = b.sext(i0, _I64)
        i1_idx = b.sext(i1, _I64)
        # LUT entries are storage_ty; widen to i32 for interp arithmetic.
        y0_s = _load1d_global(b, g_lut, i0_idx)
        y1_s = _load1d_global(b, g_lut, i1_idx)
        y0_32 = y0_s if self.storage_bits >= 32 else b.sext(y0_s, _I32)
        y1_32 = y1_s if self.storage_bits >= 32 else b.sext(y1_s, _I32)
        dy = b.sub(y1_32, y0_32)
        # dy * frac_q >> sf, all in i32
        dy_64 = b.sext(dy, _I64)
        frac_64 = b.sext(frac_q, _I64)
        dy_frac_64 = b.ashr(b.mul(dy_64, frac_64), self._ci64(sf))
        dy_frac = b.trunc(dy_frac_64, _I32)
        result_32 = b.add(y0_32, dy_frac)
        # Truncate back to storage_ty
        if self.storage_bits >= 32:
            return result_32
        return b.trunc(result_32, self.storage_ty)

    def _lower_build_phi(self, op):
        if self.phi_arr is None:
            raise RuntimeError("BuildPhi requires phi buffer")
        K, N = op.K, op.N
        tK = self.b.mul(self.t, _ci(K))
        off = 0
        if op.include_bias:
            # phi[0] = (1 << state_frac) so phi[0] * W_out_q[0] gives
            # state_scale^2 like all other contributions.
            _store1d(
                self.b, self.phi_arr, _ci(off), self._cs(1 << self.state_frac)
            )
            off += 1
        if op.include_input:
            with _loop(self.b, _ci(K), "kphi") as k:
                u_val = _load1d(self.b, self.X_arg, self.b.add(tK, k))
                _store1d(self.b, self.phi_arr, self.b.add(_ci(off), k), u_val)
            off += K
        with _loop(self.b, _ci(N), "iphi") as i:
            _store1d(
                self.b,
                self.phi_arr,
                self.b.add(_ci(off), i),
                _load1d(self.b, self.h, i),
            )

    def _lower_readout_linear(self, op):
        """Readout in i64 accumulator regardless of storage width.

        Optionally uses `@llvm.sadd.sat.i64` for accumulation when
        `saturating=True`. Final i64 → storage_ty truncation happens after
        the >> state_frac shift, with saturation to the storage range.
        """
        g_Wout = self.globals["W_out"]
        F = op.F
        tM = self.b.mul(self.t, _ci(op.M))
        sadd_i64 = (
            self.module.globals.get("llvm.sadd.sat.i64")
            if self.saturating
            else None
        )
        if self.saturating and sadd_i64 is None:
            sadd_i64 = ir.Function(
                self.module,
                ir.FunctionType(_I64, [_I64, _I64]),
                name="llvm.sadd.sat.i64",
            )

        with _loop(self.b, _ci(op.M), "m") as m:
            self.b.store(self._ci64(0), self.acc64)
            with _loop(self.b, _ci(F), "fout") as fi:
                w = _load2d_global(self.b, g_Wout, F, m, fi)
                pv = _load1d(self.b, self.phi_arr, fi)
                w64 = self.b.sext(w, _I64)
                pv64 = self.b.sext(pv, _I64)
                prod = self.b.mul(w64, pv64)
                cur = self.b.load(self.acc64)
                summed = (
                    self.b.call(sadd_i64, [cur, prod])
                    if self.saturating
                    else self.b.add(cur, prod)
                )
                self.b.store(summed, self.acc64)
            shifted = self.b.ashr(
                self.b.load(self.acc64), self._ci64(self.state_frac)
            )
            # Saturating truncation to storage_ty: clamp to storage range
            # before truncation to avoid wrap-around.
            if self.storage_bits == 32:
                y = self.b.trunc(shifted, _I32)
            else:
                lo = self._ci64(-(1 << (self.storage_bits - 1)))
                hi = self._ci64((1 << (self.storage_bits - 1)) - 1)
                clipped_lo = self.b.select(
                    self.b.icmp_signed("<", shifted, lo), lo, shifted
                )
                clipped = self.b.select(
                    self.b.icmp_signed(">", clipped_lo, hi), hi, clipped_lo
                )
                y = self.b.trunc(clipped, self.storage_ty)
            if self.logits is not None:
                _store1d(self.b, self.logits, m, y)
            else:
                _store1d(self.b, self.Y_arg, self.b.add(tM, m), y)


# ----------------------------------------------------------------------------
# Affine (asymmetric per-tensor) lowering
