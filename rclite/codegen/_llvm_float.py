"""Float (f64/f32) LLVM lowering for the RC IDL.

`_Lowerer` emits a ``void rc_predict(int64_t T, T* X, T* Y)`` kernel for
a trained `ReservoirComputer`; `emit_module` is the public entry point.
Split out of the former monolithic ``llvm.py``.
"""

from __future__ import annotations

import numpy as np
from llvmlite import ir
import llvmlite.binding as llvm

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Activation, Topology
from rclite.runtime.reference import RCExecutor
from ._llvm_common import (
    _SUPPORTED_ACTIVATIONS,
    _F32,
    _I32,
    _I64,
    _ci,
    _ci32,
    _dtype_bindings,
    _load1d,
    _load1d_global,
    _load2d_global,
    _loop,
    _loop_strided,
    _store1d,
)


class _Lowerer:
    """Walks an `rclite.ir.Module` and emits LLVM IR."""

    def __init__(self, ir_module, dtype: str):
        from rclite.ir.ops import (
            ReadoutLinear,
            BuildPhi,
            FusedStepReadout,
            Argmax,
            Softmax,
            AccumulateState,
        )

        self.ir_module = ir_module
        self.fty, _, self.np_dtype, _ = _dtype_bindings(dtype)
        self.K, self.N, self.M = ir_module.K, ir_module.N, ir_module.M

        self.module = ir.Module(name=f"rc_jit_{id(ir_module)}")
        self.module.triple = llvm.get_default_triple()

        # libm scalar functions are declared lazily and cached by name so the
        # f32/f64 variant ("tanh"/"tanhf", "exp"/"expf") is only imported when
        # an activation or head actually needs it. relu/identity import nothing.
        self._libm_cache = {}

        # Classification heads: argmax produces an i32 output; both heads make
        # the readout write to a logits scratch buffer rather than Y directly.
        flat = list(self._flatten_ops())
        self.out_int = any(isinstance(op, Argmax) for op in flat)
        self.has_head = any(isinstance(op, (Argmax, Softmax)) for op in flat)
        self.needs_state_sum = any(
            isinstance(op, AccumulateState) and op.mode == "mean"
            for op in flat
        )
        out_ty = _I32 if self.out_int else self.fty

        # Emit weight globals
        self.globals = {}
        for name, arr in ir_module.weights.items():
            self.globals[name] = self._emit_global(name, arr)

        # rc_predict function
        fnty = ir.FunctionType(
            ir.VoidType(),
            [_I64, self.fty.as_pointer(), out_ty.as_pointer()],
        )
        self.fn = ir.Function(self.module, fnty, name="rc_predict")
        self.T_arg, self.X_arg, self.Y_arg = self.fn.args
        self.T_arg.name = "T"
        self.X_arg.name = "X"
        self.Y_arg.name = "Y"

        entry = self.fn.append_basic_block("entry")
        self.b = ir.IRBuilder(entry)

        # Determine scratch sizes
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

        self.h = self.b.alloca(self.fty, size=_ci(self.N), name="h")
        self.u_pre = self.b.alloca(self.fty, size=_ci(self.K), name="u_pre")
        self.pre_arr = self.b.alloca(self.fty, size=_ci(self.N), name="pre")
        self.phi_arr = (
            self.b.alloca(self.fty, size=_ci(max(max_F, 1)), name="phi")
            if needs_phi
            else None
        )
        self.acc = self.b.alloca(self.fty, name="acc")
        # Logits scratch when a classification head consumes the readout.
        self.logits = (
            self.b.alloca(self.fty, size=_ci(max(self.M, 1)), name="logits")
            if self.has_head
            else None
        )
        # Running state sum + step count for MEAN time-pooling.
        if self.needs_state_sum:
            self.h_sum = self.b.alloca(
                self.fty, size=_ci(self.N), name="h_sum"
            )
            with _loop(self.b, _ci(self.N), "sinit") as i:
                _store1d(self.b, self.h_sum, i, self._cf(0.0))
        else:
            self.h_sum = None

        # Init h to zero
        with _loop(self.b, _ci(self.N), "init") as i:
            _store1d(self.b, self.h, i, self._cf(0.0))

        self.t = None  # current time index, valid inside a TimeLoop body
        self.row = None  # current output row (= t per-step, = 0 post-loop)

    def _cf(self, v):
        return ir.Constant(self.fty, float(v))

    def _libm(self, base):
        """Declare (once) and return an external libm scalar function.

        `base` is the f64 name ("tanh", "exp"); the f32 variant appends "f".
        """
        name = base + ("f" if self.fty == _F32 else "")
        fn = self._libm_cache.get(name)
        if fn is None:
            fn = ir.Function(
                self.module,
                ir.FunctionType(self.fty, [self.fty]),
                name=name,
            )
            self._libm_cache[name] = fn
        return fn

    def _emit_activation(self, pre_i, activation):
        """Emit `activation(pre_i)` and return the resulting value.

        Mirrors `rclite.runtime.reference._ACTIVATIONS`:
          tanh     → libm tanh/tanhf
          sigmoid  → 1 / (1 + exp(-x))   (libm exp/expf)
          relu     → max(0, x)           (fcmp + select, no libm)
          identity → x                   (no-op)
        """
        b = self.b
        if activation == Activation.TANH:
            return b.call(self._libm("tanh"), [pre_i])
        if activation == Activation.IDENTITY:
            return pre_i
        if activation == Activation.RELU:
            is_pos = b.fcmp_ordered(">", pre_i, self._cf(0.0))
            return b.select(is_pos, pre_i, self._cf(0.0))
        if activation == Activation.SIGMOID:
            neg = b.fsub(self._cf(0.0), pre_i)
            e = b.call(self._libm("exp"), [neg])
            return b.fdiv(self._cf(1.0), b.fadd(self._cf(1.0), e))
        raise NotImplementedError(
            f"LLVM backend does not support activation {activation.name}"
        )

    def _emit_global(self, name, arr):
        arr = np.asarray(arr)
        if np.issubdtype(arr.dtype, np.integer):
            # CSR index arrays (col_idx / row_ptr) are emitted as i32.
            flat = np.ascontiguousarray(arr, dtype=np.int32).reshape(-1)
            ty = ir.ArrayType(_I32, flat.size)
            init = [_ci32(int(v)) for v in flat]
        else:
            flat = np.ascontiguousarray(arr, dtype=self.np_dtype).reshape(-1)
            ty = ir.ArrayType(self.fty, flat.size)
            init = [self._cf(float(v)) for v in flat]
        g = ir.GlobalVariable(self.module, ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(ty, init)
        return g

    def _flatten_ops(self):
        from rclite.ir.ops import TimeLoop

        for op in self.ir_module.ops:
            yield op
            if isinstance(op, TimeLoop):
                yield from op.body

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
            FusedStepReadout,
            Argmax,
            Softmax,
            AccumulateState,
            FinalizeAggregate,
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
        if isinstance(op, FusedStepReadout):
            return self._lower_fused(op)
        if isinstance(op, AccumulateState):
            return self._lower_accumulate_state(op)
        if isinstance(op, FinalizeAggregate):
            return self._lower_finalize_aggregate(op)
        if isinstance(op, Argmax):
            return self._lower_argmax(op)
        if isinstance(op, Softmax):
            return self._lower_softmax(op)
        raise NotImplementedError(f"unknown op: {type(op).__name__}")

    def _lower_time_loop(self, op):
        K_unroll = op.unroll
        T = self.T_arg
        if K_unroll == 1:
            with _loop(self.b, T, "t") as t:
                self.t = t
                self.row = t
                for body_op in op.body:
                    self._lower(body_op)
            self.t = None
            self.row = None
            return
        # Unroll body by `K_unroll` over [0, T_unrolled), tail loop for remainder.
        K_const = _ci(K_unroll)
        T_unrolled = self.b.mul(self.b.sdiv(T, K_const), K_const)
        with _loop_strided(
            self.b, _ci(0), T_unrolled, K_const, "tu"
        ) as t_base:
            for k in range(K_unroll):
                self.t = (
                    t_base
                    if k == 0
                    else self.b.add(t_base, _ci(k), name=f"t_{k}")
                )
                self.row = self.t
                for body_op in op.body:
                    self._lower(body_op)
        with _loop_strided(self.b, T_unrolled, T, _ci(1), "ttail") as t:
            self.t = t
            self.row = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None
        self.row = None

    def _lower_preprocess(self, op):
        tK = self.b.mul(self.t, _ci(op.K))
        with _loop(self.b, _ci(op.K), "kpre") as k:
            x_val = _load1d(self.b, self.X_arg, self.b.add(tK, k))
            up = self.b.fmul(
                self.b.fsub(x_val, self._cf(op.offset)),
                self._cf(op.scale),
            )
            _store1d(self.b, self.u_pre, k, up)

    def _lower_reservoir_step(self, op):
        g_Win = self.globals[op.W_in_name]
        g_Wres = self.globals.get(op.W_res_name) if op.W_res_name else None

        self._emit_preactivation(op, g_Win, g_Wres)

        with _loop(self.b, _ci(op.N), "iupd") as i:
            h_old = _load1d(self.b, self.h, i)
            pre_i = _load1d(self.b, self.pre_arr, i)
            act = self._emit_activation(pre_i, op.activation)
            new_h = self.b.fadd(
                self.b.fmul(self._cf(1.0 - op.leak), h_old),
                self.b.fmul(self._cf(op.leak), act),
            )
            _store1d(self.b, self.h, i, new_h)

    def _emit_preactivation(self, op, g_Win, g_Wres):
        """Write pre[i] = bias + W_in[i]·u_pre + (W_res·h)[i] for all rows i.

        The recurrent term uses one of three kernels:
          - dense:  runtime N×N matvec (op.res_sparse is None)
          - csr:    runtime loop over each row's nonzeros (kind=='csr')
          - unroll: Python-unrolled rows with the nonzero weights baked in
                    as constants (kind=='unroll'); skips the W_res global.
        """
        b = self.b
        spec = op.res_sparse
        if spec is not None and spec.kind == "unroll":
            # Each row has a distinct nonzero set, so unroll the i-loop too.
            for i in range(op.N):
                b.store(self._cf(op.bias), self.acc)
                with _loop(b, _ci(op.K), "kin") as k:
                    w = _load2d_global(b, g_Win, op.K, _ci(i), k)
                    u_val = _load1d(b, self.u_pre, k)
                    b.store(
                        b.fadd(b.load(self.acc), b.fmul(w, u_val)), self.acc
                    )
                for j, wv in spec.rows[i]:
                    hv = _load1d(b, self.h, _ci(j))
                    # Value specialization: w==+-1 needs no fmul (fmul by an
                    # exact +-1.0 is the IEEE identity / sign flip, so
                    # fadd/fsub are bit-identical). 2**k is not specialized
                    # for floats -- there is no cheaper exact float op.
                    if wv == 1.0:
                        acc = b.fadd(b.load(self.acc), hv)
                    elif wv == -1.0:
                        acc = b.fsub(b.load(self.acc), hv)
                    else:
                        acc = b.fadd(
                            b.load(self.acc), b.fmul(self._cf(wv), hv)
                        )
                    b.store(acc, self.acc)
                _store1d(b, self.pre_arr, _ci(i), b.load(self.acc))
            return

        with _loop(b, _ci(op.N), "ipre") as i:
            b.store(self._cf(op.bias), self.acc)
            with _loop(b, _ci(op.K), "kin") as k:
                w = _load2d_global(b, g_Win, op.K, i, k)
                u_val = _load1d(b, self.u_pre, k)
                b.store(b.fadd(b.load(self.acc), b.fmul(w, u_val)), self.acc)
            if spec is not None:
                self._emit_res_contrib_csr(spec, i)
            else:
                self._emit_res_contrib(
                    op.topology,
                    op.N,
                    op.chain_weight,
                    op.chain_feedback,
                    g_Wres,
                    i,
                )
            _store1d(b, self.pre_arr, i, b.load(self.acc))

    def _emit_res_contrib_csr(self, spec, i):
        """acc += sum over row i's nonzeros of val[p] * h[col[p]] (CSR)."""
        b = self.b
        g_val = self.globals[spec.val_name]
        g_col = self.globals[spec.col_name]
        g_rowptr = self.globals[spec.rowptr_name]
        start = b.sext(_load1d_global(b, g_rowptr, i), _I64)
        end = b.sext(_load1d_global(b, g_rowptr, b.add(i, _ci(1))), _I64)
        with _loop_strided(b, start, end, _ci(1), "csr") as p:
            j = b.sext(_load1d_global(b, g_col, p), _I64)
            w = _load1d_global(b, g_val, p)
            hv = _load1d(b, self.h, j)
            b.store(b.fadd(b.load(self.acc), b.fmul(w, hv)), self.acc)

    def _emit_res_contrib(
        self, topology, N, chain_weight, chain_feedback, g_Wres, i
    ):
        b = self.b
        cf = self._cf
        if topology == Topology.DLR:
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_safe = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val = _load1d(b, self.h, i_safe)
            contrib = b.select(is_pos, b.fmul(cf(chain_weight), val), cf(0.0))
            b.store(b.fadd(b.load(self.acc), contrib), self.acc)
        elif topology == Topology.SCR:
            is_zero = b.icmp_signed("==", i, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i, _ci(1)))
            val = _load1d(b, self.h, i_prev)
            b.store(
                b.fadd(b.load(self.acc), b.fmul(cf(chain_weight), val)),
                self.acc,
            )
        elif topology == Topology.DLRB:
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val_back = _load1d(b, self.h, i_back)
            contrib_back = b.select(
                is_pos, b.fmul(cf(chain_weight), val_back), cf(0.0)
            )
            is_lt_last = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt_last, b.add(i, _ci(1)), _ci(N - 1))
            val_fwd = _load1d(b, self.h, i_fwd)
            contrib_fwd = b.select(
                is_lt_last, b.fmul(cf(chain_feedback), val_fwd), cf(0.0)
            )
            b.store(
                b.fadd(b.fadd(b.load(self.acc), contrib_back), contrib_fwd),
                self.acc,
            )
        else:
            with _loop(b, _ci(N), "jres") as j:
                w = _load2d_global(b, g_Wres, N, i, j)
                hv = _load1d(b, self.h, j)
                b.store(b.fadd(b.load(self.acc), b.fmul(w, hv)), self.acc)

    def _lower_build_phi(self, op):
        if self.phi_arr is None:
            raise RuntimeError("BuildPhi requires phi buffer")
        tK = self.b.mul(self.t, _ci(op.K))
        off = 0
        if op.include_bias:
            _store1d(self.b, self.phi_arr, _ci(off), self._cf(1.0))
            off += 1
        if op.include_input:
            with _loop(self.b, _ci(op.K), "kphi") as k:
                x_val = _load1d(self.b, self.X_arg, self.b.add(tK, k))
                _store1d(self.b, self.phi_arr, self.b.add(_ci(off), k), x_val)
            off += op.K
        with _loop(self.b, _ci(op.N), "iphi") as i:
            _store1d(
                self.b,
                self.phi_arr,
                self.b.add(_ci(off), i),
                _load1d(self.b, self.h, i),
            )

    def _lower_readout_linear(self, op):
        g_Wout = self.globals[op.W_out_name]
        tM = self.b.mul(self.row, _ci(op.M))
        with _loop(self.b, _ci(op.M), "m") as m:
            self.b.store(self._cf(0.0), self.acc)
            with _loop(self.b, _ci(op.F), "fout") as fi:
                w = _load2d_global(self.b, g_Wout, op.F, m, fi)
                pv = _load1d(self.b, self.phi_arr, fi)
                self.b.store(
                    self.b.fadd(self.b.load(self.acc), self.b.fmul(w, pv)),
                    self.acc,
                )
            if self.logits is not None:
                _store1d(self.b, self.logits, m, self.b.load(self.acc))
            else:
                _store1d(
                    self.b,
                    self.Y_arg,
                    self.b.add(tM, m),
                    self.b.load(self.acc),
                )

    def _lower_fused(self, op):
        """Step + readout in one op: no phi buffer materialization."""
        g_Win = self.globals[op.W_in_name]
        g_Wres = self.globals.get(op.W_res_name) if op.W_res_name else None
        g_Wout = self.globals[op.W_out_name]
        b = self.b
        cf = self._cf

        # Step (same as _lower_reservoir_step)
        self._emit_preactivation(op, g_Win, g_Wres)
        with _loop(b, _ci(op.N), "iupd") as i:
            h_old = _load1d(b, self.h, i)
            pre_i = _load1d(b, self.pre_arr, i)
            act = self._emit_activation(pre_i, op.activation)
            new_h = b.fadd(
                b.fmul(cf(1.0 - op.leak), h_old),
                b.fmul(cf(op.leak), act),
            )
            _store1d(b, self.h, i, new_h)

        # Readout — phi is virtual; we index W_out's columns directly.
        tM = b.mul(self.row, _ci(op.M))
        tK = b.mul(self.t, _ci(op.K))
        bias_off = 1 if op.include_bias_phi else 0
        input_off = bias_off + (op.K if op.include_input_phi else 0)

        with _loop(b, _ci(op.M), "m") as m:
            b.store(cf(0.0), self.acc)
            if op.include_bias_phi:
                w_bias = _load2d_global(b, g_Wout, op.F, m, _ci(0))
                # phi[0] is constant 1.0, so the term is just w_bias.
                b.store(b.fadd(b.load(self.acc), w_bias), self.acc)
            if op.include_input_phi:
                with _loop(b, _ci(op.K), "kfo") as k:
                    w = _load2d_global(
                        b, g_Wout, op.F, m, b.add(_ci(bias_off), k)
                    )
                    x_val = _load1d(b, self.X_arg, b.add(tK, k))
                    b.store(
                        b.fadd(b.load(self.acc), b.fmul(w, x_val)), self.acc
                    )
            with _loop(b, _ci(op.N), "ifo") as i:
                w = _load2d_global(
                    b, g_Wout, op.F, m, b.add(_ci(input_off), i)
                )
                hv = _load1d(b, self.h, i)
                b.store(b.fadd(b.load(self.acc), b.fmul(w, hv)), self.acc)
            if self.logits is not None:
                _store1d(b, self.logits, m, b.load(self.acc))
            else:
                _store1d(b, self.Y_arg, b.add(tM, m), b.load(self.acc))

    # ------------------------------------------------------------------
    # sequence-to-label time pooling

    def _washout_clamped(self, washout):
        """w = min(washout, T-1) as an i64 SSA value (loop-invariant)."""
        b = self.b
        w_const = _ci(washout)
        t_minus1 = b.sub(self.T_arg, _ci(1))
        return b.select(
            b.icmp_signed("<", w_const, self.T_arg), w_const, t_minus1
        )

    def _lower_accumulate_state(self, op):
        """mode='mean': h_sum += h for t >= min(washout, T-1).
        mode='last': nothing (the final h is the pool)."""
        if op.mode == "last":
            return
        b = self.b
        w = self._washout_clamped(op.washout)
        in_window = b.icmp_signed(">=", self.t, w)
        with _loop(b, _ci(op.N), "acc_h") as i:
            s = _load1d(b, self.h_sum, i)
            h_i = _load1d(b, self.h, i)
            add = b.select(in_window, h_i, self._cf(0.0))
            _store1d(b, self.h_sum, i, b.fadd(s, add))

    def _lower_finalize_aggregate(self, op):
        """Write the pooled state into h, then point output at row 0.

        mode='mean' divides the running sum by the pooled-step count
        (T - min(washout, T-1)); mode='last' leaves h untouched.
        """
        if op.mode == "mean":
            b = self.b
            w = self._washout_clamped(op.washout)
            count = b.sub(self.T_arg, w)
            tf = b.uitofp(count, self.fty)
            with _loop(b, _ci(op.N), "fin_h") as i:
                s = _load1d(b, self.h_sum, i)
                _store1d(b, self.h, i, b.fdiv(s, tf))
        # Sequence output is a single row.
        self.t = _ci(0)
        self.row = _ci(0)

    # ------------------------------------------------------------------
    # classification heads

    def _lower_argmax(self, op):
        """class_id = argmax_m logits[m]; write one i32 at output row."""
        b = self.b
        best_v = b.alloca(self.fty, name="best_v")
        best_i = b.alloca(_I64, name="best_i")
        b.store(_load1d(b, self.logits, _ci(0)), best_v)
        b.store(_ci(0), best_i)
        with _loop(b, _ci(op.M), "am") as m:
            v = _load1d(b, self.logits, m)
            is_gt = b.fcmp_ordered(">", v, b.load(best_v))
            b.store(b.select(is_gt, v, b.load(best_v)), best_v)
            b.store(b.select(is_gt, m, b.load(best_i)), best_i)
        cls = b.trunc(b.load(best_i), _I32)
        _store1d(b, self.Y_arg, self.row, cls)

    def _lower_softmax(self, op):
        """p[m] = exp(logits[m]-max) / sum_j exp(logits[j]-max), M floats out."""
        b = self.b
        exp_fn = self._libm("exp")
        # max
        mx = b.alloca(self.fty, name="sm_max")
        b.store(_load1d(b, self.logits, _ci(0)), mx)
        with _loop(b, _ci(op.M), "smx") as m:
            v = _load1d(b, self.logits, m)
            is_gt = b.fcmp_ordered(">", v, b.load(mx))
            b.store(b.select(is_gt, v, b.load(mx)), mx)
        # exp(v - max) into logits, accumulate sum
        b.store(self._cf(0.0), self.acc)
        with _loop(b, _ci(op.M), "sme") as m:
            v = _load1d(b, self.logits, m)
            e = b.call(exp_fn, [b.fsub(v, b.load(mx))])
            _store1d(b, self.logits, m, e)
            b.store(b.fadd(b.load(self.acc), e), self.acc)
        # normalize into Y
        tM = b.mul(self.row, _ci(op.M))
        denom = b.load(self.acc)
        with _loop(b, _ci(op.M), "smn") as m:
            e = _load1d(b, self.logits, m)
            _store1d(b, self.Y_arg, b.add(tM, m), b.fdiv(e, denom))


def emit_module(
    rc: ReservoirComputer,
    exe: RCExecutor,
    *,
    dtype: str = "f64",
    passes=None,
    head=None,
) -> ir.Module:
    """Build an rclite IR module, apply passes, and lower to LLVM IR.

    `dtype` selects f64 (host) vs f32 (Cortex-M cross-compile).
    `passes` is a list of `rclite.ir.passes.*` instances; defaults to
    `[VerifyEchoStateConstraint(strict=False), StructuralSpecialize()]`.
    `head` selects the output format: None / "logits" (raw scores),
    "proba" (softmax), or "classify" (argmax class id, i32 output).
    """
    if rc.reservoir.activation not in _SUPPORTED_ACTIVATIONS:
        raise NotImplementedError(
            f"LLVM backend supports "
            f"{', '.join(a.name for a in _SUPPORTED_ACTIVATIONS)}; "
            f"got {rc.reservoir.activation.name}"
        )

    # Import here to avoid an import cycle (rclite.ir uses runtime types).
    from rclite.ir import build_ir
    from rclite.ir.passes import (
        VerifyEchoStateConstraint,
        StructuralSpecialize,
    )

    ir_module = build_ir(rc, exe, head=head)
    if passes is None:
        passes = [
            VerifyEchoStateConstraint(strict=False),
            StructuralSpecialize(),
        ]
    for p in passes:
        ir_module = p(ir_module)
    return _Lowerer(ir_module, dtype=dtype).lower()
