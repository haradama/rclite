"""LLVM JIT backend for the RC IDL (via llvmlite).

Emits LLVM IR for a trained `ReservoirComputer` and JIT-compiles it via
llvmlite's MCJIT. The compiled module exposes a single C-ABI entry
point:

    void rc_predict(int64_t T, double* X, double* Y);

`X` is a contiguous row-major (T, K) matrix; `Y` is a (T, M) output
buffer the caller pre-allocates. Reservoir weights are embedded as
internal global constants so LLVM can constant-fold and vectorize.

Currently supports: tanh / sigmoid / relu / identity activations; any
topology (DLR/DLRB/SCR/RANDOM); include_bias / include_input readout
features; RIDGE/PINV-trained readouts.
"""

from __future__ import annotations
import ctypes
from contextlib import contextmanager

import numpy as np
from llvmlite import ir
import llvmlite.binding as llvm

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Activation, Topology
from rclite.runtime.reference import RCExecutor


_initialized = False
_all_targets_initialized = False


def _ensure_initialized() -> None:
    global _initialized
    if _initialized:
        return
    llvm.initialize_native_target()
    llvm.initialize_native_asmprinter()
    _initialized = True


def _ensure_all_targets() -> None:
    """Initialize every LLVM target/asmprinter. Required for cross-compile."""
    global _all_targets_initialized
    if _all_targets_initialized:
        return
    llvm.initialize_all_targets()
    llvm.initialize_all_asmprinters()
    _all_targets_initialized = True


_F64 = ir.DoubleType()
_F32 = ir.FloatType()
_I64 = ir.IntType(64)
_I32 = ir.IntType(32)

# Float activations the LLVM backend can emit (matches the reference runtime).
# tanh/sigmoid import libm (tanh[f]/exp[f]); relu/identity import nothing.
_SUPPORTED_ACTIVATIONS = (
    Activation.TANH,
    Activation.SIGMOID,
    Activation.RELU,
    Activation.IDENTITY,
)


def _dtype_bindings(dtype: str):
    """Return (fty, tanh_name, np_dtype, ctype) for the requested float type."""
    if dtype == "f64":
        return _F64, "tanh", np.float64, ctypes.c_double
    if dtype == "f32":
        return _F32, "tanhf", np.float32, ctypes.c_float
    raise ValueError(f"unknown dtype: {dtype!r}; expected 'f32' or 'f64'")


def _cf(x: float, fty: ir.Type = _F64) -> ir.Constant:
    return ir.Constant(fty, float(x))


def _ci(x: int) -> ir.Constant:
    return ir.Constant(_I64, int(x))


def _ci32(x: int) -> ir.Constant:
    return ir.Constant(_I32, int(x))


def _load1d(b: ir.IRBuilder, ptr, i):
    return b.load(b.gep(ptr, [i]))


def _store1d(b: ir.IRBuilder, ptr, i, val) -> None:
    b.store(val, b.gep(ptr, [i]))


def _load2d_global(b: ir.IRBuilder, g, ncols: int, i, j):
    flat = b.add(b.mul(i, _ci(ncols)), j)
    return b.load(b.gep(g, [_ci32(0), flat]))


def _load1d_global(b: ir.IRBuilder, g, i):
    """Load element i from a global array (pointer-to-[N x ty])."""
    return b.load(b.gep(g, [_ci32(0), i]))


@contextmanager
def _loop(b: ir.IRBuilder, count, name: str = "i"):
    """Emit a 0..count-1 loop. Yields the loop index value (i64)."""
    fn = b.block.function
    hdr = fn.append_basic_block(name + "_hdr")
    body = fn.append_basic_block(name + "_body")
    done = fn.append_basic_block(name + "_done")

    idx = b.alloca(_I64, name=name + "_idx")
    b.store(_ci(0), idx)
    b.branch(hdr)

    b.position_at_end(hdr)
    cond = b.icmp_signed("<", b.load(idx), count)
    b.cbranch(cond, body, done)

    b.position_at_end(body)
    cur = b.load(idx, name=name + "_v")
    try:
        yield cur
    finally:
        b.store(b.add(b.load(idx), _ci(1)), idx)
        b.branch(hdr)
        b.position_at_end(done)


@contextmanager
def _loop_strided(b: ir.IRBuilder, start, end, stride, name: str = "i"):
    """Emit a `for i = start; i < end; i += stride` loop."""
    fn = b.block.function
    hdr = fn.append_basic_block(name + "_hdr")
    body = fn.append_basic_block(name + "_body")
    done = fn.append_basic_block(name + "_done")

    idx = b.alloca(_I64, name=name + "_idx")
    b.store(start, idx)
    b.branch(hdr)

    b.position_at_end(hdr)
    cond = b.icmp_signed("<", b.load(idx), end)
    b.cbranch(cond, body, done)

    b.position_at_end(body)
    cur = b.load(idx, name=name + "_v")
    try:
        yield cur
    finally:
        b.store(b.add(b.load(idx), stride), idx)
        b.branch(hdr)
        b.position_at_end(done)


# ----------------------------------------------------------------------------
# Value specialization for baked unroll weights
#
# In the "unroll" sparse kernel each nonzero W_res weight is a compile-time
# constant baked into the IR. When that constant is +-1 or +-2**k the multiply
# can be replaced by a negate / shift (or, for floats, +-1 by add/sub), which
# removes a multiply per nonzero MAC -- the win the roadmap flags for FPU-less
# / multiplier-light cores. Exact zeros never reach here (SparsifyReservoir
# prunes them), so we only special-case the power-of-two magnitudes.


def _pow2_exp(v: int):
    """Return k if abs(int(v)) == 2**k (k >= 0), else None.

    `+-1` maps to k=0. Callers must pass an integer-valued weight (the
    quantized integer paths do); the float path checks `+-1.0` directly
    because a fractional float like 1.5 would truncate to a spurious k.
    """
    a = abs(int(v))
    if a == 0 or (a & (a - 1)) != 0:
        return None
    return a.bit_length() - 1


# ----------------------------------------------------------------------------
# IR-driven lowering


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
        import numpy as np

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


def emit_quantized_affine_module(
    qmodel, *, passes=None, head=None
) -> ir.Module:
    """Build LLVM IR for the affine integer quantized path (i8 or i16).

    Function signature is identical to the symmetric path:
        void rc_predict(int64_t T, storage_t* X, storage_t* Y);
    With `head="classify"`, Y is `int32_t*` (one class id per step).

    `qmodel` is an `AffineQuantizedModel`; weights and metadata flow
    through `build_ir_from_quantized_affine` into the IR Module, then
    `_AffineLowerer` emits the kernel using TFLM-style requantize.
    """
    from rclite.quant.affine.ir_builder import build_ir_from_quantized_affine

    ir_module = build_ir_from_quantized_affine(qmodel, head=head)
    if passes is None:
        passes = []
    for p in passes:
        ir_module = p(ir_module)
    return _AffineLowerer(ir_module).lower()


class _AffineLowerer:
    """Lower an affine-quantized rclite IR Module to LLVM IR.

    Per-step structure (bit-exact mirror of `AffineQuantizedExecutor`):

        acc_in  = sum_k q_W_in[i,k]  * q_x[k] - zp_u_pre * row_sum_W_in[i]
        acc_res = sum_j q_W_res[i,j] * q_h[j] - zp_state  * row_sum_W_res[i]
        pre[i]  = sat( zp_pre + bias_pre
                       + requantize(acc_in,  M_in_M0,  M_in_n)
                       + requantize(acc_res, M_res_M0, M_res_n) )
        a[i]    = LUT[ sext(pre[i], i32) + lut_offset ]
        h[i]    = sat( zp_state + (h[i] - zp_state)
                       + requantize(a[i] - h[i], leak_M0, leak_n) )

        y[m]    = sat( zp_output
                       + requantize(W_out[m, 0],          M_out_bias_*)   # if include_bias
                       + requantize(W_out·x  − zp·R_in,   M_out_input_*)  # if include_input
                       + requantize(W_out·h  − zp·R_st,   M_out_state_*) )

    Accumulator widths:
        storage_bits == 8  : accum = i32, product = i32
        storage_bits == 16 : accum = i64, product = i64 (matmul over N can
                              overflow i32 for N ≳ 4 with full-range i16)

    The requantize step takes its operand in i32 (after clamping the
    accumulator if needed) and uses `(x*M0 + (1<<(n-1))) >> n`. The
    rounding direction (arithmetic shift on `(prod + bias)`) exactly
    matches what `apply_multiplier_array` does on the Python side, so
    Python and JIT agree bit-for-bit.
    """

    def __init__(self, ir_module):
        from rclite.ir.ops import (
            Argmax,
            Softmax,
            AccumulateState,
        )

        self.ir_module = ir_module
        md = ir_module.metadata
        if md.get("quantization") != "affine":
            raise ValueError(
                "_AffineLowerer expects metadata['quantization']='affine'"
            )

        self.storage_bits = int(md["storage_bits"])
        self.storage_ty = ir.IntType(self.storage_bits)
        if self.storage_bits == 8:
            self.accum_ty = _I32
        elif self.storage_bits == 16:
            self.accum_ty = _I64
        else:
            raise NotImplementedError(
                f"_AffineLowerer only supports storage_bits in {{8, 16}}, "
                f"got {self.storage_bits}"
            )

        # Zero points (Python ints)
        self.zp_input = int(md["zp_input"])
        self.zp_u_pre = int(md["zp_u_pre"])
        self.zp_state = int(md["zp_state"])
        self.zp_pre = int(md["zp_pre"])
        self.zp_output = int(md["zp_output"])

        self.lut_offset = int(md["lut_offset"])
        self.bias_pre = int(md["bias_pre"])

        # LUT strategy and per-strategy precomputed.
        self.lut_kind = md.get("lut_kind", "direct")
        if self.lut_kind == "linear_interp":
            self.lut_n_entries = int(md["lut_n_entries"])
            self.lut_interp_frac_bits = int(md["lut_interp_frac_bits"])
            self.lut_idx_M0 = int(md["lut_idx_M0"])
            self.lut_idx_n = int(md["lut_idx_n"])
        elif self.lut_kind == "polynomial":
            self.poly_qf_bits = int(md["poly_qf_bits"])
            self.poly_degree = int(md.get("poly_degree", 5))
            self.poly_x_M0 = int(md["poly_x_M0"])
            self.poly_x_n = int(md["poly_x_n"])
            self.poly_back_M0 = int(md["poly_back_M0"])
            self.poly_back_n = int(md["poly_back_n"])
            self.poly_clip_qf = int(md["poly_clip_qf"])
            self.poly_one_qf = int(md["poly_one_qf"])
            self.poly_a1_qf = int(md["poly_a1_qf"])
            self.poly_a3_qf = int(md["poly_a3_qf"])
            self.poly_a5_qf = int(md["poly_a5_qf"])

        # Requantize multipliers (M0, n)
        self.M_in_M0, self.M_in_n = int(md["M_in_M0"]), int(md["M_in_n"])
        self.M_res_M0, self.M_res_n = int(md["M_res_M0"]), int(md["M_res_n"])
        self.per_channel_res = bool(md.get("per_channel_res", False))
        self.per_channel_out = bool(md.get("per_channel_out", False))
        self.leak_M0, self.leak_n = int(md["leak_M0"]), int(md["leak_n"])
        self.M_out_bias_M0 = int(md["M_out_bias_M0"])
        self.M_out_bias_n = int(md["M_out_bias_n"])
        self.M_out_input_M0 = int(md["M_out_input_M0"])
        self.M_out_input_n = int(md["M_out_input_n"])
        self.M_out_state_M0 = int(md["M_out_state_M0"])
        self.M_out_state_n = int(md["M_out_state_n"])

        self.include_bias = bool(md["include_bias"])
        self.include_input = bool(md["include_input"])

        # Topology specialisation (SCR/DLR/DLRB skip the dense W_res matmul).
        self.structured = bool(md.get("structured", False))
        self.topology_name = md.get("topology", "ESN_STANDARD")
        self.chain_weight_q = int(md.get("chain_weight_q", 0))
        self.chain_feedback_q = int(md.get("chain_feedback_q", 0))

        # Integer input preprocess (active iff input_offset != 0 or
        # input_scaling != 1). When active, the kernel computes u_pre into
        # a scratch buffer that the W_in matmul reads; otherwise the matmul
        # reads X directly.
        self.has_int_preprocess = bool(md.get("has_integer_preprocess", False))
        self.pre_M0 = int(md.get("pre_M0", 0))
        self.pre_n = int(md.get("pre_n", 0))
        self.pre_const = int(md.get("pre_const", 0))

        self.K, self.N, self.M = ir_module.K, ir_module.N, ir_module.M

        self.module = ir.Module(
            name=f"rc_affine_jit_i{self.storage_bits}_{id(ir_module)}",
        )
        self.module.triple = llvm.get_default_triple()

        # Classification head: argmax emits an int32 class id per step;
        # softmax emits M probabilities (storage type, Q.sm_prob_frac).
        # Both route the readout through a logits scratch.
        flat = list(self._flatten_ops())
        self.out_int = any(isinstance(op, Argmax) for op in flat)
        self.has_softmax = any(isinstance(op, Softmax) for op in flat)
        self.has_head = self.out_int or self.has_softmax
        # MEAN time-pooling needs a running i64 state-sum buffer.
        self.needs_state_sum = any(
            isinstance(op, AccumulateState) and op.mode == "mean"
            for op in flat
        )
        out_ty = _I32 if self.out_int else self.storage_ty
        if self.has_softmax:
            self.sm_dmin_q = int(md["sm_dmin_q"])
            self.sm_n = int(md["sm_n"])
            self.sm_idx_frac = int(md["sm_idx_frac"])
            self.sm_prob_frac = int(md["sm_prob_frac"])

        # Emit all globals (storage-typed weights + i32 precomputed row sums).
        self.globals = {}
        for name, arr in ir_module.weights.items():
            self.globals[name] = self._emit_global(name, arr)

        # void rc_predict(i64 T, storage_t* X, {storage_t|i32}* Y)
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

        # Buffers
        self.h_buf = self.b.alloca(
            self.storage_ty,
            size=_ci(self.N),
            name="h",
        )
        self.pre_buf = self.b.alloca(
            self.storage_ty,
            size=_ci(self.N),
            name="pre",
        )
        self.logits = (
            self.b.alloca(
                self.storage_ty, size=_ci(max(self.M, 1)), name="logits"
            )
            if self.has_head
            else None
        )
        self.exp_scratch = (
            self.b.alloca(_I32, size=_ci(max(self.M, 1)), name="exp_q")
            if self.has_softmax
            else None
        )
        # u_pre scratch only allocated when integer preprocess is in play —
        # otherwise the matmul reads X directly.
        if self.has_int_preprocess:
            self.u_pre_buf = self.b.alloca(
                self.storage_ty,
                size=_ci(max(self.K, 1)),
                name="u_pre",
            )
        else:
            self.u_pre_buf = None

        # Running state-sum buffer (i64) for MEAN time-pooling.
        if self.needs_state_sum:
            self.h_sum = self.b.alloca(_I64, size=_ci(self.N), name="h_sum")
            with _loop(self.b, _ci(self.N), "sinit") as i:
                _store1d(self.b, self.h_sum, i, self._ci64(0))
        else:
            self.h_sum = None

        # Initialize state to zp_state
        with _loop(self.b, _ci(self.N), "init") as i:
            _store1d(self.b, self.h_buf, i, self._cs(self.zp_state))

        self.t = None

    # ------------------------------------------------------------------
    # constants

    def _cs(self, v: int) -> ir.Constant:
        """Constant in storage_ty (i8 or i16)."""
        return ir.Constant(self.storage_ty, int(v))

    def _ca(self, v: int) -> ir.Constant:
        """Constant in accum_ty (i32 or i64)."""
        return ir.Constant(self.accum_ty, int(v))

    def _ci32(self, v: int) -> ir.Constant:
        return ir.Constant(_I32, int(v))

    def _ci64(self, v: int) -> ir.Constant:
        return ir.Constant(_I64, int(v))

    # ------------------------------------------------------------------
    # global emission

    def _emit_global(self, name, arr):
        flat = np.asarray(arr).reshape(-1)
        if flat.dtype == np.int8:
            elem_ty = ir.IntType(8)
        elif flat.dtype == np.int16:
            elem_ty = ir.IntType(16)
        elif flat.dtype == np.int32:
            elem_ty = ir.IntType(32)
        else:
            raise ValueError(
                f"_AffineLowerer global {name!r}: unsupported dtype "
                f"{flat.dtype}"
            )
        arr_ty = ir.ArrayType(elem_ty, flat.size)
        g = ir.GlobalVariable(self.module, arr_ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(
            arr_ty,
            [ir.Constant(elem_ty, int(v)) for v in flat],
        )
        return g

    # ------------------------------------------------------------------
    # core helpers

    def _clamp_to_i32(self, val_ty):
        """If accumulator is i64, clamp to i32 range and truncate; else passthrough."""
        if self.accum_ty == _I32:
            return val_ty
        return self._clamp_i64_to_i32(val_ty)

    def _clamp_i64_to_i32(self, val_i64):
        """Clamp an i64 value to the signed i32 range and truncate to i32."""
        lo = self._ci64(-(1 << 31))
        hi = self._ci64((1 << 31) - 1)
        clipped_lo = self.b.select(
            self.b.icmp_signed("<", val_i64, lo),
            lo,
            val_i64,
        )
        clipped = self.b.select(
            self.b.icmp_signed(">", clipped_lo, hi),
            hi,
            clipped_lo,
        )
        return self.b.trunc(clipped, _I32)

    def _emit_requantize_i32(self, acc_i32, M0: int, n: int):
        """Compute `(acc * M0 + (1<<(n-1))) >> n` with i64 product. Returns i32.

        Matches `apply_multiplier_array` in the Python ref bit-for-bit.
        """
        if M0 == 0:
            return self._ci32(0)
        acc_64 = self.b.sext(acc_i32, _I64)
        prod = self.b.mul(acc_64, self._ci64(M0))
        if n > 0:
            prod = self.b.add(prod, self._ci64(1 << (n - 1)))
        shr = self.b.ashr(prod, self._ci64(n))
        return self.b.trunc(shr, _I32)

    def _emit_requantize_i32_dynamic(self, acc_i32, M0_i32, n_i32):
        """Runtime-shift requantize `(acc*M0 + (1<<(n-1)))>>n`, M0/n as SSA i32.

        Per-channel variant: M0 and n are loaded per reservoir row, so the
        shift amount is dynamic. Matches `apply_multiplier_perrow` bit-for-bit
        (n==0 → no rounding bias, ashr by 0 is identity).
        """
        b = self.b
        acc_64 = b.sext(acc_i32, _I64)
        m0_64 = b.sext(M0_i32, _I64)
        n_64 = b.sext(n_i32, _I64)
        prod = b.mul(acc_64, m0_64)
        nz = b.icmp_signed(">", n_64, self._ci64(0))
        safe_sh = b.select(nz, b.sub(n_64, self._ci64(1)), self._ci64(0))
        half = b.select(nz, b.shl(self._ci64(1), safe_sh), self._ci64(0))
        prod = b.add(prod, half)
        shr = b.ashr(prod, n_64)
        return b.trunc(shr, _I32)

    def _emit_saturate_to_storage(self, val_i32):
        """Clamp i32 to signed storage range, then truncate to storage_ty."""
        lo = self._ci32(-(1 << (self.storage_bits - 1)))
        hi = self._ci32((1 << (self.storage_bits - 1)) - 1)
        clipped_lo = self.b.select(
            self.b.icmp_signed("<", val_i32, lo),
            lo,
            val_i32,
        )
        clipped = self.b.select(
            self.b.icmp_signed(">", clipped_lo, hi),
            hi,
            clipped_lo,
        )
        return self.b.trunc(clipped, self.storage_ty)

    # ------------------------------------------------------------------
    # dispatcher

    def lower(self) -> ir.Module:
        for op in self.ir_module.ops:
            self._lower(op)
        self.b.ret_void()
        return self.module

    def _flatten_ops(self):
        from rclite.ir.ops import TimeLoop

        for op in self.ir_module.ops:
            yield op
            if isinstance(op, TimeLoop):
                yield from op.body

    def _lower(self, op):
        from rclite.ir.ops import (
            TimeLoop,
            PreprocessInput,
            ReservoirStep,
            BuildPhi,
            ReadoutLinear,
            Argmax,
            Softmax,
            AccumulateState,
            FinalizeAggregate,
        )

        if isinstance(op, TimeLoop):
            return self._lower_time_loop(op)
        if isinstance(op, PreprocessInput):
            return self._lower_preprocess_affine(op)
        if isinstance(op, ReservoirStep):
            return self._lower_reservoir_step(op)
        if isinstance(op, BuildPhi):
            # Affine readout pulls X and h directly — no phi buffer needed.
            return
        if isinstance(op, ReadoutLinear):
            return self._lower_readout_linear(op)
        if isinstance(op, AccumulateState):
            return self._lower_accumulate_state(op)
        if isinstance(op, FinalizeAggregate):
            return self._lower_finalize_aggregate(op)
        if isinstance(op, Argmax):
            return self._lower_argmax(op)
        if isinstance(op, Softmax):
            return self._lower_softmax(op)
        raise NotImplementedError(
            f"{type(op).__name__} not supported in the affine path"
        )

    # ------------------------------------------------------------------
    # sequence-to-label time pooling (mirrors AffineQuantizedExecutor)

    def _washout_clamped(self, washout):
        """Return min(washout, T-1) clamped at >= 0 as an i64 SSA value."""
        b = self.b
        w_const = self._ci64(washout)
        t_minus1 = b.sub(self.T_arg, self._ci64(1))
        w = b.select(
            b.icmp_signed("<", w_const, self.T_arg), w_const, t_minus1
        )
        return b.select(b.icmp_signed("<", w, self._ci64(0)), self._ci64(0), w)

    def _lower_accumulate_state(self, op):
        """mode='mean': h_sum[i] += q_h[i] for t >= washout. 'last': no-op."""
        if op.mode == "last":
            return
        b = self.b
        w = self._washout_clamped(op.washout)
        in_window = b.icmp_signed(">=", self.t, w)
        with _loop(b, _ci(op.N), "acc_h") as i:
            s = _load1d(b, self.h_sum, i)  # i64
            h_i = b.sext(_load1d(b, self.h_buf, i), _I64)
            add = b.select(in_window, h_i, self._ci64(0))
            _store1d(b, self.h_sum, i, b.add(s, add))

    def _lower_finalize_aggregate(self, op):
        """Write the pooled state back into h_buf, then point output at row 0.

        mode='mean' divides the running sum by L = T - washout, rounding half
        away from zero (bit-exact with `AffineQuantizedExecutor._round_div`).
        mode='last' leaves the final state in place.
        """
        if op.mode == "mean":
            b = self.b
            w = self._washout_clamped(op.washout)
            L = b.sub(self.T_arg, w)  # i64, >= 1
            with _loop(b, _ci(op.N), "fin_h") as i:
                s = _load1d(b, self.h_sum, i)  # i64
                q = self._emit_round_div_i64(s, L)
                _store1d(
                    b,
                    self.h_buf,
                    i,
                    self._emit_saturate_to_storage(self._clamp_i64_to_i32(q)),
                )
        # The pooled sequence produces a single output row.
        self.t = _ci(0)

    def _emit_round_div_i64(self, s, L):
        """Round-half-away-from-zero integer division s/L (L>0), in i64."""
        b = self.b
        half = b.ashr(L, self._ci64(1))  # floor(L/2), L>0
        is_neg = b.icmp_signed("<", s, self._ci64(0))
        pos = b.sdiv(b.add(s, half), L)  # sdiv truncates toward 0
        neg_s = b.sub(self._ci64(0), s)
        neg = b.sub(self._ci64(0), b.sdiv(b.add(neg_s, half), L))
        return b.select(is_neg, neg, pos)

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

        Identical integer algorithm to the symmetric path; operates on the
        quantized logits scratch and writes Q.sm_prob_frac probabilities.
        """
        b = self.b
        g_lut = self.globals["sm_lut"]
        n = self.sm_n
        idxf = self.sm_idx_frac
        dmin = self.sm_dmin_q
        pf = self.sm_prob_frac
        M = op.M
        qmax = (1 << (self.storage_bits - 1)) - 1

        mx = b.alloca(_I32, name="sm_max")
        b.store(b.sext(_load1d(b, self.logits, _ci(0)), _I32), mx)
        with _loop(b, _ci(M), "smx") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            b.store(
                b.select(b.icmp_signed(">", v, b.load(mx)), v, b.load(mx)), mx
            )

        sum_acc = b.alloca(_I64, name="sm_sum")
        b.store(self._ci64(0), sum_acc)
        with _loop(b, _ci(M), "sme") as m:
            v = b.sext(_load1d(b, self.logits, m), _I32)
            d = b.sub(v, b.load(mx))
            d = b.select(
                b.icmp_signed("<", d, self._ci32(dmin)), self._ci32(dmin), d
            )
            num = b.sub(d, self._ci32(dmin))
            num64 = b.sext(num, _I64)
            posn = b.shl(b.mul(num64, self._ci64(n - 1)), self._ci64(idxf))
            pos = b.sdiv(posn, self._ci64(-dmin))
            i0 = b.ashr(pos, self._ci64(idxf))
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

        s = b.load(sum_acc)
        with _loop(b, _ci(M), "smn") as m:
            e = b.sext(_load1d(b, self.exp_scratch, m), _I64)
            p = b.sdiv(b.shl(e, self._ci64(pf)), s)
            p = b.select(
                b.icmp_signed(">", p, self._ci64(qmax)), self._ci64(qmax), p
            )
            tM = b.mul(self.t, _ci(M))
            _store1d(b, self.Y_arg, b.add(tM, m), b.trunc(p, self.storage_ty))

    def _lower_preprocess_affine(self, op):
        """Integer preprocess: u_pre[k] = sat(pre_const + apply_mult(q_x − zp_x)).

        Mirrors `AffineQuantizedExecutor._quantize_u_pre` step for step.
        """
        K = op.K
        if K == 0 or not self.has_int_preprocess:
            return
        t = self.t
        tK = self.b.mul(t, _ci(K))
        with _loop(self.b, _ci(K), "kpre_aff") as k:
            x_q = _load1d(self.b, self.X_arg, self.b.add(tK, k))
            centered = self.b.sub(
                self.b.sext(x_q, _I32),
                self._ci32(self.zp_input),
            )
            delta = self._emit_requantize_i32(
                centered,
                self.pre_M0,
                self.pre_n,
            )
            total = self.b.add(delta, self._ci32(self.pre_const))
            _store1d(
                self.b,
                self.u_pre_buf,
                k,
                self._emit_saturate_to_storage(total),
            )

    def _lower_time_loop(self, op):
        with _loop(self.b, self.T_arg, "t") as t:
            self.t = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None

    # ------------------------------------------------------------------
    # reservoir step

    def _lower_reservoir_step(self, op):
        g_Win = self.globals["W_in"]
        g_rs_in = self.globals["row_sum_W_in"]
        # W_res / row_sum_W_res only exist for non-structured (dense) topologies.
        g_Wres = self.globals.get("W_res")
        g_rs_res = self.globals.get("row_sum_W_res")
        K, N = op.K, op.N
        t = self.t

        # ---- Pre-act loop ----
        spec = op.res_sparse
        if spec is not None and spec.kind == "unroll":
            # Per-row nonzero sets differ → unroll the outer i-loop.
            for i in range(N):
                self._emit_affine_row(
                    op, _ci(i), g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py=i
                )
        else:
            with _loop(self.b, _ci(N), "ipre") as i:
                self._emit_affine_row(
                    op, i, g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py=None
                )

        # ---- Activation + leaky integration ----
        with _loop(self.b, _ci(N), "iact") as i:
            p = _load1d(self.b, self.pre_buf, i)  # storage_ty
            a = self._emit_activation(p)  # storage_ty

            h_old = _load1d(self.b, self.h_buf, i)
            h_c = self.b.sub(
                self.b.sext(h_old, _I32), self._ci32(self.zp_state)
            )
            a_c = self.b.sub(self.b.sext(a, _I32), self._ci32(self.zp_state))
            diff = self.b.sub(a_c, h_c)
            delta = self._emit_requantize_i32(diff, self.leak_M0, self.leak_n)
            new_h_c = self.b.add(h_c, delta)
            new_h_total = self.b.add(new_h_c, self._ci32(self.zp_state))
            new_h_q = self._emit_saturate_to_storage(new_h_total)
            _store1d(self.b, self.h_buf, i, new_h_q)

    def _const_mul_accum(self, wv: int, h):
        """wv * sext(h) in accum_ty, folding the multiply when wv==+-2**k.

        `mul(2**k, sext(h))` equals `shl(sext(h), k)` bit-for-bit in accum_ty
        (no overflow: accum_ty is wider than the storage state), and a
        negative power negates the shifted value -- bit-identical to the
        baked `mul`. Falls back to the multiply otherwise.
        """
        b = self.b
        k = _pow2_exp(wv)
        h_p = b.sext(h, self.accum_ty)
        if k is None:
            return b.mul(self._ca(int(wv)), h_p)
        if k > 0:
            h_p = b.shl(h_p, ir.Constant(self.accum_ty, k))
        if wv < 0:
            h_p = b.sub(self._ca(0), h_p)
        return h_p

    def _emit_affine_row(
        self, op, i, g_Win, g_rs_in, g_Wres, g_rs_res, spec, i_py
    ):
        """Emit pre[row i] for the affine kernel (one body of the ipre loop).

        `i` is an SSA index (a constant when unrolling). For the unrolled
        sparse kernel `i_py` is the Python row index and the recurrent
        accumulation uses the baked nonzeros in `spec.rows[i_py]`; the
        affine zero-point correction `- zp_state * row_sum_W_res[i]` and the
        requantize are unchanged (row_sum_W_res is preserved by the pass).
        """
        b, K, N, t = self.b, op.K, op.N, self.t
        # acc_in
        acc_in_var = b.alloca(self.accum_ty, name="acc_in")
        b.store(self._ca(0), acc_in_var)
        with _loop(b, _ci(K), "kin") as k:
            w = _load2d_global(b, g_Win, K, i, k)
            if self.has_int_preprocess:
                x = _load1d(b, self.u_pre_buf, k)
            else:
                x = _load1d(b, self.X_arg, b.add(b.mul(t, _ci(K)), k))
            prod = b.mul(b.sext(w, self.accum_ty), b.sext(x, self.accum_ty))
            b.store(b.add(b.load(acc_in_var), prod), acc_in_var)
        rs_in_i32 = _load1d_global(b, g_rs_in, i)
        rs_in = (
            rs_in_i32
            if self.accum_ty == _I32
            else b.sext(rs_in_i32, self.accum_ty)
        )
        acc_in_final = b.sub(
            b.load(acc_in_var), b.mul(self._ca(self.zp_u_pre), rs_in)
        )
        rq_in = self._emit_requantize_i32(
            self._clamp_to_i32(acc_in_final), self.M_in_M0, self.M_in_n
        )

        # acc_res
        if self.structured:
            acc_res_i32 = self._emit_chain_contribution(i, N)
        else:
            acc_res_var = b.alloca(self.accum_ty, name="acc_res")
            b.store(self._ca(0), acc_res_var)
            if i_py is not None:  # unrolled sparse
                for j, wv in spec.rows[i_py]:
                    h = _load1d(b, self.h_buf, _ci(j))
                    prod = self._const_mul_accum(int(wv), h)
                    b.store(b.add(b.load(acc_res_var), prod), acc_res_var)
            elif spec is not None:  # CSR
                self._emit_affine_res_csr(spec, acc_res_var, i)
            else:  # dense
                with _loop(b, _ci(N), "jres") as j:
                    w = _load2d_global(b, g_Wres, N, i, j)
                    h = _load1d(b, self.h_buf, j)
                    prod = b.mul(
                        b.sext(w, self.accum_ty), b.sext(h, self.accum_ty)
                    )
                    b.store(b.add(b.load(acc_res_var), prod), acc_res_var)
            rs_res_i32 = _load1d_global(b, g_rs_res, i)
            rs_res = (
                rs_res_i32
                if self.accum_ty == _I32
                else b.sext(rs_res_i32, self.accum_ty)
            )
            acc_res_final = b.sub(
                b.load(acc_res_var), b.mul(self._ca(self.zp_state), rs_res)
            )
            acc_res_i32 = self._clamp_to_i32(acc_res_final)
        if self.per_channel_res and not self.structured:
            # per-row (M0[i], n[i]) loaded from i32 globals → dynamic shift.
            m0_i = _load1d_global(b, self.globals["M_res_M0"], i)
            n_i = _load1d_global(b, self.globals["M_res_n"], i)
            rq_res = self._emit_requantize_i32_dynamic(acc_res_i32, m0_i, n_i)
        else:
            rq_res = self._emit_requantize_i32(
                acc_res_i32, self.M_res_M0, self.M_res_n
            )

        pre_total = b.add(
            b.add(self._ci32(self.zp_pre + self.bias_pre), rq_in), rq_res
        )
        pre_q = self._emit_saturate_to_storage(pre_total)
        _store1d(b, self.pre_buf, i, pre_q)

    def _emit_affine_res_csr(self, spec, acc_res_var, i):
        """Accumulate W_res·h over row i's nonzeros (CSR) into acc_res_var."""
        b = self.b
        g_val = self.globals[spec.val_name]
        g_col = self.globals[spec.col_name]
        g_rowptr = self.globals[spec.rowptr_name]
        start = b.sext(_load1d_global(b, g_rowptr, i), _I64)
        end = b.sext(_load1d_global(b, g_rowptr, b.add(i, _ci(1))), _I64)
        with _loop_strided(b, start, end, _ci(1), "csr") as p:
            j = b.sext(_load1d_global(b, g_col, p), _I64)
            w = _load1d_global(b, g_val, p)
            h = _load1d(b, self.h_buf, j)
            prod = b.mul(b.sext(w, self.accum_ty), b.sext(h, self.accum_ty))
            b.store(b.add(b.load(acc_res_var), prod), acc_res_var)

    # ------------------------------------------------------------------
    # structured-topology W_res contribution (SCR / DLR / DLRB)

    def _emit_chain_contribution(self, i, N):
        """Return the i32 acc_res for row `i` under a structured topology.

        Algebraic identity (since q_W_res is symmetric, zp_W_res = 0):
            sum_j q_W[i,j]·q_h[j]  −  zp_state·row_sum_W[i]
          =  cw_q · (q_h[prev] − zp_state)              for SCR/DLR(i>0)
          +  cf_q · (q_h[next] − zp_state)              for DLRB extra edge

        Each chain entry is a single i8/i16 weight, so the product fits
        in i32 without a wider accumulator.
        """
        b = self.b
        cw = self._ci32(self.chain_weight_q)
        cf = self._ci32(self.chain_feedback_q)
        zp_state_const = self._ci32(self.zp_state)
        zero32 = self._ci32(0)

        def _h_centered(idx_i64):
            h_val = _load1d(b, self.h_buf, idx_i64)
            return b.sub(b.sext(h_val, _I32), zp_state_const)

        topo = self.topology_name
        if topo == "SCR":
            # prev_idx = (i==0 ? N-1 : i-1)
            is_zero = b.icmp_signed("==", i, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i, _ci(1)))
            return b.mul(cw, _h_centered(i_prev))
        if topo == "DLR":
            # only contribute for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_safe = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            prod = b.mul(cw, _h_centered(i_safe))
            return b.select(is_pos, prod, zero32)
        if topo == "DLRB":
            # backward chain: chain_weight * h[i-1] for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            back_prod = b.mul(cw, _h_centered(i_back))
            contrib_back = b.select(is_pos, back_prod, zero32)
            # forward chain: chain_feedback * h[i+1] for i < N-1
            is_lt = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt, b.add(i, _ci(1)), _ci(N - 1))
            fwd_prod = b.mul(cf, _h_centered(i_fwd))
            contrib_fwd = b.select(is_lt, fwd_prod, zero32)
            return b.add(contrib_back, contrib_fwd)
        raise ValueError(
            f"_emit_chain_contribution: unsupported structured topology "
            f"{topo!r}"
        )

    # ------------------------------------------------------------------
    # activation — dispatch on lut_kind

    def _emit_activation(self, p_storage):
        """Compute one tanh value from `p_storage` (storage_ty), return storage_ty."""
        if self.lut_kind == "direct":
            return self._emit_act_direct(p_storage)
        if self.lut_kind == "linear_interp":
            return self._emit_act_linear_interp(p_storage)
        if self.lut_kind == "polynomial":
            return self._emit_act_polynomial(p_storage)
        raise ValueError(f"unknown lut_kind: {self.lut_kind}")

    def _emit_act_direct(self, p_storage):
        g_lut = self.globals["lut_table"]
        idx_i32 = self.b.add(
            self.b.sext(p_storage, _I32), self._ci32(self.lut_offset)
        )
        idx_i64 = self.b.sext(idx_i32, _I64)
        return _load1d_global(self.b, g_lut, idx_i64)

    def _emit_act_linear_interp(self, p_storage):
        """Subsampled table + linear interp, bit-exact mirror of Python ref."""
        g_lut = self.globals["lut_table"]
        f = self.lut_interp_frac_bits
        n = self.lut_n_entries
        # normalized = sext(p, i32) + offset, then t_q = requantize(normalized, idx_M0, idx_n)
        normalized = self.b.add(
            self.b.sext(p_storage, _I32), self._ci32(self.lut_offset)
        )
        t_q = self._emit_requantize_i32(
            normalized, self.lut_idx_M0, self.lut_idx_n
        )
        # idx = t_q >> f, clipped to [0, n-2]
        idx_raw = self.b.ashr(t_q, self._ci32(f))
        zero32 = self._ci32(0)
        n_minus2 = self._ci32(n - 2)
        idx_lo = self.b.select(
            self.b.icmp_signed("<", idx_raw, zero32),
            zero32,
            idx_raw,
        )
        idx = self.b.select(
            self.b.icmp_signed(">", idx_lo, n_minus2),
            n_minus2,
            idx_lo,
        )
        # frac = t_q - (idx << f)
        frac_q = self.b.sub(t_q, self.b.shl(idx, self._ci32(f)))

        # Load y0 = lut[idx], y1 = lut[idx + 1]; widen to i32 for the lerp math.
        idx_i64 = self.b.sext(idx, _I64)
        idx1_i64 = self.b.add(idx_i64, self._ci64(1))
        y0_s = _load1d_global(self.b, g_lut, idx_i64)
        y1_s = _load1d_global(self.b, g_lut, idx1_i64)
        y0_i32 = self.b.sext(y0_s, _I32)
        y1_i32 = self.b.sext(y1_s, _I32)

        # Lerp in i64: y0 + ((y1 - y0) * frac_q) >> f.
        dy_i32 = self.b.sub(y1_i32, y0_i32)
        dy_64 = self.b.sext(dy_i32, _I64)
        frac_64 = self.b.sext(frac_q, _I64)
        scaled_64 = self.b.ashr(self.b.mul(dy_64, frac_64), self._ci64(f))
        interp_i32 = self.b.add(y0_i32, self.b.trunc(scaled_64, _I32))
        return self._emit_saturate_to_storage(interp_i32)

    def _emit_act_polynomial(self, p_storage):
        """Odd-only minimax tanh, Horner in x², bit-exact with Python ref.

        x² = (x·x) >> qf
        inner = ((x²·a5) >> qf) + a3
        outer = ((x²·inner) >> qf) + a1
        y     = (x·outer) >> qf
        y     = clamp(y, ±one_qf)
        """
        qf = self.poly_qf_bits
        # x_qf = requantize(sext(p) - zp_pre, x_M0, x_n), widen to i64.
        centered = self.b.sub(
            self.b.sext(p_storage, _I32), self._ci32(self.zp_pre)
        )
        x_qf_i32 = self._emit_requantize_i32(
            centered, self.poly_x_M0, self.poly_x_n
        )
        x_qf = self.b.sext(x_qf_i32, _I64)
        # Clamp |x| <= x_clip_qf
        clip_pos = self._ci64(self.poly_clip_qf)
        clip_neg = self._ci64(-self.poly_clip_qf)
        x_qf = self.b.select(
            self.b.icmp_signed("<", x_qf, clip_neg), clip_neg, x_qf
        )
        x_qf = self.b.select(
            self.b.icmp_signed(">", x_qf, clip_pos), clip_pos, x_qf
        )
        qf_const = self._ci64(qf)
        a1_const = self._ci64(self.poly_a1_qf)
        a3_const = self._ci64(self.poly_a3_qf)
        a5_const = self._ci64(self.poly_a5_qf)
        # Horner in x²:  y = x · (a1 + x² · (a3 + x² · a5))
        x2_qf = self.b.ashr(self.b.mul(x_qf, x_qf), qf_const)
        inner = self.b.add(
            self.b.ashr(self.b.mul(x2_qf, a5_const), qf_const),
            a3_const,
        )
        outer = self.b.add(
            self.b.ashr(self.b.mul(x2_qf, inner), qf_const),
            a1_const,
        )
        y_qf = self.b.ashr(self.b.mul(x_qf, outer), qf_const)
        # Clamp y to ±one_qf
        one_pos = self._ci64(self.poly_one_qf)
        one_neg = self._ci64(-self.poly_one_qf)
        y_qf = self.b.select(
            self.b.icmp_signed("<", y_qf, one_neg), one_neg, y_qf
        )
        y_qf = self.b.select(
            self.b.icmp_signed(">", y_qf, one_pos), one_pos, y_qf
        )
        # Δq_state = requantize(y_qf), then +zp_state
        y_qf_i32 = self.b.trunc(y_qf, _I32)
        delta = self._emit_requantize_i32(
            y_qf_i32, self.poly_back_M0, self.poly_back_n
        )
        total = self.b.add(delta, self._ci32(self.zp_state))
        return self._emit_saturate_to_storage(total)

    # ------------------------------------------------------------------
    # readout

    def _rq_out(self, x_i32, m, name, M0_scalar, n_scalar):
        """Readout requantize: per-row (M0[m], n[m]) when per_channel_out,
        else the scalar (M0, n). `m` is the output-row SSA index."""
        if self.per_channel_out:
            m0 = _load1d_global(self.b, self.globals[name + "_M0"], m)
            nn = _load1d_global(self.b, self.globals[name + "_n"], m)
            return self._emit_requantize_i32_dynamic(x_i32, m0, nn)
        return self._emit_requantize_i32(x_i32, M0_scalar, n_scalar)

    def _lower_readout_linear(self, op):
        g_Wout = self.globals["W_out"]
        g_rs_state = self.globals["row_sum_Wout_state"]
        g_rs_input = self.globals.get("row_sum_Wout_input")
        F = op.F
        K = self.K
        N = self.N
        Mout = op.M

        off_bias = 0
        off_input = 1 if self.include_bias else 0
        off_state = off_input + (K if self.include_input else 0)

        t = self.t
        tM = self.b.mul(t, _ci(Mout))

        # The readout accumulates in i64 — W_out may be wider than the base
        # storage (mixed precision) and N can be large, so the matmul must
        # not overflow. Each block is clamped to i32 before its requantize,
        # matching the Python reference exactly.
        with _loop(self.b, _ci(Mout), "m") as m:
            y_var = self.b.alloca(_I32, name="y_acc")
            self.b.store(self._ci32(self.zp_output), y_var)

            if self.include_bias:
                w0 = _load2d_global(self.b, g_Wout, F, m, _ci(off_bias))
                clamped_b = self._clamp_i64_to_i32(self.b.sext(w0, _I64))
                rq_b = self._rq_out(
                    clamped_b,
                    m,
                    "M_out_bias",
                    self.M_out_bias_M0,
                    self.M_out_bias_n,
                )
                self.b.store(self.b.add(self.b.load(y_var), rq_b), y_var)

            if self.include_input:
                acc_var = self.b.alloca(_I64, name="acc_input_ro")
                self.b.store(self._ci64(0), acc_var)
                with _loop(self.b, _ci(K), "kin_ro") as k:
                    col = self.b.add(_ci(off_input), k)
                    w = _load2d_global(self.b, g_Wout, F, m, col)
                    x = _load1d(
                        self.b,
                        self.X_arg,
                        self.b.add(self.b.mul(t, _ci(K)), k),
                    )
                    prod = self.b.mul(
                        self.b.sext(w, _I64), self.b.sext(x, _I64)
                    )
                    self.b.store(
                        self.b.add(self.b.load(acc_var), prod),
                        acc_var,
                    )
                rs = self.b.sext(_load1d_global(self.b, g_rs_input, m), _I64)
                adj = self.b.sub(
                    self.b.load(acc_var),
                    self.b.mul(self._ci64(self.zp_input), rs),
                )
                rq_i = self._rq_out(
                    self._clamp_i64_to_i32(adj),
                    m,
                    "M_out_input",
                    self.M_out_input_M0,
                    self.M_out_input_n,
                )
                self.b.store(self.b.add(self.b.load(y_var), rq_i), y_var)

            acc_var = self.b.alloca(_I64, name="acc_state_ro")
            self.b.store(self._ci64(0), acc_var)
            with _loop(self.b, _ci(N), "jst_ro") as j:
                col = self.b.add(_ci(off_state), j)
                w = _load2d_global(self.b, g_Wout, F, m, col)
                h = _load1d(self.b, self.h_buf, j)
                prod = self.b.mul(self.b.sext(w, _I64), self.b.sext(h, _I64))
                self.b.store(
                    self.b.add(self.b.load(acc_var), prod),
                    acc_var,
                )
            rs = self.b.sext(_load1d_global(self.b, g_rs_state, m), _I64)
            adj = self.b.sub(
                self.b.load(acc_var),
                self.b.mul(self._ci64(self.zp_state), rs),
            )
            rq_s = self._rq_out(
                self._clamp_i64_to_i32(adj),
                m,
                "M_out_state",
                self.M_out_state_M0,
                self.M_out_state_n,
            )
            self.b.store(self.b.add(self.b.load(y_var), rq_s), y_var)

            y_q = self._emit_saturate_to_storage(self.b.load(y_var))
            if self.logits is not None:
                _store1d(self.b, self.logits, m, y_q)
            else:
                _store1d(self.b, self.Y_arg, self.b.add(tM, m), y_q)


class CompiledAffineRC:
    """JIT-compiled affine `AffineQuantizedModel` (host LLVM).

    Mirrors `CompiledQuantizedRC` but consumes an `AffineQuantizedModel`
    and emits via `_AffineLowerer`. `predict()` accepts float inputs,
    quantizes them via the model's input params (matching what the
    Python `AffineQuantizedExecutor.predict` does), calls the kernel,
    and dequantizes the output back to float.
    """

    name = "llvm-affine"

    def __init__(self, qmodel, opt_level: int = 3, passes=None, head=None):
        _ensure_initialized()
        self.qmodel = qmodel
        self.rc = qmodel.rc
        self.head = head or "logits"
        self._out_int = self.head == "classify"
        self._ir_text = str(
            emit_quantized_affine_module(
                qmodel,
                passes=passes,
                head=head,
            )
        )
        self._mod = llvm.parse_assembly(self._ir_text)
        self._mod.verify()

        target = llvm.Target.from_triple(llvm.get_default_triple())
        self._tm = target.create_target_machine(opt=opt_level)
        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        pto.loop_vectorization = True
        pto.slp_vectorization = True
        pb = llvm.create_pass_builder(self._tm, pto)
        pb.getModulePassManager().run(self._mod, pb)

        self._engine = llvm.create_mcjit_compiler(self._mod, self._tm)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        sw = qmodel.storage_bits
        if sw == 8:
            self._cstorage = ctypes.c_int8
            self._np_storage = np.int8
        elif sw == 16:
            self._cstorage = ctypes.c_int16
            self._np_storage = np.int16
        else:
            raise NotImplementedError(
                f"CompiledAffineRC: storage_bits {sw} not supported"
            )
        out_ptr = (
            ctypes.POINTER(ctypes.c_int32)
            if self._out_int
            else ctypes.POINTER(self._cstorage)
        )
        addr = self._engine.get_function_address("rc_predict")
        self._cfn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int64,
            ctypes.POINTER(self._cstorage),
            out_ptr,
        )(addr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Float input → JIT kernel → output.

        head="logits": dequantized float (T, M). head="classify": int32
        class indices (T,). head="proba": float (T, M) probabilities.
        """
        if X.ndim == 1:
            X = X[:, None]
        from rclite.core.profile import Aggregation

        T = X.shape[0]
        Mout = self.qmodel.M
        # Sequence pooling collapses the whole input to a single output row.
        n_rows = (
            1 if self.qmodel.rc.readout.aggregation != Aggregation.NONE else T
        )
        # Quantize input via the model's input params (matches Python ref).
        X_q = self.qmodel.config.input.quantize_array(X).astype(
            self._np_storage
        )
        X_q = np.ascontiguousarray(X_q.reshape(-1))
        if self._out_int:
            Y = np.zeros(n_rows, dtype=np.int32)
            self._cfn(
                ctypes.c_int64(T),
                X_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
                Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            return Y
        Y_q = np.zeros(n_rows * Mout, dtype=self._np_storage)
        self._cfn(
            ctypes.c_int64(T),
            X_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
            Y_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
        )
        Y_q = Y_q.reshape(n_rows, Mout)
        if self.head == "proba":
            prob_frac = min(self.qmodel.storage_bits - 1, 15)
            return Y_q.astype(np.float64) / (1 << prob_frac)
        return self.qmodel.config.output.dequantize_array(Y_q)

    @property
    def llvm_ir(self) -> str:
        return self._ir_text

    @property
    def optimized_ir(self) -> str:
        return str(self._mod)

    @property
    def assembly(self) -> str:
        return self._tm.emit_assembly(self._mod)


class CompiledRC:
    """JIT-compiled ReservoirComputer (LLVM backend).

    Mirrors the relevant subset of `RCExecutor.predict`.
    """

    name = "llvm"

    def __init__(
        self,
        rc: ReservoirComputer,
        exe: RCExecutor,
        opt_level: int = 3,
        vectorize: bool = True,
        passes=None,
        head=None,
    ):
        _ensure_initialized()
        self.rc = rc
        self.exe = exe
        self.head = head or "logits"
        self._out_int = self.head == "classify"
        self._ir_text = str(emit_module(rc, exe, passes=passes, head=head))
        self._mod = llvm.parse_assembly(self._ir_text)
        self._mod.verify()

        target = llvm.Target.from_triple(llvm.get_default_triple())
        self._tm = target.create_target_machine(opt=opt_level)

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        pto.loop_vectorization = vectorize
        pto.slp_vectorization = vectorize
        pb = llvm.create_pass_builder(self._tm, pto)
        mpm = pb.getModulePassManager()
        mpm.run(self._mod, pb)
        self._engine = llvm.create_mcjit_compiler(self._mod, self._tm)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        out_ptr = (
            ctypes.POINTER(ctypes.c_int32)
            if self._out_int
            else ctypes.POINTER(ctypes.c_double)
        )
        addr = self._engine.get_function_address("rc_predict")
        self._cfn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            out_ptr,
        )(addr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the kernel.

        head="logits"/"proba": returns float (n_rows, M).
        head="classify":       returns int32 class indices (n_rows,).
        n_rows is T for per-step readouts, or 1 for sequence aggregation
        (the whole input is pooled to a single output row).
        """
        if X.ndim == 1:
            X = X[:, None]
        from rclite.core.profile import Aggregation

        T = X.shape[0]
        M = self.rc.readout.units
        n_rows = 1 if self.rc.readout.aggregation != Aggregation.NONE else T
        X = np.ascontiguousarray(X, dtype=np.float64)
        if self._out_int:
            Y = np.zeros(n_rows, dtype=np.int32)
            self._cfn(
                ctypes.c_int64(T),
                X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            return Y
        Y = np.zeros((n_rows, M), dtype=np.float64)
        self._cfn(
            ctypes.c_int64(T),
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        return Y

    @property
    def llvm_ir(self) -> str:
        return self._ir_text

    @property
    def optimized_ir(self) -> str:
        return str(self._mod)

    @property
    def assembly(self) -> str:
        return self._tm.emit_assembly(self._mod)

    # ------------------------------------------------------------------
    # AOT: emit object file, shared library, and C header

    def emit_object(self, path: str) -> None:
        """Write a PIC ELF object file to `path` for shared-library linking."""
        target = llvm.Target.from_triple(llvm.get_default_triple())
        tm_pic = target.create_target_machine(opt=3, reloc="pic")
        obj_bytes = tm_pic.emit_object(self._mod)
        with open(path, "wb") as f:
            f.write(obj_bytes)

    def emit_shared_library(self, path: str, cc: str = "gcc") -> None:
        """Compile to a shared library via `cc -shared -lm`."""
        import os
        import subprocess
        import tempfile

        fd, obj_path = tempfile.mkstemp(suffix=".o", prefix="rc_")
        os.close(fd)
        try:
            self.emit_object(obj_path)
            cmd = [cc, "-shared", "-fPIC", "-o", path, obj_path, "-lm"]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                raise RuntimeError(
                    f"link failed: {' '.join(cmd)}\nstderr:\n{result.stderr}"
                )
        finally:
            try:
                os.unlink(obj_path)
            except FileNotFoundError:
                pass

    def emit_header(self, path: str, fn_name: str = "rc_predict") -> None:
        """Write a C header declaring the compiled function.

        The output type tracks the head this CompiledRC was built with:
        head="classify" declares `int32_t *Y` (class ids, one per row);
        head="logits"/"proba" declare `double *Y`. Sequence-aggregation
        models produce a single output row regardless of T.
        """
        from rclite.core.profile import Aggregation

        K = self.rc.input.units
        N = self.rc.reservoir.units
        M = self.rc.readout.units
        topo = self.rc.reservoir.topology.name
        trainer = self.rc.readout.trainer.name
        task = self.rc.readout.task.name
        agg = self.rc.readout.aggregation.name
        n_rows = (
            "1 (sequence-pooled)"
            if self.rc.readout.aggregation != Aggregation.NONE
            else "T"
        )
        out_decl = f"int32_t *Y" if self._out_int else "double *Y"
        out_desc = (
            "class id per row (argmax)"
            if self._out_int
            else (
                "softmax probabilities"
                if self.head == "proba"
                else "linear scores"
            )
        )
        guard = "RC_PREDICT_H"
        header = (
            f"/* Auto-generated header for compiled ReservoirComputer.\n"
            f" *\n"
            f" *   input units      = {K}\n"
            f" *   reservoir units  = {N}\n"
            f" *   output units     = {M}\n"
            f" *   topology         = {topo}\n"
            f" *   trainer          = {trainer}\n"
            f" *   task             = {task}\n"
            f" *   aggregation      = {agg}\n"
            f" *   head             = {self.head}\n"
            f" *   activation       = {self.rc.reservoir.activation.name}\n"
            f" *   leak_rate        = {self.rc.reservoir.leak_rate}\n"
            f" *   input_scaling    = {self.rc.input.input_scaling}\n"
            f" *   input_offset     = {self.rc.input.input_offset}\n"
            f" */\n"
            f"#ifndef {guard}\n"
            f"#define {guard}\n"
            f"\n"
            f"#include <stdint.h>\n"
            f"\n"
            f"#define RC_INPUT_DIM  {K}\n"
            f"#define RC_OUTPUT_DIM {M}\n"
            f"#define RC_RES_UNITS  {N}\n"
            f"#define RC_NUM_CLASSES {M if task == 'CLASSIFICATION' else 0}\n"
            f"\n"
            f"#ifdef __cplusplus\n"
            f'extern "C" {{\n'
            f"#endif\n"
            f"\n"
            f"/* Run inference over a length-T sequence.\n"
            f" *   X: row-major (T x RC_INPUT_DIM) input.   Caller-owned.\n"
            f" *   Y: {out_desc}; {n_rows} output row(s). Caller-allocated.\n"
            f" */\n"
            f"void {fn_name}(int64_t T, double *X, {out_decl});\n"
            f"\n"
            f"#ifdef __cplusplus\n"
            f"}}\n"
            f"#endif\n"
            f"\n"
            f"#endif /* {guard} */\n"
        )
        with open(path, "w") as f:
            f.write(header)


def compile_rc(rc: ReservoirComputer, exe: RCExecutor, **kwargs) -> CompiledRC:
    return CompiledRC(rc, exe, **kwargs)


class CompiledQuantizedRC:
    """JIT-compiled QuantizedModel (integer i32 path).

    Mirrors the float CompiledRC interface but consumes a `QuantizedModel`
    and emits the i32 kernel. `predict()` accepts float inputs, quantizes
    them at input_scale (no float preprocessing — the kernel applies
    `(u_raw - offset) * scale` internally in fixed point so the readout
    can see the original raw input), calls the kernel, and dequantizes the
    output back to float.
    """

    name = "llvm-quantized"

    def __init__(
        self,
        qmodel,
        opt_level: int = 3,
        passes=None,
        saturating: bool = True,
        head=None,
    ):
        _ensure_initialized()
        self.qmodel = qmodel
        self.rc = qmodel.rc
        self.saturating = saturating
        self.head = head or "logits"
        self._out_int = self.head == "classify"
        self._ir_text = str(
            emit_quantized_module(
                qmodel,
                passes=passes,
                saturating=saturating,
                head=head,
            )
        )
        self._mod = llvm.parse_assembly(self._ir_text)
        self._mod.verify()

        target = llvm.Target.from_triple(llvm.get_default_triple())
        self._tm = target.create_target_machine(opt=opt_level)

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        pto.loop_vectorization = True
        pto.slp_vectorization = True
        pb = llvm.create_pass_builder(self._tm, pto)
        mpm = pb.getModulePassManager()
        mpm.run(self._mod, pb)

        self._engine = llvm.create_mcjit_compiler(self._mod, self._tm)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        # ctypes signature depends on storage width
        sw = qmodel.target.storage_bits
        if sw == 32:
            self._cstorage = ctypes.c_int32
            self._np_storage = np.int32
        elif sw == 16:
            self._cstorage = ctypes.c_int16
            self._np_storage = np.int16
        elif sw == 8:
            self._cstorage = ctypes.c_int8
            self._np_storage = np.int8
        else:
            raise NotImplementedError(
                f"storage width {sw} not supported in JIT"
            )

        out_ptr = (
            ctypes.POINTER(ctypes.c_int32)
            if self._out_int
            else ctypes.POINTER(self._cstorage)
        )
        addr = self._engine.get_function_address("rc_predict")
        self._cfn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int64,
            ctypes.POINTER(self._cstorage),
            out_ptr,
        )(addr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """head="logits": float (T, M) at state scale. head="classify":
        int32 class indices (T,). head="proba": float (T, M) probabilities."""
        if X.ndim == 1:
            X = X[:, None]
        cfg = self.qmodel.config
        # The kernel preprocesses internally (PreprocessInput op); the caller
        # passes raw input quantized at input_scale.
        X_q = np.ascontiguousarray(
            self.qmodel.target.quantize_input_array(X, cfg).astype(
                self._np_storage
            )
        )
        T = X_q.shape[0]
        if self._out_int:
            Y = np.zeros(T, dtype=np.int32)
            self._cfn(
                ctypes.c_int64(T),
                X_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
                Y.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
            )
            return Y
        Y_q = np.zeros((T, self.qmodel.M), dtype=self._np_storage)
        self._cfn(
            ctypes.c_int64(T),
            X_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
            Y_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
        )
        if self.head == "proba":
            prob_frac = min(self.qmodel.target.storage_bits - 1, 15)
            return Y_q.astype(np.float64) / (1 << prob_frac)
        return Y_q.astype(np.float64) / cfg.state_scale

    @property
    def llvm_ir(self) -> str:
        return self._ir_text

    @property
    def optimized_ir(self) -> str:
        return str(self._mod)

    @property
    def assembly(self) -> str:
        return self._tm.emit_assembly(self._mod)

    def emit_object(self, path: str) -> None:
        target = llvm.Target.from_triple(llvm.get_default_triple())
        tm_pic = target.create_target_machine(opt=3, reloc="pic")
        with open(path, "wb") as f:
            f.write(tm_pic.emit_object(self._mod))


def compile_quantized_rc(qmodel, **kwargs) -> CompiledQuantizedRC:
    return CompiledQuantizedRC(qmodel, **kwargs)


_FASTMATH_OPS = ("fadd", "fsub", "fmul", "fdiv", "fneg")


def _add_fastmath_flags(ir_text: str) -> str:
    """Insert `fast` into every FP arithmetic instruction in the IR text.

    Matches `<op> ` followed by a float/double/vector type and rewrites it
    to `<op> fast `. The leading word boundary plus the following-type
    lookahead protect identifiers that merely start with `fadd`/etc.

    Needed only for vectorized targets: strict IEEE FP addition is not
    associative, so the LLVM loop vectorizer refuses to reorder the matmul
    reductions (and thus leaves them scalar) unless the `fadd`/`fmul`
    carry `fast` (or at least `reassoc contract`).
    """
    import re

    for op in _FASTMATH_OPS:
        pattern = re.compile(
            rf"\b{op}\s+(?=(?:float|double|half|<\s*\d+\s+x\s+"
            rf"(?:float|double|half)\s*>))"
        )
        ir_text = pattern.sub(f"{op} fast ", ir_text)
    return ir_text


class CrossCompiledRC:
    """AOT-only compiler targeting a non-host triple (e.g. Cortex-M0).

    Emits an LLVM module for the requested triple/CPU and optimizes it
    against that target machine, but does NOT JIT. Use `emit_object()`
    to write the cross-compiled object file for linking with a target
    toolchain (e.g. arm-none-eabi-gcc).
    """

    name = "llvm-cross"

    def __init__(
        self,
        rc: ReservoirComputer,
        exe: RCExecutor,
        *,
        triple: str,
        cpu: str = "",
        features: str = "",
        dtype: str = "f32",
        opt_level: int = 2,
        passes=None,
        head=None,
        vectorize: bool = False,
    ):
        _ensure_all_targets()
        self.rc = rc
        self.exe = exe
        self.triple = triple
        self.cpu = cpu
        self.dtype = dtype
        self.head = head or "logits"
        self.vectorize = vectorize

        module = emit_module(rc, exe, dtype=dtype, passes=passes, head=head)
        module.triple = triple
        ir_text = str(module)
        if vectorize:
            ir_text = _add_fastmath_flags(ir_text)
        self._ir_text = ir_text
        self._mod = llvm.parse_assembly(self._ir_text)
        self._mod.verify()

        target = llvm.Target.from_triple(triple)
        self._tm = target.create_target_machine(
            cpu=cpu,
            features=features,
            opt=opt_level,
            reloc="static",
        )

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        # Scalar targets (e.g. Cortex-M0) have no SIMD, so vector passes only
        # bloat the code. Targets with a vector ISA (e.g. wasm32 +simd128)
        # opt in via `vectorize=True` to let LLVM lower the f32 matmul inner
        # loops to packed `v128` ops (f32x4.mul / fma / load).
        pto.loop_vectorization = vectorize
        pto.slp_vectorization = vectorize
        # With weight shapes known at compile time, full unrolling eagerly
        # explodes the matmuls into thousands of scalar fmuls *before* the
        # loop vectorizer runs, defeating SIMD. Keep the loops rolled when
        # vectorizing so the vectorizer gets first crack at them.
        pto.loop_unrolling = not vectorize
        pb = llvm.create_pass_builder(self._tm, pto)
        mpm = pb.getModulePassManager()
        mpm.run(self._mod, pb)

    def emit_object(self, path: str) -> None:
        with open(path, "wb") as f:
            f.write(self._tm.emit_object(self._mod))

    def emit_assembly(self, path: str) -> None:
        with open(path, "w") as f:
            f.write(self._tm.emit_assembly(self._mod))

    @property
    def llvm_ir(self) -> str:
        return self._ir_text

    @property
    def optimized_ir(self) -> str:
        return str(self._mod)


def cross_compile_rc(
    rc: ReservoirComputer, exe: RCExecutor, **kwargs
) -> CrossCompiledRC:
    return CrossCompiledRC(rc, exe, **kwargs)
