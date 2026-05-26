"""LLVM JIT backend for the RC IDL (via llvmlite).

Emits LLVM IR for a trained `ReservoirComputer` and JIT-compiles it via
llvmlite's MCJIT. The compiled module exposes a single C-ABI entry
point:

    void rc_predict(int64_t T, double* X, double* Y);

`X` is a contiguous row-major (T, K) matrix; `Y` is a (T, M) output
buffer the caller pre-allocates. Reservoir weights are embedded as
internal global constants so LLVM can constant-fold and vectorize.

Currently supports: tanh activation; any topology (DLR/DLRB/SCR/RANDOM);
include_bias / include_input readout features; RIDGE/PINV-trained readouts.
"""
from __future__ import annotations
import ctypes
from contextlib import contextmanager
from typing import Optional

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
# IR-driven lowering


class _Lowerer:
    """Walks an `rclite.ir.Module` and emits LLVM IR."""

    def __init__(self, ir_module, dtype: str):
        from rclite.ir.ops import ReadoutLinear, BuildPhi, FusedStepReadout, TimeLoop

        self.ir_module = ir_module
        self.fty, self.tanh_name, self.np_dtype, _ = _dtype_bindings(dtype)
        self.K, self.N, self.M = ir_module.K, ir_module.N, ir_module.M

        self.module = ir.Module(name=f"rc_jit_{id(ir_module)}")
        self.module.triple = llvm.get_default_triple()

        self.libm_fn = ir.Function(
            self.module, ir.FunctionType(self.fty, [self.fty]),
            name=self.tanh_name,
        )

        # Emit weight globals
        self.globals = {}
        for name, arr in ir_module.weights.items():
            self.globals[name] = self._emit_global(name, arr)

        # rc_predict function
        fnty = ir.FunctionType(
            ir.VoidType(),
            [_I64, self.fty.as_pointer(), self.fty.as_pointer()],
        )
        self.fn = ir.Function(self.module, fnty, name="rc_predict")
        self.T_arg, self.X_arg, self.Y_arg = self.fn.args
        self.T_arg.name = "T"; self.X_arg.name = "X"; self.Y_arg.name = "Y"

        entry = self.fn.append_basic_block("entry")
        self.b = ir.IRBuilder(entry)

        # Determine scratch sizes
        needs_phi = any(
            isinstance(op, (ReadoutLinear, BuildPhi))
            for op in self._flatten_ops()
        )
        max_F = max((op.F for op in self._flatten_ops()
                      if isinstance(op, (ReadoutLinear, FusedStepReadout))),
                     default=self.N + self.K + 1)

        self.h = self.b.alloca(self.fty, size=_ci(self.N), name="h")
        self.u_pre = self.b.alloca(self.fty, size=_ci(self.K), name="u_pre")
        self.pre_arr = self.b.alloca(self.fty, size=_ci(self.N), name="pre")
        self.phi_arr = (
            self.b.alloca(self.fty, size=_ci(max(max_F, 1)), name="phi")
            if needs_phi else None
        )
        self.acc = self.b.alloca(self.fty, name="acc")

        # Init h to zero
        with _loop(self.b, _ci(self.N), "init") as i:
            _store1d(self.b, self.h, i, self._cf(0.0))

        self.t = None  # current time index, valid inside a TimeLoop body

    def _cf(self, v):
        return ir.Constant(self.fty, float(v))

    def _emit_global(self, name, arr):
        flat = np.ascontiguousarray(arr, dtype=self.np_dtype).reshape(-1)
        ty = ir.ArrayType(self.fty, flat.size)
        g = ir.GlobalVariable(self.module, ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(ty, [self._cf(float(v)) for v in flat])
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
            TimeLoop, PreprocessInput, ReservoirStep, BuildPhi,
            ReadoutLinear, FusedStepReadout,
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
        raise NotImplementedError(f"unknown op: {type(op).__name__}")

    def _lower_time_loop(self, op):
        K_unroll = op.unroll
        T = self.T_arg
        if K_unroll == 1:
            with _loop(self.b, T, "t") as t:
                self.t = t
                for body_op in op.body:
                    self._lower(body_op)
            self.t = None
            return
        # Unroll body by `K_unroll` over [0, T_unrolled), tail loop for remainder.
        K_const = _ci(K_unroll)
        T_unrolled = self.b.mul(self.b.sdiv(T, K_const), K_const)
        with _loop_strided(self.b, _ci(0), T_unrolled, K_const, "tu") as t_base:
            for k in range(K_unroll):
                self.t = (t_base if k == 0
                          else self.b.add(t_base, _ci(k), name=f"t_{k}"))
                for body_op in op.body:
                    self._lower(body_op)
        with _loop_strided(self.b, T_unrolled, T, _ci(1), "ttail") as t:
            self.t = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None

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

        with _loop(self.b, _ci(op.N), "ipre") as i:
            self.b.store(self._cf(op.bias), self.acc)
            with _loop(self.b, _ci(op.K), "kin") as k:
                w = _load2d_global(self.b, g_Win, op.K, i, k)
                u_val = _load1d(self.b, self.u_pre, k)
                self.b.store(
                    self.b.fadd(self.b.load(self.acc), self.b.fmul(w, u_val)),
                    self.acc,
                )
            self._emit_res_contrib(op.topology, op.N, op.chain_weight,
                                    op.chain_feedback, g_Wres, i)
            _store1d(self.b, self.pre_arr, i, self.b.load(self.acc))

        with _loop(self.b, _ci(op.N), "iupd") as i:
            h_old = _load1d(self.b, self.h, i)
            pre_i = _load1d(self.b, self.pre_arr, i)
            tan = self.b.call(self.libm_fn, [pre_i])
            new_h = self.b.fadd(
                self.b.fmul(self._cf(1.0 - op.leak), h_old),
                self.b.fmul(self._cf(op.leak), tan),
            )
            _store1d(self.b, self.h, i, new_h)

    def _emit_res_contrib(self, topology, N, chain_weight, chain_feedback,
                           g_Wres, i):
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
            b.store(b.fadd(b.load(self.acc),
                            b.fmul(cf(chain_weight), val)), self.acc)
        elif topology == Topology.DLRB:
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val_back = _load1d(b, self.h, i_back)
            contrib_back = b.select(is_pos,
                                     b.fmul(cf(chain_weight), val_back),
                                     cf(0.0))
            is_lt_last = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt_last, b.add(i, _ci(1)), _ci(N - 1))
            val_fwd = _load1d(b, self.h, i_fwd)
            contrib_fwd = b.select(is_lt_last,
                                    b.fmul(cf(chain_feedback), val_fwd),
                                    cf(0.0))
            b.store(b.fadd(b.fadd(b.load(self.acc), contrib_back),
                            contrib_fwd), self.acc)
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
                _store1d(self.b, self.phi_arr,
                          self.b.add(_ci(off), k), x_val)
            off += op.K
        with _loop(self.b, _ci(op.N), "iphi") as i:
            _store1d(self.b, self.phi_arr,
                      self.b.add(_ci(off), i),
                      _load1d(self.b, self.h, i))

    def _lower_readout_linear(self, op):
        g_Wout = self.globals[op.W_out_name]
        tM = self.b.mul(self.t, _ci(op.M))
        with _loop(self.b, _ci(op.M), "m") as m:
            self.b.store(self._cf(0.0), self.acc)
            with _loop(self.b, _ci(op.F), "fout") as fi:
                w = _load2d_global(self.b, g_Wout, op.F, m, fi)
                pv = _load1d(self.b, self.phi_arr, fi)
                self.b.store(
                    self.b.fadd(self.b.load(self.acc), self.b.fmul(w, pv)),
                    self.acc,
                )
            _store1d(self.b, self.Y_arg, self.b.add(tM, m),
                      self.b.load(self.acc))

    def _lower_fused(self, op):
        """Step + readout in one op: no phi buffer materialization."""
        g_Win = self.globals[op.W_in_name]
        g_Wres = self.globals.get(op.W_res_name) if op.W_res_name else None
        g_Wout = self.globals[op.W_out_name]
        b = self.b
        cf = self._cf

        # Step (same as _lower_reservoir_step)
        with _loop(b, _ci(op.N), "ipre") as i:
            b.store(cf(op.bias), self.acc)
            with _loop(b, _ci(op.K), "kin") as k:
                w = _load2d_global(b, g_Win, op.K, i, k)
                u_val = _load1d(b, self.u_pre, k)
                b.store(b.fadd(b.load(self.acc), b.fmul(w, u_val)), self.acc)
            self._emit_res_contrib(op.topology, op.N, op.chain_weight,
                                    op.chain_feedback, g_Wres, i)
            _store1d(b, self.pre_arr, i, b.load(self.acc))
        with _loop(b, _ci(op.N), "iupd") as i:
            h_old = _load1d(b, self.h, i)
            pre_i = _load1d(b, self.pre_arr, i)
            tan = b.call(self.libm_fn, [pre_i])
            new_h = b.fadd(
                b.fmul(cf(1.0 - op.leak), h_old),
                b.fmul(cf(op.leak), tan),
            )
            _store1d(b, self.h, i, new_h)

        # Readout — phi is virtual; we index W_out's columns directly.
        tM = b.mul(self.t, _ci(op.M))
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
                    w = _load2d_global(b, g_Wout, op.F, m,
                                        b.add(_ci(bias_off), k))
                    x_val = _load1d(b, self.X_arg, b.add(tK, k))
                    b.store(b.fadd(b.load(self.acc), b.fmul(w, x_val)),
                             self.acc)
            with _loop(b, _ci(op.N), "ifo") as i:
                w = _load2d_global(b, g_Wout, op.F, m,
                                    b.add(_ci(input_off), i))
                hv = _load1d(b, self.h, i)
                b.store(b.fadd(b.load(self.acc), b.fmul(w, hv)), self.acc)
            _store1d(b, self.Y_arg, b.add(tM, m), b.load(self.acc))


def emit_module(rc: ReservoirComputer, exe: RCExecutor,
                *, dtype: str = "f64", passes=None) -> ir.Module:
    """Build an rclite IR module, apply passes, and lower to LLVM IR.

    `dtype` selects f64 (host) vs f32 (Cortex-M cross-compile).
    `passes` is a list of `rclite.ir.passes.*` instances; defaults to
    `[StructuralSpecialize()]`.
    """
    if rc.reservoir.activation != Activation.TANH:
        raise NotImplementedError(
            f"LLVM backend only supports tanh; got {rc.reservoir.activation.name}"
        )

    # Import here to avoid an import cycle (rclite.ir uses runtime types).
    from rclite.ir import build_ir
    from rclite.ir.passes import StructuralSpecialize

    ir_module = build_ir(rc, exe)
    if passes is None:
        passes = [StructuralSpecialize()]
    for p in passes:
        ir_module = p(ir_module)
    return _Lowerer(ir_module, dtype=dtype).lower()


def emit_quantized_module(qmodel, *, passes=None,
                            saturating: bool = True) -> ir.Module:
    """Build LLVM IR for the integer quantized path (i32 or i16).

    Function signature:
        void rc_predict(int64_t T, storage_t* X, storage_t* Y);
    where storage_t = int32_t for `I32FixedPoint` and int16_t for `I16FixedPoint`.

    `saturating=True` wraps inner-loop accumulations and the final
    truncation with `@llvm.sadd.sat.*` and clamping selects, so overflow
    saturates instead of wrapping. Recommended for i16 (narrow range);
    cheap to leave on for i32 as well.
    """
    from rclite.quant.ir_builder import build_ir_from_quantized

    ir_module = build_ir_from_quantized(qmodel)
    if passes is None:
        passes = []
    for p in passes:
        ir_module = p(ir_module)
    return _IntLowerer(ir_module, saturating=saturating).lower()


# ----------------------------------------------------------------------------
# Integer (quantized) lowering


def _load1d_global(b: ir.IRBuilder, g, idx):
    """Load element from a 1D global array at i64/i32 index."""
    return b.load(b.gep(g, [_ci32(0), idx]))


class _IntLowerer:
    """Lower an rclite IR module under `dtype` in {'i32', 'i16'} to LLVM IR.

    Parameterized over storage_ty / accum_ty:
        i32 storage : accumulator i64    (mirage default)
        i16 storage : accumulator i32    (-OS / size-constrained)

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
    matmul accumulators (mostly relevant for i16 where overflow is realistic).
    """

    def __init__(self, ir_module, *, saturating: bool = True):
        from rclite.ir.ops import (
            TimeLoop, ReservoirStep, BuildPhi, ReadoutLinear, FusedStepReadout,
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
        else:
            raise ValueError(
                f"_IntLowerer supports dtype in {{'i32', 'i16'}}, got {dtype!r}"
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

        # Weight / LUT globals at storage_ty (i32 or i16)
        self.globals = {}
        for name, arr in ir_module.weights.items():
            self.globals[name] = self._emit_int_global(name, arr)

        fnty = ir.FunctionType(
            ir.VoidType(),
            [_I64, self.storage_ty.as_pointer(), self.storage_ty.as_pointer()],
        )
        self.fn = ir.Function(self.module, fnty, name="rc_predict")
        self.T_arg, self.X_arg, self.Y_arg = self.fn.args
        self.T_arg.name = "T"; self.X_arg.name = "X"; self.Y_arg.name = "Y"

        entry = self.fn.append_basic_block("entry")
        self.b = ir.IRBuilder(entry)

        needs_phi = any(
            isinstance(op, (ReadoutLinear, BuildPhi))
            for op in self._flatten_ops()
        )
        max_F = max(
            (op.F for op in self._flatten_ops()
              if isinstance(op, (ReadoutLinear, FusedStepReadout))),
            default=self.N + self.K + 1,
        )

        self.h = self.b.alloca(self.storage_ty, size=_ci(self.N), name="h")
        self.pre_arr = self.b.alloca(self.storage_ty, size=_ci(self.N), name="pre")
        self.u_pre = self.b.alloca(
            self.storage_ty, size=_ci(max(self.K, 1)), name="u_pre",
        )
        self.phi_arr = (
            self.b.alloca(self.storage_ty, size=_ci(max(max_F, 1)), name="phi")
            if needs_phi else None
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
        np_dtype = np.int16 if self.storage_bits == 16 else np.int32
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

    # ------------------------------------------------------------------
    # dispatcher

    def lower(self) -> ir.Module:
        for op in self.ir_module.ops:
            self._lower(op)
        self.b.ret_void()
        return self.module

    def _lower(self, op):
        from rclite.ir.ops import (
            TimeLoop, PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear,
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
        raise NotImplementedError(
            f"{type(op).__name__} not supported in the integer path"
        )

    def _lower_time_loop(self, op):
        with _loop(self.b, self.T_arg, "t") as t:
            self.t = t
            for body_op in op.body:
                self._lower(body_op)
        self.t = None

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
                diff, scale_const, self.weight_frac,
            )
            _store1d(self.b, self.u_pre, k, u_pre_val)

    def _lower_reservoir_step(self, op):
        g_Win = self.globals["W_in"]
        g_Wres = self.globals.get(op.W_res_name) if op.W_res_name else None
        g_lut = self.globals["lut_table"]
        K, N = op.K, op.N

        with _loop(self.b, _ci(N), "ipre") as i:
            # acc is accum_ty (i32 for i16 storage, i64 for i32 storage).
            # bias_q is stored at state_scale, sext-widen it.
            self.b.store(self._ca(self.bias_q), self.acc)
            with _loop(self.b, _ci(K), "kin") as k:
                w = _load2d_global(self.b, g_Win, K, i, k)
                # u_pre lives in scratch (filled by _lower_preprocess); X_arg
                # is the *raw* input — the readout in BuildPhi reads it.
                u = _load1d(self.b, self.u_pre, k)
                prod = self._fixed_mul_to_accum(w, u, self.shift_in)
                self.b.store(
                    self._accum_add(self.b.load(self.acc), prod), self.acc,
                )
            # Topology-aware reservoir contribution. For structured topologies
            # this emits O(1) scalar work per row instead of an O(N) matmul.
            self._emit_res_contrib_int(op, g_Wres, i)
            # Truncate widened accumulator back to storage_ty for pre[i].
            pre_val = self.b.trunc(self.b.load(self.acc), self.storage_ty)
            _store1d(self.b, self.pre_arr, i, pre_val)

        with _loop(self.b, _ci(N), "iupd") as i:
            pre_i = _load1d(self.b, self.pre_arr, i)
            activated = self._emit_lut_lookup(pre_i, g_lut)
            h_old = _load1d(self.b, self.h, i)
            t1 = self._fixed_mul_to_storage(h_old, self._cs(self.one_minus_leak_q),
                                              self.state_frac)
            t2 = self._fixed_mul_to_storage(activated, self._cs(self.leak_q),
                                              self.state_frac)
            new_h = self.b.add(t1, t2)
            _store1d(self.b, self.h, i, new_h)

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
            prod = self._fixed_mul_to_accum(self._cs(cw_q), val, self.shift_res)
            contrib = b.select(is_pos, prod, self._ca(0))
            b.store(self._accum_add(b.load(self.acc), contrib), self.acc)
        elif op.topology == Topology.SCR:
            # Cyclic chain: prev = (i - 1) mod N
            cw_q = int(round(op.chain_weight * weight_scale))
            is_zero = b.icmp_signed("==", i, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i, _ci(1)))
            val = _load1d(b, self.h, i_prev)
            prod = self._fixed_mul_to_accum(self._cs(cw_q), val, self.shift_res)
            b.store(self._accum_add(b.load(self.acc), prod), self.acc)
        elif op.topology == Topology.DLRB:
            cw_q = int(round(op.chain_weight * weight_scale))
            cb_q = int(round(op.chain_feedback * weight_scale))
            # Backward chain: chain_weight * h[i-1] for i > 0
            is_pos = b.icmp_signed(">", i, _ci(0))
            i_back = b.select(is_pos, b.sub(i, _ci(1)), _ci(0))
            val_back = _load1d(b, self.h, i_back)
            prod_back = self._fixed_mul_to_accum(self._cs(cw_q), val_back,
                                                  self.shift_res)
            contrib_back = b.select(is_pos, prod_back, self._ca(0))
            # Forward chain: chain_feedback * h[i+1] for i < N-1
            is_lt_last = b.icmp_signed("<", i, _ci(N - 1))
            i_fwd = b.select(is_lt_last, b.add(i, _ci(1)), _ci(N - 1))
            val_fwd = _load1d(b, self.h, i_fwd)
            prod_fwd = self._fixed_mul_to_accum(self._cs(cb_q), val_fwd,
                                                  self.shift_res)
            contrib_fwd = b.select(is_lt_last, prod_fwd, self._ca(0))
            acc_val = b.load(self.acc)
            b.store(
                self._accum_add(self._accum_add(acc_val, contrib_back),
                                 contrib_fwd),
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
            _store1d(self.b, self.phi_arr, _ci(off),
                      self._cs(1 << self.state_frac))
            off += 1
        if op.include_input:
            with _loop(self.b, _ci(K), "kphi") as k:
                u_val = _load1d(self.b, self.X_arg, self.b.add(tK, k))
                _store1d(self.b, self.phi_arr, self.b.add(_ci(off), k), u_val)
            off += K
        with _loop(self.b, _ci(N), "iphi") as i:
            _store1d(self.b, self.phi_arr, self.b.add(_ci(off), i),
                      _load1d(self.b, self.h, i))

    def _lower_readout_linear(self, op):
        """Readout in i64 accumulator regardless of storage width.

        Optionally uses `@llvm.sadd.sat.i64` for accumulation when
        `saturating=True`. Final i64 → storage_ty truncation happens after
        the >> state_frac shift, with saturation to the storage range.
        """
        g_Wout = self.globals["W_out"]
        F = op.F
        tM = self.b.mul(self.t, _ci(op.M))
        sadd_i64 = (self.module.globals.get("llvm.sadd.sat.i64")
                     if self.saturating else None)
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
                summed = (self.b.call(sadd_i64, [cur, prod])
                          if self.saturating
                          else self.b.add(cur, prod))
                self.b.store(summed, self.acc64)
            shifted = self.b.ashr(self.b.load(self.acc64),
                                    self._ci64(self.state_frac))
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
            _store1d(self.b, self.Y_arg, self.b.add(tM, m), y)


class CompiledRC:
    """JIT-compiled ReservoirComputer (LLVM backend).

    Mirrors the relevant subset of `RCExecutor.predict`.
    """

    name = "llvm"

    def __init__(self, rc: ReservoirComputer, exe: RCExecutor,
                 opt_level: int = 3, vectorize: bool = True,
                 passes=None):
        _ensure_initialized()
        self.rc = rc
        self.exe = exe
        self._ir_text = str(emit_module(rc, exe, passes=passes))
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

        addr = self._engine.get_function_address("rc_predict")
        self._cfn = ctypes.CFUNCTYPE(
            None, ctypes.c_int64,
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_double),
        )(addr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        M = self.rc.readout.units
        X = np.ascontiguousarray(X, dtype=np.float64)
        Y = np.zeros((T, M), dtype=np.float64)
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
                    f"link failed: {' '.join(cmd)}\n"
                    f"stderr:\n{result.stderr}"
                )
        finally:
            try:
                os.unlink(obj_path)
            except FileNotFoundError:
                pass

    def emit_header(self, path: str, fn_name: str = "rc_predict") -> None:
        """Write a C header declaring the compiled function."""
        K = self.rc.input.units
        N = self.rc.reservoir.units
        M = self.rc.readout.units
        topo = self.rc.reservoir.topology.name
        trainer = self.rc.readout.trainer.name
        guard = "RC_PREDICT_H"
        header = (
            f"/* Auto-generated header for compiled ReservoirComputer.\n"
            f" *\n"
            f" *   input units      = {K}\n"
            f" *   reservoir units  = {N}\n"
            f" *   output units     = {M}\n"
            f" *   topology         = {topo}\n"
            f" *   trainer          = {trainer}\n"
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
            f"\n"
            f"#ifdef __cplusplus\n"
            f"extern \"C\" {{\n"
            f"#endif\n"
            f"\n"
            f"/* Run inference over a length-T sequence.\n"
            f" *   X: row-major (T x RC_INPUT_DIM) input.   Caller-owned.\n"
            f" *   Y: row-major (T x RC_OUTPUT_DIM) output. Caller-allocated.\n"
            f" */\n"
            f"void {fn_name}(int64_t T, double *X, double *Y);\n"
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

    def __init__(self, qmodel, opt_level: int = 3, passes=None,
                 saturating: bool = True):
        _ensure_initialized()
        self.qmodel = qmodel
        self.rc = qmodel.rc
        self.saturating = saturating
        self._ir_text = str(emit_quantized_module(
            qmodel, passes=passes, saturating=saturating,
        ))
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
        else:
            raise NotImplementedError(f"storage width {sw} not supported in JIT")

        addr = self._engine.get_function_address("rc_predict")
        self._cfn = ctypes.CFUNCTYPE(
            None, ctypes.c_int64,
            ctypes.POINTER(self._cstorage),
            ctypes.POINTER(self._cstorage),
        )(addr)

    def predict(self, X: np.ndarray) -> np.ndarray:
        if X.ndim == 1:
            X = X[:, None]
        cfg = self.qmodel.config
        # The kernel preprocesses internally (PreprocessInput op); the caller
        # passes raw input quantized at input_scale.
        X_q = np.ascontiguousarray(
            self.qmodel.target.quantize_input_array(X, cfg).astype(self._np_storage)
        )
        T = X_q.shape[0]
        Y_q = np.zeros((T, self.qmodel.M), dtype=self._np_storage)
        self._cfn(
            ctypes.c_int64(T),
            X_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
            Y_q.ctypes.data_as(ctypes.POINTER(self._cstorage)),
        )
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


class CrossCompiledRC:
    """AOT-only compiler targeting a non-host triple (e.g. Cortex-M0).

    Emits an LLVM module for the requested triple/CPU and optimizes it
    against that target machine, but does NOT JIT. Use `emit_object()`
    to write the cross-compiled object file for linking with a target
    toolchain (e.g. arm-none-eabi-gcc).
    """

    name = "llvm-cross"

    def __init__(self, rc: ReservoirComputer, exe: RCExecutor, *,
                 triple: str, cpu: str = "", features: str = "",
                 dtype: str = "f32", opt_level: int = 2, passes=None):
        _ensure_all_targets()
        self.rc = rc
        self.exe = exe
        self.triple = triple
        self.cpu = cpu
        self.dtype = dtype

        module = emit_module(rc, exe, dtype=dtype, passes=passes)
        module.triple = triple
        self._ir_text = str(module)
        self._mod = llvm.parse_assembly(self._ir_text)
        self._mod.verify()

        target = llvm.Target.from_triple(triple)
        self._tm = target.create_target_machine(
            cpu=cpu, features=features, opt=opt_level, reloc="static",
        )

        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = opt_level
        pto.loop_vectorization = False  # Cortex-M0 has no SIMD; skip vector passes.
        pto.slp_vectorization = False
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


def cross_compile_rc(rc: ReservoirComputer, exe: RCExecutor, **kwargs) -> CrossCompiledRC:
    return CrossCompiledRC(rc, exe, **kwargs)
