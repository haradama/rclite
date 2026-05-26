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


def emit_module(rc: ReservoirComputer, exe: RCExecutor,
                *, dtype: str = "f64") -> ir.Module:
    """Build the LLVM IR module for the given trained reservoir computer.

    `dtype` selects the floating-point width — `f64` (host default) or
    `f32` (Cortex-M cross-compile). Weights stored as f64 in `exe` are
    cast to the chosen width when emitted as IR constants.
    """
    if rc.reservoir.activation != Activation.TANH:
        raise NotImplementedError(
            f"LLVM backend only supports tanh; got {rc.reservoir.activation.name}"
        )
    if exe.W_out is None:
        raise ValueError("Readout has not been trained — call fit() first")

    fty, tanh_name, np_dtype, _ = _dtype_bindings(dtype)

    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()

    leak = float(rc.reservoir.leak_rate)
    one_minus_leak = 1.0 - leak
    bias_val = float(rc.reservoir.bias)
    in_off = float(rc.input.input_offset)
    in_sc = float(rc.input.input_scaling)
    inc_bias = bool(rc.readout.include_bias)
    inc_input = bool(rc.readout.include_input)

    module = ir.Module(name=f"rc_jit_{id(rc)}")
    module.triple = llvm.get_default_triple()

    def cf(v):
        return ir.Constant(fty, float(v))

    def emit_global(name, arr):
        flat = np.ascontiguousarray(arr, dtype=np_dtype).reshape(-1)
        ty = ir.ArrayType(fty, flat.size)
        g = ir.GlobalVariable(module, ty, name=name)
        g.linkage = "internal"
        g.global_constant = True
        g.initializer = ir.Constant(ty, [cf(float(v)) for v in flat])
        return g

    def emit_res(b, g_Wres_, h_, i_, acc_):
        topo = rc.reservoir.topology
        if topo == Topology.DLR:
            r = float(rc.reservoir.chain_weight)
            is_pos = b.icmp_signed(">", i_, _ci(0))
            i_safe = b.select(is_pos, b.sub(i_, _ci(1)), _ci(0))
            val = _load1d(b, h_, i_safe)
            contrib = b.select(is_pos, b.fmul(cf(r), val), cf(0.0))
            b.store(b.fadd(b.load(acc_), contrib), acc_)
        elif topo == Topology.SCR:
            r = float(rc.reservoir.chain_weight)
            is_zero = b.icmp_signed("==", i_, _ci(0))
            i_prev = b.select(is_zero, _ci(N - 1), b.sub(i_, _ci(1)))
            val = _load1d(b, h_, i_prev)
            b.store(b.fadd(b.load(acc_), b.fmul(cf(r), val)), acc_)
        elif topo == Topology.DLRB:
            r = float(rc.reservoir.chain_weight)
            bw = float(rc.reservoir.chain_feedback)
            is_pos = b.icmp_signed(">", i_, _ci(0))
            i_back = b.select(is_pos, b.sub(i_, _ci(1)), _ci(0))
            val_back = _load1d(b, h_, i_back)
            contrib_back = b.select(is_pos, b.fmul(cf(r), val_back), cf(0.0))
            is_lt_last = b.icmp_signed("<", i_, _ci(N - 1))
            i_fwd = b.select(is_lt_last, b.add(i_, _ci(1)), _ci(N - 1))
            val_fwd = _load1d(b, h_, i_fwd)
            contrib_fwd = b.select(is_lt_last, b.fmul(cf(bw), val_fwd), cf(0.0))
            b.store(b.fadd(b.fadd(b.load(acc_), contrib_back), contrib_fwd), acc_)
        else:
            with _loop(b, _ci(N), "jres") as j:
                w = _load2d_global(b, g_Wres_, N, i_, j)
                hv = _load1d(b, h_, j)
                b.store(b.fadd(b.load(acc_), b.fmul(w, hv)), acc_)

    libm_fn = ir.Function(module, ir.FunctionType(fty, [fty]), name=tanh_name)

    g_Win = emit_global("rc_W_in", exe.W_in)
    is_structured = rc.reservoir.topology in (
        Topology.DLR, Topology.DLRB, Topology.SCR
    )
    g_Wres = None if is_structured else emit_global("rc_W_res", exe.W_res)
    g_Wout = emit_global("rc_W_out", exe.W_out)

    fnty = ir.FunctionType(
        ir.VoidType(),
        [_I64, fty.as_pointer(), fty.as_pointer()],
    )
    fn = ir.Function(module, fnty, name="rc_predict")
    T_arg, X_arg, Y_arg = fn.args
    T_arg.name, X_arg.name, Y_arg.name = "T", "X", "Y"

    entry = fn.append_basic_block("entry")
    b = ir.IRBuilder(entry)

    h = b.alloca(fty, size=_ci(N), name="h")
    u_pre = b.alloca(fty, size=_ci(K), name="u_pre")
    pre_arr = b.alloca(fty, size=_ci(N), name="pre")
    phi_arr = b.alloca(fty, size=_ci(F), name="phi")
    acc = b.alloca(fty, name="acc")

    with _loop(b, _ci(N), "init") as i:
        _store1d(b, h, i, cf(0.0))

    with _loop(b, T_arg, "t") as t:
        tK = b.mul(t, _ci(K))
        tM = b.mul(t, _ci(M))

        with _loop(b, _ci(K), "kpre") as k:
            x_val = _load1d(b, X_arg, b.add(tK, k))
            up = b.fmul(b.fsub(x_val, cf(in_off)), cf(in_sc))
            _store1d(b, u_pre, k, up)

        with _loop(b, _ci(N), "ipre") as i:
            b.store(cf(bias_val), acc)
            with _loop(b, _ci(K), "kin") as k:
                w = _load2d_global(b, g_Win, K, i, k)
                u_val = _load1d(b, u_pre, k)
                b.store(b.fadd(b.load(acc), b.fmul(w, u_val)), acc)
            emit_res(b, g_Wres, h, i, acc)
            _store1d(b, pre_arr, i, b.load(acc))

        with _loop(b, _ci(N), "iupd") as i:
            h_old = _load1d(b, h, i)
            pre_i = _load1d(b, pre_arr, i)
            tan = b.call(libm_fn, [pre_i])
            new_h = b.fadd(
                b.fmul(cf(one_minus_leak), h_old),
                b.fmul(cf(leak), tan),
            )
            _store1d(b, h, i, new_h)

        off = 0
        if inc_bias:
            _store1d(b, phi_arr, _ci(off), cf(1.0))
            off += 1
        if inc_input:
            with _loop(b, _ci(K), "kphi") as k:
                x_val = _load1d(b, X_arg, b.add(tK, k))
                _store1d(b, phi_arr, b.add(_ci(off), k), x_val)
            off += K
        with _loop(b, _ci(N), "iphi") as i:
            _store1d(b, phi_arr, b.add(_ci(off), i), _load1d(b, h, i))

        with _loop(b, _ci(M), "m") as m:
            b.store(cf(0.0), acc)
            with _loop(b, _ci(F), "fout") as fi:
                w = _load2d_global(b, g_Wout, F, m, fi)
                pv = _load1d(b, phi_arr, fi)
                b.store(b.fadd(b.load(acc), b.fmul(w, pv)), acc)
            _store1d(b, Y_arg, b.add(tM, m), b.load(acc))

    b.ret_void()
    return module


class CompiledRC:
    """JIT-compiled ReservoirComputer (LLVM backend).

    Mirrors the relevant subset of `RCExecutor.predict`.
    """

    name = "llvm"

    def __init__(self, rc: ReservoirComputer, exe: RCExecutor,
                 opt_level: int = 3, vectorize: bool = True):
        _ensure_initialized()
        self.rc = rc
        self.exe = exe
        self._ir_text = str(emit_module(rc, exe))
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
                 dtype: str = "f32", opt_level: int = 2):
        _ensure_all_targets()
        self.rc = rc
        self.exe = exe
        self.triple = triple
        self.cpu = cpu
        self.dtype = dtype

        module = emit_module(rc, exe, dtype=dtype)
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
