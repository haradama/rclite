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

The per-scheme IR lowerers live in sibling modules and are re-exported
here for backward compatibility:

    * `_llvm_common`  - shared types / IR-emission primitives
    * `_llvm_float`   - `_Lowerer` + `emit_module` (f64/f32)
    * `_llvm_int`     - `_IntLowerer` + `emit_quantized_module` (symmetric)
    * `_llvm_affine`  - `_AffineLowerer` + `emit_quantized_affine_module`

This module keeps the user-facing `Compiled*RC` JIT wrappers and the
`compile_*` / `cross_compile_rc` entry points.
"""

from __future__ import annotations
import ctypes

import numpy as np
import llvmlite.binding as llvm

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from ._llvm_common import (
    _ensure_initialized,
    _ensure_all_targets,
    _pow2_exp,
)
from ._llvm_float import emit_module
from ._llvm_int import emit_quantized_module
from ._llvm_affine import emit_quantized_affine_module

__all__ = [
    "emit_module",
    "emit_quantized_module",
    "emit_quantized_affine_module",
    "compile_rc",
    "CompiledRC",
    "compile_quantized_rc",
    "CompiledQuantizedRC",
    "CompiledAffineRC",
    "cross_compile_rc",
    "CrossCompiledRC",
    "_ensure_initialized",
    "_ensure_all_targets",
    "_pow2_exp",
]


class CompiledAffineRC:
    """JIT-compiled affine `AffineQuantizedModel` (host LLVM).

    Mirrors `CompiledQuantizedRC` but consumes an `AffineQuantizedModel`
    and emits via `_AffineLowerer`. `predict()` accepts float inputs,
    quantizes them via the model's input params (matching what the
    Python `AffineQuantizedExecutor.predict` does), calls the kernel,
    and dequantizes the output back to float.
    """

    name = "llvm-affine"

    def __init__(
        self, qmodel, opt_level: int = 3, passes=None, head=None, vlen=1
    ):
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
                vlen=vlen,
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
