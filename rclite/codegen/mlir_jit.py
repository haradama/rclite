"""Execute MLIR kernels via the llvmlite MCJIT backend (MLIR -> LLVM IR -> JIT).

This is the *bridge* in rclite's codegen story:

    MLIR (arith/memref/scf, built by the xDSL constructors mlir_*_xdsl)
      --mlir-opt-->        LLVM dialect
      --mlir-translate-->  LLVM IR (text)
      --llvmlite-->        MCJIT  (host execution)

Why bridge to llvmlite instead of MLIR's own ExecutionEngine: rclite kernels
bake every weight / LUT into `memref.global` constants. llvmlite's MCJIT
materialises those constant globals correctly (it is the same backend the
production `CompiledQuantizedRC` / `CompiledAffineRC` already use); the MLIR
Python bindings' ExecutionEngine in the current wheels does not (reads zero /
segfaults). This path therefore drops `llc` + `gcc` + the `.so`/dlopen step
*and* the broken ExecutionEngine — lowering+translate stay on the (nix-pinned)
CLI tools, execution unifies on the production MCJIT.

llvmlite stays the single execution substrate; MLIR is the opt-in
representation layer. Needs `mlir-opt` + `mlir-translate` on PATH (use the nix
devShell for an LLVM-20 toolchain); llvmlite is already a core dependency.
"""

from __future__ import annotations

import ctypes
import pathlib
import re
import shutil
import subprocess
import tempfile
from typing import Optional

import numpy as np

import llvmlite.binding as llvm

# mlir-opt lowering pipeline (arith/memref/scf -> LLVM dialect), shared by the
# JIT and cross-compile paths.
_LOWER_PASSES = [
    "--convert-scf-to-cf",
    "--expand-strided-metadata",
    "--finalize-memref-to-llvm",
    "--convert-cf-to-llvm",
    "--convert-arith-to-llvm",
    "--convert-func-to-llvm",
    "--reconcile-unrealized-casts",
]
# Host JIT needs only lower+translate (llvmlite does codegen); cross-compile to
# an embedded object also needs llc.
_TRANSLATE_TOOLS = ("mlir-opt", "mlir-translate")
_CROSS_TOOLS = ("mlir-opt", "mlir-translate", "llc")
_C_STORAGE = {8: ctypes.c_int8, 16: ctypes.c_int16, 32: ctypes.c_int32}
_NP_STORAGE = {8: np.int8, 16: np.int16, 32: np.int32}


class _MemRef1D(ctypes.Structure):
    """1-D MLIR memref descriptor for the `_mlir_ciface_` ABI."""

    _fields_ = [
        ("alloc", ctypes.c_void_p),
        ("align", ctypes.c_void_p),
        ("offset", ctypes.c_int64),
        ("size", ctypes.c_int64),
        ("stride", ctypes.c_int64),
    ]


def _desc(arr: np.ndarray) -> _MemRef1D:
    p = arr.ctypes.data_as(ctypes.c_void_p)
    return _MemRef1D(p, p, 0, arr.shape[0], 1)


_LLVM_READY = False


def _ensure_llvm() -> None:
    global _LLVM_READY
    if not _LLVM_READY:
        llvm.initialize_native_target()
        llvm.initialize_native_asmprinter()
        _LLVM_READY = True


def _llvm_tool_major(tool: str) -> int | None:
    """Best-effort LLVM major version from `<tool> --version`."""
    exe = shutil.which(tool)
    if exe is None:
        return None
    r = subprocess.run([exe, "--version"], capture_output=True, text=True)
    if r.returncode != 0:
        return None
    m = re.search(r"LLVM version\s+(\d+)\.", r.stdout)
    if m is None:
        return None
    return int(m.group(1))


def tools_available() -> bool:
    """True when MLIR CLI tools are present and LLVM major versions match.

    The bridge translates MLIR with external LLVM tools and then parses the
    LLVM IR using llvmlite's embedded LLVM parser. Different LLVM majors can
    emit/accept different IR attribute syntax and fail at parse time.
    """
    if not all(shutil.which(t) for t in _TRANSLATE_TOOLS):
        return False
    tool_major = _llvm_tool_major("mlir-opt")
    if tool_major is None:
        return False
    llvmlite_major = int(llvm.llvm_version_info[0])
    return tool_major == llvmlite_major


def mlir_to_llvm_ir(mlir_text: str, *, extra_passes=()) -> str:
    """Lower + translate MLIR to LLVM IR text.

    `mlir-opt <extra_passes> <_LOWER_PASSES>` then `mlir-translate
    --mlir-to-llvmir`. The same lowering pipeline the text/xDSL emitters target;
    only the final object emission (llc/gcc) is replaced — by llvmlite
    downstream. `extra_passes` are prepended (e.g. `--convert-vector-to-llvm`
    for the Stage-3 `vector`-dialect float kernels).
    """
    missing = [t for t in _TRANSLATE_TOOLS if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"MLIR->LLVMIR bridge needs {missing} on PATH")

    def run(cmd, inp):
        r = subprocess.run(cmd, input=inp, capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[:4000]}")
        return r.stdout

    lowered = run(["mlir-opt", *extra_passes, *_LOWER_PASSES], mlir_text)
    return run(["mlir-translate", "--mlir-to-llvmir", "-"], lowered)


def cross_compile_object(
    mlir_text: str,
    *,
    triple: str,
    cpu: str = "",
    features: str = "",
    extra_passes=(),
    filetype: str = "obj",
) -> bytes:
    """Cross-compile MLIR to a relocatable object for `triple` (no host link).

    The MCJIT path above is host-only; this is the embedded counterpart —
    emits a `.o` for e.g. thumbv6m (Cortex-M0), thumbv4t (GBA), wasm32 (WASM),
    the same triples the llvmlite production path serves. Uses the same
    lowering+translate, then retargets `llc`. `extra_passes` are prepended to the
    mlir-opt pipeline (e.g. `--convert-vector-to-llvm` for the `rc`-dialect
    `vector` float kernels); `filetype` is `obj` (returns object bytes) or `asm`
    (returns the textual assembly bytes — handy to confirm SIMD instructions).

    NOTE on quantized SIMD: only the genuinely non-associative parts of the
    integer kernel (the per-row *saturation* / requantize) must stay scalar. The
    i64 matvec *reduction* IS associative (no mid-loop saturation, no overflow
    for realistic sizes), so it vectorizes BIT-EXACT —
    `emit_affine_mlir_xdsl(..., vlen=N)` emits a `vector` i64 reduction whose
    integer output is byte-identical to the scalar kernel (verified host + wasm
    in `tests/mlir_affine_spike_test.py::test_vectorized_matvec_bit_exact`). The
    float `vector` kernel vectorizes likewise. One `rc`/affine dialect -> SIMD on
    wasm32 (+simd128), aarch64/armv7 (+neon), x86 (+avx), riscv (+v).
    """
    missing = [t for t in _CROSS_TOOLS if shutil.which(t) is None]
    if missing:
        raise RuntimeError(f"MLIR cross-compile needs {missing} on PATH")
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "rc.mlir").write_text(mlir_text)

        def run(cmd):
            r = subprocess.run(cmd, capture_output=True, text=True)
            if r.returncode != 0:
                raise RuntimeError(f"{cmd[0]} failed:\n{r.stderr[:3000]}")
            return r.stdout

        (td / "rc.ll.mlir").write_text(
            run(
                [
                    "mlir-opt",
                    str(td / "rc.mlir"),
                    *extra_passes,
                    *_LOWER_PASSES,
                ]
            )
        )
        (td / "rc.ll").write_text(
            run(["mlir-translate", "--mlir-to-llvmir", str(td / "rc.ll.mlir")])
        )
        out = td / ("rc.s" if filetype == "asm" else "rc.o")
        llc = [
            "llc",
            "-O2",
            f"-mtriple={triple}",
            f"-filetype={filetype}",
            str(td / "rc.ll"),
            "-o",
            str(out),
        ]
        if cpu:
            llc.append(f"-mcpu={cpu}")
        if features:
            llc.append(f"-mattr={features}")
        run(llc)
        return out.read_bytes()


class CompiledMLIRJit:
    """JIT-compile an MLIR kernel via llvmlite MCJIT and run it through ctypes.

    The lowered IR is executed in-process by MCJIT (the same llvmlite backend
    the production path uses) — no llc+gcc+.so link. Use `jit_symmetric` /
    `jit_affine` to build + JIT from a quantized model in one call.

    `mlir_text` is whatever the emitters produce (e.g.
    `emit_symmetric_mlir_xdsl(qm)` / `emit_affine_mlir_xdsl(qm, head=...)`).
    `classify=True` selects the i32 class-id output ABI.
    """

    def __init__(
        self,
        mlir_text: str,
        *,
        storage_bits: int,
        M: int,
        K: int,
        classify: bool = False,
        opt_level: int = 3,
        extra_passes=(),
    ):
        if storage_bits not in _C_STORAGE:
            raise NotImplementedError(
                f"storage width {storage_bits} unsupported"
            )
        _ensure_llvm()
        self.M, self.K = M, K
        self._classify = classify
        self._np_in = _NP_STORAGE[storage_bits]
        self._np_out = np.int32 if classify else self._np_in

        ir_text = mlir_to_llvm_ir(mlir_text, extra_passes=extra_passes)
        self._mod = llvm.parse_assembly(ir_text)
        self._mod.verify()
        tm = llvm.Target.from_triple(
            llvm.get_default_triple()
        ).create_target_machine(opt=opt_level)
        self._engine = llvm.create_mcjit_compiler(self._mod, tm)
        self._engine.finalize_object()
        self._engine.run_static_constructors()

        # The c-interface wrapper takes (i64 T, memref* X, memref* Y); the memref
        # descriptor (_MemRef1D) is element-type agnostic, so one signature fits
        # every storage width / head.
        addr = self._engine.get_function_address("_mlir_ciface_rc_predict")
        self._fn = ctypes.CFUNCTYPE(
            None,
            ctypes.c_int64,
            ctypes.POINTER(_MemRef1D),
            ctypes.POINTER(_MemRef1D),
        )(addr)

    def predict_q(self, X_q: np.ndarray) -> np.ndarray:
        """Run the JIT kernel on pre-quantized inputs (storage dtype)."""
        X_q = np.ascontiguousarray(X_q, dtype=self._np_in).reshape(-1)
        T = X_q.size // self.K
        out_len = T if self._classify else T * self.M
        Y = np.zeros(out_len, dtype=self._np_out)
        dx, dy = _desc(X_q), _desc(Y)
        self._fn(ctypes.c_int64(T), ctypes.byref(dx), ctypes.byref(dy))
        return Y if self._classify else Y.reshape(T, self.M)


def jit_symmetric(
    qmodel,
    *,
    head: Optional[str] = None,
    sparse: Optional[str] = None,
) -> CompiledMLIRJit:
    """Build the symmetric MLIR (xDSL constructor) and JIT it via llvmlite."""
    from rclite.codegen.mlir_symmetric_xdsl import emit_symmetric_mlir_xdsl

    mlir_text = emit_symmetric_mlir_xdsl(qmodel, head=head, sparse=sparse)
    return CompiledMLIRJit(
        mlir_text,
        storage_bits=qmodel.target.storage_bits,
        M=qmodel.M,
        K=qmodel.K,
        classify=head == "classify",
    )


def jit_affine(
    qmodel,
    *,
    head: Optional[str] = None,
    sparse: Optional[str] = None,
    vlen: int = 1,
) -> CompiledMLIRJit:
    """Build the affine MLIR (xDSL constructor) and JIT it via llvmlite.

    `vlen > 1` vectorizes the dense W_res i64 matvec reduction (`vector` dialect,
    `--convert-vector-to-llvm`). The i64 sum is associative, so the output is
    BIT-EXACT with the scalar kernel — quantized SIMD that keeps host<->device
    integer equality (verified in `tests/mlir_affine_spike_test.py`)."""
    from rclite.codegen.mlir_affine_xdsl import emit_affine_mlir_xdsl

    mlir_text = emit_affine_mlir_xdsl(
        qmodel, head=head, sparse=sparse, vlen=vlen
    )
    return CompiledMLIRJit(
        mlir_text,
        storage_bits=qmodel.storage_bits,
        M=qmodel.M,
        K=qmodel.K,
        classify=head == "classify",
        extra_passes=["--convert-vector-to-llvm"] if vlen > 1 else (),
    )
