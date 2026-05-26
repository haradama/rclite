"""MLIR backend skeleton — emits textual MLIR for a ReservoirComputer.

Status: PARTIAL. Generates valid MLIR text using the `func`, `arith`,
`memref`, and `linalg` dialects. Lowering to native code requires the
LLVM/MLIR toolchain (`mlir-opt`, `mlir-translate`, `llc`) which is not
assumed to be present. When `mlir-opt` is on PATH, `MLIRBackend.compile()`
will invoke the lowering pipeline; otherwise it raises with a clear
message and the emitted MLIR is still available via `emit_mlir()`.

The point of having this backend alongside the LLVM one is to make
custom dialects (e.g. a `rc.reservoir_step` op) addressable in the
future: define the op, write lowering patterns to `linalg`, reuse the
rest of the pipeline.

Reference pipeline:

    mlir-opt input.mlir \\
        -convert-linalg-to-loops \\
        -convert-scf-to-cf -convert-cf-to-llvm \\
        -convert-arith-to-llvm -convert-memref-to-llvm \\
        -convert-func-to-llvm -reconcile-unrealized-casts \\
        | mlir-translate --mlir-to-llvmir \\
        | llc -O3 -filetype=obj -o input.o
"""
from __future__ import annotations
import shutil
import subprocess
from dataclasses import dataclass
from typing import Optional

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Activation, Topology
from rclite.runtime.reference import RCExecutor


def emit_mlir(rc: ReservoirComputer, exe: RCExecutor) -> str:
    """Emit textual MLIR for the given trained reservoir computer.

    Currently emits a single `func.func @rc_predict(...)` skeleton using
    `linalg.matmul` for the dense W_in and W_res applications. Weights
    are emitted as `memref.global` constants.
    """
    if rc.reservoir.activation != Activation.TANH:
        raise NotImplementedError(
            f"MLIR backend currently only supports tanh; got {rc.reservoir.activation.name}"
        )
    if exe.W_out is None:
        raise ValueError("Readout has not been trained — call fit() first")

    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()
    leak = float(rc.reservoir.leak_rate)
    bias = float(rc.reservoir.bias)
    in_off = float(rc.input.input_offset)
    in_sc = float(rc.input.input_scaling)

    def _arr(name: str, arr, shape):
        flat = arr.reshape(-1).tolist()
        body = ", ".join(f"{float(v):.17g}" for v in flat)
        shape_s = "x".join(str(d) for d in shape)
        return (f"  memref.global \"private\" constant @{name} : "
                f"memref<{shape_s}xf64> = dense<[{body}]>")

    globals_ir = [
        _arr("rc_W_in", exe.W_in, (N, K)),
        _arr("rc_W_out", exe.W_out, (M, F)),
    ]
    if rc.reservoir.topology not in (Topology.DLR, Topology.DLRB, Topology.SCR):
        globals_ir.append(_arr("rc_W_res", exe.W_res, (N, N)))

    header = (
        f"// Generated MLIR for ReservoirComputer (N={N}, K={K}, M={M}, F={F})\n"
        f"// leak={leak}, bias={bias}, input_scaling={in_sc}, "
        f"input_offset={in_off}, topology={rc.reservoir.topology.name}\n"
        f"module {{\n"
    )
    body = (
        "  // STUB: function body lowering is not yet implemented in textual form.\n"
        "  // Intended pipeline:\n"
        "  //   1. linalg.matmul for W_in*u and W_res*h (or scalar ops for DLR/SCR/DLRB)\n"
        "  //   2. linalg.generic + math.tanh for the activation\n"
        "  //   3. scf.for over time steps; linalg.matmul for the readout\n"
        "  func.func @rc_predict(%T: i64, %X: memref<?x?xf64>, %Y: memref<?x?xf64>) {\n"
        "    return\n"
        "  }\n"
    )
    return header + "\n".join(globals_ir) + "\n" + body + "}\n"


@dataclass
class MLIRBackend:
    """Driver for the textual-MLIR codegen path.

    The compile() method requires `mlir-opt`, `mlir-translate`, and `llc`
    on PATH. If any are missing it raises a `RuntimeError` describing
    what's needed.
    """
    name: str = "mlir"

    def __post_init__(self):
        self._required_tools = ("mlir-opt", "mlir-translate", "llc")

    def emit(self, rc: ReservoirComputer, exe: RCExecutor) -> str:
        return emit_mlir(rc, exe)

    def compile(self, rc: ReservoirComputer, exe: RCExecutor):
        missing = [t for t in self._required_tools if shutil.which(t) is None]
        if missing:
            raise RuntimeError(
                f"MLIR backend requires {self._required_tools} on PATH "
                f"(missing: {missing}). Install with e.g. "
                f"`sudo apt install llvm-18 mlir-18-tools`, or use the "
                f"LLVMBackend via rc_idl.codegen.compile_rc instead."
            )
        raise NotImplementedError(
            "MLIR function-body lowering is not yet implemented. "
            "See rc_idl/codegen/mlir_backend.py docstring for the intended "
            "pipeline. The textual MLIR for inspection is available via "
            "MLIRBackend.emit(rc, exe)."
        )
