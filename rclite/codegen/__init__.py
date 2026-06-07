"""Code generation backends: LLVM (active), MLIR (skeleton).

Compiles a trained `ReservoirComputer` into native machine code so inference
runs without the Python interpreter or numpy in the loop.
"""

try:
    import llvmlite  # noqa: F401
except ImportError as e:
    raise ImportError(
        "rclite.codegen requires llvmlite. Install with `pip install llvmlite`."
    ) from e

from .llvm import (
    compile_rc,
    CompiledRC,
    emit_module,
    cross_compile_rc,
    CrossCompiledRC,
)
from .mlir import MLIRBackend, emit_mlir

__all__ = [
    "compile_rc",
    "CompiledRC",
    "emit_module",
    "cross_compile_rc",
    "CrossCompiledRC",
    "MLIRBackend",
    "emit_mlir",
]
