"""rclite IR — high-level operations and RC-aware optimization passes.

Analogous to rc-bench's MLIR `rc` dialect, but realized in Python so it
can be folded into the existing llvmlite-based lowering with no new
toolchain dependency. The textual `to_mlir_text()` printer emits ops in
a form compatible with the rc-bench dialect for future cross-tool use.
"""
from .module import Module
from .ops import (
    Op,
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear,
    FusedStepReadout, TimeLoop,
)
from .builder import build_ir
from .printer import to_mlir_text
from .passes import StructuralSpecialize, FuseStepReadout, TimeUnroll

__all__ = [
    "Module",
    "Op",
    "PreprocessInput", "ReservoirStep", "BuildPhi", "ReadoutLinear",
    "FusedStepReadout", "TimeLoop",
    "build_ir", "to_mlir_text",
    "StructuralSpecialize", "FuseStepReadout", "TimeUnroll",
]
