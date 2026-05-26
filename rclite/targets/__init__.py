"""Deployment targets.

Each `Target` is responsible for lowering an IDL `ReservoirComputer` into
native code for a specific platform, plus an optional runner that executes
the result on an emulator or host process.

Available targets:
    HostTarget        — native x86_64 Linux (LLVM JIT + shared library)
    CortexM0Target    — ARMv6-M Thumb cross-compile (configurable board)
    Microbit          — convenience preset: CortexM0Target on BBC micro:bit v1
"""
from .target import Target, CompiledArtifact, RunResult
from .host import HostTarget
from .cortex_m0 import CortexM0Target, CortexM0Board, MicrobitV1, Microbit

__all__ = [
    "Target", "CompiledArtifact", "RunResult",
    "HostTarget",
    "CortexM0Target", "CortexM0Board", "MicrobitV1", "Microbit",
]
