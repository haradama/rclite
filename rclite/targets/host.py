"""Native host target (Linux x86_64).

Emits a shared library (.so) plus C header for in-process use from Python
(ctypes) or from a C program linking with the library. Keeps the JIT
engine alive so the same target instance can also be used for in-process
inference via `artifact.metadata['jit']`.
"""

from __future__ import annotations
import pathlib

from rclite.codegen import compile_rc
from .target import Target, CompiledArtifact


class HostTarget(Target):
    """Native host (LLVM JIT + emit_shared_library)."""

    name = "host-native"

    def __init__(self, dtype: str = "f64"):
        self.dtype = dtype

    def compile(
        self,
        rc,
        exe,
        *,
        output_dir,
        lib_name: str = "rc",
        emit_shared_library: bool = True,
        emit_header: bool = True,
        **_,
    ) -> CompiledArtifact:
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        jit = compile_rc(rc, exe)
        sources, binary = [], None

        if emit_header:
            hdr = out / "rc_predict.h"
            jit.emit_header(str(hdr))
            sources.append(hdr)

        if emit_shared_library:
            so = out / f"lib{lib_name}.so"
            jit.emit_shared_library(str(so))
            binary = so

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=binary,
            sources=sources,
            metadata={"jit": jit, "dtype": self.dtype},
        )
