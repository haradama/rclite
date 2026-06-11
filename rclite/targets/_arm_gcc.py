"""Shared arm-none-eabi-gcc toolchain mechanics for the ARM cross targets.

The Cortex-M0 and GBA targets drive the *same* toolchain: cross-compile a
quantized LLVM module to ``rc_predict.o``, assemble/compile C sources, link
with newlib-nano, and read ``size`` output. Only the per-board flags, startup
files and link libraries differ — those stay in the target classes, while the
mechanical invocations live here so they are written (and fixed) once.
"""

from __future__ import annotations
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Sequence, Tuple

import numpy as np


# storage_bits -> (C scalar type, numpy dtype) for the embedded test arrays.
_STORAGE = {
    32: ("int32_t", np.int32),
    16: ("int16_t", np.int16),
    8: ("int8_t", np.int8),
}


def storage_types(
    sw: int,
    *,
    allowed: Sequence[int] = (8, 16, 32),
    context: str = "compile",
) -> Tuple[str, type]:
    """Map ``storage_bits`` to ``(c_type, np_dtype)`` or raise for unsupported."""
    if sw not in allowed:
        raise NotImplementedError(
            f"{context}: storage_bits={sw} not supported"
        )
    return _STORAGE[sw]


def require_tool(
    name: str, *, hint: str = "install gcc-arm-none-eabi"
) -> None:
    """Raise a helpful error if ``name`` is not on PATH."""
    if shutil.which(name) is None:
        raise RuntimeError(f"{name} not found on PATH — {hint}")


def size_tool(cc: str) -> str:
    """Derive the ``size`` tool name from the compiler driver name."""
    return cc.replace("gcc", "size")


def run_tool(cmd, *, error: str) -> None:
    """Run a build subprocess, raising ``RuntimeError(error: stderr)`` on failure."""
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"{error}: {cp.stderr}")


def read_size(cc: str, elf: Path) -> Optional[str]:
    """Return ``arm-none-eabi-size`` output for ``elf``, or None if unavailable.

    Size reporting is best-effort metadata, so a missing/failing ``size`` tool
    is swallowed — but only the toolchain-level failures (tool absent, non-zero
    exit), not arbitrary exceptions.
    """
    try:
        sz = subprocess.run(
            [size_tool(cc), str(elf)],
            capture_output=True,
            text=True,
            check=True,
        )
        return sz.stdout.strip()
    except (OSError, subprocess.SubprocessError):
        return None


def cross_object(ll_mod, *, triple: str, cpu: str, out: Path) -> Path:
    """Optimize a quantized LLVM module for ``triple``/``cpu`` and emit objects.

    Writes ``rc_predict.o`` (and ``rc_predict.s`` for inspection) into ``out``
    and returns the object path. Loop/SLP vectorization is left off: the ARM
    cross targets (Cortex-M0, ARMv4T) have no SIMD to exploit.
    """
    import llvmlite.binding as llvm
    from rclite.codegen.llvm import _ensure_all_targets

    ll_mod.triple = triple
    _ensure_all_targets()
    mod = llvm.parse_assembly(str(ll_mod))
    mod.verify()
    target = llvm.Target.from_triple(triple)
    tm = target.create_target_machine(cpu=cpu, opt=2, reloc="static")
    pto = llvm.create_pipeline_tuning_options()
    pto.speed_level = 2
    pto.loop_vectorization = False
    pto.slp_vectorization = False
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod, pb)
    rc_o = out / "rc_predict.o"
    with open(rc_o, "wb") as f:
        f.write(tm.emit_object(mod))
    with open(out / "rc_predict.s", "w") as f:
        f.write(tm.emit_assembly(mod))
    return rc_o
