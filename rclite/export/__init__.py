"""Target-agnostic export of a quantized reservoir computer.

`export_bundle(qmodel, out_dir)` writes a ready-to-build bundle that is both
a portable C99 library and a Cargo crate:

    rc_kernel.c   pure-integer kernel (PROGMEM on AVR)
    rc_model.h    dims, storage type, float<->quant helpers, decl
    Cargo.toml    crate manifest (default `std`, optional `no_std`)
    build.rs      compiles the C kernel via the `cc` crate
    src/lib.rs    safe Rust FFI wrapper + metadata constants
    README.md     usage for both entry points

Both quantization families are supported: the asymmetric *affine* path
(`AffineQuantizedModel`) and the symmetric *Q-format* path
(`QuantizedModel`). The emitted C is bit-exact with the respective Python
reference executor / LLVM JIT.

`export_optimized_object(qmodel, target=...)` ships the opposite contract: a
*pre-optimized* relocatable object (`rc_kernel.o`) that rclite compiled through
MLIR/LLVM — vectorized for the target's SIMD ISA — plus a thin C header. The
caller links the object instead of compiling source, so it gets the SIMD
without its own compiler reproducing it (affine models only).
"""

from __future__ import annotations

from .info import KernelInfo, info_from_affine, info_from_symmetric
from .c_header import emit_c_header
from .c_kernel_symmetric import emit_symmetric_kernel_c
from .rust import emit_rust_lib, emit_cargo_toml, emit_build_rs
from .bundle import export_bundle
from .optimized_object import (
    export_optimized_object,
    OptimizedObjectBundle,
    SimdTarget,
    SIMD_TARGETS,
)
from .avr_object import export_avr_object, AvrObjectBundle
from .gba_object import export_gba_object, GbaObjectBundle

__all__ = [
    "export_bundle",
    "export_optimized_object",
    "OptimizedObjectBundle",
    "SimdTarget",
    "SIMD_TARGETS",
    "export_avr_object",
    "AvrObjectBundle",
    "export_gba_object",
    "GbaObjectBundle",
    "KernelInfo",
    "info_from_affine",
    "info_from_symmetric",
    "emit_c_header",
    "emit_symmetric_kernel_c",
    "emit_rust_lib",
    "emit_cargo_toml",
    "emit_build_rs",
]
