"""WebAssembly deployment target (wasm32-wasip1, runnable via wasmtime).

`WasmTarget(simd=True)` is the default; pass `simd=False` for a scalar
baseline (useful for ablation benchmarks).
"""
from .target import WasmTarget


class Wasmtime(WasmTarget):
    """Convenience preset: wasm32-wasip1 module with SIMD128 enabled,
    runnable via the wasmtime CLI."""
    def __init__(self, dtype: str = "f32", *, simd: bool = True):
        super().__init__(dtype=dtype, simd=simd)


__all__ = ["WasmTarget", "Wasmtime"]
