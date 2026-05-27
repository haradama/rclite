"""WebAssembly deployment target (wasm32-wasip1, runnable via wasmtime)."""
from .target import WasmTarget


class Wasmtime(WasmTarget):
    """Convenience preset: wasm32-wasip1 module runnable via the wasmtime CLI."""
    def __init__(self, dtype: str = "f32"):
        super().__init__(dtype=dtype)


__all__ = ["WasmTarget", "Wasmtime"]
