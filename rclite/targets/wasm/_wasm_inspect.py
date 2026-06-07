"""Minimal WebAssembly binary inspector.

Parses just enough of the wasm module format to list a module's imports and
exports -- used to verify that a browser/reactor build has the expected
surface (no WASI imports, `rc_predict` + `memory` exported) without pulling
in an external tool like `wasm-objdump`.
"""

from __future__ import annotations
import struct
from typing import NamedTuple


class WasmModuleInfo(NamedTuple):
    imports: list[str]  # "module.name" entries
    exports: list[str]  # exported names


def _uleb(data: bytes, i: int) -> tuple[int, int]:
    result = shift = 0
    while True:
        b = data[i]
        i += 1
        result |= (b & 0x7F) << shift
        shift += 7
        if not (b & 0x80):
            return result, i


def inspect_wasm(path: str | bytes) -> WasmModuleInfo:
    """Return the import/export surface of a wasm module on disk (or bytes)."""
    if isinstance(path, (bytes, bytearray)):
        data = bytes(path)
    else:
        with open(path, "rb") as f:
            data = f.read()
    if data[:4] != b"\0asm":
        raise ValueError("not a WebAssembly binary (bad magic)")
    version = struct.unpack("<I", data[4:8])[0]
    if version != 1:
        raise ValueError(f"unsupported wasm version {version}")

    imports: list[str] = []
    exports: list[str] = []
    i = 8
    n = len(data)
    while i < n:
        section_id = data[i]
        i += 1
        size, i = _uleb(data, i)
        body = data[i : i + size]
        i += size
        if section_id == 2:  # import section
            j = 0
            count, j = _uleb(body, j)
            for _ in range(count):
                mlen, j = _uleb(body, j)
                mod = body[j : j + mlen].decode("utf-8", "replace")
                j += mlen
                nlen, j = _uleb(body, j)
                nm = body[j : j + nlen].decode("utf-8", "replace")
                j += nlen
                kind = body[j]
                j += 1
                if kind == 0:  # func: typeidx
                    _, j = _uleb(body, j)
                elif kind == 1:  # table: elemtype + limits
                    j += 1
                    flags = body[j]
                    j += 1
                    _, j = _uleb(body, j)
                    if flags & 1:
                        _, j = _uleb(body, j)
                elif kind == 2:  # memory: limits
                    flags = body[j]
                    j += 1
                    _, j = _uleb(body, j)
                    if flags & 1:
                        _, j = _uleb(body, j)
                elif kind == 3:  # global: valtype + mutability
                    j += 2
                imports.append(f"{mod}.{nm}")
        elif section_id == 7:  # export section
            j = 0
            count, j = _uleb(body, j)
            for _ in range(count):
                nlen, j = _uleb(body, j)
                nm = body[j : j + nlen].decode("utf-8", "replace")
                j += nlen
                j += 1  # export kind
                _, j = _uleb(body, j)  # export index
                exports.append(nm)
    return WasmModuleInfo(imports=imports, exports=exports)
