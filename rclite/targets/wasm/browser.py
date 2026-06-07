"""Browser / JS WebAssembly target (zero-WASI "reactor" modules).

The default :class:`WasmTarget` builds a WASI command module (a `_start`
that prints via `fd_write`) meant for `wasmtime`. Browsers, Node, Deno and
edge runtimes instead want a *reactor*: a module with no WASI imports that
simply **exports** `rc_predict` + the linear `memory`, so JS can write inputs
into memory, call the kernel, and read outputs back.

This target links the cross-compiled kernel object with `wasm-ld` (or the
`rust-lld` that ships inside rustc, used in `-flavor wasm` mode -- so no extra
toolchain is required beyond what the WASI target already needs) into such a
reactor, and emits a small ES-module loader (`rclite.js`) plus a runnable
`index.html` demo.

  - f32 build: imports `env.tanhf` (the loader wires it to `Math.tanh`).
  - quantized i8/i16/i32 build: **zero imports** (tanh is a LUT) -- a fully
    self-contained module.
"""
from __future__ import annotations
import os
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from rclite.codegen.templating import render_template
from ..target import CompiledArtifact, RunResult
from .target import WasmTarget, _SUPPORT_DIR
from ._wasm_inspect import inspect_wasm


def _resolve_wasm_ld() -> list[str]:
    """Return the argv prefix for a wasm linker.

    Prefers `wasm-ld` on PATH; otherwise falls back to the `rust-lld` bundled
    in the active rustc sysroot, invoked as `rust-lld -flavor wasm`.
    """
    found = shutil.which("wasm-ld")
    if found:
        return [found]
    try:
        sysroot = subprocess.run(
            ["rustc", "--print", "sysroot"],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except Exception:
        sysroot = ""
    if sysroot:
        for root, _dirs, files in os.walk(sysroot):
            if "rust-lld" in files:
                return [str(pathlib.Path(root) / "rust-lld"), "-flavor", "wasm"]
    raise RuntimeError(
        "no wasm linker found -- install `wasm-ld` (LLVM/lld) or ensure "
        "rustc ships `rust-lld` (rustup component `llvm-tools` / the wasm "
        "target)"
    )


class BrowserWasm(WasmTarget):
    """WebAssembly reactor target for browsers / JS runtimes."""

    def __init__(self, dtype: str = "f32", *,
                 simd: bool = True,
                 rustc: str = "rustc",
                 rust_target: str = "wasm32-wasip1",
                 opt_level: str = "2",
                 loader_name: str = "rclite.js",
                 wasm_name: Optional[str] = None):
        super().__init__(dtype=dtype, simd=simd, rustc=rustc,
                         rust_target=rust_target, opt_level=opt_level)
        self.loader_name = loader_name
        self._wasm_name_override = wasm_name
        self.name = (f"wasm32/browser+simd128" if simd else "wasm32/browser")

    # ------------------------------------------------------------------ link

    def _link_reactor(self, rc_o: pathlib.Path, wasm: pathlib.Path, *,
                      import_tanhf: bool) -> None:
        ld = _resolve_wasm_ld()
        # `--allow-undefined` lets the f32 kernel's `tanhf` become an import
        # for JS to satisfy. It is also required for wasm-ld to synthesize the
        # exported linear `memory`; it is harmless for the integer kernel,
        # which has no undefined symbols (verified: zero imports).
        cmd = ld + [
            "--no-entry",
            "--allow-undefined",
            "--export=rc_predict",
            "--export=memory",
            "--export=__heap_base",
            "--strip-debug",
            str(rc_o), "-o", str(wasm),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"wasm-ld reactor link failed:\n  cmd: {' '.join(cmd)}\n"
                f"  stderr:\n{cp.stderr}"
            )

    # --------------------------------------------------------------- emit JS

    def _emit_loader_and_demo(self, out: pathlib.Path, *, dtype: str,
                              K: int, M: int, array_ctor: str, bytes_per: int,
                              has_tanhf: bool, input_scale: float,
                              state_scale: float, in_offset: float,
                              in_scaling: float, wasm_name: str,
                              demo_input: np.ndarray, T: int,
                              imports: list[str], wasm_size: int) -> list[pathlib.Path]:
        loader = render_template(
            _SUPPORT_DIR / "browser_loader.js",
            DTYPE=dtype,
            K=str(K),
            M=str(M),
            ARRAY=array_ctor,
            BYTES=str(bytes_per),
            HAS_TANHF="true" if has_tanhf else "false",
            INPUT_SCALE=repr(float(input_scale)),
            STATE_SCALE=repr(float(state_scale)),
            IN_OFFSET=repr(float(in_offset)),
            IN_SCALING=repr(float(in_scaling)),
        )
        loader_path = out / self.loader_name
        loader_path.write_text(loader)

        demo_flat = np.ascontiguousarray(demo_input, dtype=np.float32).ravel()
        html = render_template(
            _SUPPORT_DIR / "browser_index.html",
            DTYPE=dtype,
            K=str(K),
            M=str(M),
            T=str(T),
            WASM_NAME=wasm_name,
            WASM_SIZE=str(wasm_size),
            LOADER_NAME=self.loader_name,
            IMPORTS=", ".join(imports) or "(none)",
            DEMO_INPUT=", ".join(f"{v:.6g}" for v in demo_flat),
        )
        html_path = out / "index.html"
        html_path.write_text(html)
        return [loader_path, html_path]

    # ---------------------------------------------------------- public: f32

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                **_) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Browser deployment needs `test_inputs` for the demo page"
            )
        self._check_toolchain()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cc_obj = cross_compile_rc(
            rc, exe, triple=self.triple, dtype=self.dtype,
            features=self._features(), vectorize=self.simd,
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))

        wasm_name = self._wasm_name_override or "rc.wasm"
        wasm = out / wasm_name
        self._link_reactor(rc_o, wasm, import_tanhf=True)
        info = inspect_wasm(str(wasm))

        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        x_in = np.ascontiguousarray(test_inputs)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        K = rc.input.units
        M = rc.readout.units
        T = x_in.shape[0]

        sources = self._emit_loader_and_demo(
            out, dtype="f32", K=K, M=M, array_ctor="Float32Array",
            bytes_per=4, has_tanhf=True, input_scale=1.0, state_scale=1.0,
            in_offset=rc.input.input_offset, in_scaling=rc.input.input_scaling,
            wasm_name=wasm_name, demo_input=x_in, T=T,
            imports=info.imports, wasm_size=wasm.stat().st_size,
        )

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=wasm,
            sources=[*sources, hdr],
            objects=[rc_o],
            metadata={
                "triple": self.triple,
                "dtype": "f32",
                "simd": self.simd,
                "browser": True,
                "reactor": True,
                "imports": info.imports,
                "exports": info.exports,
                "K": K, "M": M, "T": T,
                "wasm_size": wasm.stat().st_size,
            },
        )

    # ---------------------------------------------------- public: quantized

    def compile_quantized(self, qmodel, *,
                          output_dir,
                          test_inputs: np.ndarray,
                          **_) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Browser quantized deployment needs `test_inputs` for the demo"
            )
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sw = qmodel.target.storage_bits
        array_ctor = {8: "Int8Array", 16: "Int16Array", 32: "Int32Array"}.get(sw)
        bytes_per = {8: 1, 16: 2, 32: 4}.get(sw)
        if array_ctor is None:
            raise NotImplementedError(
                f"BrowserWasm.compile_quantized: storage_bits={sw} not "
                f"supported (expected 8 / 16 / 32)"
            )

        rc_o = self._compile_quantized_object(qmodel, out)
        wasm_name = self._wasm_name_override or "rc_q.wasm"
        wasm = out / wasm_name
        # integer kernel: tanh is a LUT, so there is nothing to import
        self._link_reactor(rc_o, wasm, import_tanhf=False)
        info = inspect_wasm(str(wasm))

        rc = qmodel.rc
        cfg = qmodel.config
        x_in = np.ascontiguousarray(test_inputs)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        K = rc.input.units
        M = qmodel.M
        T = x_in.shape[0]

        sources = self._emit_loader_and_demo(
            out, dtype=f"i{sw}", K=K, M=M, array_ctor=array_ctor,
            bytes_per=bytes_per, has_tanhf=False,
            input_scale=cfg.input_scale, state_scale=cfg.state_scale,
            in_offset=rc.input.input_offset, in_scaling=rc.input.input_scaling,
            wasm_name=wasm_name, demo_input=x_in, T=T,
            imports=info.imports, wasm_size=wasm.stat().st_size,
        )

        return CompiledArtifact(
            target_name=self.name + f"/quantized-i{sw}",
            output_dir=out,
            binary=wasm,
            sources=sources,
            objects=[rc_o],
            metadata={
                "triple": self.triple,
                "dtype": f"i{sw}",
                "quantized": True,
                "browser": True,
                "reactor": True,
                "imports": info.imports,
                "exports": info.exports,
                "K": K, "M": M, "T": T,
                "state_frac": cfg.state_frac,
                "wasm_size": wasm.stat().st_size,
            },
        )

    # --------------------------------------------------------------- runner

    def run(self, artifact: CompiledArtifact, *,
            wasmtime: str = "wasmtime",
            timeout: float = 60.0,
            **_) -> RunResult:
        """Smoke-check the reactor.

        A reactor has no `_start`, so it can't be "run" as a WASI command.
        We instead verify the export surface and -- when the module has zero
        imports (the quantized build) -- instantiate it under `wasmtime
        --invoke` to confirm `rc_predict` is callable in a non-WASI host.
        """
        info = inspect_wasm(str(artifact.binary))
        missing = [s for s in ("rc_predict", "memory") if s not in info.exports]
        if missing:
            return RunResult(
                success=False,
                output=f"missing exports: {missing}; have {info.exports}",
                returncode=1,
            )
        if info.imports:
            # f32: needs env.tanhf supplied by a JS host -- can't exec here.
            return RunResult(
                success=True,
                output=(f"reactor OK (structure-only check)\n"
                        f"imports={info.imports}\nexports={info.exports}"),
                returncode=0,
            )
        if shutil.which(wasmtime) is None:
            return RunResult(
                success=True,
                output=(f"reactor OK (no wasmtime to exec)\n"
                        f"exports={info.exports}"),
                returncode=0,
            )
        cp = subprocess.run(
            [wasmtime, "run", "--invoke", "rc_predict",
             str(artifact.binary), "1", "1024", "2048"],
            capture_output=True, text=True, timeout=timeout,
        )
        ok = cp.returncode == 0
        return RunResult(
            success=ok,
            output=(f"wasmtime --invoke rc_predict -> exit {cp.returncode}\n"
                    f"exports={info.exports}\n{cp.stderr}"),
            returncode=cp.returncode,
        )
