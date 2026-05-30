"""WebAssembly (wasm32-wasip1) cross-compilation target.

Lowers the IDL to a wasm32 module using:
  - llvmlite cross-compile (wasm32-wasip1, f32 by default)
  - rustc (wasm32-wasip1) to link the `rc_predict.o` object with a small
    Rust harness `main.rs` that embeds the test inputs / host reference
    outputs, calls `rc_predict`, and writes a comparison report via WASI
    stdout
  - The Rust harness pulls in wasi-libc for `tanhf` and friends, so no
    extra toolchain (clang/wasi-sdk) is required.

Runner uses `wasmtime` to execute the resulting `.wasm` module.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from ..target import Target, CompiledArtifact, RunResult


_SUPPORT_DIR = pathlib.Path(__file__).parent / "support"


class WasmTarget(Target):
    """WebAssembly cross-compile target (wasm32-wasip1)."""

    triple = "wasm32-wasip1"

    def __init__(self, dtype: str = "f32",
                 rustc: str = "rustc",
                 rust_target: str = "wasm32-wasip1",
                 rust_edition: str = "2024",
                 opt_level: str = "2"):
        if dtype != "f32":
            raise ValueError(
                f"WasmTarget currently only supports dtype='f32', got {dtype!r}"
            )
        self.dtype = dtype
        self.rustc = rustc
        self.rust_target = rust_target
        self.rust_edition = rust_edition
        self.opt_level = opt_level
        self.name = f"wasm32/{rust_target}"

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                **_) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Wasm deployment needs `test_inputs` to embed in main.rs"
            )
        if shutil.which(self.rustc) is None:
            raise RuntimeError(
                f"{self.rustc} not found on PATH -- install rustup and "
                f"`rustup target add {self.rust_target}`"
            )

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Cross-compile rc_predict.o for wasm32.
        cc_obj = cross_compile_rc(
            rc, exe, triple=self.triple, dtype=self.dtype,
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))

        # 2. C header (host JIT renders the same metadata).
        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        # 3. f32 host reference for embedded comparison.
        x_in = np.ascontiguousarray(test_inputs)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        if expected_outputs is None:
            expected_outputs = host_jit.predict(x_in).astype(np.float32)
        y_ref = np.ascontiguousarray(expected_outputs, dtype=np.float32)
        if y_ref.ndim == 1:
            y_ref = y_ref[:, None]

        T = x_in.shape[0]
        K = rc.input.units
        M = rc.readout.units
        if x_in.shape[1] != K:
            raise ValueError(
                f"test_inputs has K={x_in.shape[1]} columns, "
                f"but the model expects K={K}"
            )
        if y_ref.shape != (T, M):
            raise ValueError(
                f"expected_outputs shape {y_ref.shape} != ({T}, {M})"
            )

        x_flat = x_in.astype(np.float32).ravel()
        y_flat = y_ref.ravel()

        # 4. Render main.rs from template.
        tmpl = (_SUPPORT_DIR / "main_template.rs").read_text()
        main_path = out / "main.rs"
        main_path.write_text(
            tmpl
            .replace("@@T@@", str(T))
            .replace("@@K@@", str(K))
            .replace("@@M@@", str(M))
            .replace("@@X_VALUES@@", ", ".join(f"{v:.9g}_f32" for v in x_flat))
            .replace("@@Y_VALUES@@", ", ".join(f"{v:.9g}_f32" for v in y_flat))
        )

        # 5. Link via rustc: rustc compiles main.rs against rust-std
        #    (which bundles wasi-libc for `tanhf`/`memcpy`/...), then wasm-ld
        #    pulls rc_predict.o in through `-C link-arg`.
        wasm = out / "rc.wasm"
        cmd = [
            self.rustc,
            "--edition", self.rust_edition,
            "--target", self.rust_target,
            "-C", f"opt-level={self.opt_level}",
            "-C", f"link-arg={rc_o}",
            "-o", str(wasm),
            str(main_path),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"rustc link failed:\n  cmd: {' '.join(cmd)}\n"
                f"  stderr:\n{cp.stderr}"
            )

        metadata = {
            "triple": self.triple,
            "rust_target": self.rust_target,
            "dtype": self.dtype,
            "T": T, "K": K, "M": M,
            "wasm_size": wasm.stat().st_size,
        }

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=wasm,
            sources=[main_path, hdr],
            objects=[rc_o],
            metadata=metadata,
        )

    def compile_bench(self, rc, exe, *,
                      output_dir,
                      test_inputs: np.ndarray,
                      expected_outputs: Optional[np.ndarray] = None,
                      repeats: int = 25,
                      warmup: int = 3,
                      **_) -> CompiledArtifact:
        """Compile a benchmark harness that times rc_predict internally.

        The resulting `.wasm` runs `repeats` measured inferences (after
        `warmup` discarded ones) using the WASI `std::time::Instant` clock
        and prints best/median/mean wall-clock per inference plus a parity
        report against the embedded host reference. The Python driver in
        `benchmarks/host/compare_wasm.py` parses the keyed `RCLITE_BENCH:` lines.
        """
        if test_inputs is None:
            raise ValueError(
                "Wasm benchmark needs `test_inputs` to embed in main.rs"
            )
        if shutil.which(self.rustc) is None:
            raise RuntimeError(
                f"{self.rustc} not found on PATH -- install rustup and "
                f"`rustup target add {self.rust_target}`"
            )

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cc_obj = cross_compile_rc(
            rc, exe, triple=self.triple, dtype=self.dtype,
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))

        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        x_in = np.ascontiguousarray(test_inputs)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        if expected_outputs is None:
            expected_outputs = host_jit.predict(x_in).astype(np.float32)
        y_ref = np.ascontiguousarray(expected_outputs, dtype=np.float32)
        if y_ref.ndim == 1:
            y_ref = y_ref[:, None]

        T = x_in.shape[0]
        K = rc.input.units
        M = rc.readout.units
        if x_in.shape[1] != K:
            raise ValueError(
                f"test_inputs has K={x_in.shape[1]} columns, "
                f"but the model expects K={K}"
            )
        if y_ref.shape != (T, M):
            raise ValueError(
                f"expected_outputs shape {y_ref.shape} != ({T}, {M})"
            )

        x_flat = x_in.astype(np.float32).ravel()
        y_flat = y_ref.ravel()

        tmpl = (_SUPPORT_DIR / "bench_template.rs").read_text()
        main_path = out / "bench.rs"
        main_path.write_text(
            tmpl
            .replace("@@T@@", str(T))
            .replace("@@K@@", str(K))
            .replace("@@M@@", str(M))
            .replace("@@REPEATS@@", str(repeats))
            .replace("@@WARMUP@@", str(warmup))
            .replace("@@X_VALUES@@", ", ".join(f"{v:.9g}_f32" for v in x_flat))
            .replace("@@Y_VALUES@@", ", ".join(f"{v:.9g}_f32" for v in y_flat))
        )

        wasm = out / "rc_bench.wasm"
        cmd = [
            self.rustc,
            "--edition", self.rust_edition,
            "--target", self.rust_target,
            "-C", f"opt-level={self.opt_level}",
            "-C", f"link-arg={rc_o}",
            "-o", str(wasm),
            str(main_path),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"rustc link failed:\n  cmd: {' '.join(cmd)}\n"
                f"  stderr:\n{cp.stderr}"
            )

        return CompiledArtifact(
            target_name=self.name + "/bench",
            output_dir=out,
            binary=wasm,
            sources=[main_path, hdr],
            objects=[rc_o],
            metadata={
                "triple": self.triple,
                "rust_target": self.rust_target,
                "dtype": self.dtype,
                "T": T, "K": K, "M": M,
                "repeats": repeats, "warmup": warmup,
                "wasm_size": wasm.stat().st_size,
                "bench": True,
            },
        )

    def run(self, artifact: CompiledArtifact, *,
            wasmtime: str = "wasmtime",
            timeout: float = 60.0,
            **_) -> RunResult:
        if shutil.which(wasmtime) is None:
            raise RuntimeError(f"{wasmtime} not found on PATH")
        cp = subprocess.run(
            [wasmtime, str(artifact.binary)],
            capture_output=True, text=True, timeout=timeout,
        )
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(success=success, output=output, returncode=cp.returncode)
