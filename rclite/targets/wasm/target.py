"""WebAssembly (wasm32-wasip1) cross-compilation target.

Lowers the IDL to a wasm32 module using:
  - llvmlite cross-compile (wasm32-wasip1, f32 by default), optionally with
    the WebAssembly SIMD128 proposal enabled (`features="+simd128"` plus
    LLVM loop/SLP vectorization). With SIMD on, the W_in / W_out / W_res
    matmul inner loops get lowered to `v128` instructions (`f32x4.mul` etc).
  - rustc (wasm32-wasip1) links the `rc_predict.o` object with a small
    Rust harness (`main.rs` for verification, `bench.rs` for measurement)
    that embeds test inputs / host reference outputs, calls `rc_predict`,
    and writes a report via WASI stdout.
  - The Rust harness pulls in wasi-libc for `tanhf` / `memcpy` / ..., so
    no extra C toolchain (clang / wasi-sdk) is needed.

Runner uses `wasmtime` to execute the resulting `.wasm` module. wasmtime
enables the SIMD proposal by default; pass `simd=False` to compile a
scalar baseline for comparison.
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

    def __init__(self, dtype: str = "f32", *,
                 simd: bool = True,
                 rustc: str = "rustc",
                 rust_target: str = "wasm32-wasip1",
                 rust_edition: str = "2024",
                 opt_level: str = "2"):
        if dtype != "f32":
            raise ValueError(
                f"WasmTarget currently only supports dtype='f32', got {dtype!r}"
            )
        self.dtype = dtype
        self.simd = simd
        self.rustc = rustc
        self.rust_target = rust_target
        self.rust_edition = rust_edition
        self.opt_level = opt_level
        self.name = (f"wasm32/{rust_target}+simd128" if simd
                     else f"wasm32/{rust_target}")

    # ------------------------------------------------------------------
    # internal helpers

    def _features(self) -> str:
        # `+simd128` enables WebAssembly SIMD instructions in the wasm32
        # backend. With LLVM loop vectorization on, the f32 matmul inner
        # loops get auto-vectorized to v128 ops (f32x4.mul / fma / load).
        return "+simd128" if self.simd else ""

    def _check_toolchain(self) -> None:
        if shutil.which(self.rustc) is None:
            raise RuntimeError(
                f"{self.rustc} not found on PATH -- install rustup and "
                f"`rustup target add {self.rust_target}`"
            )

    def _build_object(self, rc, exe, out: pathlib.Path) -> pathlib.Path:
        cc_obj = cross_compile_rc(
            rc, exe,
            triple=self.triple,
            dtype=self.dtype,
            features=self._features(),
            vectorize=self.simd,
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))
        # Persist the human-readable assembly so users can eyeball the
        # generated v128 instructions.
        cc_obj.emit_assembly(str(out / "rc_predict.s"))
        return rc_o

    def _check_inputs(self, rc, test_inputs: np.ndarray,
                       expected_outputs: Optional[np.ndarray], host_jit):
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
        return x_in.astype(np.float32), y_ref, T, K, M

    def _link_rustc(self, main_path: pathlib.Path, rc_o: pathlib.Path,
                     wasm: pathlib.Path) -> None:
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

    # ------------------------------------------------------------------
    # public API

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                **_) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Wasm deployment needs `test_inputs` to embed in main.rs"
            )
        self._check_toolchain()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rc_o = self._build_object(rc, exe, out)
        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        x_in, y_ref, T, K, M = self._check_inputs(
            rc, test_inputs, expected_outputs, host_jit,
        )
        x_flat = x_in.ravel()
        y_flat = y_ref.ravel()

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

        wasm = out / "rc.wasm"
        self._link_rustc(main_path, rc_o, wasm)

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=wasm,
            sources=[main_path, hdr],
            objects=[rc_o],
            metadata={
                "triple": self.triple,
                "rust_target": self.rust_target,
                "dtype": self.dtype,
                "simd": self.simd,
                "features": self._features(),
                "T": T, "K": K, "M": M,
                "wasm_size": wasm.stat().st_size,
            },
        )

    def compile_bench(self, rc, exe, *,
                      output_dir,
                      test_inputs: np.ndarray,
                      expected_outputs: Optional[np.ndarray] = None,
                      repeats: int = 25,
                      warmup: int = 3,
                      **_) -> CompiledArtifact:
        """Compile a benchmark harness that times rc_predict internally.

        Runs `repeats` measured inferences (after `warmup` discarded ones)
        using the WASI `std::time::Instant` clock and prints best/median/
        mean wall time per inference plus a parity report against the
        embedded host reference. The output uses keyed `RCLITE_BENCH:`
        lines so a Python driver can parse the numbers back without
        depending on float-formatting round-trips.
        """
        self._check_toolchain()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        rc_o = self._build_object(rc, exe, out)
        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        x_in, y_ref, T, K, M = self._check_inputs(
            rc, test_inputs, expected_outputs, host_jit,
        )
        x_flat = x_in.ravel()
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
        self._link_rustc(main_path, rc_o, wasm)

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
                "simd": self.simd,
                "features": self._features(),
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
        # wasmtime enables the SIMD proposal by default; we pass `-W simd=y`
        # explicitly so a misconfigured wasmtime invocation fails loudly
        # instead of silently rejecting our v128 instructions.
        cmd = [wasmtime]
        if artifact.metadata.get("simd"):
            cmd.extend(["-W", "simd=y"])
        cmd.append(str(artifact.binary))
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
        )
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(success=success, output=output, returncode=cp.returncode)
