"""WebAssembly (wasm32-wasip1) cross-compilation target.

Lowers the IDL to a wasm32 module using:
  - llvmlite cross-compile (wasm32-wasip1, f32 by default). With `simd=True`
    (the default) the backend gets `features="+simd128"` plus LLVM loop/SLP
    vectorization, so the f32 W_in / W_res / W_out matmul inner loops are
    lowered to packed `v128` ops (`f32x4.mul` / `f32x4.add` / `v128.load`).
    Pass `simd=False` for a scalar baseline (ablation benchmarks).
  - rustc (wasm32-wasip1) to link the `rc_predict.o` object with a small
    Rust harness `main.rs` that embeds the test inputs / host reference
    outputs, calls `rc_predict`, and writes a comparison report via WASI
    stdout
  - The Rust harness pulls in wasi-libc for `tanhf` and friends, so no
    extra toolchain (clang/wasi-sdk) is required.

Runner uses `wasmtime` to execute the resulting `.wasm` module; wasmtime
enables the SIMD proposal by default, so the vectorized module runs
unchanged.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from rclite.ir import sparse_passes
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

    def _features(self) -> str:
        # `+simd128` enables the WebAssembly SIMD proposal in the wasm32
        # backend. Combined with LLVM loop/SLP vectorization (vectorize=True),
        # the f32 W_in / W_res / W_out matmul inner loops get auto-vectorized
        # to `v128` instructions. wasmtime enables the SIMD proposal by
        # default, so the resulting module runs unchanged.
        return "+simd128" if self.simd else ""

    def _check_toolchain(self) -> None:
        if shutil.which(self.rustc) is None:
            raise RuntimeError(
                f"{self.rustc} not found on PATH -- install rustup and "
                f"`rustup target add {self.rust_target}`"
            )

    def _link_rustc(self, main_path: pathlib.Path, rc_o: pathlib.Path,
                    wasm: pathlib.Path) -> None:
        """rustc compiles the harness against rust-std (bundling wasi-libc
        for `tanhf`/`memcpy`/...), then wasm-ld pulls `rc_predict.o` in via
        `-C link-arg`."""
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

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                sparse=False,
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
            features=self._features(), vectorize=self.simd,
            passes=sparse_passes(sparse, include_structural=True),
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))
        # Persist the human-readable assembly so users can eyeball the
        # generated v128 instructions (`grep f32x4 rc_predict.s`).
        cc_obj.emit_assembly(str(out / "rc_predict.s"))

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
            "simd": self.simd,
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

    def _compile_quantized_object(self, qmodel, out: pathlib.Path,
                                  sparse=False) -> pathlib.Path:
        """Cross-compile the integer (symmetric fixed-point) kernel to a
        wasm32 object.

        Vectorization stays OFF: the quantized path uses saturating integer
        arithmetic (which is not associative), so reordering the matmul
        reductions would break the bit-exact guarantee against the host
        kernel. The integer kernel needs no libm (tanh is a LUT).
        """
        import llvmlite.binding as llvm
        from rclite.codegen.llvm import (
            emit_quantized_module, _ensure_all_targets,
        )
        ll_mod = emit_quantized_module(
            qmodel, passes=sparse_passes(sparse, include_structural=False))
        ll_mod.triple = self.triple
        _ensure_all_targets()
        mod = llvm.parse_assembly(str(ll_mod))
        mod.verify()
        target = llvm.Target.from_triple(self.triple)
        tm = target.create_target_machine(
            opt=int(self.opt_level), reloc="static",
        )
        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = int(self.opt_level)
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

    def compile_quantized(self, qmodel, *,
                          output_dir,
                          test_inputs: np.ndarray,
                          sparse=False,
                          **_) -> CompiledArtifact:
        """Cross-compile a *symmetric-quantized* (i8 / i16 / i32) model to a
        wasm32 module. The integer kernel takes storage_t inputs (at
        input_scale) and returns storage_t outputs (at state_scale) using
        pure integer arithmetic -- no libm, no soft-float. The Rust harness
        embeds the host kernel's reference and verifies an exact
        (max |diff| == 0) match via WASI stdout.

        Affine-quantized models (``AffineQuantizedModel``) are not yet
        supported here -- use a symmetric ``quantize_model(...)``.
        """
        if test_inputs is None:
            raise ValueError(
                "Wasm quantized deployment needs `test_inputs` to embed"
            )
        if not hasattr(qmodel, "target"):
            raise NotImplementedError(
                "WasmTarget.compile_quantized supports symmetric quantized "
                "models (quantize_model). Affine models are not yet supported."
            )
        self._check_toolchain()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sw = qmodel.target.storage_bits
        storage_rs = {8: "i8", 16: "i16", 32: "i32"}.get(sw)
        np_storage = {8: np.int8, 16: np.int16, 32: np.int32}.get(sw)
        if storage_rs is None:
            raise NotImplementedError(
                f"compile_quantized: storage_bits={sw} not supported "
                f"(expected 8 / 16 / 32)"
            )

        rc_o = self._compile_quantized_object(qmodel, out, sparse=sparse)

        # The kernel takes RAW input quantized at input_scale and applies the
        # `(u - offset) * scaling` preprocessing internally (PreprocessInput
        # op), so the embedded X matches exactly what the host kernel is fed.
        #
        # The reference is the *host* quantized kernel's output (same design
        # as the f32 path, which embeds `host_jit.predict`): cross-compiled
        # wasm vs host JIT is a cross-platform integer-determinism check that
        # is bit-exact by construction. (We deliberately do NOT use the
        # Python QuantizedExecutor here -- it can diverge from the compiled
        # kernel for narrow i8/i16 storage.)
        rc = qmodel.rc
        cfg = qmodel.config
        x_in = np.ascontiguousarray(test_inputs)
        if x_in.ndim == 1:
            x_in = x_in[:, None]
        T = x_in.shape[0]
        K = x_in.shape[1]
        M = qmodel.M
        if K != rc.input.units:
            raise ValueError(
                f"test_inputs has K={K} columns, but the model expects "
                f"K={rc.input.units}"
            )

        X_q = np.ascontiguousarray(
            qmodel.target.quantize_input_array(x_in, cfg).astype(np_storage)
        )
        if X_q.ndim == 1:
            X_q = X_q[:, None]

        from rclite.codegen.llvm import CompiledQuantizedRC
        host_kernel = CompiledQuantizedRC(qmodel)
        # predict() decodes to float = Y_q / state_scale; recover the exact
        # integer (values are < 2**31, representable in f64).
        y_float = host_kernel.predict(x_in)
        if y_float.ndim == 1:
            y_float = y_float[:, None]
        Y_ref_q = np.rint(y_float * cfg.state_scale).astype(np_storage)

        tmpl = (_SUPPORT_DIR / "main_template_q.rs").read_text()
        main_path = out / "main_q.rs"
        main_path.write_text(
            tmpl
            .replace("@@T@@", str(T))
            .replace("@@K@@", str(K))
            .replace("@@M@@", str(M))
            .replace("@@STORAGE_T@@", storage_rs)
            .replace("@@STATE_FRAC@@", str(cfg.state_frac))
            .replace("@@X_VALUES_Q@@",
                     ", ".join(str(int(v)) for v in X_q.ravel()))
            .replace("@@Y_VALUES_Q@@",
                     ", ".join(str(int(v)) for v in Y_ref_q.ravel()))
        )

        wasm = out / "rc_q.wasm"
        self._link_rustc(main_path, rc_o, wasm)

        return CompiledArtifact(
            target_name=self.name + f"/quantized-i{sw}",
            output_dir=out,
            binary=wasm,
            sources=[main_path],
            objects=[rc_o],
            metadata={
                "triple": self.triple,
                "rust_target": self.rust_target,
                "dtype": f"i{sw}",
                "quantized": True,
                "state_frac": cfg.state_frac,
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
            features=self._features(), vectorize=self.simd,
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))
        # Persist the human-readable assembly so users can eyeball the
        # generated v128 instructions (`grep f32x4 rc_predict.s`).
        cc_obj.emit_assembly(str(out / "rc_predict.s"))

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
                "simd": self.simd,
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
