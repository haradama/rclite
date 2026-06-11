"""Cortex-M0 (ARMv6-M / Thumb) cross-compilation target.

Lowers the IDL to a Cortex-M0 ELF using:
  - llvmlite cross-compile (thumbv6m-none-eabi, cortex-m0, f32)
  - arm-none-eabi-gcc for assembly + linking
  - newlib-nano libc + libgcc (soft-float)
  - ARM semihosting for stdout / EXIT (BKPT #0xAB)
  - A board's linker script for memory layout

Runner uses qemu-system-arm with the board's QEMU machine name.
"""

from __future__ import annotations
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from rclite.codegen.llvm import (
    emit_quantized_module,
    emit_quantized_affine_module,
)
from rclite.codegen.templating import render_template
from rclite.ir import sparse_passes
from rclite.targets.arduino import emit_affine_kernel_c
from ..target import (
    Target,
    CompiledArtifact,
    RunResult,
    affine_reference_outputs,
    symmetric_reference_outputs,
)
from .._arm_gcc import (
    cross_object,
    read_size,
    require_tool,
    run_tool,
    storage_types,
)
from .boards import CortexM0Board


_SUPPORT_DIR = pathlib.Path(__file__).parent / "support"

# LLVM emits the libgcc compiler-rt names; arm-none-eabi-gcc's v6-m libgcc
# provides only the AAPCS-named variants. Bridge them at link time.
_AEABI_ALIASES = [
    "-Wl,--defsym=__addsf3=__aeabi_fadd",
    "-Wl,--defsym=__subsf3=__aeabi_fsub",
    "-Wl,--defsym=__mulsf3=__aeabi_fmul",
    "-Wl,--defsym=__divsf3=__aeabi_fdiv",
    "-Wl,--defsym=__adddf3=__aeabi_dadd",
    "-Wl,--defsym=__subdf3=__aeabi_dsub",
    "-Wl,--defsym=__muldf3=__aeabi_dmul",
    "-Wl,--defsym=__divdf3=__aeabi_ddiv",
]


class CortexM0Target(Target):
    """Cortex-M0 cross-compile target."""

    triple = "thumbv6m-none-eabi"
    cpu = "cortex-m0"

    def __init__(
        self,
        board: CortexM0Board,
        dtype: str = "f32",
        cc: str = "arm-none-eabi-gcc",
    ):
        self.board = board
        self.dtype = dtype
        self.cc = cc
        self.name = f"cortex-m0/{board.name}"

    # -- shared build mechanics --------------------------------------------

    def _cflags(self, out) -> list:
        return [
            f"-mcpu={self.cpu}",
            "-mthumb",
            "-O2",
            "-g",
            "-ffunction-sections",
            "-fdata-sections",
            "-Wall",
            f"-I{out}",
        ]

    def _stage_support(self, out):
        """Copy startup.c + the board linker script next to the build."""
        startup_path = out / "startup.c"
        linker_path = out / self.board.linker_script
        shutil.copy(_SUPPORT_DIR / "startup.c", startup_path)
        shutil.copy(_SUPPORT_DIR / self.board.linker_script, linker_path)
        return startup_path, linker_path

    def _compile_sources(self, out, srcs) -> None:
        """Compile each C source to a sibling .o with the standard cflags."""
        cflags = self._cflags(out)
        for src in srcs:
            run_tool(
                [
                    self.cc,
                    "-c",
                    *cflags,
                    str(src),
                    "-o",
                    str(src.with_suffix(".o")),
                ],
                error=f"compile failed for {src.name}",
            )

    def _link_elf(
        self, out, rc_o, linker_path, *, with_float, extra_objects=()
    ):
        """Link startup.o + main.o + rc_predict.o into rc.elf; return its path."""
        elf = out / "rc.elf"
        link_cmd = [
            self.cc,
            f"-mcpu={self.cpu}",
            "-mthumb",
            "-T",
            str(linker_path),
            "-nostartfiles",
            "-Wl,--gc-sections",
            f"-Wl,-Map={out / 'rc.map'}",
            "--specs=nosys.specs",
            *_AEABI_ALIASES,
            str(out / "startup.o"),
            str(out / "main.o"),
            str(rc_o),
            *[str(o) for o in extra_objects],
            "-o",
            str(elf),
        ]
        link_cmd += (
            ["-lm", "-lgcc", "-lc", "-lnosys"]
            if with_float
            else ["-lgcc", "-lc", "-lnosys"]
        )
        run_tool(link_cmd, error="link failed")
        return elf

    def compile(
        self,
        rc,
        exe,
        *,
        output_dir,
        test_inputs: Optional[np.ndarray] = None,
        expected_outputs: Optional[np.ndarray] = None,
        sparse=False,
        **_,
    ) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Cortex-M0 deployment needs `test_inputs` to embed in main.c"
            )
        require_tool(self.cc)

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Cross-compile rc_predict.o for Cortex-M0.
        cc_obj = cross_compile_rc(
            rc,
            exe,
            triple=self.triple,
            cpu=self.cpu,
            dtype=self.dtype,
            passes=sparse_passes(sparse, include_structural=True),
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))
        cc_obj.emit_assembly(str(out / "rc_predict.s"))

        # 2. C header (use the host JIT to render it; same metadata).
        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        # 3. f32 host reference for embedded "expected" comparison.
        if expected_outputs is None:
            expected_outputs = host_jit.predict(test_inputs).astype(np.float32)

        # 4. Render main.c from template. @@T_LEN@@ is the step count T (not
        # T*K / T*M); X and Y are embedded row-major as (T, K) and (T, M),
        # the dims coming from rc_predict.h's RC_INPUT_DIM / RC_OUTPUT_DIM.
        T = test_inputs.shape[0]
        x_flat = np.ascontiguousarray(test_inputs, dtype=np.float32).ravel()
        y_flat = np.ascontiguousarray(
            expected_outputs, dtype=np.float32
        ).ravel()
        main_path = out / "main.c"
        main_path.write_text(
            render_template(
                _SUPPORT_DIR / "main_template.c",
                T_LEN=str(T),
                X_VALUES=", ".join(f"{v:.9g}f" for v in x_flat),
                Y_VALUES=", ".join(f"{v:.9g}f" for v in y_flat),
            )
        )

        # 5. Stage startup + linker, then assemble + link.
        startup_path, linker_path = self._stage_support(out)
        self._compile_sources(out, (startup_path, main_path))
        elf = self._link_elf(out, rc_o, linker_path, with_float=True)

        metadata = {
            "board": self.board,
            "triple": self.triple,
            "cpu": self.cpu,
            "dtype": self.dtype,
        }
        sz = read_size(self.cc, elf)
        if sz is not None:
            metadata["size"] = sz

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=elf,
            sources=[main_path, hdr, startup_path, linker_path],
            objects=[rc_o, out / "startup.o", out / "main.o"],
            metadata=metadata,
        )

    def compile_quantized(
        self, qmodel, *, output_dir, test_inputs: np.ndarray, sparse=False, **_
    ) -> CompiledArtifact:
        """Cross-compile a quantized model. The kernel takes storage_t inputs
        already at input_scale (preprocessed). main.c embeds the
        storage_t-encoded input/reference arrays and uses pure integer
        arithmetic — no libm tanhf, no soft-float. The storage width is
        picked from `qmodel.target.storage_bits` (32 / 16 / 8)."""
        require_tool(self.cc)

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = qmodel.config
        sw = qmodel.target.storage_bits
        storage_t, np_storage = storage_types(sw, context="compile_quantized")

        # Cross-compile the quantized kernel to rc_predict.o.
        rc_o = cross_object(
            emit_quantized_module(
                qmodel, passes=sparse_passes(sparse, include_structural=False)
            ),
            triple=self.triple,
            cpu=self.cpu,
            out=out,
        )

        # Quantize test inputs + bit-exact references (Python QuantizedExecutor
        # reproduces the kernel's integer arithmetic exactly).
        X_q, Y_ref_q, _ = symmetric_reference_outputs(
            qmodel, test_inputs, np_storage
        )

        # Render main.c from template
        T = len(X_q)
        x_lit = ", ".join(str(int(v)) for v in X_q.ravel())
        y_lit = ", ".join(str(int(v)) for v in Y_ref_q.ravel())
        main_c = render_template(
            _SUPPORT_DIR / "main_template_q.c",
            T_LEN=str(T),
            RC_K=str(qmodel.K),
            RC_M=str(qmodel.M),
            STATE_FRAC=str(cfg.state_frac),
            STORAGE_T=storage_t,
            X_VALUES_Q=x_lit,
            Y_VALUES_Q=y_lit,
        )
        main_path = out / "main.c"
        main_path.write_text(main_c)

        # Stage startup + linker, then assemble + link.
        startup_path, linker_path = self._stage_support(out)
        self._compile_sources(out, (startup_path, main_path))
        elf = self._link_elf(out, rc_o, linker_path, with_float=False)

        metadata = {
            "board": self.board,
            "triple": self.triple,
            "cpu": self.cpu,
            "dtype": f"i{sw}",
            "state_frac": cfg.state_frac,
            "quantized": True,
        }
        sz = read_size(self.cc, elf)
        if sz is not None:
            metadata["size"] = sz

        return CompiledArtifact(
            target_name=self.name + "/quantized",
            output_dir=out,
            binary=elf,
            sources=[main_path, startup_path, linker_path],
            objects=[rc_o, out / "startup.o", out / "main.o"],
            metadata=metadata,
        )

    def compile_affine_quantized(
        self,
        qmodel,
        *,
        output_dir,
        test_inputs: np.ndarray,
        sparse=False,
        kernel_backend: str = "llvm",
        **_,
    ) -> CompiledArtifact:
        """Cross-compile an `AffineQuantizedModel` to a Cortex-M0 ELF.

        Storage width (i8 / i16) and the LUT strategy (DIRECT /
        LINEAR_INTERP / POLYNOMIAL) flow through `emit_quantized_affine_module`
        from the model. The test driver embeds the input samples and the
        bit-exact reference outputs computed by `AffineQuantizedExecutor`,
        all as quantized integers, so the on-device verification stays in
        pure integer arithmetic.
        """
        require_tool(self.cc)

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sw = qmodel.storage_bits
        storage_t, np_storage = storage_types(
            sw, allowed=(8, 16), context="compile_affine_quantized"
        )

        kernel_sources = []
        extra_link_objects = []

        if kernel_backend == "llvm":
            # First-class LLVM path: persist IR as a build artifact and
            # cross-compile to rc_predict.o for the target triple.
            ll_mod = emit_quantized_affine_module(
                qmodel, passes=sparse_passes(sparse, include_structural=False)
            )
            ll_mod.triple = self.triple
            (out / "rc_kernel.ll").write_text(str(ll_mod))
            kernel_sources.append(out / "rc_kernel.ll")
            rc_o = cross_object(
                ll_mod, triple=self.triple, cpu=self.cpu, out=out
            )
            kernel_kind = "llvm_ir"
        elif kernel_backend == "c":
            # Portable-C kernel path for apples-to-apples product builds.
            # main_template_q_affine expects rc_predict(int64_t,...), so wrap
            # the C emitter's int32_t-T entrypoint with a tiny adapter.
            kernel_c_path = out / "rc_kernel.c"
            kernel_c_path.write_text(
                emit_affine_kernel_c(qmodel, sparse=sparse)
            )
            kernel_sources.append(kernel_c_path)

            rc_i32_o = out / "rc_predict_i32.o"
            run_tool(
                [
                    self.cc,
                    "-c",
                    *self._cflags(out),
                    "-Drc_predict=rc_predict_i32",
                    str(kernel_c_path),
                    "-o",
                    str(rc_i32_o),
                ],
                error=f"compile failed for {kernel_c_path.name}",
            )

            wrapper_path = out / "rc_wrapper.c"
            wrapper_path.write_text(
                "#include <stdint.h>\n"
                f"typedef {storage_t} rc_storage_t;\n"
                "extern void rc_predict_i32(int32_t T, const rc_storage_t *X, rc_storage_t *Y);\n"
                "void rc_predict(int64_t T, rc_storage_t *X, rc_storage_t *Y) {\n"
                "    rc_predict_i32((int32_t)T, (const rc_storage_t *)X, Y);\n"
                "}\n"
            )
            wrapper_o = out / "rc_wrapper.o"
            run_tool(
                [
                    self.cc,
                    "-c",
                    *self._cflags(out),
                    str(wrapper_path),
                    "-o",
                    str(wrapper_o),
                ],
                error=f"compile failed for {wrapper_path.name}",
            )

            rc_o = rc_i32_o
            extra_link_objects.append(str(wrapper_o))
            kernel_sources.extend([wrapper_path])
            kernel_kind = "portable_c"
        else:
            raise ValueError(
                f"unknown kernel_backend {kernel_backend!r}; expected 'llvm' or 'c'"
            )

        # Quantize input + bit-exact references via the affine executor
        # (bit-exact with the JIT; mirrors CompiledAffineRC.predict).
        X_q, Y_ref_q, _ = affine_reference_outputs(
            qmodel, test_inputs, np_storage
        )
        T = X_q.shape[0]

        # Render the affine main.c
        x_lit = ", ".join(str(int(v)) for v in X_q.ravel())
        y_lit = ", ".join(str(int(v)) for v in Y_ref_q.ravel())
        main_c = render_template(
            _SUPPORT_DIR / "main_template_q_affine.c",
            T_LEN=str(T),
            RC_K=str(qmodel.K),
            RC_M=str(qmodel.M),
            STORAGE_T=storage_t,
            LUT_KIND=qmodel.lut_strategy.kind.value,
            X_VALUES_Q=x_lit,
            Y_VALUES_Q=y_lit,
        )
        main_path = out / "main.c"
        main_path.write_text(main_c)

        # Stage startup + linker, then assemble + link.
        startup_path, linker_path = self._stage_support(out)
        self._compile_sources(out, (startup_path, main_path))
        elf = self._link_elf(
            out,
            rc_o,
            linker_path,
            with_float=False,
            extra_objects=extra_link_objects,
        )

        metadata = {
            "board": self.board,
            "triple": self.triple,
            "cpu": self.cpu,
            "dtype": f"i{sw}",
            "quantized": True,
            "affine": True,
            "lut_kind": qmodel.lut_strategy.kind.value,
            "kernel_backend": kernel_kind,
        }
        sz = read_size(self.cc, elf)
        if sz is not None:
            metadata["size"] = sz

        objects = [rc_o, out / "startup.o", out / "main.o"]
        for obj in extra_link_objects:
            objects.append(pathlib.Path(obj))

        return CompiledArtifact(
            target_name=self.name + "/affine",
            output_dir=out,
            binary=elf,
            sources=[main_path, startup_path, linker_path, *kernel_sources],
            objects=objects,
            metadata=metadata,
        )

    def run(
        self,
        artifact: CompiledArtifact,
        *,
        qemu: str = "qemu-system-arm",
        timeout: float = 60.0,
        **_,
    ) -> RunResult:
        if shutil.which(qemu) is None:
            raise RuntimeError(f"{qemu} not found on PATH")
        board = artifact.metadata.get("board")
        if board is None:
            raise RuntimeError(
                "artifact missing board metadata; nothing to run"
            )
        cp = subprocess.run(
            [
                qemu,
                "-M",
                board.qemu_machine,
                "-nographic",
                "-semihosting",
                "-kernel",
                str(artifact.binary),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        # ARM semihosting writes the program's stdout to QEMU's stderr.
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(
            success=success, output=output, returncode=cp.returncode
        )
