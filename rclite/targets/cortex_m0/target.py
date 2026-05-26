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
from ..target import Target, CompiledArtifact, RunResult
from .boards import CortexM0Board, MicrobitV1


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

    def __init__(self, board: CortexM0Board, dtype: str = "f32",
                 cc: str = "arm-none-eabi-gcc"):
        self.board = board
        self.dtype = dtype
        self.cc = cc
        self.name = f"cortex-m0/{board.name}"

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                **_) -> CompiledArtifact:
        if test_inputs is None:
            raise ValueError(
                "Cortex-M0 deployment needs `test_inputs` to embed in main.c"
            )
        if shutil.which(self.cc) is None:
            raise RuntimeError(
                f"{self.cc} not found on PATH — install gcc-arm-none-eabi"
            )

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # 1. Cross-compile rc_predict.o for Cortex-M0.
        cc_obj = cross_compile_rc(
            rc, exe, triple=self.triple, cpu=self.cpu, dtype=self.dtype,
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

        # 4. Render main.c from template.
        tmpl = (_SUPPORT_DIR / "main_template.c").read_text()
        x_flat = np.ascontiguousarray(test_inputs, dtype=np.float32).ravel()
        y_flat = np.ascontiguousarray(expected_outputs, dtype=np.float32).ravel()
        main_path = out / "main.c"
        main_path.write_text(
            tmpl
            .replace("@@T_LEN@@", str(len(x_flat)))
            .replace("@@X_VALUES@@", ", ".join(f"{v:.9g}f" for v in x_flat))
            .replace("@@Y_VALUES@@", ", ".join(f"{v:.9g}f" for v in y_flat))
        )

        # 5. Stage startup + linker script next to the sources.
        startup_path = out / "startup.c"
        linker_path = out / self.board.linker_script
        shutil.copy(_SUPPORT_DIR / "startup.c", startup_path)
        shutil.copy(_SUPPORT_DIR / self.board.linker_script, linker_path)

        # 6. Assemble + link.
        cflags = [
            f"-mcpu={self.cpu}", "-mthumb", "-O2", "-g",
            "-ffunction-sections", "-fdata-sections", "-Wall",
            f"-I{out}",
        ]
        for src in (startup_path, main_path):
            obj = src.with_suffix(".o")
            cp = subprocess.run([self.cc, "-c", *cflags, str(src), "-o", str(obj)],
                                capture_output=True, text=True)
            if cp.returncode != 0:
                raise RuntimeError(f"compile failed for {src.name}: {cp.stderr}")

        elf = out / "rc.elf"
        link_cmd = [
            self.cc, f"-mcpu={self.cpu}", "-mthumb",
            "-T", str(linker_path),
            "-nostartfiles",
            "-Wl,--gc-sections",
            f"-Wl,-Map={out / 'rc.map'}",
            "--specs=nosys.specs",
            *_AEABI_ALIASES,
            str(out / "startup.o"),
            str(out / "main.o"),
            str(rc_o),
            "-o", str(elf),
            "-lm", "-lgcc", "-lc", "-lnosys",
        ]
        cp = subprocess.run(link_cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"link failed: {cp.stderr}")

        metadata = {
            "board": self.board, "triple": self.triple,
            "cpu": self.cpu, "dtype": self.dtype,
        }
        try:
            sz = subprocess.run(
                [self.cc.replace("gcc", "size"), str(elf)],
                capture_output=True, text=True, check=True,
            )
            metadata["size"] = sz.stdout.strip()
        except Exception:
            pass

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=elf,
            sources=[main_path, hdr, startup_path, linker_path],
            objects=[rc_o, out / "startup.o", out / "main.o"],
            metadata=metadata,
        )

    def run(self, artifact: CompiledArtifact, *,
            qemu: str = "qemu-system-arm",
            timeout: float = 60.0,
            **_) -> RunResult:
        if shutil.which(qemu) is None:
            raise RuntimeError(f"{qemu} not found on PATH")
        board = artifact.metadata.get("board")
        if board is None:
            raise RuntimeError("artifact missing board metadata; nothing to run")
        cp = subprocess.run(
            [qemu, "-M", board.qemu_machine, "-nographic", "-semihosting",
             "-kernel", str(artifact.binary)],
            capture_output=True, text=True, timeout=timeout,
        )
        # ARM semihosting writes the program's stdout to QEMU's stderr.
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(success=success, output=output, returncode=cp.returncode)
