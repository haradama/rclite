"""Cortex-M0(+) (ARMv6-M / Thumb) cross-compilation target.

Lowers the IDL to an ARMv6-M ELF using:
  - llvmlite cross-compile (thumbv6m-none-eabi, board-specified mcpu, f32)
  - arm-none-eabi-gcc for assembly + linking
  - newlib-nano libc + libgcc (soft-float)
  - ARM semihosting for stdout / EXIT (BKPT #0xAB)
  - The board's linker script for memory layout

The board declares its `-mcpu=` value (`cortex-m0` for nRF51/micro:bit v1,
`cortex-m0plus` for RP2040/Pico) so the same target class handles both.

`run()` selects an emulator in this priority order:
  1. qemu-system-arm     — if `board.qemu_machine` is set
  2. wokwi-cli           — if `board.wokwi_part` is set and `WOKWI_CLI_TOKEN`
                           is in the environment (free token from
                           https://wokwi.com/dashboard/ci)
  3. on-device SWD       — error with picoprobe/openocd instructions
"""
from __future__ import annotations
import os
import json
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from rclite.codegen.llvm import emit_quantized_module
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
    """Cortex-M0(+) cross-compile target."""

    triple = "thumbv6m-none-eabi"

    def __init__(self, board: CortexM0Board, dtype: str = "f32",
                 cc: str = "arm-none-eabi-gcc"):
        self.board = board
        self.cpu = board.cpu
        self.dtype = dtype
        self.cc = cc
        self.name = f"{board.cpu}/{board.name}"

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

        # 4. Render main.c from the board's template.
        tmpl = (_SUPPORT_DIR / self.board.main_template).read_text()
        x_flat = np.ascontiguousarray(test_inputs, dtype=np.float32).ravel()
        y_flat = np.ascontiguousarray(expected_outputs, dtype=np.float32).ravel()
        main_path = out / "main.c"
        main_path.write_text(
            tmpl
            .replace("@@T_LEN@@", str(len(x_flat)))
            .replace("@@X_VALUES@@", ", ".join(f"{v:.9g}f" for v in x_flat))
            .replace("@@Y_VALUES@@", ", ".join(f"{v:.9g}f" for v in y_flat))
        )

        # 5. Stage startup + linker script + per-board boot stubs.
        startup_path = out / "startup.c"
        linker_path = out / self.board.linker_script
        shutil.copy(_SUPPORT_DIR / "startup.c", startup_path)
        shutil.copy(_SUPPORT_DIR / self.board.linker_script, linker_path)
        extra_asm_objs = self._compile_extra_asm(out)

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
            *[str(o) for o in extra_asm_objs],
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

        sources = [main_path, hdr, startup_path, linker_path]
        sources.extend(self._stage_wokwi_files(out, elf))

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=elf,
            sources=sources,
            objects=[rc_o, out / "startup.o", out / "main.o"],
            metadata=metadata,
        )

    def compile_quantized(self, qmodel, *,
                            output_dir,
                            test_inputs: np.ndarray,
                            **_) -> CompiledArtifact:
        """Cross-compile a quantized model. The kernel takes i32 inputs
        already at input_scale (preprocessed). main.c embeds the i32-encoded
        input/reference arrays and uses pure integer arithmetic — no libm
        tanhf, no soft-float."""
        import llvmlite.binding as llvm
        if shutil.which(self.cc) is None:
            raise RuntimeError(
                f"{self.cc} not found on PATH — install gcc-arm-none-eabi"
            )

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = qmodel.config

        # Cross-compile the i32 kernel
        ll_mod = emit_quantized_module(qmodel)
        ll_mod.triple = self.triple
        from rclite.codegen.llvm import _ensure_all_targets
        _ensure_all_targets()
        mod = llvm.parse_assembly(str(ll_mod))
        mod.verify()
        target = llvm.Target.from_triple(self.triple)
        tm = target.create_target_machine(cpu=self.cpu, opt=2, reloc="static")
        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = 2
        pto.loop_vectorization = False
        pto.slp_vectorization = False
        pb = llvm.create_pass_builder(tm, pto)
        pb.getModulePassManager().run(mod, pb)
        rc_o = out / "rc_predict.o"
        with open(rc_o, "wb") as f:
            f.write(tm.emit_object(mod))
        with open(out / "rc_predict.s", "w") as f:
            f.write(tm.emit_assembly(mod))

        # Quantize test inputs (matches what CompiledQuantizedRC.predict does)
        rc = qmodel.rc
        u_pre = (test_inputs - rc.input.input_offset) * rc.input.input_scaling
        X_q = qmodel.target.quantize_input_array(u_pre, cfg).astype(np.int32)

        # Reference outputs (bit-exact via Python QuantizedExecutor)
        from rclite.quant.executor import QuantizedExecutor
        qexe = QuantizedExecutor(qmodel)
        Y_ref_q = np.zeros((test_inputs.shape[0], qmodel.M), dtype=np.int32)
        for t in range(test_inputs.shape[0]):
            qexe.step_q(X_q[t] if X_q.ndim > 1 else np.array([X_q[t]], dtype=np.int32))
            # phi-style readout uses raw input passthrough scaling — match the
            # kernel's BuildPhi by feeding the (preprocessed-quantized) X_q here.
            from rclite.quant._intops import trunc_i32
            phi_input = X_q[t] if X_q.ndim > 1 else np.array([X_q[t]], dtype=np.int32)
            Y_ref_q[t] = qexe.predict_one_q(phi_input, qexe.state_q)

        # Render main.c from the board's quantized template
        tmpl_path = _SUPPORT_DIR / self.board.main_template_q
        tmpl = tmpl_path.read_text()
        T = len(X_q)
        x_lit = ", ".join(str(int(v)) for v in X_q.ravel())
        y_lit = ", ".join(str(int(v)) for v in Y_ref_q.ravel())
        main_c = (tmpl
                  .replace("@@T_LEN@@", str(T))
                  .replace("@@STATE_FRAC@@", str(cfg.state_frac))
                  .replace("@@X_VALUES_Q@@", x_lit)
                  .replace("@@Y_VALUES_Q@@", y_lit))
        main_path = out / "main.c"
        main_path.write_text(main_c)

        # Stage startup + linker + per-board boot stubs
        startup_path = out / "startup.c"
        linker_path = out / self.board.linker_script
        shutil.copy(_SUPPORT_DIR / "startup.c", startup_path)
        shutil.copy(_SUPPORT_DIR / self.board.linker_script, linker_path)
        extra_asm_objs = self._compile_extra_asm(out)

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
            *[str(o) for o in extra_asm_objs],
            "-o", str(elf),
            "-lgcc", "-lc", "-lnosys",  # no -lm: integer path has no FP
        ]
        cp = subprocess.run(link_cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"link failed: {cp.stderr}")

        metadata = {
            "board": self.board, "triple": self.triple,
            "cpu": self.cpu, "dtype": "i32",
            "state_frac": cfg.state_frac,
            "quantized": True,
        }
        try:
            sz = subprocess.run(
                [self.cc.replace("gcc", "size"), str(elf)],
                capture_output=True, text=True, check=True,
            )
            metadata["size"] = sz.stdout.strip()
        except Exception:
            pass

        sources = [main_path, startup_path, linker_path]
        sources.extend(self._stage_wokwi_files(out, elf))

        return CompiledArtifact(
            target_name=self.name + "/quantized",
            output_dir=out,
            binary=elf,
            sources=sources,
            objects=[rc_o, out / "startup.o", out / "main.o"],
            metadata=metadata,
        )

    def _compile_extra_asm(self, out: pathlib.Path) -> list[pathlib.Path]:
        """Stage and assemble any board-specific .S sources (e.g. RP2040
        boot2 + flash entry stub). Returns the list of produced .o paths."""
        objs: list[pathlib.Path] = []
        for asm_name in self.board.extra_asm:
            src = out / asm_name
            shutil.copy(_SUPPORT_DIR / asm_name, src)
            obj = src.with_suffix(".o")
            cp = subprocess.run(
                [self.cc, "-c", f"-mcpu={self.cpu}", "-mthumb",
                 str(src), "-o", str(obj)],
                capture_output=True, text=True,
            )
            if cp.returncode != 0:
                raise RuntimeError(
                    f"assemble failed for {asm_name}: {cp.stderr}"
                )
            objs.append(obj)
        return objs

    def _stage_wokwi_files(self, out: pathlib.Path,
                            elf: pathlib.Path) -> list[pathlib.Path]:
        """Drop a diagram.json + wokwi.toml into `out` if the board has a
        Wokwi part. Caller-pulled by run() when QEMU isn't available.

        `$serialMonitor` is a virtual built-in that wires straight to the
        wokwi-cli serial output (captured by --serial-log-file / matched
        by --expect-text). We connect Pico GP0=TX to it so UART writes
        are surfaced; GP1=RX is wired for completeness."""
        if not self.board.wokwi_part:
            return []
        diagram = {
            "version": 1,
            "author": "rclite",
            "editor": "wokwi",
            "parts": [{
                "type": self.board.wokwi_part,
                "id": "mcu",
                "top": 0, "left": 0, "attrs": {},
            }],
            "connections": [
                ["$serialMonitor:RX", "mcu:GP0", "", []],
                ["$serialMonitor:TX", "mcu:GP1", "", []],
            ],
            "dependencies": {},
        }
        diagram_path = out / "diagram.json"
        diagram_path.write_text(json.dumps(diagram, indent=2))
        toml_path = out / "wokwi.toml"
        # `firmware` is the binary loaded into the simulator (Wokwi accepts
        # ELF directly for ARM Cortex-M); `elf` is for symbol info / GDB.
        toml_path.write_text(
            f'[wokwi]\nversion = 1\nfirmware = "{elf.name}"\nelf = "{elf.name}"\n'
        )
        return [diagram_path, toml_path]

    def run(self, artifact: CompiledArtifact, *,
            qemu: str = "qemu-system-arm",
            wokwi: str = "wokwi-cli",
            timeout: float = 60.0,
            **_) -> RunResult:
        board = artifact.metadata.get("board")
        if board is None:
            raise RuntimeError("artifact missing board metadata; nothing to run")
        if board.qemu_machine:
            return self._run_qemu(artifact, board, qemu, timeout)
        if board.wokwi_part and os.environ.get("WOKWI_CLI_TOKEN"):
            return self._run_wokwi(artifact, board, wokwi, timeout)
        if board.wokwi_part and shutil.which(wokwi):
            raise RuntimeError(
                f"{board.name} can be simulated by wokwi-cli, but "
                f"WOKWI_CLI_TOKEN is not set. Get a free token at "
                f"https://wokwi.com/dashboard/ci and export it before "
                f"calling run()."
            )
        raise RuntimeError(
            f"{board.name} ({board.soc}) has no QEMU machine model and no "
            f"usable Wokwi setup — flash to hardware via SWD (e.g. picoprobe + "
            f"openocd: `openocd -f interface/cmsis-dap.cfg -f target/rp2040.cfg`, "
            f"then `gdb-multiarch {artifact.binary} -ex 'target remote :3333' "
            f"-ex 'load' -ex 'monitor arm semihosting enable' -ex 'continue'`)"
        )

    def _run_qemu(self, artifact: CompiledArtifact, board: CortexM0Board,
                  qemu: str, timeout: float) -> RunResult:
        if shutil.which(qemu) is None:
            raise RuntimeError(f"{qemu} not found on PATH")
        cp = subprocess.run(
            [qemu, "-M", board.qemu_machine, "-nographic", "-semihosting",
             "-kernel", str(artifact.binary)],
            capture_output=True, text=True, timeout=timeout,
        )
        # ARM semihosting writes the program's stdout to QEMU's stderr.
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(success=success, output=output, returncode=cp.returncode)

    def _run_wokwi(self, artifact: CompiledArtifact, board: CortexM0Board,
                   wokwi: str, timeout: float) -> RunResult:
        if shutil.which(wokwi) is None:
            raise RuntimeError(
                f"{wokwi} not found on PATH (install from "
                f"https://github.com/wokwi/wokwi-cli/releases)"
            )
        if not (artifact.output_dir / "diagram.json").exists():
            raise RuntimeError(
                "Wokwi diagram.json missing from artifact output_dir — "
                "rebuild with this version of CortexM0Target"
            )
        cp = subprocess.run(
            [wokwi, str(artifact.output_dir),
             "--timeout", str(int(timeout * 1000)),
             "--expect-text", "EMULATOR_EXIT"],
            capture_output=True, text=True, timeout=timeout + 10,
        )
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (cp.returncode == 0) and ("EMULATOR_EXIT" in output)
        return RunResult(success=success, output=output, returncode=cp.returncode)
