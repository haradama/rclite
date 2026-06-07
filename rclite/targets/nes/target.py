"""Nintendo Entertainment System (MOS 6502 / NROM) deployment target.

Lowers the IDL to an iNES cartridge image (`.nes`) via the llvm-mos toolchain:

  - the affine quantized kernel is emitted as portable C (`emit_affine_kernel_c`,
    shared with the Arduino target) — llvm-mos puts the `const` weight / LUT
    tables in PRG-ROM, so only the small state buffers live in the 2 KB of RAM;
  - `mos-nes-nrom-clang` (llvm-mos-sdk) compiles the kernel + harness and links
    the crt0, the reset/NMI/IRQ vectors and the iNES header into a `.nes`.

The 6502 has no FPU and no hardware multiply, so — exactly as on the Arduino
Uno — only the affine quantized path is supported, and a structured topology
(SCR / DLR / DLRB) is strongly recommended so the dense `W_res` matmul is never
materialised. `allow_i32_accum=True` is passed to the C emitter: the kernel
then uses i32 (instead of i64) accumulators wherever the worst case provably
fits, which avoids the 6502's very costly 64-bit libcalls. It stays bit-exact
with the host reference on a modern clang such as llvm-mos.

Verification uses the de-facto NES test protocol (blargg): the harness writes
a result byte + message string to PRG-RAM at $6000 (8 KB of PRG-RAM is mapped
there via `MAPPER_PRG_RAM_KB`), and the runner prints the message and exits
with the result byte (0 == pass). Two headless backends are supported:
`run(emulator="mesen")` uses Mesen2's purpose-built `--testrunner`, and
`run(emulator="fceux")` drives FCEUX with a Lua watcher under `xvfb-run`
(apt-installable, ships Lua). `"auto"` (default) prefers Mesen, else FCEUX.
A `.nes` built this way also runs on real hardware and any cycle-accurate
emulator.
"""

from __future__ import annotations
import os
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from ..target import Target, CompiledArtifact, RunResult
from ..arduino.emit_c import emit_affine_kernel_c
from ...codegen.templating import render_template


_SUPPORT_DIR = pathlib.Path(__file__).parent / "support"


class NesTarget(Target):
    """Nintendo Entertainment System (6502 / NROM) affine-quantized target."""

    name = "nes/6502"
    mapper = "nrom"

    def __init__(self, cc: str = "mos-nes-nrom-clang"):
        self.cc = cc

    def _require_cc(self):
        if shutil.which(self.cc) is None:
            raise RuntimeError(
                f"{self.cc} not found on PATH — install the llvm-mos SDK "
                "(https://github.com/llvm-mos/llvm-mos-sdk) and put its bin/ "
                "on PATH"
            )

    def compile(self, rc, exe, **_):
        raise NotImplementedError(
            "NesTarget only supports the affine quantized path; "
            "call compile_affine_quantized(qmodel, ...)"
        )

    # ------------------------------------------------------------------

    def compile_affine_quantized(
        self,
        qmodel,
        *,
        output_dir,
        test_inputs: np.ndarray,
        tol: int = 1,
        build: Optional[bool] = None,
        sparse=None,
    ) -> CompiledArtifact:
        """Emit the kernel + harness and link them into a `.nes` cartridge.

        `build` forces (True) or skips (False) the llvm-mos link step.
        Default: build iff `mos-nes-nrom-clang` is on PATH.
        """
        sw = qmodel.storage_bits
        if sw == 8:
            storage_t, np_storage = "int8_t", np.int8
        elif sw == 16:
            storage_t, np_storage = "int16_t", np.int16
        else:
            raise NotImplementedError(
                f"NES target supports i8/i16 storage, got i{sw}"
            )

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        # Portable affine kernel. i32 accumulators where they provably fit:
        # llvm-mos is a modern clang (no avr-gcc 7.x widening-MAC bug), so this
        # is bit-exact and dodges the 6502's expensive 64-bit libcalls.
        kernel_c = emit_affine_kernel_c(
            qmodel, allow_i32_accum=True, sparse=sparse
        )
        kernel_path = out / "rc_kernel.c"
        kernel_path.write_text(kernel_c)

        # Quantize test inputs + bit-exact reference outputs via the affine
        # executor (the same path the C kernel reproduces exactly).
        from rclite.quant.affine.executor import AffineQuantizedExecutor

        cfg = qmodel.config
        X = test_inputs[:, None] if test_inputs.ndim == 1 else test_inputs
        X_q = cfg.input.quantize_array(X).astype(np_storage)
        qexe = AffineQuantizedExecutor(qmodel)
        T = X.shape[0]
        Y_ref_q = np.zeros((T, qmodel.M), dtype=np_storage)
        for t in range(T):
            x_raw_q = qexe._quantize_raw_input(X[t])
            u_pre_q = qexe._quantize_u_pre(X[t])
            qexe.step_q(u_pre_q)
            Y_ref_q[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(
                np_storage
            )

        x_flat = np.ascontiguousarray(X_q).ravel()
        y_flat = np.ascontiguousarray(Y_ref_q).ravel()

        main_path = out / "main.c"
        main_path.write_text(
            render_template(
                _SUPPORT_DIR / "main_template_q_affine.c",
                T_STEPS=str(T),
                X_LEN=str(len(x_flat)),
                Y_LEN=str(len(y_flat)),
                STORAGE_T=storage_t,
                LUT_KIND=qmodel.lut_strategy.kind.value,
                TOL=str(int(tol)),
                X_VALUES_Q=", ".join(str(int(v)) for v in x_flat),
                Y_VALUES_Q=", ".join(str(int(v)) for v in y_flat),
            )
        )

        metadata = {
            "cpu": "6502",
            "mapper": self.mapper,
            "dtype": f"i{sw}",
            "w_out_dtype": f"i{qmodel.w_out_storage_bits}",
            "topology": qmodel.rc.reservoir.topology.name,
            "lut_kind": qmodel.lut_strategy.kind.value,
            "quantized": True,
            "affine": True,
            "tol": int(tol),
        }

        if build is None:
            build = shutil.which(self.cc) is not None
        binary = None
        if build:
            binary = self._build_rom(out, kernel_path, main_path)
            if binary.exists():
                metadata["rom_bytes"] = binary.stat().st_size

        return CompiledArtifact(
            target_name=self.name + "/affine",
            output_dir=out,
            binary=binary,
            sources=[main_path, kernel_path],
            objects=[],
            metadata=metadata,
        )

    # ------------------------------------------------------------------

    def _build_rom(
        self,
        out: pathlib.Path,
        kernel_path: pathlib.Path,
        main_path: pathlib.Path,
    ) -> pathlib.Path:
        """Compile + link the kernel and harness into a `.nes` with llvm-mos."""
        self._require_cc()
        rom = out / "rc.nes"
        cmd = [
            self.cc,
            "-Os",
            "-flto",
            str(main_path),
            str(kernel_path),
            "-o",
            str(rom),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"llvm-mos link failed:\n  {' '.join(cmd)}\n"
                f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
            )
        return rom

    # -- runner ------------------------------------------------------------

    def run(
        self,
        artifact: CompiledArtifact,
        *,
        emulator: str = "auto",
        frames: int = 30000,
        timeout: float = 120.0,
        **_,
    ) -> RunResult:
        """Run the cartridge headlessly and report the blargg ($6000) verdict.

        Two backends speak the same de-facto NES test protocol — the harness
        writes a result byte + message string to PRG-RAM at $6000, the runner
        prints the message and exits with the result byte (0 == pass):

          * ``"mesen"``  — Mesen2's purpose-built ``--testrunner`` (cleanest);
          * ``"fceux"``  — FCEUX driven by ``support/fceux_testrunner.lua``,
            run under ``xvfb-run`` (FCEUX is a Qt app). apt-installable and
            ships Lua, so it is the practical default on Linux without Mesen.

        ``emulator="auto"`` (default) picks Mesen if present, else FCEUX.
        ``frames`` bounds how long the runner waits for the test to finish.
        """
        if emulator not in ("auto", "mesen", "fceux"):
            raise ValueError(f"unknown emulator {emulator!r}")

        mesen_bin = (
            shutil.which("Mesen")
            or shutil.which("mesen")
            or shutil.which("Mesen2")
        )
        fceux_bin = shutil.which("fceux") or shutil.which("/usr/games/fceux")

        if emulator == "mesen" or (emulator == "auto" and mesen_bin):
            if mesen_bin is None:
                raise RuntimeError(
                    "Mesen not found on PATH — install Mesen2 "
                    "(https://github.com/SourMesen/Mesen2)"
                )
            return self._run_mesen(mesen_bin, artifact, frames, timeout)

        if emulator == "fceux" or (emulator == "auto" and fceux_bin):
            if fceux_bin is None:
                raise RuntimeError(
                    "fceux not found on PATH — install it (e.g. "
                    "`apt install fceux`); it ships Lua scripting"
                )
            return self._run_fceux(fceux_bin, artifact, frames, timeout)

        raise RuntimeError(
            "no NES emulator found on PATH — install Mesen2 (--testrunner) or "
            "fceux (`apt install fceux`, driven via Lua under xvfb-run)"
        )

    def _run_mesen(self, mesen_bin, artifact, frames, timeout) -> RunResult:
        cmd = [mesen_bin, "--testrunner", str(artifact.binary), str(frames)]
        cp = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout
        )
        output = (cp.stdout or "") + (cp.stderr or "")
        success = (
            cp.returncode == 0
            and "TEST_PASS" in output
            and "TEST_FAIL" not in output
        )
        return RunResult(
            success=success, output=output, returncode=cp.returncode
        )

    def _run_fceux(self, fceux_bin, artifact, frames, timeout) -> RunResult:
        lua = _SUPPORT_DIR / "fceux_testrunner.lua"
        cmd = [
            fceux_bin,
            "--no-config",
            "1",
            "--loadlua",
            str(lua),
            str(artifact.binary),
        ]
        if shutil.which("xvfb-run") is not None:
            cmd = ["xvfb-run", "-a"] + cmd
        env = dict(os.environ, RCLITE_MAX_FRAMES=str(frames))
        try:
            cp = subprocess.run(
                cmd, capture_output=True, text=True, timeout=timeout, env=env
            )
            output = (cp.stdout or "") + (cp.stderr or "")
            returncode = cp.returncode
        except subprocess.TimeoutExpired as e:
            output = (
                (e.stdout or "") + (e.stderr or "")
                if isinstance(e.stdout, str)
                else "TIMEOUT\n"
            )
            returncode = 99
        # The verdict is the message the harness wrote to $6004 (printed by the
        # Lua watcher), not the exit code: FCEUX's Qt teardown can segfault on
        # exit under xvfb after a clean os.exit(), so we key off the message.
        success = (
            "TEST_PASS" in output
            and "TEST_FAIL" not in output
            and "TIMEOUT" not in output
        )
        return RunResult(success=success, output=output, returncode=returncode)
