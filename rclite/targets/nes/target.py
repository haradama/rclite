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

from rclite.codegen.llvm import emit_quantized_affine_module
from rclite.ir import sparse_passes

from ..target import (
    Target,
    CompiledArtifact,
    RunResult,
    affine_reference_outputs,
)
from ..arduino.emit_c import emit_affine_kernel_c
from ...codegen.templating import render_template


_SUPPORT_DIR = pathlib.Path(__file__).parent / "support"


class NesTarget(Target):
    """Nintendo Entertainment System (6502 / NROM) affine-quantized target."""

    name = "nes/6502"
    mapper = "nrom"

    # Optimization profiles -> llvm-mos flags. -O3 currently trips over some
    # auto-vectorized forms on this path, so "speed" pins -O2 and disables
    # vectorization for a stable result.
    _OPT_PROFILES = {
        "size": ["-Oz"],
        "speed": ["-O2", "-fno-vectorize", "-fno-slp-vectorize"],
    }
    # kernel_backend -> (rc_predict T-argument type, metadata kernel kind)
    _BACKENDS = {
        "c": ("int32_t", "portable_c"),
        "llvm": ("int64_t", "llvm_ir"),
    }

    def __init__(
        self,
        cc: str = "mos-nes-nrom-clang",
        ld: Optional[str] = None,
    ):
        self.cc = cc
        # Keep backward compatibility: if no linker is specified, use the
        # compiler driver for final link as before.
        self.ld = ld or cc

    @staticmethod
    def _require(tool: str):
        if shutil.which(tool) is None:
            raise RuntimeError(
                f"{tool} not found on PATH — install the llvm-mos SDK "
                "(https://github.com/llvm-mos/llvm-mos-sdk) and put its bin/ "
                "on PATH"
            )

    def _profile_flags(self, profile: str) -> list[str]:
        try:
            return self._OPT_PROFILES[profile]
        except KeyError:
            raise ValueError(
                f"llvm_opt_profile must be 'size' or 'speed', got {profile!r}"
            )

    @staticmethod
    def _run(cmd: list[str], what: str):
        """Run a toolchain command, raising with full output on failure."""
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"llvm-mos {what} failed:\n  {' '.join(cmd)}\n"
                f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
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
        kernel_backend: str = "c",
        llvm_opt_profile: str = "size",
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

        try:
            predict_t, kernel_kind = self._BACKENDS[kernel_backend]
        except KeyError:
            raise ValueError(
                f"kernel_backend must be 'c' or 'llvm', got {kernel_backend!r}"
            )
        # Validate the profile up front (fail before emitting anything).
        self._profile_flags(llvm_opt_profile)

        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        if kernel_backend == "llvm":
            ll_mod = emit_quantized_affine_module(
                qmodel, passes=sparse_passes(sparse, include_structural=False)
            )
            ll_mod.triple = "mos-nes-nrom"
            kernel_path = out / "rc_kernel.ll"
            kernel_path.write_text(str(ll_mod))
        else:
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
        X_q, Y_ref_q, _ = affine_reference_outputs(
            qmodel, test_inputs, np_storage
        )
        T = X_q.shape[0]
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
                PREDICT_T=predict_t,
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
            "kernel_backend": kernel_kind,
            "llvm_opt_profile": llvm_opt_profile,
        }

        if build is None:
            build = shutil.which(self.cc) is not None
        binary = None
        objects = []
        if build:
            binary, objects = self._build_rom(
                out,
                kernel_path,
                main_path,
                kernel_backend=kernel_backend,
                llvm_opt_profile=llvm_opt_profile,
            )
            if binary.exists():
                metadata["rom_bytes"] = binary.stat().st_size

        return CompiledArtifact(
            target_name=self.name + "/affine",
            output_dir=out,
            binary=binary,
            sources=[main_path, kernel_path],
            objects=objects,
            metadata=metadata,
        )

    # ------------------------------------------------------------------

    def _build_rom(
        self,
        out: pathlib.Path,
        kernel_path: pathlib.Path,
        main_path: pathlib.Path,
        *,
        kernel_backend: str,
        llvm_opt_profile: str,
    ) -> tuple[pathlib.Path, list[pathlib.Path]]:
        """Compile + link the kernel and harness into a `.nes` with llvm-mos."""
        self._require(self.cc)
        if self.ld != self.cc:
            self._require(self.ld)
        rom = out / "rc.nes"
        objects = []
        profile_flags = self._profile_flags(llvm_opt_profile)

        if kernel_backend == "llvm":
            # First-class LLVM path with an explicit optimization stage:
            #   rc_kernel.ll -> rc_kernel.opt.ll -> rc_kernel.o
            # This mirrors the intent used on Cortex-M0 where the optimized
            # object is materialized before final link.
            kernel_opt_ll = out / "rc_kernel.opt.ll"
            kernel_o = out / "rc_kernel.o"
            self._run(
                [
                    self.cc,
                    "-S",
                    "-emit-llvm",
                    *profile_flags,
                    "-flto",
                    "-x",
                    "ir",
                    str(kernel_path),
                    "-o",
                    str(kernel_opt_ll),
                ],
                "optimize",
            )
            self._run(
                [
                    self.cc,
                    "-c",
                    *profile_flags,
                    "-flto",
                    str(kernel_opt_ll),
                    "-o",
                    str(kernel_o),
                ],
                "compile",
            )
            objects.append(kernel_o)
            link_inputs = [str(main_path), str(kernel_o)]
        elif kernel_backend == "c":
            link_inputs = [str(main_path), str(kernel_path)]
        else:
            raise ValueError(f"unknown kernel_backend {kernel_backend!r}")

        self._run(
            [self.ld, *profile_flags, "-flto", *link_inputs, "-o", str(rom)],
            "link",
        )
        return rom, objects

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
