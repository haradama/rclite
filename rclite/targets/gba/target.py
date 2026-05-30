"""Game Boy Advance (ARM7TDMI / ARMv4T / Thumb) cross-compilation target.

Lowers the IDL to a GBA cartridge image (.gba) using:
  - llvmlite cross-compile (thumbv4t-none-eabi, arm7tdmi)
  - arm-none-eabi-gcc for the ARM crt0 + Thumb sources + linking
  - newlib-nano libc + libgcc (soft-float for the f32 path)
  - mGBA debug-log MMIO for stdout (no semihosting on the GBA)
  - a GBA linker script (ROM @ 0x08000000, EWRAM, IWRAM stack)

ARMv4T has no FPU and no Thumb-2/saturation instructions, so the integer
quantized paths (especially `compile_affine_quantized`) are recommended; the
f32 path works but runs through slow soft-float.

Runner uses mGBA under xvfb. The GBA has no clean program-exit, so the driver
loops forever after printing TEST_PASS / TEST_FAIL and the runner stops the
emulator with a timeout (a timeout is treated as "ran without crashing").
"""
from __future__ import annotations
import os
import pathlib
import shutil
import signal
import subprocess
from typing import Optional

import numpy as np

from rclite.codegen import compile_rc, cross_compile_rc
from rclite.codegen.llvm import emit_quantized_module, emit_quantized_affine_module
from rclite.ir import sparse_passes
from ..target import Target, CompiledArtifact, RunResult


_SUPPORT_DIR = pathlib.Path(__file__).parent / "support"

# LLVM emits the compiler-rt float names; arm-none-eabi-gcc's v4t libgcc may
# only export the AAPCS-named variants. Bridge them at link time (f32 path only).
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

_LINKER_SCRIPT = "gba.ld"


class GbaTarget(Target):
    """Game Boy Advance (ARM7TDMI) cross-compile target."""

    triple = "thumbv4t-none-eabi"
    cpu = "arm7tdmi"

    def __init__(self, dtype: str = "f32", cc: str = "arm-none-eabi-gcc"):
        self.dtype = dtype
        self.cc = cc
        self.objcopy = cc.replace("gcc", "objcopy")
        self.name = "gba/arm7tdmi"

    # -- internal helpers --------------------------------------------------

    def _require_cc(self):
        if shutil.which(self.cc) is None:
            raise RuntimeError(
                f"{self.cc} not found on PATH — install gcc-arm-none-eabi"
            )

    def _cross_object(self, ll_mod, out: pathlib.Path) -> pathlib.Path:
        """Optimize a quantized LLVM module for thumbv4t and emit rc_predict.o."""
        import llvmlite.binding as llvm
        from rclite.codegen.llvm import _ensure_all_targets

        ll_mod.triple = self.triple
        _ensure_all_targets()
        mod = llvm.parse_assembly(str(ll_mod))
        mod.verify()
        target = llvm.Target.from_triple(self.triple)
        tm = target.create_target_machine(cpu=self.cpu, opt=2, reloc="static")
        pto = llvm.create_pipeline_tuning_options()
        pto.speed_level = 2
        pto.loop_vectorization = False   # ARMv4T has no SIMD
        pto.slp_vectorization = False
        pb = llvm.create_pass_builder(tm, pto)
        pb.getModulePassManager().run(mod, pb)
        rc_o = out / "rc_predict.o"
        with open(rc_o, "wb") as f:
            f.write(tm.emit_object(mod))
        with open(out / "rc_predict.s", "w") as f:
            f.write(tm.emit_assembly(mod))
        return rc_o

    def _build_rom(self, out: pathlib.Path, rc_o: pathlib.Path,
                   main_path: pathlib.Path, *, with_float: bool) -> pathlib.Path:
        """Assemble crt0, compile main, link the ELF, then objcopy to .gba.

        Returns the path to the runnable cartridge image (rc.gba).
        """
        crt0_path = out / "crt0.s"
        linker_path = out / _LINKER_SCRIPT
        shutil.copy(_SUPPORT_DIR / "crt0.s", crt0_path)
        shutil.copy(_SUPPORT_DIR / _LINKER_SCRIPT, linker_path)

        common = [f"-mcpu={self.cpu}", "-mthumb-interwork", "-O2", "-g",
                  "-ffunction-sections", "-fdata-sections", "-Wall", f"-I{out}"]

        # crt0 is ARM; main is Thumb.
        crt0_o = out / "crt0.o"
        cp = subprocess.run(
            [self.cc, "-c", f"-mcpu={self.cpu}", "-marm", "-mthumb-interwork",
             str(crt0_path), "-o", str(crt0_o)],
            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"assemble failed for crt0.s: {cp.stderr}")

        main_o = out / "main.o"
        cp = subprocess.run([self.cc, "-c", *common, "-mthumb",
                             str(main_path), "-o", str(main_o)],
                            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"compile failed for main.c: {cp.stderr}")

        elf = out / "rc.elf"
        link_cmd = [
            self.cc, f"-mcpu={self.cpu}", "-mthumb", "-mthumb-interwork",
            "-T", str(linker_path),
            "-nostartfiles",
            "-Wl,--gc-sections",
            f"-Wl,-Map={out / 'rc.map'}",
            "--specs=nosys.specs",
        ]
        if with_float:
            link_cmd += _AEABI_ALIASES
        link_cmd += [str(crt0_o), str(main_o), str(rc_o), "-o", str(elf)]
        link_cmd += ["-lm", "-lgcc", "-lc", "-lnosys"] if with_float \
            else ["-lgcc", "-lc", "-lnosys"]
        cp = subprocess.run(link_cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"link failed: {cp.stderr}")

        rom = out / "rc.gba"
        if shutil.which(self.objcopy) is None:
            raise RuntimeError(
                f"{self.objcopy} not found on PATH — install gcc-arm-none-eabi"
            )
        cp = subprocess.run(
            [self.objcopy, "-O", "binary", str(elf), str(rom)],
            capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(f"objcopy failed: {cp.stderr}")
        return rom

    def _size(self, elf: pathlib.Path) -> Optional[str]:
        try:
            sz = subprocess.run(
                [self.cc.replace("gcc", "size"), str(elf)],
                capture_output=True, text=True, check=True)
            return sz.stdout.strip()
        except Exception:
            return None

    # -- compile entry points ----------------------------------------------

    def compile(self, rc, exe, *,
                output_dir,
                test_inputs: Optional[np.ndarray] = None,
                expected_outputs: Optional[np.ndarray] = None,
                tol: float = 1e-2,
                sparse=False,
                **_) -> CompiledArtifact:
        """Cross-compile the f32 (soft-float) kernel to a GBA cartridge."""
        if test_inputs is None:
            raise ValueError(
                "GBA deployment needs `test_inputs` to embed in main.c"
            )
        self._require_cc()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        cc_obj = cross_compile_rc(
            rc, exe, triple=self.triple, cpu=self.cpu, dtype=self.dtype,
            passes=sparse_passes(sparse, include_structural=True),
        )
        rc_o = out / "rc_predict.o"
        cc_obj.emit_object(str(rc_o))
        cc_obj.emit_assembly(str(out / "rc_predict.s"))

        host_jit = compile_rc(rc, exe)
        hdr = out / "rc_predict.h"
        host_jit.emit_header(str(hdr))

        if expected_outputs is None:
            expected_outputs = host_jit.predict(test_inputs).astype(np.float32)

        tmpl = (_SUPPORT_DIR / "main_template.c").read_text()
        x_flat = np.ascontiguousarray(test_inputs, dtype=np.float32).ravel()
        y_flat = np.ascontiguousarray(expected_outputs, dtype=np.float32).ravel()
        main_path = out / "main.c"
        # @@T_LEN@@ is the step count T; X/Y are embedded row-major (T, K)/(T, M)
        # with the dims coming from rc_predict.h's RC_INPUT_DIM / RC_OUTPUT_DIM.
        main_path.write_text(
            tmpl
            .replace("@@T_LEN@@", str(test_inputs.shape[0]))
            .replace("@@TOLF@@", f"{tol:.9g}")
            .replace("@@X_VALUES@@", ", ".join(f"{v:.9g}f" for v in x_flat))
            .replace("@@Y_VALUES@@", ", ".join(f"{v:.9g}f" for v in y_flat))
        )

        shutil.copy(_SUPPORT_DIR / "mgba_log.h", out / "mgba_log.h")
        rom = self._build_rom(out, rc_o, main_path, with_float=True)

        elf = out / "rc.elf"
        metadata = {
            "triple": self.triple, "cpu": self.cpu, "dtype": self.dtype,
            "elf": elf, "tol": tol,
        }
        sz = self._size(elf)
        if sz is not None:
            metadata["size"] = sz

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=rom,
            sources=[main_path, hdr, out / "crt0.s", out / _LINKER_SCRIPT],
            objects=[rc_o, out / "crt0.o", out / "main.o"],
            metadata=metadata,
        )

    def compile_quantized(self, qmodel, *,
                          output_dir,
                          test_inputs: np.ndarray,
                          tol: int = 1,
                          sparse=False,
                          **_) -> CompiledArtifact:
        """Cross-compile a symmetric (Q-format) quantized model to a .gba."""
        sw = qmodel.target.storage_bits
        # Known limitation: on multi-input (K>1) models the LLVM thumbv4t
        # backend miscompiles the narrow-storage (i8/i16) input-accumulation
        # loop, so the device result diverges from the host reference (the
        # identical kernel IR is bit-exact on Cortex-M0, and i32 symmetric /
        # affine / float are all bit-exact on the GBA). Steer K>1 i8/i16 to a
        # verified-working path rather than emit silently-wrong numbers. This
        # is a model-capability check, so it runs before the toolchain probe.
        if qmodel.K > 1 and sw in (8, 16):
            raise NotImplementedError(
                f"GBA symmetric i{sw} quantization does not support "
                f"multi-input models (K={qmodel.K}) — the thumbv4t backend "
                "miscompiles the i8/i16 input loop. Use the affine path "
                "(compile_affine_quantized) or i32 symmetric (I32FixedPoint), "
                "both bit-exact for multi-input on the GBA."
            )
        self._require_cc()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)
        cfg = qmodel.config
        if sw == 32:
            storage_t, np_storage = "int32_t", np.int32
        elif sw == 16:
            storage_t, np_storage = "int16_t", np.int16
        elif sw == 8:
            storage_t, np_storage = "int8_t", np.int8
        else:
            raise NotImplementedError(
                f"compile_quantized: storage_bits={sw} not supported"
            )

        rc_o = self._cross_object(emit_quantized_module(
            qmodel, passes=sparse_passes(sparse, include_structural=False)), out)

        rc = qmodel.rc
        u_pre = (test_inputs - rc.input.input_offset) * rc.input.input_scaling
        X_q = qmodel.target.quantize_input_array(u_pre, cfg).astype(np_storage)

        from rclite.quant.executor import QuantizedExecutor
        qexe = QuantizedExecutor(qmodel)
        Y_ref_q = np.zeros((test_inputs.shape[0], qmodel.M), dtype=np_storage)
        for t in range(test_inputs.shape[0]):
            x_row = (X_q[t] if X_q.ndim > 1
                     else np.array([X_q[t]], dtype=np_storage))
            qexe.step_q(x_row.astype(np.int32))
            Y_ref_q[t] = qexe.predict_one_q(x_row.astype(np.int32),
                                            qexe.state_q).astype(np_storage)

        tmpl = (_SUPPORT_DIR / "main_template_q.c").read_text()
        T = len(X_q)
        main_path = out / "main.c"
        main_path.write_text(
            tmpl
            .replace("@@T_LEN@@", str(T))
            .replace("@@RC_K@@", str(qmodel.K))
            .replace("@@RC_M@@", str(qmodel.M))
            .replace("@@STATE_FRAC@@", str(cfg.state_frac))
            .replace("@@STORAGE_T@@", storage_t)
            .replace("@@TOL@@", str(int(tol)))
            .replace("@@X_VALUES_Q@@", ", ".join(str(int(v)) for v in X_q.ravel()))
            .replace("@@Y_VALUES_Q@@", ", ".join(str(int(v)) for v in Y_ref_q.ravel()))
        )

        shutil.copy(_SUPPORT_DIR / "mgba_log.h", out / "mgba_log.h")
        rom = self._build_rom(out, rc_o, main_path, with_float=False)

        elf = out / "rc.elf"
        metadata = {
            "triple": self.triple, "cpu": self.cpu, "dtype": f"i{sw}",
            "state_frac": cfg.state_frac, "quantized": True,
            "elf": elf, "tol": int(tol),
        }
        sz = self._size(elf)
        if sz is not None:
            metadata["size"] = sz

        return CompiledArtifact(
            target_name=self.name + "/quantized",
            output_dir=out,
            binary=rom,
            sources=[main_path, out / "crt0.s", out / _LINKER_SCRIPT],
            objects=[rc_o, out / "crt0.o", out / "main.o"],
            metadata=metadata,
        )

    def compile_affine_quantized(self, qmodel, *,
                                 output_dir,
                                 test_inputs: np.ndarray,
                                 tol: int = 1,
                                 sparse=False,
                                 **_) -> CompiledArtifact:
        """Cross-compile an `AffineQuantizedModel` to a GBA cartridge.

        The recommended path for the GBA: pure-integer, no FPU/soft-float.
        """
        self._require_cc()
        out = pathlib.Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        sw = qmodel.storage_bits
        if sw == 8:
            storage_t, np_storage = "int8_t", np.int8
        elif sw == 16:
            storage_t, np_storage = "int16_t", np.int16
        else:
            raise NotImplementedError(
                f"compile_affine_quantized: storage_bits={sw} not supported"
            )

        rc_o = self._cross_object(emit_quantized_affine_module(
            qmodel, passes=sparse_passes(sparse, include_structural=False)), out)

        cfg = qmodel.config
        X_q = cfg.input.quantize_array(test_inputs).astype(np_storage)

        from rclite.quant.affine.executor import AffineQuantizedExecutor
        qexe = AffineQuantizedExecutor(qmodel)
        test_inputs_2d = test_inputs[:, None] if test_inputs.ndim == 1 else test_inputs
        T = test_inputs_2d.shape[0]
        Y_ref_q = np.zeros((T, qmodel.M), dtype=np_storage)
        for t in range(T):
            x_raw = test_inputs_2d[t]
            x_raw_q = qexe._quantize_raw_input(x_raw)
            u_pre_q = qexe._quantize_u_pre(x_raw)
            qexe.step_q(u_pre_q)
            Y_ref_q[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(np_storage)

        tmpl = (_SUPPORT_DIR / "main_template_q_affine.c").read_text()
        main_path = out / "main.c"
        main_path.write_text(
            tmpl
            .replace("@@T_LEN@@", str(T))
            .replace("@@RC_K@@", str(qmodel.K))
            .replace("@@RC_M@@", str(qmodel.M))
            .replace("@@STORAGE_T@@", storage_t)
            .replace("@@LUT_KIND@@", qmodel.lut_strategy.kind.value)
            .replace("@@TOL@@", str(int(tol)))
            .replace("@@X_VALUES_Q@@", ", ".join(str(int(v)) for v in X_q.ravel()))
            .replace("@@Y_VALUES_Q@@", ", ".join(str(int(v)) for v in Y_ref_q.ravel()))
        )

        shutil.copy(_SUPPORT_DIR / "mgba_log.h", out / "mgba_log.h")
        rom = self._build_rom(out, rc_o, main_path, with_float=False)

        elf = out / "rc.elf"
        metadata = {
            "triple": self.triple, "cpu": self.cpu, "dtype": f"i{sw}",
            "quantized": True, "affine": True,
            "lut_kind": qmodel.lut_strategy.kind.value,
            "elf": elf, "tol": int(tol),
        }
        sz = self._size(elf)
        if sz is not None:
            metadata["size"] = sz

        return CompiledArtifact(
            target_name=self.name + "/affine",
            output_dir=out,
            binary=rom,
            sources=[main_path, out / "crt0.s", out / _LINKER_SCRIPT],
            objects=[rc_o, out / "crt0.o", out / "main.o"],
            metadata=metadata,
        )

    # -- runner ------------------------------------------------------------

    def run(self, artifact: CompiledArtifact, *,
            mgba: str = "mgba",
            timeout: float = 5.0,
            log_level: int = 15,
            **_) -> RunResult:
        """Run the cartridge in mGBA and grep its debug log for TEST_PASS.

        The GBA has no clean program exit, so the driver loops forever after
        printing its verdict and we stop the emulator with a timeout. A timeout
        therefore means "ran for `timeout`s without crashing" — the healthy path
        for the GBA — and is treated as success-eligible (returncode 124).

        We run mGBA headlessly via SDL's dummy video/audio drivers (no X server
        needed). The real X path (xvfb) makes mGBA reset the console partway
        through a long compute burst; the dummy driver does not, and combined
        with the VBlank-IRQ crt0 the full rc_predict runs to completion.

        mGBA's debug log is block-buffered, so we wrap it in `stdbuf -oL` to
        line-buffer the output; otherwise a handful of log lines would sit
        unflushed and be lost when we stop it. The whole tree runs in its own
        session so we can signal the process group to stop it reliably.
        """
        mgba_bin = shutil.which(mgba) or shutil.which("/usr/games/mgba")
        if mgba_bin is None:
            raise RuntimeError(f"{mgba} not found on PATH")

        cmd = []
        if shutil.which("stdbuf") is not None:
            cmd += ["stdbuf", "-oL", "-eL"]
        cmd += [mgba_bin, "-l", str(log_level), str(artifact.binary)]

        env = dict(os.environ)
        env.setdefault("SDL_VIDEODRIVER", "dummy")
        env.setdefault("SDL_AUDIODRIVER", "dummy")

        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, start_new_session=True, env=env,
        )
        timed_out = False
        try:
            output, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            timed_out = True
            # SIGTERM the group first (lets mGBA shut down), then make sure.
            self._kill_group(proc, signal.SIGTERM)
            try:
                output, _ = proc.communicate(timeout=5)
            except subprocess.TimeoutExpired:
                self._kill_group(proc, signal.SIGKILL)
                output, _ = proc.communicate()

        output = output or ""
        returncode = 124 if timed_out else proc.returncode
        success = (returncode in (0, 124)
                   and "TEST_PASS" in output
                   and "TEST_FAIL" not in output)
        return RunResult(success=success, output=output, returncode=returncode)

    @staticmethod
    def _kill_group(proc: subprocess.Popen, sig: int) -> None:
        try:
            os.killpg(os.getpgid(proc.pid), sig)
        except ProcessLookupError:
            pass
