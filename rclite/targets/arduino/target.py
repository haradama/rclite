"""Arduino Uno (ATmega328P / 8-bit AVR) deployment target.

Emits a self-contained Arduino sketch directory:

    <output_dir>/
      sketch/
        sketch.ino     — Serial harness: runs the model, reports parity + timing
        rc_kernel.c    — portable affine kernel (weights in Flash via PROGMEM)

and (when `arduino-cli` is available) compiles it for `arduino:avr:uno`,
returning the Flash / SRAM usage.

Two paths:

  * `compile_affine_quantized` — affine inference kernel (the practical fit
    for steady-state inference on a 2 KB-SRAM board).
  * `compile_symmetric_online` — on-device online learning (integer LMS / NLMS)
    via the symmetric kernel: the readout adapts in SRAM as samples stream in
    (`rc_train_step`). The i64 shifts/divide are software-emulated on AVR, so
    this trades cycles for the "train where you deploy" capability.

Use a structured topology (SCR / DLR / DLRB) or `sparse=` so the dense W_res
matrix is never materialised — that is what keeps the model inside the Uno's
memory.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
from typing import Optional

import numpy as np

from ..target import Target, CompiledArtifact, RunResult
from .emit_c import emit_affine_kernel_c


_TEMPLATE_DIR = pathlib.Path(__file__).parent / "templates"
_FQBN = "arduino:avr:uno"


class ArduinoUnoTarget(Target):
    """Arduino Uno (ATmega328P) affine-quantized deployment target."""

    name = "arduino/uno"
    fqbn = _FQBN

    def __init__(self, arduino_cli: str = "arduino-cli"):
        self.arduino_cli = arduino_cli

    def compile(self, rc, exe, **_):
        raise NotImplementedError(
            "ArduinoUnoTarget only supports the affine quantized path; "
            "call compile_affine_quantized(qmodel, ...)"
        )

    def run(self, artifact, **_):
        raise NotImplementedError(
            "On-device run requires a physical Uno or an AVR simulator; "
            "flash the generated sketch with arduino-cli upload."
        )

    # ------------------------------------------------------------------

    def compile_affine_quantized(self, qmodel, *,
                                  output_dir,
                                  test_inputs: np.ndarray,
                                  build: Optional[bool] = None,
                                  ) -> CompiledArtifact:
        """Emit the sketch + kernel; compile for Uno if arduino-cli is present.

        `build` forces (True) or skips (False) the arduino-cli compile.
        Default: build iff arduino-cli is on PATH.
        """
        if qmodel.storage_bits not in (8, 16):
            raise NotImplementedError(
                f"Arduino target supports i8/i16 storage, got {qmodel.storage_bits}"
            )
        out = pathlib.Path(output_dir)
        sketch_dir = out / "sketch"
        sketch_dir.mkdir(parents=True, exist_ok=True)

        storage_t = "int8_t" if qmodel.storage_bits == 8 else "int16_t"
        np_storage = np.int8 if qmodel.storage_bits == 8 else np.int16

        # Kernel source
        kernel_c = emit_affine_kernel_c(qmodel)
        (sketch_dir / "rc_kernel.c").write_text(kernel_c)

        # Quantize test inputs + bit-exact reference outputs via the Python
        # executor (same path the host JIT / C kernel reproduce exactly).
        from rclite.core.profile import Aggregation
        from rclite.quant.affine.executor import AffineQuantizedExecutor
        cfg = qmodel.config
        X = test_inputs
        if X.ndim == 1:
            X = X[:, None]
        X_q = cfg.input.quantize_array(X).astype(np_storage)
        qexe = AffineQuantizedExecutor(qmodel)
        T = X.shape[0]
        pooled = qmodel.rc.readout.aggregation != Aggregation.NONE
        if pooled:
            # Sequence-to-label: the kernel pools the whole window and emits a
            # single readout row.
            n_rows = 1
            Y_ref_q = qexe.predict_pooled_q(X)[None, :].astype(np_storage)
        else:
            n_rows = T
            Y_ref_q = np.zeros((T, qmodel.M), dtype=np_storage)
            for t in range(T):
                x_raw_q = qexe._quantize_raw_input(X[t])
                u_pre_q = qexe._quantize_u_pre(X[t])
                qexe.step_q(u_pre_q)
                Y_ref_q[t] = qexe.predict_one_q(
                    x_raw_q, qexe.state_q).astype(np_storage)

        # Render sketch.ino
        tmpl = (_TEMPLATE_DIR / "sketch.ino").read_text()
        x_lit = ", ".join(str(int(v)) for v in X_q.ravel())
        y_lit = ", ".join(str(int(v)) for v in Y_ref_q.ravel())
        ino = (tmpl
               .replace("@@T@@", str(T))
               .replace("@@NROWS@@", str(n_rows))
               .replace("@@RC_K@@", str(qmodel.K))
               .replace("@@RC_M@@", str(qmodel.M))
               .replace("@@STORAGE_T@@", storage_t)
               .replace("@@X_VALUES@@", x_lit)
               .replace("@@Y_VALUES@@", y_lit))
        sketch_ino = sketch_dir / "sketch.ino"
        sketch_ino.write_text(ino)

        metadata = {
            "fqbn": self.fqbn,
            "dtype": f"i{qmodel.storage_bits}",
            "w_out_dtype": f"i{qmodel.w_out_storage_bits}",
            "topology": qmodel.rc.reservoir.topology.name,
            "lut_kind": qmodel.lut_strategy.kind.value,
            "affine": True,
        }

        if build is None:
            build = shutil.which(self.arduino_cli) is not None
        binary = None
        if build:
            binary, sizes = self._arduino_build(sketch_dir, out)
            metadata.update(sizes)

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=binary,
            sources=[sketch_ino, sketch_dir / "rc_kernel.c"],
            objects=[],
            metadata=metadata,
        )

    # ------------------------------------------------------------------

    def compile_symmetric_online(self, qmodel, *,
                                  output_dir,
                                  X: np.ndarray,
                                  Y: np.ndarray,
                                  learning_rate: float = 1e-2,
                                  normalized: bool = False,
                                  delta: float = 1.0,
                                  warmup: int = 0,
                                  sparse=None,
                                  build: Optional[bool] = None,
                                  ) -> CompiledArtifact:
        """Emit + (optionally) build an **on-device online-learning** sketch.

        Unlike `compile_affine_quantized` (inference only), this wires the
        symmetric online kernel (`rc_train_step` / `rc_infer_step` /
        `rc_export_W_out`). The reservoir and the *initial* readout are baked
        into rc_kernel.c; the sketch streams (X, Y) through the kernel and
        verifies, on-device, that the per-step predictions and the final
        learned readout are bit-exact with `IntegerLMSLearner`.

        `X` / `Y` are float input / target sequences (T, K) / (T, M). The first
        `warmup` steps run inference-only (no update). `normalized=True` selects
        NLMS. Use a structured topology (SCR/DLR/DLRB) or `sparse=` so the dense
        W_res stays out of the Uno's Flash/SRAM.
        """
        from rclite.export.c_kernel_symmetric import (
            emit_symmetric_online_kernel_c,
        )
        from rclite.quant.online import collect_training_stream

        storage_bits = qmodel.target.storage_bits
        if storage_bits not in (16, 32):
            raise NotImplementedError(
                "symmetric online kernel supports i16/i32 storage "
                f"(I16/I32FixedPoint); got i{storage_bits}"
            )
        if qmodel.rc.readout.aggregation.name != "NONE":
            raise NotImplementedError(
                "online learning targets a per-step regression readout; "
                "sequence pooling is not an online trainer"
            )
        out = pathlib.Path(output_dir)
        sketch_dir = out / "sketch"
        sketch_dir.mkdir(parents=True, exist_ok=True)

        storage_t = "int16_t" if storage_bits == 16 else "int32_t"
        pgm_rd = "word" if storage_bits == 16 else "dword"

        # Emit the kernel from the INITIAL weights, *before* the reference loop
        # mutates rc_W_out (collect_training_stream learns in place).
        kernel_c = emit_symmetric_online_kernel_c(
            qmodel, learning_rate, normalized=normalized, delta=delta,
            sparse=sparse)
        (sketch_dir / "rc_kernel.c").write_text(kernel_c)

        X = np.asarray(X, dtype=np.float64)
        Y = np.asarray(Y, dtype=np.float64)
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        stream = collect_training_stream(
            qmodel, X, Y, learning_rate=learning_rate,
            normalized=normalized, delta=delta, warmup=warmup)

        T = int(X.shape[0])
        F = qmodel.F

        def _lit(arr):
            return ", ".join(str(int(v)) for v in np.asarray(arr).ravel())

        tmpl = (_TEMPLATE_DIR / "sketch_online.ino").read_text()
        ino = (tmpl
               .replace("@@T@@", str(T))
               .replace("@@RC_K@@", str(qmodel.K))
               .replace("@@RC_M@@", str(qmodel.M))
               .replace("@@RC_F@@", str(F))
               .replace("@@STORAGE_T@@", storage_t)
               .replace("@@PGM_RD@@", pgm_rd)
               .replace("@@U_VALUES@@", _lit(stream["u_q"]))
               .replace("@@YT_VALUES@@", _lit(stream["y_target_q"]))
               .replace("@@WARM_VALUES@@", _lit(stream["warm"]))
               .replace("@@YP_REF@@", _lit(stream["y_pred_q"]))
               .replace("@@WOUT_REF@@", _lit(stream["final_W_out"])))
        sketch_ino = sketch_dir / "sketch.ino"
        sketch_ino.write_text(ino)

        metadata = {
            "fqbn": self.fqbn,
            "dtype": f"i{storage_bits}",
            "topology": qmodel.rc.reservoir.topology.name,
            "online": True,
            "trainer": "NLMS" if normalized else "LMS",
            "learning_rate": float(learning_rate),
            "steps": T,
            "warmup": int(warmup),
        }

        if build is None:
            build = shutil.which(self.arduino_cli) is not None
        binary = None
        if build:
            binary, sizes = self._arduino_build(sketch_dir, out)
            metadata.update(sizes)

        return CompiledArtifact(
            target_name=self.name,
            output_dir=out,
            binary=binary,
            sources=[sketch_ino, sketch_dir / "rc_kernel.c"],
            objects=[],
            metadata=metadata,
        )

    # ------------------------------------------------------------------

    def _arduino_build(self, sketch_dir: pathlib.Path,
                        out: pathlib.Path) -> tuple:
        if shutil.which(self.arduino_cli) is None:
            raise RuntimeError(
                f"{self.arduino_cli} not found on PATH — install arduino-cli "
                f"and run `arduino-cli core install arduino:avr`"
            )
        build_dir = out / "build"
        cmd = [
            self.arduino_cli, "compile",
            "--fqbn", self.fqbn,
            "--output-dir", str(build_dir),
            str(sketch_dir),
        ]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise RuntimeError(
                f"arduino-cli compile failed:\n  {' '.join(cmd)}\n"
                f"stdout:\n{cp.stdout}\nstderr:\n{cp.stderr}"
            )
        sizes = self._parse_sizes(cp.stdout)
        hexes = list(build_dir.glob("*.hex"))
        binary = hexes[0] if hexes else None
        return binary, sizes

    @staticmethod
    def _parse_sizes(stdout: str) -> dict:
        """Pull Flash / SRAM usage out of arduino-cli's compile summary."""
        sizes: dict = {}
        for line in stdout.splitlines():
            low = line.lower()
            # "Sketch uses 4096 bytes (12%) of program storage space."
            if "program storage" in low:
                for tok in line.replace("(", " ").replace(")", " ").split():
                    if tok.isdigit():
                        sizes["flash_bytes"] = int(tok)
                        break
            # "Global variables use 308 bytes (15%) of dynamic memory"
            if "dynamic memory" in low or "global variables" in low:
                for tok in line.replace("(", " ").replace(")", " ").split():
                    if tok.isdigit():
                        sizes["sram_bytes"] = int(tok)
                        break
        return sizes
