"""Deploy a quantized reservoir to an Arduino Uno (ATmega328P, 8-bit AVR).

  1. Train a float ESN on Mackey-Glass.
  2. QAT-quantize affine, mixed precision (i8 reservoir + i16 readout) — the
     accuracy sweet spot for i8-class storage.
  3. Emit a portable C kernel (weights in Flash via PROGMEM) + an Arduino
     sketch, and compile it for `arduino:avr:uno` via arduino-cli.
  4. Report Flash / SRAM usage.

A structured topology (SCR) is used so the dense W_res never materialises —
that is what keeps the model inside the Uno's 2 KB SRAM. Flash the produced
sketch with `arduino-cli upload` (or open it in the Arduino IDE); it runs the
model on an embedded test sequence and prints parity + timing over Serial.
"""
from __future__ import annotations
import pathlib
import shutil
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    search_quantization_affine, quantize_model_affine,
    AffineQuantizedExecutor, LUTStrategy,
)
from rclite.targets import ArduinoUnoTarget

from examples.forecasting.mackey_glass_esn import mackey_glass


BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "arduino_uno"


def main() -> None:
    series = mackey_glass(n=2500)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=60, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=42),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=200, include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    sample = X[n_train:n_train + 10]
    sample_Y = Y[n_train:n_train + 10]
    print(f"[1/3] trained float ESN (N={rc.reservoir.units}, SCR)")

    print("[2/3] QAT affine (i8 reservoir + i16 W_out) + interp-64 LUT")
    res = search_quantization_affine(
        rc, exe, X[:n_train], Y[:n_train], sample, sample_Y,
        storage_bits=8, w_out_storage_bits=16,
        lut_strategy=LUTStrategy.linear_interp(64), n_iterations=1,
    )
    qm = res.best_qmodel   # QAT-refit W_out + interp-64 LUT baked in

    qexe = AffineQuantizedExecutor(qm)
    Y_q = qexe.predict(np.concatenate([X[:n_train], sample], axis=0))[-len(sample):]
    sig = float(np.std(sample_Y))
    nrmse = float(np.sqrt(np.mean((Y_q - sample_Y) ** 2))) / sig * 100
    print(f"      sample NRMSE = {nrmse:.2f}%")

    print("[3/3] emit sketch + compile for arduino:avr:uno")
    target = ArduinoUnoTarget()
    art = target.compile_affine_quantized(
        qm, output_dir=BUILD, test_inputs=sample,
    )
    print(f"      sketch: {art.sources[0].relative_to(BUILD.parent.parent)}")
    md = art.metadata
    print(f"      storage={md['dtype']}  W_out={md['w_out_dtype']}  "
          f"topology={md['topology']}  lut={md['lut_kind']}")
    if "flash_bytes" in md:
        print(f"      Flash: {md['flash_bytes']} bytes / 32768  "
              f"({md['flash_bytes'] / 32768 * 100:.1f}%)")
        print(f"      SRAM : {md['sram_bytes']} bytes / 2048   "
              f"({md['sram_bytes'] / 2048 * 100:.1f}%)")
        print(f"      binary: {art.binary.relative_to(BUILD.parent.parent)}")
        print("\n[ok] fits Arduino Uno — flash with:  arduino-cli upload "
              f"-p <PORT> --fqbn arduino:avr:uno {art.sources[0].parent}")
    elif shutil.which("arduino-cli") is None:
        print("      arduino-cli not on PATH — sketch emitted but not compiled.")


if __name__ == "__main__":
    main()
