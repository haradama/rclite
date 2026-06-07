"""End-to-end QAT deploy demo: train float → QAT search → cross-compile
i32 kernel → run on QEMU micro:bit.

  1. Float training on Mackey-Glass.
  2. QAT search over state_frac with W_out refitting (mirage style).
  3. Build the integer kernel via emit_quantized_module (LUT-based tanh).
  4. Cross-compile to Cortex-M0 + ARM Thumb, link with newlib-nano
     (no libm, no soft-FP — pure i32 arithmetic on device).
  5. Run on qemu-system-arm -M microbit; output via semihosting.
"""

from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    search_quantization,
    TanhLUTSpec,
)
from rclite.targets import Microbit

from examples.forecasting.mackey_glass_esn import mackey_glass


BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "microbit_q"


def train_esn():
    series = mackey_glass(n=2500)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    rc = ReservoirComputer(
        # input_offset=0 / input_scaling=1 keeps the integer kernel preprocess-free.
        input=InputNode(
            units=1,
            input_offset=0.0,
            input_scaling=1.0,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=60,
            activation=Activation.TANH,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    return (
        rc,
        exe,
        X[n_train : n_train + 10],
        Y[n_train : n_train + 10],
        X[:n_train],
        Y[:n_train],
    )


def main() -> None:
    rc, exe, sample_X, sample_Y, train_X, train_Y = train_esn()
    print(
        f"[1/3] train float ESN (N={rc.reservoir.units}, "
        f"topology={rc.reservoir.topology.name})"
    )

    Y_f32 = exe.predict(sample_X)
    print(
        f"      float baseline MSE on sample: "
        f"{float(np.mean((Y_f32 - sample_Y) ** 2)):.6e}"
    )

    print(f"[2/3] QAT search over state_frac")
    result = search_quantization(
        rc,
        exe,
        train_X,
        train_Y,
        sample_X,
        sample_Y,
        state_frac_range=(12, 22),
        lut=TanhLUTSpec(xmin=-4, xmax=4, n=256),
    )
    print(f"      best={result.best_config}")
    print(f"      best MSE={result.best_mse:.6e}")
    qmodel = result.best_qmodel

    print(f"[3/3] cross-compile i32 kernel + qemu run")
    target = Microbit()
    artifact = target.compile_quantized(
        qmodel,
        output_dir=BUILD,
        test_inputs=sample_X,
    )
    print(f"      binary: {artifact.binary.relative_to(BUILD.parent.parent)}")
    if "size" in artifact.metadata:
        print(artifact.metadata["size"])

    print(
        f"\n$ qemu-system-arm -M {target.board.qemu_machine} "
        f"-nographic -semihosting -kernel rc.elf\n"
    )
    rresult = target.run(artifact)
    print(rresult.output, end="")
    if not rresult.success:
        sys.exit(f"\nFAIL (returncode={rresult.returncode})")
    print("\n[ok] EMULATOR_EXIT observed — quantized kernel ran on QEMU")


if __name__ == "__main__":
    main()
