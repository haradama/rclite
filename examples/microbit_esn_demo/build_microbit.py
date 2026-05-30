"""Deploy a trained ESN to BBC micro:bit v1 (Cortex-M0) via the Target API.

This script is intentionally short: the Target abstraction owns the
cross-compile + link + qemu pipeline. To target a different Cortex-M0 board
write a new `CortexM0Board` and reuse `CortexM0Target` unchanged.
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.targets import Microbit

from examples.forecasting.mackey_glass_esn import mackey_glass


BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "microbit"


def train_esn():
    series = mackey_glass(n=2500)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=float(X[:n_train].mean()),
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=80, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=42, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=200, include_bias=True, include_input=True,
                             name="out"),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    return rc, exe, X[n_train:n_train + 10]


def main() -> None:
    rc, exe, sample_X = train_esn()

    target = Microbit()
    print(f"target = {target.name}  (board={target.board.soc}, "
          f"flash={target.board.flash_kb}K, ram={target.board.ram_kb}K, "
          f"dtype={target.dtype})")

    artifact = target.compile(rc, exe, output_dir=BUILD, test_inputs=sample_X)
    print(f"compiled -> {artifact.binary.relative_to(BUILD.parent.parent)}")
    if "size" in artifact.metadata:
        print(artifact.metadata["size"])

    print(f"\n$ qemu-system-arm -M {target.board.qemu_machine} -nographic "
          f"-semihosting -kernel {artifact.binary.name}\n")
    result = target.run(artifact)
    print(result.output, end="")
    if not result.success:
        sys.exit(f"\nFAIL (returncode={result.returncode})")
    print("\n[ok] EMULATOR_EXIT observed — QEMU exited via semihosting")


if __name__ == "__main__":
    main()
