"""Symmetric i8 QAT deploy demo + size comparison across {i32, i16, i8}.

  1. Train one float ESN on Mackey-Glass.
  2. For each storage width, QAT-search a sensible state_frac and
     cross-compile a Cortex-M0 ELF.
  3. Run the i8 binary on qemu-system-arm (the deployable artifact).
  4. Print a size + accuracy table comparing all three widths.

The point of this example is to show what footprint reduction the
symmetric i8 path buys. Accuracy will be noticeably worse than i32/i16
because symmetric Q-format saturates W_out coefficients (typical readout
magnitudes exceed [-1, 1] / 2^state_frac at state_frac ≤ 6). The
asymmetric `I8Affine` path — when implemented — fixes that with
per-tensor scales.
"""
from __future__ import annotations
import pathlib
import subprocess
import sys
import shutil

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec,
    I32FixedPoint, I16FixedPoint, I8Symmetric,
    search_quantization,
)
from rclite.targets import Microbit

from examples.forecasting.mackey_glass_esn import mackey_glass


BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "microbit_i8"


# Per-storage QAT search recipe. input_frac / weight_frac are picked to keep
# typical MG values (|x| ~1.4, |w| ~1.0) well inside the storage range.
_RECIPES = [
    # (label, target_factory, state_frac_range, input_frac, weight_frac, lut_n)
    ("i32", I32FixedPoint,   (12, 20),  14, 14, 256),
    ("i16", I16FixedPoint,   (8, 12),    8,  8, 128),
    ("i8",  I8Symmetric,     (3,  6),    4,  4,  32),
]


def train_esn():
    series = mackey_glass(n=2500)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=60, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=42, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=200, include_bias=True, include_input=True,
                             name="out"),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    return rc, exe, X[n_train:n_train + 10], Y[n_train:n_train + 10], \
            X[:n_train], Y[:n_train]


def parse_size(elf_path: pathlib.Path, cc: str) -> dict:
    """Run arm-none-eabi-size and return text/data/bss as ints."""
    size_cmd = cc.replace("gcc", "size")
    cp = subprocess.run([size_cmd, str(elf_path)],
                         capture_output=True, text=True, check=True)
    # Output: "   text    data     bss     dec     hex filename"
    lines = cp.stdout.strip().splitlines()
    if len(lines) < 2:
        return {}
    parts = lines[1].split()
    return {
        "text": int(parts[0]),
        "data": int(parts[1]),
        "bss":  int(parts[2]),
        "total": int(parts[3]),
    }


def build_one(rc, exe, label, target_factory, sf_range, input_frac,
               weight_frac, lut_n, train_X, train_Y, sample_X, sample_Y):
    """QAT search → cross-compile → return (size_dict, mse, qmodel)."""
    target = target_factory()
    result = search_quantization(
        rc, exe, train_X, train_Y, sample_X, sample_Y,
        target=target,
        state_frac_range=sf_range,
        input_frac=input_frac, weight_frac=weight_frac,
        lut=TanhLUTSpec(xmin=-4, xmax=4, n=lut_n),
    )
    qmodel = result.best_qmodel

    out = BUILD / label
    out.mkdir(parents=True, exist_ok=True)
    microbit = Microbit()
    artifact = microbit.compile_quantized(
        qmodel, output_dir=out, test_inputs=sample_X,
    )
    sizes = parse_size(artifact.binary, microbit.cc)
    return sizes, result.best_mse, qmodel, microbit, artifact


def main() -> None:
    rc, exe, sample_X, sample_Y, train_X, train_Y = train_esn()
    print(f"[1/3] train float ESN (N={rc.reservoir.units}, "
          f"topology={rc.reservoir.topology.name})")

    Y_f32 = exe.predict(sample_X)
    mse_float = float(np.mean((Y_f32 - sample_Y) ** 2))
    print(f"      float baseline MSE on sample: {mse_float:.6e}\n")

    rows = []
    for label, factory, sf_range, ifrac, wfrac, lut_n in _RECIPES:
        print(f"[2/3] QAT + cross-compile: storage={label} "
              f"state_frac in {sf_range}, input_frac={ifrac}, weight_frac={wfrac}")
        sizes, mse, qmodel, microbit, artifact = build_one(
            rc, exe, label, factory, sf_range, ifrac, wfrac, lut_n,
            train_X, train_Y, sample_X, sample_Y,
        )
        rows.append((label, qmodel, sizes, mse, artifact, microbit))
        print(f"      best state_frac={qmodel.config.state_frac}, "
              f"MSE={mse:.4e}, ELF total={sizes.get('total', '?')} bytes")

    # Show the size + accuracy table side by side.
    print("\n" + "=" * 66)
    print(f"{'storage':<8} {'state_frac':>10} {'text':>8} {'data':>6} "
          f"{'bss':>6} {'total':>8} {'MSE':>14}")
    print("-" * 66)
    for label, qm, sizes, mse, _, _ in rows:
        print(f"{label:<8} {qm.config.state_frac:>10} "
              f"{sizes.get('text', 0):>8} {sizes.get('data', 0):>6} "
              f"{sizes.get('bss', 0):>6} {sizes.get('total', 0):>8} "
              f"{mse:>14.4e}")
    print("=" * 66)
    base_total = rows[0][2].get("total", 1)
    for label, _, sizes, _, _, _ in rows[1:]:
        total = sizes.get("total", 0)
        delta = total - base_total
        ratio = total / base_total if base_total else 0.0
        sign = "-" if delta < 0 else "+"
        print(f"  {label} vs i32: {sign}{abs(delta):>5} bytes "
              f"({ratio*100:5.1f}% of i32)")
    print()

    # Run the i8 binary on QEMU as the headline deploy demo.
    label_i8, _, _, _, art_i8, mb_i8 = rows[-1]
    print(f"[3/3] qemu-system-arm run on i8 binary "
          f"({art_i8.binary.relative_to(BUILD.parent.parent)})")
    if shutil.which("qemu-system-arm") is None:
        print("      qemu-system-arm not found on PATH — skipping run.")
        return
    rresult = mb_i8.run(art_i8)
    # Truncate per-step output; the comparison line at the bottom is
    # what's worth showing.
    tail = "\n".join(rresult.output.splitlines()[-12:])
    print(tail)
    if not rresult.success:
        sys.exit(f"\nFAIL (returncode={rresult.returncode})")
    print("\n[ok] EMULATOR_EXIT observed — i8 quantized kernel ran on QEMU")


if __name__ == "__main__":
    main()
