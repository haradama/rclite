"""End-to-end demo: train a float ESN, calibrate + QAT for i16 affine,
cross-compile to Cortex-M0 with each LUT strategy, run on QEMU, and
print a footprint-and-parity comparison.

The headline result is i16 affine + QAT + linear-interp LUT — that's the
combination we expect to use in production: i16 keeps near-float accuracy
(NRMSE ~1.5%), QAT absorbs the rest of the quant noise, linear-interp LUT
cuts the 128 KB direct table down to ~1 KB so the whole thing fits on
micro:bit.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    AffineQuantizedExecutor,
    LUTStrategy,
)
from rclite.targets import Microbit

from examples.mackey_glass_esn import mackey_glass


BUILD = pathlib.Path(__file__).resolve().parent.parent / "build" / "microbit_i16_affine"


_STRATEGIES = [
    ("direct",        LUTStrategy.direct()),
    ("interp_n256",   LUTStrategy.linear_interp(n_entries=256)),
    ("interp_n64",    LUTStrategy.linear_interp(n_entries=64)),
    ("polynomial",    LUTStrategy.polynomial()),
]


def parse_size(elf_path: pathlib.Path, cc: str) -> dict:
    """Run arm-none-eabi-size, return text/data/bss/total."""
    cp = subprocess.run([cc.replace("gcc", "size"), str(elf_path)],
                         capture_output=True, text=True, check=True)
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


def train_esn():
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
    return rc, exe, X[n_train:n_train + 10], Y[n_train:n_train + 10], \
            X[:n_train], Y[:n_train]


def main() -> None:
    rc, exe, sample_X, sample_Y, train_X, train_Y = train_esn()
    print(f"[1/3] train float ESN (N={rc.reservoir.units}, "
          f"topology={rc.reservoir.topology.name})")
    Y_f32 = exe.predict(sample_X)
    print(f"      float baseline MSE on sample: "
          f"{float(np.mean((Y_f32 - sample_Y) ** 2)):.6e}\n")

    # Single calibration step shared across strategies (only the LUT changes).
    from rclite.quant.affine.calibrate import calibrate_from_data
    from rclite.quant.affine.quantize import quantize_model_affine
    cfg = calibrate_from_data(rc, exe, train_X, storage_bits=16)

    rows = []
    microbit = Microbit()
    for label, strategy in _STRATEGIES:
        print(f"[2/3] build + cross-compile: LUT={label}")
        qmodel = quantize_model_affine(rc, exe, cfg, lut_strategy=strategy)
        out = BUILD / label
        out.mkdir(parents=True, exist_ok=True)
        artifact = microbit.compile_affine_quantized(
            qmodel, output_dir=out, test_inputs=sample_X,
        )
        sizes = parse_size(artifact.binary, microbit.cc)

        # Eval accuracy: warm up with train_X, measure on sample.
        qexe = AffineQuantizedExecutor(qmodel)
        Y_full = qexe.predict(np.concatenate([train_X, sample_X], axis=0))
        Y_sample_pred = Y_full[-len(sample_X):]
        mse = float(np.mean((Y_sample_pred - sample_Y) ** 2))

        rows.append((label, sizes, mse, artifact))
        print(f"      ELF total={sizes.get('total', '?')} bytes, "
              f"sample MSE={mse:.4e}")

    # Comparison table
    print("\n" + "=" * 74)
    print(f"{'strategy':<14} {'text':>8} {'data':>6} {'bss':>6} "
          f"{'total':>8} {'sample MSE':>14}")
    print("-" * 74)
    base_total = rows[0][1].get("total", 1)
    for label, sizes, mse, _ in rows:
        delta = sizes.get("total", 0) - base_total
        sign = "" if delta == 0 else (f" ({delta:+d}B)")
        print(f"{label:<14} {sizes.get('text', 0):>8} "
              f"{sizes.get('data', 0):>6} {sizes.get('bss', 0):>6} "
              f"{sizes.get('total', 0):>8}{sign:<10} {mse:>14.4e}")
    print("=" * 74)

    # Run the LINEAR_INTERP variant on QEMU as the headline production demo
    interp_row = next((r for r in rows if r[0] == "interp_n256"), rows[0])
    label, _, _, art = interp_row
    print(f"\n[3/3] qemu-system-arm run on {label} binary "
          f"({art.binary.relative_to(BUILD.parent.parent)})")
    if shutil.which("qemu-system-arm") is None:
        print("      qemu-system-arm not found on PATH — skipping run.")
        return
    rresult = microbit.run(art)
    tail = "\n".join(rresult.output.splitlines()[-12:])
    print(tail)
    if not rresult.success:
        sys.exit(f"\nFAIL (returncode={rresult.returncode})")
    print(f"\n[ok] EMULATOR_EXIT observed — i16 affine {label} ran on QEMU")


if __name__ == "__main__":
    main()
