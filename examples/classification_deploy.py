"""Deploy a quantized reservoir classifier as a portable C kernel.

End-to-end Phase 3 classification path:

  1. Train a per-step 3-class classifier (band of a smoothed signal).
  2. Affine-quantize to int8 (the MCU-friendly storage).
  3. export_bundle(head="classify") → a self-contained C kernel whose
     `rc_predict(T, X, Y)` writes one int32 class id per step, plus a Rust
     crate exposing `classify()`.
  4. Compile the generated C with host gcc and run it — the same integer
     kernel that ships to a microcontroller — and check it agrees with the
     LLVM reference and the float model.

Also exports a `head="proba"` bundle (softmax via exp LUT) emitting
Q-format class probabilities. Requires gcc on PATH; no MCU toolchain.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer, Task,
)
from rclite.runtime import RCExecutor
from rclite.quant import calibrate_from_data, quantize_model_affine
from rclite.export import export_bundle
from rclite.codegen.llvm import CompiledAffineRC


CLASS_NAMES = ["low", "mid", "high"]


def make_dataset(n=1500, seed=0):
    """Per-step 3-class task: which band a smoothed random walk sits in."""
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.92 * u[t - 1] + 0.08 * rng.standard_normal()
    X = u[:, None]
    y = np.ones(n, dtype=int)        # mid
    y[u > 0.25] = 2                  # high
    y[u < -0.25] = 0                 # low
    return X, y


def build_and_train():
    X, y = make_dataset()
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=60, activation=Activation.TANH, spectral_radius=0.9,
            leak_rate=0.3, density=0.2, topology=Topology.RANDOM, seed=7,
            name="res",
        ),
        readout=ReadoutNode(
            units=3, activation=Activation.IDENTITY, trainer=Trainer.RIDGE,
            regularization=1e-3, washout=100, task=Task.CLASSIFICATION,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    n_tr = 1000
    exe.fit(X[:n_tr], y[:n_tr])
    return rc, exe, X, y, n_tr


def compile_and_run_c(bundle_dir, qx_flat, T, out_ctype, n_out):
    """Compile rc_kernel.c + a tiny main against rc_model.h and run it."""
    body = ", ".join(str(int(v)) for v in qx_flat)
    main = "\n".join([
        "#include <stdint.h>", "#include <stdio.h>", '#include "rc_model.h"',
        "int main(void){",
        f"  rc_storage_t X[{len(qx_flat)}] = {{ {body} }};",
        f"  {out_ctype} Y[{n_out}];",
        f"  rc_predict({T}, X, Y);",
        f"  for (int i=0;i<{n_out};i++) printf(\"%d\\n\", (int)Y[i]);",
        "  return 0; }",
    ])
    (bundle_dir / "main.c").write_text(main)
    subprocess.run(
        ["gcc", "-O2", "-std=c99", "-I", str(bundle_dir),
         "-o", str(bundle_dir / "a.out"),
         str(bundle_dir / "main.c"), str(bundle_dir / "rc_kernel.c")],
        check=True, capture_output=True, text=True,
    )
    out = subprocess.run([str(bundle_dir / "a.out")], capture_output=True,
                         text=True).stdout
    return np.array([int(v) for v in out.strip().split("\n")], dtype=np.int64)


def main() -> None:
    if shutil.which("gcc") is None:
        sys.exit("this demo needs gcc on PATH")

    rc, exe, X, y, n_tr = build_and_train()
    Xte, yte = X[n_tr:], y[n_tr:]
    float_acc = float(np.mean(exe.predict_classes(Xte) == yte))
    print(f"[1/4] trained 3-class per-step classifier "
          f"(N={rc.reservoir.units}); float test acc = {float_acc:.3f}")

    cfg = calibrate_from_data(rc, exe, X[:n_tr], storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg)
    print(f"[2/4] affine-quantized to int8 (storage_bits={qm.storage_bits})")

    T, K, M = Xte.shape[0], qm.K, qm.M
    qx = qm.config.input.quantize_array(Xte).astype(np.int8).reshape(-1)

    # --- classify bundle: int32 class id per step ---
    jit_cls = CompiledAffineRC(qm, head="classify").predict(Xte)
    with tempfile.TemporaryDirectory() as td:
        out = export_bundle(qm, pathlib.Path(td) / "clf", name="rc_clf",
                            head="classify")
        files = sorted(p.name for p in out.iterdir() if p.is_file())
        kernel_bytes = (out / "rc_kernel.c").stat().st_size
        c_cls = compile_and_run_c(out, qx, T, "int32_t", T)
    c_acc = float(np.mean(c_cls == yte))
    print(f"[3/4] export_bundle(head='classify') → {files}")
    print(f"      rc_kernel.c = {kernel_bytes} bytes")
    print(f"      C kernel test acc = {c_acc:.3f}  "
          f"(C == LLVM: {np.array_equal(c_cls, jit_cls)})")

    # --- proba bundle: Q-format probabilities per step ---
    pf = min(qm.storage_bits - 1, 15)
    storage_ctype = {8: "int8_t", 16: "int16_t", 32: "int32_t"}[qm.storage_bits]
    jit_p = CompiledAffineRC(qm, head="proba").predict(Xte)
    with tempfile.TemporaryDirectory() as td:
        out = export_bundle(qm, pathlib.Path(td) / "prob", name="rc_prob",
                            head="proba")
        c_p = compile_and_run_c(out, qx, T, storage_ctype, T * M).reshape(T, M)
    c_p = c_p.astype(np.float64) / (1 << pf)
    agree = np.array_equal((jit_p * (1 << pf)).round().astype(int),
                           (c_p * (1 << pf)).round().astype(int))
    print(f"[4/4] export_bundle(head='proba') → softmax probabilities (Q{pf})")
    print(f"      C == LLVM: {agree}; example row: "
          f"{dict(zip(CLASS_NAMES, np.round(c_p[0], 3)))}")
    print("\n[ok] the generated rc_predict() emits class ids / probabilities "
          "on-device with no float or libm in the loop.")


if __name__ == "__main__":
    main()
