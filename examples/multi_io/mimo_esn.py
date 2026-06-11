"""End-to-end multi-input / multi-output (MIMO) ESN.

The whole rclite pipeline is generic over the input dimension K
(`input.units`) and the output dimension M (`readout.units`). This demo drives
a genuine MIMO model — **3 input channels, 2 output channels** — through every
stage and checks bit-exactness at each hop:

    1. reference runtime  (numpy)         -> (T, 2) predictions
    2. LLVM JIT codegen   (float)         -> bit-exact with the runtime
    3. symmetric i16 quantization + C     -> generated rc_kernel.c is
       export, compiled with host gcc        bit-exact with the quantized
                                              executor

The synthetic task: two outputs that each mix *different*, time-delayed input
channels, so the readout genuinely has to use all three inputs and produce two
decorrelated outputs — something a scalar (single-in/single-out) model cannot
represent.

Run from the repo root:

    python examples/multi_io/mimo_esn.py

The C-export stage is skipped automatically when `gcc` is not on PATH.
Artifacts (when gcc is present) land in ./build/ next to this file.
"""

from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc
from rclite.quant import (
    QuantConfig,
    TanhLUTSpec,
    I16FixedPoint,
    quantize_model,
    QuantizedExecutor,
)
from rclite.export import export_bundle, emit_symmetric_kernel_c


HERE = pathlib.Path(__file__).resolve().parent
REPO = HERE.parents[1]

K, N, M = 3, 80, 2  # 3 inputs, 80 reservoir units, 2 outputs


def make_data(T: int, seed: int = 0):
    """Three input channels; two outputs mixing delayed, distinct channels."""
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, K)) * 0.3
    Y = np.zeros((T, M))
    for t in range(2, T):
        Y[t, 0] = 0.6 * X[t - 1, 0] + 0.3 * X[t - 2, 1] - 0.1 * X[t, 2]
        Y[t, 1] = -0.4 * X[t - 1, 2] + 0.25 * X[t, 0] + 0.2 * X[t - 2, 0]
    return X, Y


def build() -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(
            units=K,
            input_scaling=0.5,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=N,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=50,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )


def nrmse(pred, ref, washout):
    p, r = pred[washout:], ref[washout:]
    rmse = np.sqrt(np.mean((p - r) ** 2, axis=0))
    return rmse / (np.std(r, axis=0) + 1e-12)


# ----------------------------------------------------------------- C harness


def _emit_mimo_c_main(ctype, q_x, T, K, M):
    """A row-major (T, K) -> (T, M) C driver for the quantized kernel.

    Doubles as the reference pattern for the embedded demo harnesses: X is
    laid out X[t*K + k], Y as Y[t*M + m]; rc_predict receives the step count
    T, never T*K or T*M.
    """
    body = ", ".join(str(int(v)) for v in q_x.reshape(-1))
    return "\n".join(
        [
            "#include <stdint.h>",
            "#include <stdio.h>",
            f"#define RC_T {T}",
            f"#define RC_K {K}",
            f"#define RC_M {M}",
            f"extern void rc_predict(int32_t, const {ctype}*, {ctype}*);",
            f"static const {ctype} X[RC_T * RC_K] = {{ {body} }};",
            f"static {ctype} Y[RC_T * RC_M];",
            "int main(void){",
            "  rc_predict(RC_T, X, Y);",
            "  for (int t = 0; t < RC_T; t++) {",
            "    for (int m = 0; m < RC_M; m++)",
            '      printf("%d\\n", (int)Y[t * RC_M + m]);',
            "  }",
            "  return 0; }",
        ]
    )


def quantized_reference(qm, cfg, X):
    qexe = QuantizedExecutor(qm)
    qexe.reset()
    out = np.zeros((X.shape[0], qm.M), dtype=np.int64)
    qx = np.zeros((X.shape[0], qm.K), dtype=np.int64)
    for t in range(X.shape[0]):
        u_raw_q = qm.target.quantize_input_array(X[t], cfg)
        qx[t] = u_raw_q
        u_pre_q = qexe._preprocess_q(u_raw_q)
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(u_raw_q, qexe.state_q)
    return qx, out


def main() -> None:
    X, Y = make_data(800)
    n_train = 600
    rc = build()
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])

    Xte, Yte = X[n_train:], Y[n_train:]

    print(f"MIMO ESN: K={K} inputs -> M={M} outputs, N={N} reservoir units")
    print(f"  W_in shape  = {exe.W_in.shape}   (N, K)")
    print(f"  W_out shape = {exe.W_out.shape}   (M, F)")

    # 1. reference runtime
    Y_np = exe.predict(Xte)
    nr = nrmse(Y_np, Yte, washout=50)
    print(f"\n[1] runtime predict shape = {Y_np.shape}")
    print(
        f"    per-output NRMSE = "
        + ", ".join(f"y{m}={nr[m]:.4f}" for m in range(M))
    )

    # 2. LLVM JIT codegen, bit-exact with runtime
    Y_jit = compile_rc(rc, exe).predict(Xte)
    jit_diff = float(np.max(np.abs(Y_jit - Y_np)))
    print(
        f"\n[2] JIT predict shape = {Y_jit.shape}, "
        f"max|JIT - runtime| = {jit_diff:.2e}"
    )

    # 3. quantize (symmetric i16) + export C, compile with gcc, bit-exact
    cfg = QuantConfig(state_frac=10, input_frac=8, weight_frac=8)
    qm = quantize_model(
        rc, exe, cfg, target=I16FixedPoint(), lut=TanhLUTSpec(n=256)
    )
    qx, qy_ref = quantized_reference(qm, cfg, Xte)
    print(
        f"\n[3] quantized i16: qm.K={qm.K}, qm.M={qm.M}, "
        f"q_y shape = {qy_ref.shape}"
    )

    if shutil.which("gcc") is None:
        print("    (skipping C export: gcc not on PATH)")
        return

    build_dir = HERE / "build"
    build_dir.mkdir(parents=True, exist_ok=True)
    export_bundle(qm, build_dir, name="mimo_rc")
    print(
        f"    export_bundle -> {build_dir.relative_to(REPO)}/ "
        "(rc_kernel.c, mimo_rc.h, Cargo crate)"
    )

    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(emit_symmetric_kernel_c(qm))
        (td / "main.c").write_text(
            _emit_mimo_c_main("int16_t", qx, Xte.shape[0], qm.K, qm.M)
        )
        r = subprocess.run(
            [
                "gcc",
                "-O2",
                "-std=c99",
                "-o",
                str(td / "a.out"),
                str(td / "main.c"),
                str(td / "kernel.c"),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            sys.exit("gcc failed:\n" + r.stderr)
        out = subprocess.run(
            [str(td / "a.out")], capture_output=True, text=True, check=True
        ).stdout
    qy_c = np.array(
        [int(v) for v in out.strip().split("\n")], dtype=np.int64
    ).reshape(Xte.shape[0], M)
    c_diff = int(np.max(np.abs(qy_c - qy_ref)))
    print(
        f"    C kernel q_y shape = {qy_c.shape}, "
        f"max|C - quant executor| = {c_diff}  "
        f"({'bit-exact' if c_diff == 0 else 'MISMATCH'})"
    )

    # First few decoded steps, both output channels.
    scale = 1 << cfg.state_frac
    print("\n      t |        y0 (C / ref)        |        y1 (C / ref)")
    print("    ----+----------------------------+---------------------------")
    for t in range(5):
        print(
            f"    {t:3d} | {qy_c[t, 0] / scale:9.5f} / {qy_ref[t, 0] / scale:9.5f}"
            f"      | {qy_c[t, 1] / scale:9.5f} / {qy_ref[t, 1] / scale:9.5f}"
        )


if __name__ == "__main__":
    main()
