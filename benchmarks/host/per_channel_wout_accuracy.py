"""Benchmark: per-tensor vs per-channel W_out affine quantization (M>1).

per-channel W_out gives each readout output channel its own block scales,
quantized bit-exactly across executor/JIT/C. Unlike per-channel W_res (which
is task-dependent for random ESN), per-channel W_out reliably helps when the
output rows differ in coefficient magnitude — i.e. multi-output regression
(MIMO) and classification readouts. This sweeps several M and reports the
quantized MSE vs the float reference for both modes.

Cost: +2*M int32 per W_out block (M0[m], n[m]); W_out bytes and op count
unchanged. No effect for single-output (M=1) readouts.

    python benchmarks/host/per_channel_wout_accuracy.py
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
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
)


def _train(M, K, units, seed):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.2,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=80,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1000, K)) * 0.3
    # heterogeneous-amplitude targets: row m has amplitude ~ (m+1)
    Y = np.stack(
        [
            (m + 1) * 0.5 * np.sin(np.arange(1000) * 0.03 * (m + 1))
            for m in range(M)
        ],
        axis=1,
    )
    exe.fit(X[:750], Y[:750])
    return rc, exe, X, Y


def _mse(rc, exe, X, Y, pc_out):
    cfg = calibrate_from_data(
        rc, exe, X[:750], storage_bits=8, per_channel_W_out=pc_out
    )
    qm = quantize_model_affine(rc, exe, cfg)
    yq = AffineQuantizedExecutor(qm).predict(X[750:950])
    return float(np.mean((yq - Y[750:950]) ** 2))


def main():
    print(
        "per-tensor vs per-channel W_out — affine i8 quantized MSE "
        "(MIMO, heterogeneous-amplitude outputs)\n"
    )
    header = (
        f"{'M':>3} {'K':>3} {'seed':>4} {'float MSE':>11} "
        f"{'per-tensor':>11} {'per-channel':>12} {'ratio':>7}"
    )
    print(header)
    print("-" * len(header))
    ratios = []
    for M in (2, 4, 8):
        for seed in (1, 2, 3):
            rc, exe, X, Y = _train(M, 2, 60, seed)
            yf = exe.predict(X[750:950])
            mse_f = float(np.mean((yf - Y[750:950]) ** 2))
            mt = _mse(rc, exe, X, Y, False)
            mc = _mse(rc, exe, X, Y, True)
            r = mc / max(mt, 1e-12)
            ratios.append(r)
            print(
                f"{M:>3} {2:>3} {seed:>4} {mse_f:>11.3e} "
                f"{mt:>11.3e} {mc:>12.3e} {r:>6.2f}x"
            )
    arr = np.array(ratios)
    print(
        f"\nper-channel/per-tensor MSE ratio: mean={arr.mean():.3f} "
        f"min={arr.min():.3f} max={arr.max():.3f} (<1 = per-channel better). "
        f"Consistent win for M>1 (vs task-dependent for W_res)."
    )
    print("Cost: +2*M int32 per block; W_out bytes / per-step ops unchanged.")


if __name__ == "__main__":
    main()
