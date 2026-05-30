"""Benchmark: per-tensor vs per-channel W_res affine quantization accuracy.

per-channel gives each reservoir row its own W_res scale (per-output-channel),
quantized bit-exactly across executor/JIT/C. This sweeps several ESN
configs and reports the quantized MSE vs the float reference for both modes.

KEY FINDING (honest): for *random* ESN reservoirs the rows are statistically
homogeneous, so per-row scales barely differ from the single per-tensor scale
— per-channel helps on some seeds and slightly hurts on others (it is NOT a
guaranteed win here). The payoff is larger for heterogeneous / structured /
trained weight rows. Cost is negligible: 2*N extra int32 (M0[i], n[i]); the
W_res storage and the per-step op count are unchanged.

    python benchmarks/host/per_channel_accuracy.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
)


def _train(units, density, seed):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=100,
                            include_bias=True, include_input=False, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    s = np.sin(np.arange(1100) * 0.05) + 0.1 * rng.standard_normal(1100)
    X, Y = s[:-1, None], s[1:, None]
    exe.fit(X[:800], Y[:800])
    return rc, exe, X, Y


def _mse(rc, exe, X, Y, sb, per_channel):
    cfg = calibrate_from_data(rc, exe, X[:800], storage_bits=sb,
                              per_channel_W_res=per_channel)
    qm = quantize_model_affine(rc, exe, cfg)
    yq = AffineQuantizedExecutor(qm).predict(X[800:1000])
    return float(np.mean((yq - Y[800:1000]) ** 2))


def main():
    print("per-tensor vs per-channel W_res — affine i8 quantized MSE "
          "(float-ref target)\n")
    header = (f"{'N':>4} {'dens':>5} {'seed':>4} {'float MSE':>11} "
              f"{'per-tensor':>11} {'per-channel':>12} {'ratio':>7}")
    print(header)
    print("-" * len(header))
    ratios = []
    for units, density in [(80, 0.1), (120, 0.1), (120, 0.3)]:
        for seed in (1, 2, 3):
            rc, exe, X, Y = _train(units, density, seed)
            yf = exe.predict(X[800:1000])
            mse_f = float(np.mean((yf - Y[800:1000]) ** 2))
            mt = _mse(rc, exe, X, Y, 8, False)
            mc = _mse(rc, exe, X, Y, 8, True)
            r = mc / max(mt, 1e-12)
            ratios.append(r)
            print(f"{units:>4} {density:>5.2f} {seed:>4} {mse_f:>11.3e} "
                  f"{mt:>11.3e} {mc:>12.3e} {r:>6.2f}x")
    arr = np.array(ratios)
    print(f"\nper-channel/per-tensor MSE ratio: mean={arr.mean():.3f} "
          f"min={arr.min():.3f} max={arr.max():.3f} "
          f"(<1 = per-channel better). Mixed for random ESN — task-dependent.")
    print("Cost: +2*N int32 (M0,n); W_res bytes and per-step ops unchanged.")


if __name__ == "__main__":
    main()
