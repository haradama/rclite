"""Benchmark numpy runtime vs LLVM JIT on Mackey-Glass inference."""

from __future__ import annotations
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc

from examples.forecasting.mackey_glass_esn import mackey_glass


def build_esn(
    units: int, input_offset: float, topology: Topology = Topology.ESN_STANDARD
) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=1.0,
            input_offset=input_offset,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            spectral_radius=0.95,
            leak_rate=0.3,
            density=0.05,
            topology=topology,
            chain_weight=0.9,
            chain_feedback=0.05,
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


def time_fn(fn, *args, repeats: int = 5):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best, out


def main() -> None:
    series = mackey_glass(n=4000)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te = X[n_train:]
    input_offset = float(X_tr.mean())

    print(
        f"{'topology':<10} {'units':>6}  {'numpy [ms]':>12}  {'jit [ms]':>12}  "
        f"{'speedup':>8}  {'compile [ms]':>14}  {'max |diff|':>12}"
    )
    for topo in (
        Topology.ESN_STANDARD,
        Topology.SCR,
        Topology.DLR,
        Topology.DLRB,
    ):
        for N in [100, 500, 1000]:
            rc = build_esn(N, input_offset, topology=topo)
            exe = RCExecutor(rc)
            exe.fit(X_tr, Y_tr)

            t_cmp_0 = time.perf_counter()
            jit = compile_rc(rc, exe)
            compile_ms = (time.perf_counter() - t_cmp_0) * 1000

            t_np, Y_np = time_fn(exe.predict, X_te)
            t_ji, Y_ji = time_fn(jit.predict, X_te)
            max_diff = float(np.max(np.abs(Y_np - Y_ji)))
            speedup = t_np / t_ji

            print(
                f"{topo.name:<10} {N:>6}  {t_np * 1000:>12.2f}  "
                f"{t_ji * 1000:>12.2f}  {speedup:>7.2f}x  "
                f"{compile_ms:>14.1f}  {max_diff:>12.2e}"
            )


if __name__ == "__main__":
    main()
