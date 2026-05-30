"""Benchmark the IR optimization passes on Mackey-Glass inference.

Compiles the same ESN under several pass schedules and times the host
JIT inference loop. Used to verify that RC-specific structural rewrites
(StructuralSpecialize, FuseStepReadout, TimeUnroll) yield measurable
improvements over the unoptimized lowering.
"""
from __future__ import annotations
import sys
import pathlib
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc
from rclite.ir import StructuralSpecialize, FuseStepReadout, TimeUnroll

from examples.forecasting.mackey_glass_esn import mackey_glass


def build_esn(units: int, topology: Topology, input_offset: float
              ) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=input_offset,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, activation=Activation.TANH,
                                 spectral_radius=0.95, leak_rate=0.3,
                                 density=0.05, topology=topology,
                                 chain_weight=0.9, chain_feedback=0.05,
                                 seed=42, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=200, include_bias=True,
                             include_input=True, name="out"),
    )


def time_fn(fn, *args, repeats: int = 7):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        out = fn(*args)
        dt = time.perf_counter() - t0
        best = min(best, dt)
    return best, out


SCHEDULES = {
    "baseline (no passes)": [],
    "structural only":      [StructuralSpecialize()],
    "+ fuse":               [StructuralSpecialize(), FuseStepReadout()],
    "+ fuse + unroll-2":    [StructuralSpecialize(), FuseStepReadout(), TimeUnroll(K=2)],
    "+ fuse + unroll-4":    [StructuralSpecialize(), FuseStepReadout(), TimeUnroll(K=4)],
    "+ fuse + unroll-8":    [StructuralSpecialize(), FuseStepReadout(), TimeUnroll(K=8)],
}


def main() -> None:
    series = mackey_glass(n=4000)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te = X[n_train:]
    input_offset = float(X_tr.mean())

    for topo in (Topology.ESN_STANDARD, Topology.SCR):
        print(f"\n=== {topo.name}  (N=200) ===")
        rc = build_esn(200, topo, input_offset)
        exe = RCExecutor(rc)
        exe.fit(X_tr, Y_tr)

        baseline_ns = None
        Y_ref = exe.predict(X_te)
        print(f"{'pass schedule':<28} {'predict [ms]':>13} {'rel':>7} "
              f"{'max |diff|':>13}")
        for label, passes in SCHEDULES.items():
            jit = compile_rc(rc, exe, passes=list(passes))
            t, Y = time_fn(jit.predict, X_te)
            diff = float(np.max(np.abs(Y - Y_ref)))
            if baseline_ns is None:
                baseline_ns = t
            rel = baseline_ns / t
            print(f"{label:<28} {t * 1000:>13.3f} {rel:>6.2f}x "
                  f"{diff:>13.3e}")


if __name__ == "__main__":
    main()
