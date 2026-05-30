"""Benchmark: dense vs sparse-specialized W_res kernel (host LLVM JIT).

The reservoir's recurrent matrix W_res is a compile-time constant and is
typically sparse (density ~0.1). The dense kernel spends N*N MACs per step,
~90% of them multiplying by exact zero. `SparsifyReservoir` skips those.

This script trains the same ESN once, compiles it with the dense kernel and
with the sparse kernel (auto strategy), verifies the outputs are bit-exact,
and reports:

  - W_res MAC count per step: dense (N*N) vs sparse (nnz) and the reduction.
  - host JIT wall-clock predict time (median of repeated runs).

Usage:
    python benchmarks/host/sparse_speedup.py
"""
from __future__ import annotations
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.ir import StructuralSpecialize, FuseStepReadout, SparsifyReservoir
from rclite.codegen import compile_rc


def _train(units: int, density: int, seed: int = 7):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.4, input_scaling=1.1,
                        name="in"),
        reservoir=ReservoirNode(units=units, activation=Activation.TANH,
                                topology=Topology.RANDOM, spectral_radius=0.9,
                                leak_rate=0.35, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=100,
                            include_bias=True, include_input=False,
                            name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1500, 1)) * 0.3 + 0.4
    Y = np.sin(np.arange(1500) * 0.05)[:, None]
    exe.fit(X, Y)
    return rc, exe


def _median_predict_time(compiled, X, repeats: int = 7) -> float:
    compiled.predict(X)  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compiled.predict(X)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _bench(units: int, density: float):
    rc, exe = _train(units, density)
    N = rc.reservoir.units
    nnz = int(np.count_nonzero(exe.W_res))

    base = [StructuralSpecialize(), FuseStepReadout()]
    dense = compile_rc(rc, exe, passes=base)
    sparse = compile_rc(rc, exe, passes=base + [SparsifyReservoir()])

    rng = np.random.default_rng(0)
    X = rng.standard_normal((2000, 1)) * 0.3 + 0.4
    Y_dense = dense.predict(X)
    Y_sparse = sparse.predict(X)
    max_diff = float(np.max(np.abs(Y_dense - Y_sparse)))

    t_dense = _median_predict_time(dense, X)
    t_sparse = _median_predict_time(sparse, X)

    dense_macs = N * N
    return {
        "N": N, "density": density, "nnz": nnz,
        "dense_macs": dense_macs, "sparse_macs": nnz,
        "mac_reduction": dense_macs / max(nnz, 1),
        "t_dense_ms": t_dense * 1e3, "t_sparse_ms": t_sparse * 1e3,
        "speedup": t_dense / t_sparse if t_sparse else float("nan"),
        "max_diff": max_diff,
    }


def main():
    print("dense vs sparse W_res kernel (host LLVM JIT, T=2000 steps)\n")
    header = (f"{'N':>4} {'dens':>5} {'nnz':>6} {'denseMAC':>9} "
              f"{'sparseMAC':>9} {'MAC↓':>6} {'dense ms':>9} "
              f"{'sparse ms':>10} {'speedup':>8} {'bit-exact':>10}")
    print(header)
    print("-" * len(header))
    configs = [(100, 0.1), (200, 0.1), (300, 0.05), (300, 0.1), (400, 0.1)]
    for units, density in configs:
        r = _bench(units, density)
        exact = "yes" if r["max_diff"] == 0.0 else f"NO({r['max_diff']:.1e})"
        print(f"{r['N']:>4} {r['density']:>5.2f} {r['nnz']:>6} "
              f"{r['dense_macs']:>9} {r['sparse_macs']:>9} "
              f"{r['mac_reduction']:>5.1f}x {r['t_dense_ms']:>9.3f} "
              f"{r['t_sparse_ms']:>10.3f} {r['speedup']:>7.2f}x {exact:>10}")
    print("\nMAC↓ = dense MAC / sparse MAC (per step). speedup = host wall-clock.")
    print("bit-exact = sparse output identical to dense kernel (atol=0).")


if __name__ == "__main__":
    main()
