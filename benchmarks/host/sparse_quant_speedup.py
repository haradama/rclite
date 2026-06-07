"""Benchmark: dense vs sparse W_res in the QUANTIZED integer kernels (host JIT).

Mirrors `sparse_speedup.py` for the integer paths. W_res is a compile-time
constant and typically sparse (~density), so the dense N*N integer matvec
wastes ~90% of its MACs on zeros. `SparsifyReservoir` skips them. Reports the
per-step W_res MAC reduction and the host wall-clock predict speedup, for the
symmetric (i32) and affine (i8) integer kernels — verifying bit-exactness.

Usage:
    python benchmarks/host/sparse_quant_speedup.py
"""

from __future__ import annotations
import pathlib
import statistics
import sys
import time

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
from rclite.quant import QuantConfig, TanhLUTSpec, quantize_model
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.codegen.llvm import CompiledQuantizedRC, CompiledAffineRC
from rclite.ir import SparsifyReservoir


def _train(units, density, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.35,
            density=density,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=100,
            include_bias=True,
            include_input=False,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((1600, 1)) * 0.2
    exe.fit(X[:1400], np.sin(np.arange(1400) * 0.05)[:, None])
    return rc, exe, X[:1400]


def _median_time(compiled, X, repeats=7):
    compiled.predict(X)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compiled.predict(X)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _bench_row(kind, units, density):
    rc, exe, Xtrain = _train(units, density)
    N = rc.reservoir.units
    nnz = int(np.count_nonzero(exe.W_res))
    X = np.random.default_rng(0).standard_normal((2000, 1)) * 0.2

    if kind == "symmetric":
        qm = quantize_model(
            rc,
            exe,
            QuantConfig(state_frac=16, input_frac=12, weight_frac=12),
            lut=TanhLUTSpec(n=128),
        )
        dense = CompiledQuantizedRC(qm)
        sparse = CompiledQuantizedRC(qm, passes=[SparsifyReservoir()])
    else:  # affine i8
        cfg = calibrate_from_data(rc, exe, Xtrain, storage_bits=8)
        qm = quantize_model_affine(rc, exe, cfg)
        dense = CompiledAffineRC(qm)
        sparse = CompiledAffineRC(qm, passes=[SparsifyReservoir()])

    Yd, Ys = dense.predict(X), sparse.predict(X)
    max_diff = int(np.max(np.abs(Yd.astype(np.int64) - Ys.astype(np.int64))))
    td, tsp = _median_time(dense, X), _median_time(sparse, X)
    return {
        "kind": kind,
        "N": N,
        "density": density,
        "nnz": nnz,
        "mac_reduction": (N * N) / max(nnz, 1),
        "t_dense_ms": td * 1e3,
        "t_sparse_ms": tsp * 1e3,
        "speedup": td / tsp if tsp else float("nan"),
        "exact": max_diff == 0,
        "max_diff": max_diff,
    }


def main():
    print(
        "dense vs sparse W_res — quantized integer kernels (host JIT, T=2000)\n"
    )
    header = (
        f"{'kernel':>10} {'N':>4} {'dens':>5} {'nnz':>6} {'MAC↓':>6} "
        f"{'dense ms':>9} {'sparse ms':>10} {'speedup':>8} {'bit-exact':>10}"
    )
    print(header)
    print("-" * len(header))
    for kind in ("symmetric", "affine"):
        for units, density in [(100, 0.1), (200, 0.1), (300, 0.1)]:
            r = _bench_row(kind, units, density)
            exact = "yes" if r["exact"] else f"NO({r['max_diff']})"
            print(
                f"{r['kind']:>10} {r['N']:>4} {r['density']:>5.2f} "
                f"{r['nnz']:>6} {r['mac_reduction']:>5.1f}x "
                f"{r['t_dense_ms']:>9.3f} {r['t_sparse_ms']:>10.3f} "
                f"{r['speedup']:>7.2f}x {exact:>10}"
            )
    print("\nMAC↓ = dense (N*N) / sparse (nnz) W_res MACs per step.")
    print("bit-exact = sparse integer output identical to dense kernel.")


if __name__ == "__main__":
    main()
