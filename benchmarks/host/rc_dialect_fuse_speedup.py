"""Benchmark: FuseStepReadout structural optimization (host LLVM JIT).

This quantifies the runtime payoff of the `FuseStepReadout` structural
optimization — the same rewrite the Stage-1 `rc`-dialect spike expresses as an
xDSL pattern (`rclite.codegen.rc_dialect_xdsl.FuseStepReadoutPattern`).

Unfused (baseline): per step the reservoir writes a phi feature buffer
`phi = [1?, u?, h]` (size F = 1 + K + N), and the readout then reads it back —
an F-wide store + F*M reload round-trip every step.

Fused (optimized): the readout's inner loop indexes `h` (and the input/bias)
directly, so the phi buffer is never materialised.

Both paths run the identical math, so the outputs are bit-exact (atol=0); only
the memory-traffic differs. The script trains one ESN per size, JIT-compiles it
both ways, checks parity, and reports the host wall-clock speedup.

Usage:
    python benchmarks/host/rc_dialect_fuse_speedup.py
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
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.ir import StructuralSpecialize, FuseStepReadout
from rclite.codegen import compile_rc


def _train(units: int, K: int, M: int, seed: int = 7):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            topology=Topology.RANDOM,
            spectral_radius=0.9,
            leak_rate=0.35,
            density=0.2,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=100,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((1500, K)) * 0.3
    Y = np.stack(
        [np.sin(np.arange(1500) * 0.05 * (m + 1)) for m in range(M)], axis=1
    )
    exe.fit(X, Y)
    return rc, exe


def _median_predict_time(compiled, X, repeats: int = 9) -> float:
    compiled.predict(X)  # warm
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compiled.predict(X)
        ts.append(time.perf_counter() - t0)
    return statistics.median(ts)


def _bench(units: int, K: int, M: int, T: int):
    rc, exe = _train(units, K, M)
    N = rc.reservoir.units
    F = 1 + K + N  # phi width eliminated by fusion

    # Same passes except the structural optimization under test.
    unfused = compile_rc(rc, exe, passes=[StructuralSpecialize()])
    fused = compile_rc(
        rc, exe, passes=[StructuralSpecialize(), FuseStepReadout()]
    )

    rng = np.random.default_rng(0)
    X = rng.standard_normal((T, K)) * 0.3
    Y_u = unfused.predict(X)
    Y_f = fused.predict(X)
    max_diff = float(np.max(np.abs(Y_u - Y_f)))

    t_u = _median_predict_time(unfused, X)
    t_f = _median_predict_time(fused, X)
    return {
        "N": N,
        "F": F,
        "t_unfused_ms": t_u * 1e3,
        "t_fused_ms": t_f * 1e3,
        "speedup": t_u / t_f if t_f else float("nan"),
        "max_diff": max_diff,
    }


def main():
    K, M, T = 4, 8, 2000
    print(
        f"FuseStepReadout structural opt — host LLVM JIT "
        f"(K={K}, M={M}, T={T} steps)\n"
    )
    header = (
        f"{'N':>4} {'phiF':>5} {'unfused ms':>11} {'fused ms':>10} "
        f"{'speedup':>8} {'bit-exact':>10}"
    )
    print(header)
    print("-" * len(header))
    for units in (64, 128, 256, 512, 1024):
        r = _bench(units, K, M, T)
        exact = "yes" if r["max_diff"] == 0.0 else f"NO({r['max_diff']:.1e})"
        print(
            f"{r['N']:>4} {r['F']:>5} {r['t_unfused_ms']:>11.3f} "
            f"{r['t_fused_ms']:>10.3f} {r['speedup']:>7.2f}x {exact:>10}"
        )
    print(
        "\nspeedup = unfused / fused host wall-clock. "
        "bit-exact = identical output (atol=0)."
    )
    print(
        "The fused kernel is what the rc-dialect FuseStepReadout xDSL pattern "
        "produces."
    )


if __name__ == "__main__":
    main()
