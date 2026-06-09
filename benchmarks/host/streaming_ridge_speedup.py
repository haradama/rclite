"""Benchmark: ridge training baseline vs streaming accumulation.

Compares:
  - baseline: RCExecutor.fit(..., materialize_states=True)
  - streaming: RCExecutor.fit(..., materialize_states=False)

Metrics:
  - fit wall-clock time (median over repeats)
  - peak RSS (ru_maxrss) measured in isolated subprocesses
  - numerical parity (W_out max|diff| and prediction max|diff|)

Usage:
    python benchmarks/host/streaming_ridge_speedup.py
    python benchmarks/host/streaming_ridge_speedup.py --T 20000 --N 300
"""

from __future__ import annotations

import argparse
import json
import pathlib
import statistics
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    Activation,
    InputNode,
    ReadoutNode,
    ReservoirComputer,
    ReservoirNode,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor


def _build_model(N: int, seed: int = 7):
    return ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_offset=0.4,
            input_scaling=1.1,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=N,
            activation=Activation.TANH,
            topology=Topology.ESN_STANDARD,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.1,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=100,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )


def _make_data(T: int, seed: int = 123):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.35 + 0.2
    Y = np.sin(np.arange(T) * 0.03)[:, None]
    Xe = rng.standard_normal((2000, 1)) * 0.35 + 0.2
    return X, Y, Xe


def _worker(mode: str, T: int, N: int, seed: int):
    import resource

    X, Y, _ = _make_data(T, seed=seed)
    rc = _build_model(N, seed=seed)
    exe = RCExecutor(rc)
    mat = mode == "baseline"

    t0 = time.perf_counter()
    exe.fit(X, Y, materialize_states=mat)
    fit_s = time.perf_counter() - t0

    rss_kb = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    out = {
        "mode": mode,
        "fit_s": fit_s,
        "rss_kb": rss_kb,
    }
    print(json.dumps(out))


def _run_worker(mode: str, T: int, N: int, seed: int):
    cmd = [
        sys.executable,
        str(pathlib.Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--T",
        str(T),
        "--N",
        str(N),
        "--seed",
        str(seed),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True, check=True)
    lines = [ln.strip() for ln in cp.stdout.splitlines() if ln.strip()]
    return json.loads(lines[-1])


def _parity(T: int, N: int, seed: int):
    X, Y, Xe = _make_data(T, seed=seed)
    rc1 = _build_model(N, seed=seed)
    rc2 = _build_model(N, seed=seed)
    e1 = RCExecutor(rc1)
    e2 = RCExecutor(rc2)

    e1.fit(X, Y, materialize_states=True)
    e2.fit(X, Y, materialize_states=False)

    w_diff = float(np.max(np.abs(e1.W_out - e2.W_out)))
    y1 = e1.predict(Xe)
    y2 = e2.predict(Xe)
    y_diff = float(np.max(np.abs(y1 - y2)))
    return w_diff, y_diff


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker", action="store_true")
    ap.add_argument(
        "--mode", choices=["baseline", "streaming"], default="baseline"
    )
    ap.add_argument("--T", type=int, default=12000)
    ap.add_argument("--N", type=int, default=240)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repeats", type=int, default=5)
    args = ap.parse_args()

    if args.worker:
        _worker(args.mode, args.T, args.N, args.seed)
        return

    base_runs = [
        _run_worker("baseline", args.T, args.N, args.seed + i)
        for i in range(args.repeats)
    ]
    strm_runs = [
        _run_worker("streaming", args.T, args.N, args.seed + i)
        for i in range(args.repeats)
    ]

    base_t = [r["fit_s"] for r in base_runs]
    strm_t = [r["fit_s"] for r in strm_runs]
    base_rss = [r["rss_kb"] for r in base_runs]
    strm_rss = [r["rss_kb"] for r in strm_runs]

    med_base_t = statistics.median(base_t)
    med_strm_t = statistics.median(strm_t)
    med_base_rss = int(statistics.median(base_rss))
    med_strm_rss = int(statistics.median(strm_rss))

    w_diff, y_diff = _parity(args.T, args.N, args.seed)

    print("ridge training baseline vs streaming\n")
    print(f"config: T={args.T}, N={args.N}, repeats={args.repeats}")
    print(f"baseline   fit median: {med_base_t * 1e3:9.2f} ms")
    print(f"streaming  fit median: {med_strm_t * 1e3:9.2f} ms")
    print(
        f"time ratio (baseline/streaming): {med_base_t / max(med_strm_t, 1e-30):.2f}x"
    )
    print()
    print(f"baseline   median peak RSS: {med_base_rss:9d} KB")
    print(f"streaming  median peak RSS: {med_strm_rss:9d} KB")
    print(
        f"RSS ratio (baseline/streaming): {med_base_rss / max(med_strm_rss, 1):.2f}x"
    )
    print()
    print(f"W_out max|diff|:   {w_diff:.3e}")
    print(f"predict max|diff|: {y_diff:.3e}")


if __name__ == "__main__":
    main()
