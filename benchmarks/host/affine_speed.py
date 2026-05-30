"""Speed benchmarks for the affine quantization path.

Measures:
  1. Per-step inference latency across all execution paths
     (float Python, symmetric Python, affine Python, symmetric JIT).
  2. Setup cost (calibration, model build).
  3. QAT search cost vs single-pass calibration.

Run with:
    uv run python benchmarks/affine_speed.py
"""
from __future__ import annotations
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec,
    I32FixedPoint, I16FixedPoint, I8Symmetric,
    quantize_model, QuantizedExecutor,
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    search_quantization_affine,
)
from rclite.codegen.llvm import CompiledQuantizedRC, CompiledAffineRC
from examples.forecasting.mackey_glass_esn import mackey_glass


def time_call(fn, n_runs=5):
    """Run fn() n_runs times, return (best_s, mean_s)."""
    times = []
    for _ in range(n_runs):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times), float(np.mean(times))


def fmt_us_per_step(seconds: float, T: int) -> str:
    return f"{seconds * 1e6 / T:8.2f} µs/step"


def build_setup(N: int = 80, n_train: int = 2000, n_eval: int = 200, seed: int = 42):
    """Train a Mackey-Glass ESN and return everything the bench needs."""
    series = mackey_glass(n=n_train + n_eval + 600)
    X, Y = series[:-1, None], series[1:, None]
    rc = ReservoirComputer(
        input=InputNode(units=1, input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=N, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=seed),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=300, include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])
    return rc, exe, X, Y, n_train, n_eval


def bench_inference(rc, exe, X, Y, n_train, n_eval, n_runs=5):
    """Time per-step inference across all execution paths."""
    print(f"Reservoir: N={rc.reservoir.units}, K={rc.input.units}, "
          f"M={rc.readout.units}, topology={rc.reservoir.topology.name}")
    print(f"Inference input: T={n_eval} steps, n_runs={n_runs}")
    print()

    # Build one quantized model per path
    cfg_i32_sym = QuantConfig(state_frac=18, input_frac=14, weight_frac=14)
    qm_i32_sym = quantize_model(rc, exe, cfg_i32_sym, target=I32FixedPoint(),
                                  lut=TanhLUTSpec(n=128))
    cfg_i16_sym = QuantConfig(state_frac=10, input_frac=8, weight_frac=8)
    qm_i16_sym = quantize_model(rc, exe, cfg_i16_sym, target=I16FixedPoint(),
                                  lut=TanhLUTSpec(n=128))
    cfg_i8_sym = QuantConfig(state_frac=5, input_frac=4, weight_frac=4)
    qm_i8_sym = quantize_model(rc, exe, cfg_i8_sym, target=I8Symmetric(),
                                lut=TanhLUTSpec(n=32))
    cfg_i16_aff = calibrate_from_data(rc, exe, X[:n_train], storage_bits=16)
    qm_i16_aff = quantize_model_affine(rc, exe, cfg_i16_aff)
    cfg_i8_aff = calibrate_from_data(rc, exe, X[:n_train], storage_bits=8)
    qm_i8_aff = quantize_model_affine(rc, exe, cfg_i8_aff)

    eval_X = X[n_train:n_train + n_eval]

    qexe_i32_s = QuantizedExecutor(qm_i32_sym)
    qexe_i16_s = QuantizedExecutor(qm_i16_sym)
    qexe_i8_s  = QuantizedExecutor(qm_i8_sym)
    qexe_i16_a = AffineQuantizedExecutor(qm_i16_aff)
    qexe_i8_a  = AffineQuantizedExecutor(qm_i8_aff)
    jit_i32_s = CompiledQuantizedRC(qm_i32_sym)
    jit_i16_s = CompiledQuantizedRC(qm_i16_sym)
    jit_i8_s  = CompiledQuantizedRC(qm_i8_sym)
    jit_i16_a = CompiledAffineRC(qm_i16_aff)
    jit_i8_a  = CompiledAffineRC(qm_i8_aff)

    paths = [
        ('Python float (RCExecutor)',  lambda: exe.predict(eval_X)),
        ('Python i32 sym',             lambda: (qexe_i32_s.reset(), qexe_i32_s.predict(eval_X))),
        ('Python i16 sym',             lambda: (qexe_i16_s.reset(), qexe_i16_s.predict(eval_X))),
        ('Python i8  sym',             lambda: (qexe_i8_s.reset(),  qexe_i8_s.predict(eval_X))),
        ('Python i16 affine',          lambda: (qexe_i16_a.reset(), qexe_i16_a.predict(eval_X))),
        ('Python i8  affine',          lambda: (qexe_i8_a.reset(),  qexe_i8_a.predict(eval_X))),
        ('JIT i32 sym (host LLVM)',    lambda: jit_i32_s.predict(eval_X)),
        ('JIT i16 sym (host LLVM)',    lambda: jit_i16_s.predict(eval_X)),
        ('JIT i8  sym (host LLVM)',    lambda: jit_i8_s.predict(eval_X)),
        ('JIT i16 affine (host LLVM)', lambda: jit_i16_a.predict(eval_X)),
        ('JIT i8  affine (host LLVM)', lambda: jit_i8_a.predict(eval_X)),
    ]

    print("=" * 70)
    print(f"{'Path':<28} {'best':>16} {'mean':>16}  {'rel':>5}")
    print("-" * 70)
    results = []
    for name, fn in paths:
        fn()  # warmup
        best, mean = time_call(fn, n_runs=n_runs)
        results.append((name, best, mean))
    fastest = min(r[1] for r in results)
    for name, best, mean in results:
        rel = best / fastest
        print(f"{name:<28} {fmt_us_per_step(best, n_eval)} {fmt_us_per_step(mean, n_eval)}  {rel:>4.1f}x")
    print()


def bench_setup(rc, exe, X, n_train, n_runs=3):
    """Time calibration and quantization (one-time setup cost)."""
    print("=" * 70)
    print(f"Setup cost (T_train={n_train}, n_runs={n_runs})")
    print("-" * 70)

    cfg_i8  = calibrate_from_data(rc, exe, X[:n_train], storage_bits=8)
    cfg_i16 = calibrate_from_data(rc, exe, X[:n_train], storage_bits=16)

    items = [
        ('calibrate i8',       lambda: calibrate_from_data(rc, exe, X[:n_train], storage_bits=8)),
        ('calibrate i16',      lambda: calibrate_from_data(rc, exe, X[:n_train], storage_bits=16)),
        ('quantize_model i8',  lambda: quantize_model_affine(rc, exe, cfg_i8)),
        ('quantize_model i16', lambda: quantize_model_affine(rc, exe, cfg_i16)),
    ]
    for name, fn in items:
        fn()
        best, mean = time_call(fn, n_runs=n_runs)
        print(f"  {name:<22} best={best * 1000:6.1f} ms  mean={mean * 1000:6.1f} ms")
    print()


def bench_qat(rc, exe, X, Y, n_train, n_eval, n_runs=2):
    """Time the QAT search at various iteration counts."""
    print("=" * 70)
    print(f"QAT search cost (T_train={n_train}, T_eval={n_eval}, n_runs={n_runs})")
    print("-" * 70)
    eval_X = X[n_train:n_train + n_eval]
    eval_Y = Y[n_train:n_train + n_eval]

    cases = [
        ('search i8  n_iter=0', 8,  0),
        ('search i8  n_iter=1', 8,  1),
        ('search i8  n_iter=3', 8,  3),
        ('search i16 n_iter=1', 16, 1),
    ]
    for name, sb, n_iter in cases:
        def fn(sb=sb, n_iter=n_iter):
            search_quantization_affine(rc, exe, X[:n_train], Y[:n_train],
                                        eval_X, eval_Y,
                                        storage_bits=sb, n_iterations=n_iter)
        fn()
        best, mean = time_call(fn, n_runs=n_runs)
        print(f"  {name:<22} best={best:5.2f} s   mean={mean:5.2f} s")
    print()


def main():
    rc, exe, X, Y, n_train, n_eval = build_setup()
    bench_inference(rc, exe, X, Y, n_train, n_eval)
    bench_setup(rc, exe, X, n_train)
    bench_qat(rc, exe, X, Y, n_train, n_eval)


if __name__ == "__main__":
    main()
