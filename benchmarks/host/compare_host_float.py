"""Benchmark: rclite (LLVM JIT) vs naive hand-written C scratch kernel.

Trains an ESN with rclite, extracts the trained weights, emits a naive
3-loop C kernel with those weights baked in, compiles it with gcc -O3,
and times both implementations on identical input. Verifies output parity
within float tolerance.

Both kernels expose the same C ABI:
    void rc_predict(int64_t T, double *X, double *Y);

so the rclite shared library and the scratch .so are drop-in replacements.
"""
from __future__ import annotations
import ctypes
import pathlib
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parents[1]
BUILD = ROOT / "build" / "bench_host_float"
TEMPLATE = HERE / "scratch_c" / "rc_naive_template.c"


def _fmt_doubles(arr: np.ndarray) -> str:
    """Comma-separated double constants, row-major flatten."""
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    return ", ".join(f"{v:.17g}" for v in flat)


def render_scratch_c(rc: ReservoirComputer, exe: RCExecutor,
                       out_path: pathlib.Path) -> None:
    """Fill the naive C template with the trained weights."""
    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()
    leak = float(rc.reservoir.leak_rate)
    bias = float(rc.reservoir.bias)

    tmpl = TEMPLATE.read_text()
    text = (
        tmpl
        .replace("@@N@@", str(N))
        .replace("@@K@@", str(K))
        .replace("@@M@@", str(M))
        .replace("@@F@@", str(F))
        .replace("@@LEAK@@", f"{leak:.17g}")
        .replace("@@BIAS@@", f"{bias:.17g}")
        .replace("@@INPUT_OFFSET@@", f"{float(rc.input.input_offset):.17g}")
        .replace("@@INPUT_SCALING@@", f"{float(rc.input.input_scaling):.17g}")
        .replace("@@INCLUDE_BIAS@@", "1" if rc.readout.include_bias else "0")
        .replace("@@INCLUDE_INPUT@@", "1" if rc.readout.include_input else "0")
        .replace("@@W_IN_VALUES@@", _fmt_doubles(exe.W_in))
        .replace("@@W_RES_VALUES@@", _fmt_doubles(exe.W_res))
        .replace("@@W_OUT_VALUES@@", _fmt_doubles(exe.W_out))
    )
    out_path.write_text(text)


def build_scratch_so(c_path: pathlib.Path, so_path: pathlib.Path,
                      cc: str = "gcc") -> None:
    cmd = [cc, "-O3", "-march=native", "-shared", "-fPIC",
           str(c_path), "-o", str(so_path), "-lm"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            f"scratch build failed:\n{' '.join(cmd)}\n{cp.stderr}"
        )


def load_predict(so_path: pathlib.Path):
    lib = ctypes.CDLL(str(so_path))
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rc_predict.restype = None
    return lib


def time_fn(fn, *args, repeats: int = 7) -> float:
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn(*args)
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def build_esn(N: int, topology: Topology, input_offset: float,
               seed: int = 42) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=input_offset, name="in"),
        reservoir=ReservoirNode(units=N, activation=Activation.TANH,
                                 spectral_radius=0.95, leak_rate=0.3,
                                 density=0.05, topology=topology,
                                 chain_weight=0.9, chain_feedback=0.05,
                                 seed=seed, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=200, include_bias=True,
                             include_input=True, name="out"),
    )


def run_one(topology: Topology, N: int, X_tr, Y_tr, X_te,
             input_offset: float) -> dict:
    out_dir = BUILD / f"{topology.name}_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = build_esn(N, topology, input_offset)
    exe = RCExecutor(rc)
    exe.fit(X_tr, Y_tr)

    # rclite JIT — uses structural specialization for SCR/DLR/DLRB.
    jit = compile_rc(rc, exe)
    Y_jit = jit.predict(X_te)

    # Naive C scratch
    c_path = out_dir / "rc_naive.c"
    so_path = out_dir / "librc_naive.so"
    render_scratch_c(rc, exe, c_path)
    t_compile_0 = time.perf_counter()
    build_scratch_so(c_path, so_path)
    compile_ms = (time.perf_counter() - t_compile_0) * 1000

    lib = load_predict(so_path)
    X_c = np.ascontiguousarray(X_te, dtype=np.float64)
    Y_c = np.zeros_like(Y_jit)

    def call_naive():
        lib.rc_predict(
            ctypes.c_int64(X_c.shape[0]),
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )

    call_naive()
    max_diff = float(np.max(np.abs(Y_c - Y_jit)))

    t_naive = time_fn(call_naive, repeats=7)
    t_jit = time_fn(jit.predict, X_te, repeats=7)

    so_size = so_path.stat().st_size

    return {
        "topology": topology.name,
        "N": N,
        "naive_ms": t_naive * 1000,
        "jit_ms": t_jit * 1000,
        "speedup": (t_naive / t_jit) if t_jit > 0 else float("nan"),
        "max_diff": max_diff,
        "naive_so_kb": so_size / 1024,
        "naive_compile_ms": compile_ms,
    }


def main() -> None:
    if shutil.which("gcc") is None:
        sys.exit("error: gcc required")
    BUILD.mkdir(parents=True, exist_ok=True)

    # Mackey-Glass-ish setup
    from examples.forecasting.mackey_glass_esn import mackey_glass
    series = mackey_glass(n=3000)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te = X[n_train:]
    input_offset = float(X_tr.mean())

    cases = []
    for topology in (Topology.ESN_STANDARD, Topology.SCR, Topology.DLR):
        for N in (100, 250, 500):
            cases.append((topology, N))

    print(f"Benchmark: naive 3-loop C (gcc -O3 -march=native) vs rclite LLVM JIT")
    print(f"Inference: T={X_te.shape[0]} samples, double-precision, K={1}, M={1}")
    print()
    header = (
        f"{'topology':<14} {'N':>5} "
        f"{'naive [ms]':>11} {'jit [ms]':>10} {'speedup':>9} "
        f"{'max |diff|':>11} {'.so [KB]':>9} {'compile':>9}"
    )
    print(header)
    print("-" * len(header))

    for topology, N in cases:
        row = run_one(topology, N, X_tr, Y_tr, X_te, input_offset)
        print(
            f"{row['topology']:<14} {row['N']:>5} "
            f"{row['naive_ms']:>11.3f} {row['jit_ms']:>10.3f} "
            f"{row['speedup']:>8.2f}x {row['max_diff']:>11.2e} "
            f"{row['naive_so_kb']:>9.1f} {row['naive_compile_ms']:>8.0f}ms"
        )


if __name__ == "__main__":
    main()
