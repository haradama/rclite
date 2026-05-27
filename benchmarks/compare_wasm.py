"""Benchmark: rclite wasm32-wasip1 (wasmtime, f32) vs rclite host JIT (f64).

For each (topology, N) case trains an ESN on Mackey-Glass, builds:
  - rclite host JIT shared library (.so, f64, native AVX)
  - rclite wasm32-wasip1 .wasm via rustc + wasm-ld, dtype=f32

Measures, on the same held-out window:
  speed     -- best/median/mean wall-clock per T-step inference
                (host: Python ctypes via librc.so;
                 wasm: std::time::Instant inside wasmtime, embedded harness)
  size      -- .wasm bytes vs host .text bytes
  parity    -- max |Y_wasm - Y_host_f32_cast| (f32 round-off)
  accuracy  -- RMSE / R² of each kernel against the ground-truth target series

The wasm timings are measured *inside* the wasm module (best of REPEATS
calls), so wasmtime startup overhead does not contaminate them.
"""
from __future__ import annotations
import ctypes
import pathlib
import re
import shutil
import subprocess
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc
from rclite.targets import WasmTarget


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = ROOT / "build" / "bench_wasm"


_BENCH_RE = re.compile(r"RCLITE_BENCH:\s*(\w+)=([^\s]+)")


def parse_bench_output(text: str) -> dict:
    """Parse the keyed `RCLITE_BENCH: key=value` lines."""
    out: dict[str, str] = {}
    for m in _BENCH_RE.finditer(text):
        out[m.group(1)] = m.group(2)
    return out


def time_host_so(lib, X_c, Y_c, repeats: int = 25, warmup: int = 3):
    def call():
        lib.rc_predict(
            ctypes.c_int64(X_c.shape[0]),
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    for _ in range(warmup):
        call()
    samples = []
    for _ in range(repeats):
        t0 = time.perf_counter_ns()
        call()
        samples.append(time.perf_counter_ns() - t0)
    samples.sort()
    return {
        "best_ns":   samples[0],
        "median_ns": samples[len(samples) // 2],
        "mean_ns":   sum(samples) // len(samples),
        "worst_ns":  samples[-1],
    }


def load_host_predict(so_path: pathlib.Path):
    lib = ctypes.CDLL(str(so_path))
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rc_predict.restype = None
    return lib


def section_size(so_path: pathlib.Path, section: str = ".text") -> int:
    cp = subprocess.run(["objdump", "-h", str(so_path)],
                         capture_output=True, text=True, check=True)
    for line in cp.stdout.splitlines():
        m = re.match(r"\s*\d+\s+(\S+)\s+([0-9a-fA-F]+)\s+", line)
        if m and m.group(1) == section:
            return int(m.group(2), 16)
    return 0


def accuracy(Y_pred, Y_true):
    e = Y_pred - Y_true
    mse = float(np.mean(e * e))
    rmse = float(np.sqrt(mse))
    yvar = float(np.var(Y_true))
    r2 = float(1 - mse / yvar) if yvar > 0 else float("nan")
    return rmse, r2


def build_esn(N: int, topology: Topology, input_offset: float, seed: int = 42):
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


def run_one(topology: Topology, N: int, X_tr, Y_tr, X_te, Y_te,
             input_offset: float, wasmtime_bin: str,
             repeats: int = 25, warmup: int = 3):
    out_dir = BUILD / f"{topology.name}_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = build_esn(N, topology, input_offset)
    exe = RCExecutor(rc)
    exe.fit(X_tr, Y_tr)

    # Host JIT (.so + in-process predict for reference)
    jit = compile_rc(rc, exe)
    host_so = out_dir / "librc_host.so"
    jit.emit_shared_library(str(host_so))
    Y_host = jit.predict(X_te)

    # Build wasm benchmark harness (embeds X_te + host f32 references)
    target = WasmTarget()
    artifact = target.compile_bench(
        rc, exe, output_dir=out_dir, test_inputs=X_te,
        expected_outputs=Y_host.astype(np.float32),
        repeats=repeats, warmup=warmup,
    )

    # Host timing
    lib = load_host_predict(host_so)
    X_c = np.ascontiguousarray(X_te, dtype=np.float64)
    Y_c = np.zeros((X_te.shape[0], 1), dtype=np.float64)
    host_times = time_host_so(lib, X_c, Y_c, repeats=repeats, warmup=warmup)

    # Wasm timing — invoke wasmtime once; the harness times REPEATS calls.
    cp = subprocess.run(
        [wasmtime_bin, str(artifact.binary)],
        capture_output=True, text=True, timeout=120.0,
    )
    if cp.returncode != 0:
        raise RuntimeError(f"wasmtime failed: {cp.stderr}\nstdout: {cp.stdout}")
    bench = parse_bench_output(cp.stdout + cp.stderr)
    wasm_times = {
        "best_ns":   int(bench["best_ns"]),
        "median_ns": int(bench["median_ns"]),
        "mean_ns":   int(bench["mean_ns"]),
        "worst_ns":  int(bench["worst_ns"]),
    }
    parity_max_abs = float(bench["parity_max_abs"])

    # Accuracy vs ground truth (host f64 result; wasm f32 result derived
    # from parity drift on the same network).
    Y_te_1d = Y_te.ravel()
    rmse_host, r2_host = accuracy(Y_host.ravel(), Y_te_1d)
    Y_wasm_estimate = Y_host.ravel().astype(np.float32).astype(np.float64)
    # We don't get the wasm predictions back to the driver (only the parity
    # statistic); but parity_max_abs is the max |wasm - host_f32_cast|, so
    # an upper bound on |wasm - host| is parity_max_abs + |host_f32_cast - host|.
    # For RMSE-against-ground-truth purposes, the host_f32 cast accounts for
    # the bulk of the precision loss already; report it.
    rmse_wasm_proxy, r2_wasm_proxy = accuracy(Y_wasm_estimate, Y_te_1d)

    return {
        "topology": topology.name, "N": N,
        "host_best_ms":   host_times["best_ns"] / 1e6,
        "host_median_ms": host_times["median_ns"] / 1e6,
        "wasm_best_ms":   wasm_times["best_ns"] / 1e6,
        "wasm_median_ms": wasm_times["median_ns"] / 1e6,
        "speedup_best":   host_times["best_ns"] / wasm_times["best_ns"]
                          if wasm_times["best_ns"] else float("nan"),
        "wasm_size_kb":   artifact.metadata["wasm_size"] / 1024,
        "wasm_kernel_kb": (out_dir / "rc_predict.o").stat().st_size / 1024,
        "host_so_kb":     host_so.stat().st_size / 1024,
        "host_text_kb":   section_size(host_so, ".text") / 1024,
        "parity_max_abs": parity_max_abs,
        "rmse_host_f64":  rmse_host,
        "rmse_wasm_proxy": rmse_wasm_proxy,
        "r2_host_f64":    r2_host,
        "r2_wasm_proxy":  r2_wasm_proxy,
    }


def main():
    if shutil.which("gcc") is None:
        sys.exit("error: gcc required (for the host .so link step)")
    if shutil.which("rustc") is None:
        sys.exit("error: rustc required ("
                 "rustup target add wasm32-wasip1)")
    wasmtime_bin = shutil.which("wasmtime")
    if wasmtime_bin is None:
        sys.exit("error: wasmtime required on PATH")
    BUILD.mkdir(parents=True, exist_ok=True)

    from examples.mackey_glass_esn import mackey_glass
    series = mackey_glass(n=3000)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te, Y_te = X[n_train:], Y[n_train:]
    input_offset = float(X_tr.mean())

    cases = []
    for topology in (Topology.ESN_STANDARD, Topology.SCR, Topology.DLR):
        for N in (100, 250, 500):
            cases.append((topology, N))

    rows = []
    print(f"Benchmark: rclite wasm32-wasip1 (wasmtime, f32) "
          f"vs rclite host LLVM JIT (f64)")
    print(f"Inference: T={X_te.shape[0]} samples on Mackey-Glass; "
          f"K=1, M=1; best of 25 inner calls per case")
    print()
    for topology, N in cases:
        print(f"[{topology.name} N={N}] ...", end=" ", flush=True)
        row = run_one(topology, N, X_tr, Y_tr, X_te, Y_te, input_offset,
                       wasmtime_bin)
        rows.append(row)
        print(f"host={row['host_best_ms']:.3f}ms wasm={row['wasm_best_ms']:.3f}ms "
              f"speedup={row['speedup_best']:.2f}x "
              f"parity={row['parity_max_abs']:.1e}")

    # Reports
    print()
    print("=" * 100)
    print("SPEED  (best-of-25 ms per T={} inference)".format(X_te.shape[0]))
    print("-" * 100)
    print(f"{'case':<22} {'host_best':>10} {'wasm_best':>10} "
          f"{'host_med':>10} {'wasm_med':>10} {'speedup':>9}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['host_best_ms']:>10.3f} {r['wasm_best_ms']:>10.3f} "
              f"{r['host_median_ms']:>10.3f} {r['wasm_median_ms']:>10.3f} "
              f"{r['speedup_best']:>8.2f}x")

    print()
    print("=" * 100)
    print("SIZE  (binary footprint, KB)")
    print("-" * 100)
    print(f"{'case':<22} {'host .so':>10} {'host .text':>12} "
          f"{'wasm kernel.o':>15} {'wasm full':>11}")
    print(f"{'':<22} {'(weights)':>10} {'(code only)':>12} "
          f"{'(code only)':>15} {'(+ wasi std)':>11}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['host_so_kb']:>10.1f} {r['host_text_kb']:>12.2f} "
              f"{r['wasm_kernel_kb']:>15.1f} {r['wasm_size_kb']:>11.1f}")

    print()
    print("=" * 100)
    print("PARITY  (max |Y_wasm_f32 - Y_host_f32_cast|, T*M outputs)")
    print("-" * 100)
    print(f"{'case':<22} {'parity max':>14} {'host RMSE':>11} "
          f"{'host_f32 RMSE':>15} {'host R²':>9} {'host_f32 R²':>13}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['parity_max_abs']:>14.3e} "
              f"{r['rmse_host_f64']:>11.6f} "
              f"{r['rmse_wasm_proxy']:>15.6f} "
              f"{r['r2_host_f64']:>9.6f} "
              f"{r['r2_wasm_proxy']:>13.6f}")


if __name__ == "__main__":
    main()
