"""Benchmark: rclite wasm32 (SIMD vs scalar) vs rclite host JIT.

For each (topology, N) case trains an ESN on Mackey-Glass, then builds
three kernels backed by the same trained weights:

  host       -- rclite LLVM JIT, f64, native AVX (Linux .so via ctypes)
  wasm_scl   -- wasm32-wasip1, f32, scalar (no +simd128, no vectorization)
  wasm_simd  -- wasm32-wasip1, f32, +simd128 + LLVM auto-vectorization

Measurements per case:
  speed      -- best/median ms per T-step inference. Wasm timings are
                taken INSIDE the wasm module via std::time::Instant, so
                wasmtime startup overhead does not contaminate them.
  size       -- bare kernel object bytes (rc_predict.o) and full .wasm
  parity     -- max |Y_wasm_f32 - Y_host_f32_cast|, RMSE
  v128 count -- number of v128 ops in the SIMD-build assembly (sanity check)
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
_V128_RE = re.compile(r"\b(?:v128\.\w+|f32x4\.\w+|i32x4\.\w+|i16x8\.\w+|i8x16\.\w+)\b")


def parse_bench_output(text: str) -> dict:
    out: dict[str, str] = {}
    for m in _BENCH_RE.finditer(text):
        out[m.group(1)] = m.group(2)
    return out


def count_v128_ops(asm_path: pathlib.Path) -> int:
    if not asm_path.exists():
        return 0
    return len(_V128_RE.findall(asm_path.read_text()))


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


def run_wasm(wasmtime_bin: str, wasm_path: pathlib.Path,
              enable_simd: bool) -> dict:
    cmd = [wasmtime_bin]
    if enable_simd:
        cmd.extend(["-W", "simd=y"])
    cmd.append(str(wasm_path))
    cp = subprocess.run(cmd, capture_output=True, text=True, timeout=180.0)
    if cp.returncode != 0:
        raise RuntimeError(
            f"wasmtime failed on {wasm_path.name}:\n"
            f"  stderr: {cp.stderr}\n  stdout: {cp.stdout}"
        )
    return parse_bench_output(cp.stdout + cp.stderr)


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


def run_one(topology: Topology, N: int, X_tr, Y_tr, X_te,
             input_offset: float, wasmtime_bin: str,
             repeats: int = 25, warmup: int = 3):
    out_dir = BUILD / f"{topology.name}_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = build_esn(N, topology, input_offset)
    exe = RCExecutor(rc)
    exe.fit(X_tr, Y_tr)

    # Host JIT — produces the f64 reference predictions.
    jit = compile_rc(rc, exe)
    host_so = out_dir / "librc_host.so"
    jit.emit_shared_library(str(host_so))
    Y_host = jit.predict(X_te)
    Y_host_f32 = Y_host.astype(np.float32)

    # Build the two wasm flavors in separate subdirs so their
    # rc_predict.o / .s / .wasm don't clobber each other.
    scl_dir = out_dir / "scalar"
    simd_dir = out_dir / "simd"
    art_scl = WasmTarget(simd=False).compile_bench(
        rc, exe, output_dir=scl_dir, test_inputs=X_te,
        expected_outputs=Y_host_f32, repeats=repeats, warmup=warmup,
    )
    art_simd = WasmTarget(simd=True).compile_bench(
        rc, exe, output_dir=simd_dir, test_inputs=X_te,
        expected_outputs=Y_host_f32, repeats=repeats, warmup=warmup,
    )

    # Host timing
    lib = load_host_predict(host_so)
    X_c = np.ascontiguousarray(X_te, dtype=np.float64)
    Y_c = np.zeros((X_te.shape[0], 1), dtype=np.float64)
    host_t = time_host_so(lib, X_c, Y_c, repeats=repeats, warmup=warmup)

    # Wasm timing (inside the wasm; wasmtime invoked once per flavor)
    b_scl = run_wasm(wasmtime_bin, art_scl.binary, enable_simd=False)
    b_simd = run_wasm(wasmtime_bin, art_simd.binary, enable_simd=True)

    return {
        "topology": topology.name, "N": N,
        "host_best_ms": host_t["best_ns"] / 1e6,
        "host_med_ms":  host_t["median_ns"] / 1e6,
        "scl_best_ms":  int(b_scl["best_ns"]) / 1e6,
        "scl_med_ms":   int(b_scl["median_ns"]) / 1e6,
        "simd_best_ms": int(b_simd["best_ns"]) / 1e6,
        "simd_med_ms":  int(b_simd["median_ns"]) / 1e6,
        "simd_speedup_vs_scl": (int(b_scl["best_ns"]) / int(b_simd["best_ns"])
                                  if int(b_simd["best_ns"]) > 0 else float("nan")),
        "simd_speedup_vs_host": (host_t["best_ns"] / int(b_simd["best_ns"])
                                   if int(b_simd["best_ns"]) > 0 else float("nan")),
        "kernel_scl_kb":  (scl_dir / "rc_predict.o").stat().st_size / 1024,
        "kernel_simd_kb": (simd_dir / "rc_predict.o").stat().st_size / 1024,
        "wasm_scl_kb":    art_scl.metadata["wasm_size"] / 1024,
        "wasm_simd_kb":   art_simd.metadata["wasm_size"] / 1024,
        "host_so_kb":     host_so.stat().st_size / 1024,
        "parity_scl":     float(b_scl["parity_max_abs"]),
        "parity_simd":    float(b_simd["parity_max_abs"]),
        "v128_ops":       count_v128_ops(simd_dir / "rc_predict.s"),
    }


def main():
    if shutil.which("gcc") is None:
        sys.exit("error: gcc required (host .so link step)")
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
    X_te = X[n_train:]
    input_offset = float(X_tr.mean())

    cases = []
    for topology in (Topology.ESN_STANDARD, Topology.SCR, Topology.DLR):
        for N in (100, 250, 500):
            cases.append((topology, N))

    rows = []
    print(f"Benchmark: wasm32-wasip1 SIMD vs scalar vs host JIT (f64 LLVM)")
    print(f"Inference: T={X_te.shape[0]} on Mackey-Glass; K=1, M=1; "
          f"best-of-25 inner calls per case")
    print()
    for topology, N in cases:
        print(f"[{topology.name} N={N}] ...", end=" ", flush=True)
        row = run_one(topology, N, X_tr, Y_tr, X_te, input_offset, wasmtime_bin)
        rows.append(row)
        print(f"host={row['host_best_ms']:.2f}ms "
              f"scl={row['scl_best_ms']:.2f}ms "
              f"simd={row['simd_best_ms']:.2f}ms "
              f"simd/scl={row['simd_speedup_vs_scl']:.2f}x "
              f"v128={row['v128_ops']}")

    print()
    print("=" * 100)
    print(f"SPEED  (best-of-25 ms per T={X_te.shape[0]} inference)")
    print("-" * 100)
    print(f"{'case':<22} {'host':>10} {'wasm scl':>10} {'wasm SIMD':>11} "
          f"{'SIMD/scl':>10} {'SIMD/host':>11}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['host_best_ms']:>10.3f} {r['scl_best_ms']:>10.3f} "
              f"{r['simd_best_ms']:>11.3f} "
              f"{r['simd_speedup_vs_scl']:>9.2f}x "
              f"{r['simd_speedup_vs_host']:>10.2f}x")

    print()
    print("=" * 100)
    print("SIZE  (KB)")
    print("-" * 100)
    print(f"{'case':<22} {'host .so':>10} {'scl kern':>10} {'simd kern':>11} "
          f"{'scl .wasm':>11} {'simd .wasm':>12}")
    print(f"{'':<22} {'(weights)':>10} {'.o':>10} {'.o':>11} "
          f"{'+ wasi std':>11} {'+ wasi std':>12}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['host_so_kb']:>10.1f} "
              f"{r['kernel_scl_kb']:>10.1f} {r['kernel_simd_kb']:>11.1f} "
              f"{r['wasm_scl_kb']:>11.1f} {r['wasm_simd_kb']:>12.1f}")

    print()
    print("=" * 100)
    print("PARITY  (max |Y_wasm_f32 - Y_host_f32_cast| over T*M outputs)")
    print("-" * 100)
    print(f"{'case':<22} {'wasm scl':>14} {'wasm SIMD':>14} "
          f"{'v128 ops in .s':>18}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['parity_scl']:>14.3e} {r['parity_simd']:>14.3e} "
              f"{r['v128_ops']:>18d}")


if __name__ == "__main__":
    main()
