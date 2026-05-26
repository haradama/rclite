"""Extended benchmark: speed + size + dynamic instruction count + accuracy.

For each (topology, N) case:
  - trains an ESN, extracts weights
  - emits a naive 3-loop C kernel, gcc -O3 -march=native → librc_naive.so
  - emits an rclite JIT-compiled shared library → librc_rclite.so
  - measures:
      time   : best of N runs (wall clock, ms)
      size   : .text section + rc_predict function size (bytes)
      ir     : callgrind retired-instruction count per inference
      acc    : RMSE / R² vs ground-truth target on the held-out window
      diff   : max |Y_naive - Y_rclite| (numeric parity check)
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
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = ROOT / "build" / "bench_host_full"
TEMPLATE = HERE / "scratch_c" / "rc_naive_template.c"
DRIVER_C = HERE / "scratch_c" / "driver.c"


# ---------------------------------------------------------------- C-emission


def _fmt_doubles(arr):
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    return ", ".join(f"{v:.17g}" for v in flat)


def render_scratch_c(rc, exe, out_path):
    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()
    tmpl = TEMPLATE.read_text()
    out_path.write_text(
        tmpl
        .replace("@@N@@", str(N))
        .replace("@@K@@", str(K))
        .replace("@@M@@", str(M))
        .replace("@@F@@", str(F))
        .replace("@@LEAK@@", f"{float(rc.reservoir.leak_rate):.17g}")
        .replace("@@BIAS@@", f"{float(rc.reservoir.bias):.17g}")
        .replace("@@INPUT_OFFSET@@", f"{float(rc.input.input_offset):.17g}")
        .replace("@@INPUT_SCALING@@", f"{float(rc.input.input_scaling):.17g}")
        .replace("@@INCLUDE_BIAS@@", "1" if rc.readout.include_bias else "0")
        .replace("@@INCLUDE_INPUT@@", "1" if rc.readout.include_input else "0")
        .replace("@@W_IN_VALUES@@", _fmt_doubles(exe.W_in))
        .replace("@@W_RES_VALUES@@", _fmt_doubles(exe.W_res))
        .replace("@@W_OUT_VALUES@@", _fmt_doubles(exe.W_out))
    )


def build_so_naive(c_path, so_path, cc="gcc"):
    # `-march=x86-64-v3` = AVX2/FMA/BMI baseline (no AVX-512). This keeps
    # the instruction set comparable with rclite's MCJIT default and stays
    # within Valgrind 3.22's supported opcode set.
    cmd = [cc, "-O3", "-march=x86-64-v3", "-shared", "-fPIC",
           str(c_path), "-o", str(so_path), "-lm"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"build failed:\n{cp.stderr}")


def build_driver(driver_bin, cc="gcc"):
    cmd = [cc, "-O2", str(DRIVER_C), "-o", str(driver_bin), "-ldl", "-lm"]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"driver build failed:\n{cp.stderr}")


# ---------------------------------------------------------------- measurements


def load_predict(so_path):
    lib = ctypes.CDLL(str(so_path))
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rc_predict.restype = None
    return lib


def time_so(lib, X_c, Y_c, repeats=7):
    def call():
        lib.rc_predict(
            ctypes.c_int64(X_c.shape[0]),
            X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        call()
        dt = time.perf_counter() - t0
        if dt < best:
            best = dt
    return best


def section_size(so_path, section=".text"):
    cp = subprocess.run(["objdump", "-h", str(so_path)],
                         capture_output=True, text=True, check=True)
    for line in cp.stdout.splitlines():
        m = re.match(r"\s*\d+\s+(\S+)\s+([0-9a-fA-F]+)\s+", line)
        if m and m.group(1) == section:
            return int(m.group(2), 16)
    return 0


def predict_fn_size(so_path):
    """Size of the rc_predict function as reported by `nm --print-size`."""
    cp = subprocess.run(["nm", "--print-size", "--radix=d", str(so_path)],
                         capture_output=True, text=True, check=True)
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == "rc_predict":
            return int(parts[1])
    return 0


def static_instr_count(so_path, symbol="rc_predict"):
    """Count instructions in the disassembled `symbol` (rough static count)."""
    cp = subprocess.run(
        ["objdump", "-d", f"--disassemble={symbol}", str(so_path)],
        capture_output=True, text=True,
    )
    if cp.returncode != 0:
        return 0
    count = 0
    in_fn = False
    for line in cp.stdout.splitlines():
        if line.startswith("0") and f"<{symbol}>:" in line:
            in_fn = True
            continue
        if not in_fn:
            continue
        if line.endswith(":"):
            in_fn = False
            continue
        if re.match(r"^\s+[0-9a-f]+:\s+[0-9a-f]", line):
            count += 1
    return count


_CALLGRIND_TOTAL_RE = re.compile(r"^summary:\s+(\d+)", re.MULTILINE)


def callgrind_instructions(driver_bin, so_path, T, n_calls=5, out_dir=None):
    """Run driver under callgrind. Returns dynamic instructions per inference."""
    out_dir = pathlib.Path(out_dir or BUILD)
    out_file = out_dir / "callgrind.out"
    cmd = [
        "valgrind", "--tool=callgrind",
        "--cache-sim=no", "--branch-sim=no",
        f"--callgrind-out-file={out_file}",
        str(driver_bin), str(so_path), str(T), str(n_calls),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"callgrind failed:\n{cp.stderr}")
    text = out_file.read_text()
    m = _CALLGRIND_TOTAL_RE.search(text)
    if not m:
        raise RuntimeError("could not parse callgrind output")
    total = int(m.group(1))
    # The 1 warmup + n_calls measured calls; total / (1 + n_calls) per call.
    # The dlopen / init overhead is small (~hundreds of K) compared to predict.
    per_call = total // (1 + n_calls)
    return per_call


def accuracy(Y_pred, Y_true):
    e = Y_pred - Y_true
    mse = float(np.mean(e * e))
    rmse = float(np.sqrt(mse))
    yvar = float(np.var(Y_true))
    r2 = float(1 - mse / yvar) if yvar > 0 else float("nan")
    return mse, rmse, r2


# ---------------------------------------------------------------- model


def build_esn(N, topology, input_offset, seed=42):
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


def run_one(topology, N, X_tr, Y_tr, X_te, Y_te, input_offset,
             driver_bin, want_ir=True):
    out_dir = BUILD / f"{topology.name}_N{N}"
    out_dir.mkdir(parents=True, exist_ok=True)

    rc = build_esn(N, topology, input_offset)
    exe = RCExecutor(rc)
    exe.fit(X_tr, Y_tr)

    # Build naive C → .so
    naive_c = out_dir / "rc_naive.c"
    naive_so = out_dir / "librc_naive.so"
    render_scratch_c(rc, exe, naive_c)
    build_so_naive(naive_c, naive_so)

    # Build rclite JIT → .so via emit_shared_library
    jit = compile_rc(rc, exe)
    rclite_so = out_dir / "librc_rclite.so"
    jit.emit_shared_library(str(rclite_so))

    # Load both
    lib_n = load_predict(naive_so)
    lib_r = load_predict(rclite_so)

    X_c = np.ascontiguousarray(X_te, dtype=np.float64)
    Y_n = np.zeros((X_te.shape[0], 1), dtype=np.float64)
    Y_r = np.zeros_like(Y_n)
    lib_n.rc_predict(X_c.shape[0],
                       X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       Y_n.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    lib_r.rc_predict(X_c.shape[0],
                       X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       Y_r.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    parity = float(np.max(np.abs(Y_n - Y_r)))

    # Speed
    t_n = time_so(lib_n, X_c, Y_n)
    t_r = time_so(lib_r, X_c, Y_r)

    # Size
    text_n = section_size(naive_so, ".text")
    text_r = section_size(rclite_so, ".text")
    fn_n = predict_fn_size(naive_so)
    fn_r = predict_fn_size(rclite_so)
    insns_static_n = static_instr_count(naive_so)
    insns_static_r = static_instr_count(rclite_so)

    # Accuracy vs ground truth
    Y_te_1d = Y_te.ravel()
    mse_n, rmse_n, r2_n = accuracy(Y_n.ravel(), Y_te_1d)
    mse_r, rmse_r, r2_r = accuracy(Y_r.ravel(), Y_te_1d)

    # Dynamic instruction count (callgrind)
    ir_n = ir_r = None
    if want_ir:
        ir_n = callgrind_instructions(driver_bin, naive_so, X_te.shape[0],
                                        n_calls=2, out_dir=out_dir)
        ir_r = callgrind_instructions(driver_bin, rclite_so, X_te.shape[0],
                                        n_calls=2, out_dir=out_dir)

    return {
        "topology": topology.name, "N": N,
        "t_naive_ms": t_n * 1000, "t_rclite_ms": t_r * 1000,
        "speedup": t_n / t_r if t_r > 0 else float("nan"),
        "text_naive_b": text_n, "text_rclite_b": text_r,
        "fn_naive_b": fn_n, "fn_rclite_b": fn_r,
        "insns_static_naive": insns_static_n,
        "insns_static_rclite": insns_static_r,
        "ir_naive": ir_n, "ir_rclite": ir_r,
        "rmse_naive": rmse_n, "rmse_rclite": rmse_r,
        "r2_naive": r2_n, "r2_rclite": r2_r,
        "parity_max_diff": parity,
    }


def main():
    if shutil.which("gcc") is None:
        sys.exit("error: gcc required")
    if shutil.which("valgrind") is None:
        sys.exit("error: valgrind required (sudo apt install valgrind)")
    BUILD.mkdir(parents=True, exist_ok=True)

    driver_bin = BUILD / "driver"
    build_driver(driver_bin)

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
    for topology, N in cases:
        print(f"[{topology.name} N={N}] ...", end=" ", flush=True)
        row = run_one(topology, N, X_tr, Y_tr, X_te, Y_te, input_offset,
                       driver_bin)
        rows.append(row)
        print(f"naive={row['t_naive_ms']:.1f}ms rclite={row['t_rclite_ms']:.1f}ms"
              f" speedup={row['speedup']:.2f}x parity={row['parity_max_diff']:.1e}")

    # Final report
    print()
    print("=" * 100)
    print("SPEED (best of 7, ms per inference of T=999 samples)")
    print("-" * 100)
    print(f"{'case':<22} {'naive':>10} {'rclite':>10} {'speedup':>9}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['t_naive_ms']:>10.3f} {r['t_rclite_ms']:>10.3f} "
              f"{r['speedup']:>8.2f}x")

    print()
    print("=" * 100)
    print("SIZE (bytes)")
    print("-" * 100)
    print(f"{'case':<22} {'naive .text':>13} {'rclite .text':>14} "
          f"{'naive fn':>11} {'rclite fn':>11} {'naive insn':>12} {'rclite insn':>12}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['text_naive_b']:>13,} {r['text_rclite_b']:>14,} "
              f"{r['fn_naive_b']:>11,} {r['fn_rclite_b']:>11,} "
              f"{r['insns_static_naive']:>12,} {r['insns_static_rclite']:>12,}")

    print()
    print("=" * 100)
    print("DYNAMIC INSTRUCTIONS PER INFERENCE (callgrind, T=999)")
    print("-" * 100)
    print(f"{'case':<22} {'naive (Ir)':>14} {'rclite (Ir)':>14} {'ratio':>9}")
    for r in rows:
        if r['ir_naive'] is None:
            continue
        ratio = r['ir_naive'] / r['ir_rclite'] if r['ir_rclite'] > 0 else float("nan")
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['ir_naive']:>14,} {r['ir_rclite']:>14,} {ratio:>8.2f}x")

    print()
    print("=" * 100)
    print("ACCURACY vs ground-truth Mackey-Glass (RMSE, R²)")
    print("-" * 100)
    print(f"{'case':<22} {'naive RMSE':>11} {'rclite RMSE':>12} "
          f"{'naive R²':>10} {'rclite R²':>11} {'parity':>10}")
    for r in rows:
        print(f"{r['topology']+' N='+str(r['N']):<22} "
              f"{r['rmse_naive']:>11.6f} {r['rmse_rclite']:>12.6f} "
              f"{r['r2_naive']:>10.6f} {r['r2_rclite']:>11.6f} "
              f"{r['parity_max_diff']:>10.1e}")


if __name__ == "__main__":
    main()
