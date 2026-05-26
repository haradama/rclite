"""4-way benchmark: float vs QAT-quantized × naive C vs rclite.

For each (topology, N) case, builds and measures four kernels:

    naive_f   — hand-written 3-loop double-precision C    (libm tanh)
    rclite_f  — rclite LLVM JIT float                     (libm tanh)
    naive_q   — hand-written 3-loop i32 fixed-point C     (LUT tanh)
    rclite_q  — rclite LLVM JIT i32 fixed-point           (LUT tanh)

The quantized pair share weights from a QAT search (W_out refit on the
quantized state trajectory, mirage-style), so the two C ABI shapes are
the same and instruction counts are comparable.

Metrics: wall-clock speed, .text & rc_predict sizes, dynamic instruction
count (callgrind), RMSE / R² against ground-truth Mackey-Glass, and
parity between the two backends within each precision class.
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
from rclite.codegen.llvm import emit_quantized_module, _ensure_initialized
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I32FixedPoint,
    quantize_model, QuantizedExecutor, search_quantization,
)
import llvmlite.binding as llvm


HERE = pathlib.Path(__file__).resolve().parent
ROOT = HERE.parent
BUILD = ROOT / "build" / "bench_4way"
TPL_F = HERE / "scratch_c" / "rc_naive_template.c"
TPL_Q = HERE / "scratch_c" / "rc_naive_q_template.c"
DRIVER_F = HERE / "scratch_c" / "driver.c"
DRIVER_Q = HERE / "scratch_c" / "driver_q.c"


# ---------------------------------------------------------------- C emit


def _fmt_doubles(arr):
    flat = np.ascontiguousarray(arr, dtype=np.float64).ravel()
    return ", ".join(f"{v:.17g}" for v in flat)


def _fmt_int32(arr):
    flat = np.asarray(arr).astype(np.int64).ravel()
    return ", ".join(str(int(v)) for v in flat)


def render_naive_f_c(rc, exe, out_path):
    K, N, M = rc.input.units, rc.reservoir.units, rc.readout.units
    F = exe._feature_dim()
    text = (
        TPL_F.read_text()
        .replace("@@N@@", str(N)).replace("@@K@@", str(K))
        .replace("@@M@@", str(M)).replace("@@F@@", str(F))
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
    out_path.write_text(text)


def render_naive_q_c(qmodel, out_path):
    rc = qmodel.rc
    cfg = qmodel.config
    K, N, M, F = qmodel.K, qmodel.N, qmodel.M, qmodel.F
    leak_q = qmodel.target.quantize_state(rc.reservoir.leak_rate, cfg)
    one_minus_leak_q = (1 << cfg.state_frac) - leak_q
    bias_q = qmodel.target.quantize_state(rc.reservoir.bias, cfg)
    lut_xmin_q = int(qmodel.lut.xmin * cfg.state_scale)
    lut_xmax_q = int(qmodel.lut.xmax * cfg.state_scale)
    text = (
        TPL_Q.read_text()
        .replace("@@N@@", str(N)).replace("@@K@@", str(K))
        .replace("@@M@@", str(M)).replace("@@F@@", str(F))
        .replace("@@STATE_FRAC@@", str(cfg.state_frac))
        .replace("@@INPUT_FRAC@@", str(cfg.input_frac))
        .replace("@@WEIGHT_FRAC@@", str(cfg.weight_frac))
        .replace("@@LEAK_Q@@", str(int(leak_q)))
        .replace("@@ONE_MINUS_LEAK_Q@@", str(int(one_minus_leak_q)))
        .replace("@@BIAS_Q@@", str(int(bias_q)))
        .replace("@@INCLUDE_BIAS@@", "1" if rc.readout.include_bias else "0")
        .replace("@@INCLUDE_INPUT@@", "1" if rc.readout.include_input else "0")
        .replace("@@LUT_N@@", str(qmodel.lut.n))
        .replace("@@LUT_XMIN_Q@@", str(lut_xmin_q))
        .replace("@@LUT_XMAX_Q@@", str(lut_xmax_q))
        .replace("@@W_IN_VALUES_Q@@", _fmt_int32(qmodel.W_in_q))
        .replace("@@W_RES_VALUES_Q@@", _fmt_int32(qmodel.W_res_q))
        .replace("@@W_OUT_VALUES_Q@@", _fmt_int32(qmodel.W_out_q))
        .replace("@@LUT_TABLE_Q@@", _fmt_int32(qmodel.lut_table_q))
    )
    out_path.write_text(text)


def gcc_build(c_path, so_path, libs=("-lm",), cc="gcc"):
    cmd = [cc, "-O3", "-march=x86-64-v3", "-shared", "-fPIC",
           str(c_path), "-o", str(so_path), *libs]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"build failed:\n{cp.stderr}")


def build_drivers(out_dir, cc="gcc"):
    df, dq = out_dir / "driver_f", out_dir / "driver_q"
    subprocess.run([cc, "-O2", str(DRIVER_F), "-o", str(df), "-ldl", "-lm"],
                   check=True, capture_output=True)
    subprocess.run([cc, "-O2", str(DRIVER_Q), "-o", str(dq), "-ldl", "-lm"],
                   check=True, capture_output=True)
    return df, dq


def emit_rclite_q_so(qmodel, so_path):
    """Cross-link a rclite quantized module to a PIC shared library."""
    _ensure_initialized()
    mod_text = str(emit_quantized_module(qmodel))
    mod = llvm.parse_assembly(mod_text)
    mod.verify()
    target = llvm.Target.from_triple(llvm.get_default_triple())
    tm = target.create_target_machine(opt=3, reloc="pic")
    # Optimize
    pto = llvm.create_pipeline_tuning_options()
    pto.speed_level = 3
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(mod, pb)

    obj_path = pathlib.Path(str(so_path) + ".o")
    with open(obj_path, "wb") as f:
        f.write(tm.emit_object(mod))
    subprocess.run(["gcc", "-shared", "-fPIC", "-o", str(so_path), str(obj_path)],
                   check=True, capture_output=True)
    obj_path.unlink()


# ---------------------------------------------------------------- measurements


def load_predict_f(so_path):
    lib = ctypes.CDLL(str(so_path))
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rc_predict.restype = None
    return lib


def load_predict_q(so_path):
    lib = ctypes.CDLL(str(so_path))
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    ]
    lib.rc_predict.restype = None
    return lib


def time_so_f(lib, X_c, Y_c, repeats=7):
    def call():
        lib.rc_predict(ctypes.c_int64(X_c.shape[0]),
                       X_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       Y_c.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); call(); dt = time.perf_counter() - t0
        if dt < best: best = dt
    return best


def time_so_q(lib, X_q, Y_q, repeats=7):
    def call():
        lib.rc_predict(ctypes.c_int64(X_q.shape[0]),
                       X_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                       Y_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter(); call(); dt = time.perf_counter() - t0
        if dt < best: best = dt
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
    cp = subprocess.run(["nm", "--print-size", "--radix=d", str(so_path)],
                         capture_output=True, text=True, check=True)
    for line in cp.stdout.splitlines():
        parts = line.split()
        if len(parts) >= 4 and parts[-1] == "rc_predict":
            return int(parts[1])
    return 0


_CG_RE = re.compile(r"^summary:\s+(\d+)", re.MULTILINE)


def callgrind_ir(driver_bin, so_path, T, n_calls=2, out_dir=BUILD):
    out_file = pathlib.Path(out_dir) / f"cg_{so_path.stem}.out"
    cmd = [
        "valgrind", "--tool=callgrind",
        "--cache-sim=no", "--branch-sim=no",
        f"--callgrind-out-file={out_file}",
        str(driver_bin), str(so_path), str(T), str(n_calls),
    ]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"callgrind failed:\n{cp.stderr[:500]}")
    m = _CG_RE.search(out_file.read_text())
    if not m:
        raise RuntimeError("callgrind output unparseable")
    return int(m.group(1)) // (1 + n_calls)


def rmse_r2(Y_pred, Y_true):
    e = Y_pred - Y_true
    mse = float(np.mean(e * e))
    rmse = float(np.sqrt(mse))
    yvar = float(np.var(Y_true))
    r2 = float(1 - mse / yvar) if yvar > 0 else float("nan")
    return rmse, r2


# ---------------------------------------------------------------- model


def build_esn(N, topology, input_offset, seed=42):
    # input_offset=0 keeps the bench fair: the scratch C naive_q template
    # takes preprocessed-and-quantized input, while rclite_q's kernel now
    # preprocesses internally. Lifting the offset would require updating the
    # C template too — out of scope for this benchmark.
    return ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=0.0, name="in"),
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
             driver_f, driver_q):
    out = BUILD / f"{topology.name}_N{N}"
    out.mkdir(parents=True, exist_ok=True)

    # === Train float ===
    rc = build_esn(N, topology, input_offset)
    exe = RCExecutor(rc)
    exe.fit(X_tr, Y_tr)

    # === Float artefacts ===
    naive_f_c = out / "rc_naive_f.c"
    naive_f_so = out / "librc_naive_f.so"
    render_naive_f_c(rc, exe, naive_f_c)
    gcc_build(naive_f_c, naive_f_so, libs=("-lm",))

    rclite_f_so = out / "librc_rclite_f.so"
    compile_rc(rc, exe).emit_shared_library(str(rclite_f_so))

    # === QAT search → quantized model ===
    qat = search_quantization(
        rc, exe, X_tr, Y_tr, X_te, Y_te,
        state_frac_range=(12, 22), lut=TanhLUTSpec(xmin=-4, xmax=4, n=256),
        target=I32FixedPoint(),
    )
    qmodel = qat.best_qmodel

    # === Quantized artefacts ===
    naive_q_c = out / "rc_naive_q.c"
    naive_q_so = out / "librc_naive_q.so"
    render_naive_q_c(qmodel, naive_q_c)
    gcc_build(naive_q_c, naive_q_so, libs=())  # no -lm: integer kernel

    rclite_q_so = out / "librc_rclite_q.so"
    emit_rclite_q_so(qmodel, rclite_q_so)

    # === Load + warmup ===
    lib_nf = load_predict_f(naive_f_so); lib_rf = load_predict_f(rclite_f_so)
    lib_nq = load_predict_q(naive_q_so); lib_rq = load_predict_q(rclite_q_so)

    X_f = np.ascontiguousarray(X_te, dtype=np.float64)
    Y_nf = np.zeros((X_te.shape[0], 1)); Y_rf = np.zeros_like(Y_nf)
    lib_nf.rc_predict(X_f.shape[0],
                       X_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       Y_nf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))
    lib_rf.rc_predict(X_f.shape[0],
                       X_f.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
                       Y_rf.ctypes.data_as(ctypes.POINTER(ctypes.c_double)))

    # Quantize input for both quantized backends (offset=0 forced above)
    u_pre = (X_te - rc.input.input_offset) * rc.input.input_scaling
    X_q = qmodel.target.quantize_input_array(u_pre, qmodel.config).astype(np.int32)
    X_q = np.ascontiguousarray(X_q)
    Y_nq = np.zeros((X_te.shape[0], 1), dtype=np.int32)
    Y_rq = np.zeros_like(Y_nq)
    lib_nq.rc_predict(X_q.shape[0],
                       X_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                       Y_nq.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    lib_rq.rc_predict(X_q.shape[0],
                       X_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                       Y_rq.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    Y_nq_f = Y_nq.astype(np.float64) / qmodel.config.state_scale
    Y_rq_f = Y_rq.astype(np.float64) / qmodel.config.state_scale

    # === Speed ===
    t_nf = time_so_f(lib_nf, X_f, Y_nf)
    t_rf = time_so_f(lib_rf, X_f, Y_rf)
    t_nq = time_so_q(lib_nq, X_q, Y_nq)
    t_rq = time_so_q(lib_rq, X_q, Y_rq)

    # === Size ===
    sizes = {
        "naive_f": (section_size(naive_f_so), predict_fn_size(naive_f_so)),
        "rclite_f": (section_size(rclite_f_so), predict_fn_size(rclite_f_so)),
        "naive_q": (section_size(naive_q_so), predict_fn_size(naive_q_so)),
        "rclite_q": (section_size(rclite_q_so), predict_fn_size(rclite_q_so)),
    }

    # === IR ===
    ir_nf = callgrind_ir(driver_f, naive_f_so, X_te.shape[0], out_dir=out)
    ir_rf = callgrind_ir(driver_f, rclite_f_so, X_te.shape[0], out_dir=out)
    ir_nq = callgrind_ir(driver_q, naive_q_so, X_te.shape[0], out_dir=out)
    ir_rq = callgrind_ir(driver_q, rclite_q_so, X_te.shape[0], out_dir=out)

    # === Accuracy vs ground truth ===
    Yt = Y_te.ravel()
    acc_nf = rmse_r2(Y_nf.ravel(), Yt)
    acc_rf = rmse_r2(Y_rf.ravel(), Yt)
    acc_nq = rmse_r2(Y_nq_f.ravel(), Yt)
    acc_rq = rmse_r2(Y_rq_f.ravel(), Yt)

    return {
        "topology": topology.name, "N": N,
        "state_frac": qmodel.config.state_frac,
        "t": {"naive_f": t_nf * 1000, "rclite_f": t_rf * 1000,
              "naive_q": t_nq * 1000, "rclite_q": t_rq * 1000},
        "size": sizes,
        "ir": {"naive_f": ir_nf, "rclite_f": ir_rf,
               "naive_q": ir_nq, "rclite_q": ir_rq},
        "acc": {"naive_f": acc_nf, "rclite_f": acc_rf,
                "naive_q": acc_nq, "rclite_q": acc_rq},
        "parity_f": float(np.max(np.abs(Y_nf - Y_rf))),
        "parity_q": int(np.max(np.abs(Y_nq.astype(np.int64) - Y_rq.astype(np.int64)))),
    }


def main():
    if shutil.which("gcc") is None: sys.exit("error: gcc required")
    if shutil.which("valgrind") is None: sys.exit("error: valgrind required")
    BUILD.mkdir(parents=True, exist_ok=True)
    driver_f, driver_q = build_drivers(BUILD)

    from examples.mackey_glass_esn import mackey_glass
    series = mackey_glass(n=3000)
    X, Y = series[:-1, None], series[1:, None]
    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te, Y_te = X[n_train:], Y[n_train:]
    input_offset = float(X_tr.mean())

    # 4-way × callgrind is expensive. Trim to representative cases.
    cases = []
    for topology in (Topology.ESN_STANDARD, Topology.SCR):
        for N in (100, 250):
            cases.append((topology, N))

    rows = []
    for topology, N in cases:
        print(f"[{topology.name} N={N}] ...", end=" ", flush=True)
        row = run_one(topology, N, X_tr, Y_tr, X_te, Y_te, input_offset,
                       driver_f, driver_q)
        rows.append(row)
        t = row["t"]
        print(f"nf={t['naive_f']:.1f}ms rf={t['rclite_f']:.1f}ms "
              f"nq={t['naive_q']:.1f}ms rq={t['rclite_q']:.1f}ms "
              f"sf=Q.{row['state_frac']} pq={row['parity_q']}")

    # ============= REPORT =============

    def label(r): return f"{r['topology']} N={r['N']}"

    print()
    print("=" * 110)
    print("SPEED (ms per inference of T=999 samples; best of 7)")
    print("-" * 110)
    print(f"{'case':<22} {'naive_f':>10} {'rclite_f':>10} "
          f"{'naive_q':>10} {'rclite_q':>10} "
          f"{'rf/nf':>7} {'rq/nq':>7} {'rq/rf':>7}")
    for r in rows:
        t = r["t"]
        print(f"{label(r):<22} "
              f"{t['naive_f']:>10.2f} {t['rclite_f']:>10.2f} "
              f"{t['naive_q']:>10.2f} {t['rclite_q']:>10.2f} "
              f"{t['naive_f']/t['rclite_f']:>6.2f}x "
              f"{t['naive_q']/t['rclite_q']:>6.2f}x "
              f"{t['rclite_f']/t['rclite_q']:>6.2f}x")

    print()
    print("=" * 110)
    print("rc_predict FUNCTION SIZE (bytes)")
    print("-" * 110)
    print(f"{'case':<22} {'naive_f':>10} {'rclite_f':>10} "
          f"{'naive_q':>10} {'rclite_q':>10}")
    for r in rows:
        s = r["size"]
        print(f"{label(r):<22} "
              f"{s['naive_f'][1]:>10,} {s['rclite_f'][1]:>10,} "
              f"{s['naive_q'][1]:>10,} {s['rclite_q'][1]:>10,}")

    print()
    print("=" * 110)
    print("DYNAMIC INSTRUCTIONS PER INFERENCE (callgrind Ir, T=999)")
    print("-" * 110)
    print(f"{'case':<22} {'naive_f':>14} {'rclite_f':>14} "
          f"{'naive_q':>14} {'rclite_q':>14} {'rq/rf':>7}")
    for r in rows:
        ir = r["ir"]
        print(f"{label(r):<22} "
              f"{ir['naive_f']:>14,} {ir['rclite_f']:>14,} "
              f"{ir['naive_q']:>14,} {ir['rclite_q']:>14,} "
              f"{ir['rclite_q']/ir['rclite_f']:>6.2f}x")

    print()
    print("=" * 110)
    print("ACCURACY vs ground truth (RMSE, R²)")
    print("-" * 110)
    print(f"{'case':<22} {'naive_f':>21} {'rclite_f':>21} "
          f"{'naive_q':>21} {'rclite_q':>21}")
    for r in rows:
        a = r["acc"]
        def fmt(x): return f"{x[0]:.5f}/{x[1]:.4f}"
        print(f"{label(r):<22} "
              f"{fmt(a['naive_f']):>21} {fmt(a['rclite_f']):>21} "
              f"{fmt(a['naive_q']):>21} {fmt(a['rclite_q']):>21}")

    print()
    print("=" * 110)
    print("PARITY (max |Y_naive - Y_rclite| within each precision class)")
    print("-" * 110)
    print(f"{'case':<22} {'float parity':>14} {'int parity (state units)':>26}")
    for r in rows:
        print(f"{label(r):<22} {r['parity_f']:>14.2e} {r['parity_q']:>26d}")


if __name__ == "__main__":
    main()
