"""WebAssembly (wasm32-wasip1) performance bench, unified schema.

Matrix: dtype in {float (f32), i8, i16, i32} x kernel in {dense, csr,
value-spec unroll}, all via the LLVM cross-compile path. Speed = wasmtime
fuel per step (a DETERMINISTIC op-count proxy), measured two-point (run with
REPEATS = R1 and R2; the fuel difference cancels WASI startup / parity /
argv overhead). Columns are shared with the Cortex-M0 and AVR benches
(benchmarks/_perf_schema.py); cells this target cannot measure are blank.

Requires rustc (+ `rustup target add wasm32-wasip1`) and the `wasmtime`
Python package.

    .venv/bin/python benchmarks/wasm_target/bench_wasm.py [--json o.json] [--md o.md]
"""

from __future__ import annotations
import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from rclite.targets import WasmTarget
import _perf_kernels as K
import _perf_schema as S

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "bench_fuel.rs"
T_SEQ = 64
R1, R2 = 5, 25  # two-point repeat counts (diff = 20)


def _lit(arr, dtype):
    if dtype == "float":
        return ", ".join(repr(float(v)) for v in arr.ravel())
    return ", ".join(str(int(v)) for v in arr.ravel())


def _render(dtype, qm_or_rcexe, x_seq, workdir):
    X, Y, eps, npd, Kk, M, T = K.reference_data(dtype, qm_or_rcexe, x_seq)
    storage = "f32" if dtype == "float" else dtype
    src = (
        HARNESS.read_text()
        .replace("@@T@@", str(T))
        .replace("@@K@@", str(Kk))
        .replace("@@M@@", str(M))
        .replace("@@STORAGE_T@@", storage)
        .replace("@@EPS@@", repr(float(eps)))
        .replace("@@X_VALUES@@", _lit(X, dtype))
        .replace("@@Y_VALUES@@", _lit(Y, dtype))
    )
    p = workdir / "bench_fuel.rs"
    p.write_text(src)
    return p, T


def _measure_fuel(wasm_path, reps, stdout_path):
    import wasmtime

    cfg = wasmtime.Config()
    cfg.consume_fuel = True
    engine = wasmtime.Engine(cfg)
    store = wasmtime.Store(engine)
    INIT = 10**16
    store.set_fuel(INIT)
    wasi = wasmtime.WasiConfig()
    wasi.argv = ["bench", str(reps)]
    wasi.stdout_file = str(stdout_path)
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    module = wasmtime.Module.from_file(engine, str(wasm_path))
    inst = linker.instantiate(store, module)
    try:
        inst.exports(store)["_start"](store)
    except wasmtime.ExitTrap as e:
        if e.code != 0:
            raise
    used = INIT - store.get_fuel()
    txt = pathlib.Path(stdout_path).read_text()
    parity = (
        "OK"
        if "parity=OK" in txt
        else ("FAIL" if "parity=FAIL" in txt else "NA")
    )
    return used, parity


def _build_and_run(target, dtype, qm_or_rcexe, x_seq, sparse, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    rc_o = K.build_object(
        dtype,
        qm_or_rcexe,
        sparse,
        triple="wasm32-wasip1",
        cpu="",
        out_path=workdir / "rc_predict.o",
    )
    main_rs, T = _render(dtype, qm_or_rcexe, x_seq, workdir)
    wasm = workdir / "bench.wasm"
    target._link_rustc(main_rs, rc_o, wasm)
    f1, _ = _measure_fuel(wasm, R1, workdir / "o1.txt")
    f2, parity = _measure_fuel(wasm, R2, workdir / "o2.txt")
    fuel_per_step = round((f2 - f1) / ((R2 - R1) * T))
    return wasm.stat().st_size, fuel_per_step, parity == "OK"


def run(sizes):
    target = WasmTarget()
    target._check_toolchain()
    rows = []
    for units, density in sizes:
        rc, exe, x_seq, y_true, x_cal = K.train_model(units, density, T_SEQ)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        qms = {
            b: K.quant_model(b, rc, exe, x_cal) for b in ("i8", "i16", "i32")
        }
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for dtype in S.DTYPES:
                src = (rc, exe) if dtype == "float" else qms[dtype]
                mse = K.accuracy_mse(dtype, src, x_seq, y_true)
                for kernel in S.KERNELS:
                    wd = td / f"{dtype}_{kernel}"
                    wb, fps, par = _build_and_run(
                        target, dtype, src, x_seq, K.KERNEL_SPARSE[kernel], wd
                    )
                    rows.append(
                        S.row(
                            N=N,
                            density=density,
                            nnz=nnz,
                            dtype=dtype,
                            kernel=kernel,
                            ops_per_step=fps,
                            parity=par,
                            wasm_B=wb,
                            mse=mse,
                            wres_B=K.wres_bytes(
                                dtype, src, K.KERNEL_SPARSE[kernel], N
                            ),
                        )
                    )
    return rows


TARGET = "WebAssembly (wasm32-wasip1) — float, affine i8/i16, symmetric i32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    try:
        import wasmtime  # noqa: F401
    except ImportError:
        print("Need the `wasmtime` Python package. Aborting.")
        return 1

    sizes = [(64, 0.1)] if args.quick else [(64, 0.1), (128, 0.1)]
    rows = run(sizes)
    print(S.fmt_text(TARGET, rows, unit="wasmtime fuel"))

    if args.md:
        args.md.write_text(
            S.fmt_md(
                TARGET,
                rows,
                unit="wasmtime fuel",
                note="Quant scheme: i8/i16 = affine (calibrated), i32 = symmetric "
                "(affine i32 overflows the i64 requantize). wasm B = full "
                "module bytes (dominated by the Rust std/WASI runtime; only "
                "variant-to-variant deltas reflect the kernel).",
            )
        )
        print(f"\nwrote {args.md}")
    ok = S.all_parity_ok(rows)
    if not ok:
        print("\nERROR: a variant failed parity.")
    if args.json:
        args.json.write_text(
            json.dumps(dict(target="wasm32-wasip1", rows=rows), indent=2)
        )
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
