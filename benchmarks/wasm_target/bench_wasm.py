"""WebAssembly (wasm32-wasip1) benchmark: dense vs CSR vs value-spec unroll
W_res (symmetric-quantized i8), measured by deterministic wasmtime fuel.

Wasm uses the same LLVM cross-compile path as the Cortex-M0 bench
(`emit_quantized_module` + `SparsifyReservoir`), so all three kernels —
including the +-1/+-2**k value-specialized unroll — are measured
apples-to-apples.

  * ACCURACY — each kernel is bit-exact with the host quantized kernel
    (the harness asserts max|Y - Y_ref| == 0 → parity OK).
  * SPEED    — wasmtime fuel per inference step. Fuel is a DETERMINISTIC
    op-count proxy (one unit per executed wasm operation). We run each
    module twice (REPEATS = R1, R2) and divide the fuel difference by
    (R2-R1)*T, which cancels WASI startup / parity / argv overhead exactly.
  * SIZE     — .wasm module bytes.

Requires rustc (+ `rustup target add wasm32-wasip1`), wasm-ld, and the
`wasmtime` Python package (run with the project venv).

    .venv/bin/python benchmarks/wasm_target/bench_wasm.py [--json out.json] [--md out.md]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import QuantConfig, TanhLUTSpec, I8Symmetric, quantize_model
from rclite.codegen.llvm import CompiledQuantizedRC
from rclite.targets import WasmTarget

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "bench_fuel.rs"
T_SEQ = 64                 # embedded sequence length
R1, R2 = 5, 25             # two-point repeat counts (diff = 20 reps)
VARIANTS = [("dense", None), ("csr", "csr"), ("unroll", "unroll")]


def _model(units, density, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=60,
                            include_bias=True, include_input=False, name="out"),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((400, 1)) * 0.15
    exe.fit(X[:340], np.sin(np.arange(340) * 0.1)[:, None])
    return rc, exe, X[340:340 + T_SEQ]


def _sym_qmodel(rc, exe):
    # i8 symmetric requires small fixed-point fracs (so values fit in i8).
    cfg = QuantConfig(state_frac=5, input_frac=6, weight_frac=6)
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128),
                          target=I8Symmetric())


def _render_harness(qm, x_seq, workdir):
    """Embed quantized input + host reference; return the harness .rs path."""
    cfg = qm.config
    x_in = x_seq if x_seq.ndim > 1 else x_seq[:, None]
    T, K, M = x_in.shape[0], x_in.shape[1], qm.M
    X_q = np.ascontiguousarray(
        qm.target.quantize_input_array(x_in, cfg).astype(np.int8))
    # Reference = host quantized kernel (bit-exact by construction with the
    # cross-compiled kernel), recovered to exact integers.
    y_float = CompiledQuantizedRC(qm).predict(x_in)
    if y_float.ndim == 1:
        y_float = y_float[:, None]
    Y_ref_q = np.rint(y_float * cfg.state_scale).astype(np.int8)

    tmpl = HARNESS.read_text()
    src = (tmpl
           .replace("@@T@@", str(T)).replace("@@K@@", str(K))
           .replace("@@M@@", str(M)).replace("@@STORAGE_T@@", "i8")
           .replace("@@X_VALUES_Q@@",
                    ", ".join(str(int(v)) for v in X_q.ravel()))
           .replace("@@Y_VALUES_Q@@",
                    ", ".join(str(int(v)) for v in Y_ref_q.ravel())))
    p = workdir / "bench_fuel.rs"
    p.write_text(src)
    return p, T


def _measure_fuel(wasm_path, reps, stdout_path):
    """Run the WASI module with `reps` and return (fuel_used, parity_str)."""
    import wasmtime
    cfg = wasmtime.Config()
    cfg.consume_fuel = True
    engine = wasmtime.Engine(cfg)
    store = wasmtime.Store(engine)
    INIT = 10 ** 15
    store.set_fuel(INIT)
    wasi = wasmtime.WasiConfig()
    wasi.argv = ["bench", str(reps)]
    wasi.stdout_file = str(stdout_path)
    store.set_wasi(wasi)
    linker = wasmtime.Linker(engine)
    linker.define_wasi()
    module = wasmtime.Module.from_file(engine, str(wasm_path))
    inst = linker.instantiate(store, module)
    start = inst.exports(store)["_start"]
    try:
        start(store)
    except wasmtime.ExitTrap as e:
        if e.code != 0:
            raise
    used = INIT - store.get_fuel()
    parity = "NA"
    txt = pathlib.Path(stdout_path).read_text()
    if "parity=OK" in txt:
        parity = "OK"
    elif "parity=FAIL" in txt:
        parity = "FAIL"
    return used, parity


def _build_and_run(target, qm, x_seq, sparse, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    rc_o = target._compile_quantized_object(qm, workdir, sparse=sparse)
    main_rs, T = _render_harness(qm, x_seq, workdir)
    wasm = workdir / "bench.wasm"
    target._link_rustc(main_rs, rc_o, wasm)

    f1, _ = _measure_fuel(wasm, R1, workdir / "o1.txt")
    f2, parity = _measure_fuel(wasm, R2, workdir / "o2.txt")
    fuel_per_step = (f2 - f1) / ((R2 - R1) * T)
    return wasm.stat().st_size, round(fuel_per_step), parity == "OK"


def run(sizes):
    target = WasmTarget()
    target._check_toolchain()
    results = []
    for units, density in sizes:
        rc, exe, x_seq = _model(units, density)
        qm = _sym_qmodel(rc, exe)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        variants = {}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for label, strat in VARIANTS:
                sz, fps, par = _build_and_run(target, qm, x_seq, strat,
                                              td / label)
                variants[label] = dict(wasm_bytes=sz, fuel_per_step=fps,
                                       parity=par)
        results.append(dict(N=N, density=density, nnz=nnz, variants=variants))
    return results


def _fmt_table(results):
    hdr = (f"{'N':>4} {'dens':>5} {'nnz':>6} {'variant':>7} {'wasm B':>8} "
           f"{'fuel/step':>10} {'speedup':>8} {'parity':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        base = r["variants"]["dense"]["fuel_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            f = v["fuel_per_step"]
            sp = (f"{base / f:.2f}x" if label != "dense" and f > 0 else "-")
            lines.append(
                f"{r['N']:>4} {r['density']:>5.2f} {r['nnz']:>6} {label:>7} "
                f"{v['wasm_bytes']:>8} {f:>10} {sp:>8} "
                f"{'OK' if v['parity'] else 'FAIL':>7}")
    return "\n".join(lines)


def _fmt_md(results):
    lines = [
        "### WebAssembly (wasm32-wasip1) — dense vs CSR vs value-spec unroll "
        "(symmetric i8)",
        "",
        "`fuel/step` = wasmtime fuel (a **deterministic** op-count proxy, one "
        "unit per wasm op) via a two-point measurement that cancels startup "
        "overhead. `speedup` = dense / variant. wasm B = full module bytes "
        "(dominated by the Rust std/WASI runtime baseline; only the "
        "variant-to-variant delta reflects the kernel).",
        "",
        "| N | density | nnz | variant | wasm B | fuel/step | speedup | parity |",
        "|--:|--:|--:|:--|--:|--:|--:|:--:|",
    ]
    for r in results:
        base = r["variants"]["dense"]["fuel_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            f = v["fuel_per_step"]
            sp = (f"{base / f:.2f}×" if label != "dense" and f > 0 else "–")
            lines.append(
                f"| {r['N']} | {r['density']:.2f} | {r['nnz']} | "
                f"{'**' + label + '**' if label == 'unroll' else label} | "
                f"{v['wasm_bytes']} | {f} | {sp} | "
                f"{'✅' if v['parity'] else '❌'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    try:
        import wasmtime  # noqa: F401
    except ImportError:
        print("Need the `wasmtime` Python package (pip install wasmtime). "
              "Aborting.")
        return 1

    sizes = [(64, 0.1), (128, 0.1)] if args.quick else [
        (64, 0.1), (96, 0.1), (128, 0.1)]

    print("WebAssembly (wasm32-wasip1) — dense vs CSR vs value-spec unroll, "
          "symmetric i8\n")
    results = run(sizes)
    print(_fmt_table(results))
    print("\nfuel/step = wasmtime fuel, deterministic op-count proxy "
          "(two-point, cancels startup); speedup = dense/variant.")

    all_vars = [v for r in results for v in r["variants"].values()]
    all_ok = all(v["parity"] for v in all_vars)
    measured = all(v["fuel_per_step"] > 0 for v in all_vars)

    if args.md:
        args.md.write_text(_fmt_md(results))
        print(f"\nwrote {args.md}")
    if not all_ok:
        print("\nERROR: a variant failed parity (PARITY_FAIL).")
    if not measured:
        print("\nERROR: a variant produced no fuel measurement.")
    if args.json:
        args.json.write_text(json.dumps(
            dict(target="wasm32-wasip1", path="llvm-symmetric-i8",
                 results=results), indent=2))
        print(f"wrote {args.json}")
    return 0 if (all_ok and measured) else 1


if __name__ == "__main__":
    sys.exit(main())
