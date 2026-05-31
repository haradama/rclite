"""Cortex-M0 firmware benchmark via the LLVM codegen path:
dense vs CSR-sparse vs value-specialized unroll W_res (affine i8).

bench.py measures the hand-written C kernel template, where "unroll" degrades
to CSR (the 2KB-SRAM Arduino path keeps code size constant). This script
instead drives the production LLVM cross-compile path
(`emit_quantized_affine_module` + `SparsifyReservoir`), so all three kernels
are measured apples-to-apples on the same backend -- including the
**value-specialized unroll** kernel, where baked +-1/+-2**k W_res weights
fold the multiply into a negate/shift (see `_pow2_exp` /
`_fixed_const_mul_to_accum` / `_const_mul_accum` in codegen/llvm.py).

  * ACCURACY — all three are bit-exact with the host AffineQuantizedExecutor
    (the firmware asserts max|Y - Y_ref| == 0 and prints PARITY_OK).
  * SPEED    — SysTick ticks per inference step under `qemu -icount shift=0`.
    The count is DETERMINISTIC (bit-stable run to run, unlike semihosting
    SYS_ELAPSED which reads host wall-clock here). Ticks are proportional to
    executed instructions (SysTick runs at the CPU-clock rate, ~1 tick per
    ~62 instructions on nRF51), so SPEEDUP RATIOS equal instruction-count
    ratios. It is an op-count proxy, NOT silicon cycles.
  * SIZE     — Flash (text+data) and static RAM (data+bss) per variant.

Requires arm-none-eabi-gcc + qemu-system-arm (same as bench.py / CI).

    python benchmarks/sparse_mcu/bench_llvm.py [--json out.json]
"""
from __future__ import annotations
import argparse
import json
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np
import llvmlite.binding as llvm

from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.codegen.llvm import (
    emit_quantized_affine_module, _ensure_all_targets,
)
from rclite.ir import sparse_passes
from rclite.targets.cortex_m0.target import _AEABI_ALIASES, _SUPPORT_DIR

# Reuse the model + reference-data + W_res-byte helpers from bench.py.
from bench import _train, _emit_data_h, _wres_bytes, _have_tools, T_FW

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "main_bench.c"
CC = "arm-none-eabi-gcc"
SIZE = "arm-none-eabi-size"
QEMU = "qemu-system-arm"
TRIPLE, CPU = "thumbv6m-none-eabi", "cortex-m0"
ARCH = ["-mcpu=cortex-m0", "-mthumb"]
CFLAGS = ARCH + ["-O2", "-std=c99", "-ffunction-sections", "-fdata-sections"]

# variant label -> SparsifyReservoir strategy ("dense" => no pass)
VARIANTS = [("dense", None), ("csr", "csr"), ("unroll", "unroll")]


def _emit_kernel_object(qm, strategy, out: pathlib.Path) -> pathlib.Path:
    """Affine LLVM kernel for `strategy` -> optimized thumbv6m object.

    Mirrors CortexM0Target.compile_affine_quantized's emit/optimize step so
    the measured object is the one the target ships. `strategy=None` =>
    no SparsifyReservoir => dense N*N matvec.
    """
    passes = [] if strategy is None else sparse_passes(
        strategy, include_structural=False)
    mod_ir = emit_quantized_affine_module(qm, passes=passes)
    mod_ir.triple = TRIPLE
    _ensure_all_targets()
    m = llvm.parse_assembly(str(mod_ir))
    m.verify()
    tgt = llvm.Target.from_triple(TRIPLE)
    tm = tgt.create_target_machine(cpu=CPU, opt=2, reloc="static")
    pto = llvm.create_pipeline_tuning_options()
    pto.speed_level = 2
    pto.loop_vectorization = False
    pto.slp_vectorization = False
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(m, pb)
    rc_o = out / "rc_predict.o"
    rc_o.write_bytes(tm.emit_object(m))
    return rc_o


def _build_and_run(qm, x_seq, strategy, workdir: pathlib.Path):
    """Build + run one variant; return (flash, ram, ticks_per_step, parity)."""
    workdir.mkdir(parents=True, exist_ok=True)
    rc_o = _emit_kernel_object(qm, strategy, workdir)
    (workdir / "rc_data.h").write_text(_emit_data_h(qm, x_seq))
    shutil.copy(_SUPPORT_DIR / "startup.c", workdir / "startup.c")
    shutil.copy(_SUPPORT_DIR / "nrf51.ld", workdir / "nrf51.ld")
    shutil.copy(HARNESS, workdir / "main_bench.c")

    objs = []
    for src in ("startup.c", "main_bench.c"):
        o = workdir / (src[:-2] + ".o")
        subprocess.run([CC, "-c", *CFLAGS, "-I", str(workdir),
                        str(workdir / src), "-o", str(o)],
                       check=True, capture_output=True, text=True)
        objs.append(str(o))
    elf = workdir / "bench.elf"
    subprocess.run(
        [CC, *ARCH, "-T", str(workdir / "nrf51.ld"), "-nostartfiles",
         "-Wl,--gc-sections", "--specs=nosys.specs", *_AEABI_ALIASES,
         *objs, str(rc_o), "-o", str(elf), "-lgcc", "-lc", "-lnosys"],
        check=True, capture_output=True, text=True)

    sz = subprocess.run([SIZE, str(elf)], check=True, capture_output=True,
                        text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sz[0]), int(sz[1]), int(sz[2])

    cp = subprocess.run(
        [QEMU, "-M", "microbit", "-nographic", "-semihosting",
         "-icount", "shift=0", "-kernel", str(elf)],
        capture_output=True, text=True, timeout=120)
    out = cp.stdout + cp.stderr
    m = re.search(r"ticks_per_step:\s*(-?\d+)", out)
    ticks = int(m.group(1)) if m else -1  # -1 = SysTick 24-bit overflow / missing
    parity = "PARITY_OK" in out
    return text + data, data + bss, ticks, parity


def run(sizes):
    results = []
    for units, density in sizes:
        rc, exe, X, Y = _train(units, density)
        cfg = calibrate_from_data(rc, exe, X[:900], storage_bits=8)
        qm = quantize_model_affine(rc, exe, cfg)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        x_seq = X[900:900 + T_FW]

        variants = {}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for label, strat in VARIANTS:
                fl, ram, ticks, par = _build_and_run(
                    qm, x_seq, strat, td / label)
                # unroll bakes weights into .text (counted in Flash), so its
                # separate W_res *table* is 0 bytes; dense=N*N, csr=val+idx.
                wb = 0 if label == "unroll" else _wres_bytes(qm, strat)
                variants[label] = dict(
                    flash=fl, ram=ram, wres_bytes=wb,
                    ticks_per_step=ticks, parity=par)
        results.append(dict(N=N, density=density, nnz=nnz, variants=variants))
    return results


def _fmt_table(results):
    lines = []
    hdr = (f"{'N':>4} {'dens':>5} {'nnz':>6} {'variant':>7} {'Flash B':>8} "
           f"{'RAM B':>6} {'Wres B':>7} {'ticks/step':>11} {'speedup':>8} "
           f"{'parity':>7}")
    lines.append(hdr)
    lines.append("-" * len(hdr))
    for r in results:
        base = r["variants"]["dense"]["ticks_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            tps = v["ticks_per_step"]
            sp = (f"{base / tps:.2f}x"
                  if label != "dense" and tps > 0 else "-")
            lines.append(
                f"{r['N']:>4} {r['density']:>5.2f} {r['nnz']:>6} {label:>7} "
                f"{v['flash']:>8} {v['ram']:>6} {v['wres_bytes']:>7} "
                f"{tps:>11} {sp:>8} {'OK' if v['parity'] else 'FAIL':>7}")
    return "\n".join(lines)


def _fmt_md(results):
    """GitHub-flavored markdown table for $GITHUB_STEP_SUMMARY."""
    lines = [
        "### Cortex-M0 QEMU perf — dense vs CSR vs value-spec unroll "
        "(affine i8)",
        "",
        "`ticks/step` = SysTick under `qemu -icount shift=0` — a "
        "**deterministic** op-count proxy (ticks ∝ executed instructions; "
        "**not** silicon cycles). `speedup` = dense / variant ticks, which "
        "equals the instruction-count ratio. Wres B = W_res table bytes "
        "(unroll bakes weights into `.text`).",
        "",
        "| N | density | nnz | variant | Flash B | RAM B | Wres B | "
        "ticks/step | speedup | parity |",
        "|--:|--:|--:|:--|--:|--:|--:|--:|--:|:--:|",
    ]
    for r in results:
        base = r["variants"]["dense"]["ticks_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            tps = v["ticks_per_step"]
            sp = (f"{base / tps:.2f}×" if label != "dense" and tps > 0
                  else "–")
            lines.append(
                f"| {r['N']} | {r['density']:.2f} | {r['nnz']} | "
                f"{'**' + label + '**' if label == 'unroll' else label} | "
                f"{v['flash']} | {v['ram']} | {v['wres_bytes']} | {tps} | "
                f"{sp} | {'✅' if v['parity'] else '❌'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None,
                    help="write machine-readable results to this path")
    ap.add_argument("--md", type=pathlib.Path, default=None,
                    help="write a markdown table (for GITHUB_STEP_SUMMARY)")
    ap.add_argument("--quick", action="store_true",
                    help="fewer/smaller sizes for a fast CI smoke")
    args = ap.parse_args()

    if not _have_tools():
        print("Need arm-none-eabi-gcc + qemu-system-arm on PATH. Aborting.")
        return 1

    sizes = [(64, 0.1), (128, 0.1)] if args.quick else [
        (64, 0.1), (96, 0.1), (128, 0.1)]

    print("Cortex-M0 (nRF51) firmware via LLVM codegen — "
          "dense vs CSR vs value-spec unroll, affine i8\n")
    results = run(sizes)
    table = _fmt_table(results)
    print(table)
    print("\nticks/step = SysTick under qemu -icount shift=0 — deterministic "
          "op-count proxy (ticks proportional to instructions, NOT silicon "
          "cycles). speedup = dense/variant = instruction-count ratio.")
    print("Wres B = bytes for the W_res representation (dense N*N vs CSR "
          "val+col+rowptr; unroll bakes weights into code, 0 table bytes).")

    all_vars = [v for r in results for v in r["variants"].values()]
    all_ok = all(v["parity"] for v in all_vars)
    measured = all(v["ticks_per_step"] > 0 for v in all_vars)

    # Write artifacts before the gate so they surface even on failure.
    if args.md:
        args.md.write_text(_fmt_md(results))
        print(f"\nwrote {args.md}")
    if not all_ok:
        print("\nERROR: a variant failed on-device parity (PARITY_FAIL).")
    if not measured:
        print("\nERROR: a variant overflowed the 24-bit SysTick window "
              "(ticks_per_step=-1); lower T_TIME or N.")
    all_ok = all_ok and measured

    if args.json:
        args.json.write_text(json.dumps(
            dict(target="cortex-m0", path="llvm", dtype="affine-i8",
                 results=results), indent=2))
        print(f"\nwrote {args.json}")
    return 0 if all_ok else 1


if __name__ == "__main__":
    sys.exit(main())
