"""ATmega328P (Arduino Uno) firmware benchmark: dense vs CSR-sparse W_res
(affine i8), via the emit_affine_kernel_c C kernel under simavr.

The Arduino/turnkey C template targets 2KB-SRAM 8-bit AVR, where the
value-specialized unroll kernel does not exist (it is LLVM-only; "unroll"
degrades to CSR here), so this measures the two AVR-relevant kernels:

  * ACCURACY — both are bit-exact with the host AffineQuantizedExecutor
    (firmware asserts max|Y - Y_ref| == 0 → parity OK).
  * SPEED    — AVR cycles per inference step, counted by simavr
    (cycle-accurate, fully DETERMINISTIC — see sim_driver.c).
  * SIZE     — Flash (text+data) and static RAM (data+bss) from avr-size;
    the headline AVR metric (does a dense reservoir fit a Uno?).

Requires avr-gcc + avr-libc and a host gcc + libsimavr-dev (for the driver).

    python benchmarks/avr_mcu/bench_avr.py [--json out.json] [--md out.md]
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sparse_mcu"))

import numpy as np

from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.targets.arduino import emit_affine_kernel_c

import bench as _b  # benchmarks/sparse_mcu/bench.py — reuse model + data helpers

# Smaller embedded sequence than the Cortex-M0 bench: keeps AVR SRAM use and
# simavr wall-time modest. _emit_data_h reads this module global at call time.
_b.T_FW = 64

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "main_bench.c"
DRIVER_SRC = HERE / "sim_driver.c"
AVR_GCC = "avr-gcc"
AVR_SIZE = "avr-size"
MMCU = "atmega328p"
CFLAGS = [f"-mmcu={MMCU}", "-Os", "-std=c99", "-ffunction-sections",
          "-fdata-sections", "-Wl,--gc-sections"]

VARIANTS = [("dense", None), ("csr", "csr")]


def _have_tools():
    return (shutil.which(AVR_GCC) and shutil.which(AVR_SIZE)
            and shutil.which("gcc"))


def _build_driver(workdir: pathlib.Path) -> pathlib.Path:
    drv = workdir / "sim_driver"
    cp = subprocess.run(["gcc", str(DRIVER_SRC), "-lsimavr", "-o", str(drv)],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(
            "failed to build simavr driver (need libsimavr-dev):\n" + cp.stderr)
    return drv


def _build_and_run(qm, x_seq, sparse, driver, workdir: pathlib.Path):
    """Build one AVR variant, run under simavr; return (flash, ram, cyc, par)."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "rc_kernel.c").write_text(
        emit_affine_kernel_c(qm, sparse=sparse))
    (workdir / "rc_data.h").write_text(_b._emit_data_h(qm, x_seq))
    shutil.copy(HARNESS, workdir / "main_bench.c")
    elf = workdir / "fw.elf"
    cp = subprocess.run(
        [AVR_GCC, *CFLAGS, "-I", str(workdir),
         str(workdir / "rc_kernel.c"), str(workdir / "main_bench.c"),
         "-o", str(elf)],
        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"avr-gcc failed ({sparse}):\n{cp.stderr}")

    sz = subprocess.run([AVR_SIZE, str(elf)], check=True, capture_output=True,
                        text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sz[0]), int(sz[1]), int(sz[2])

    cp = subprocess.run([str(driver), str(elf)], capture_output=True,
                        text=True, timeout=300)
    out = cp.stdout + cp.stderr
    m = re.search(r"avr_cycles:\s*(\d+)", out)
    total = int(m.group(1)) if m else -1
    parity = "parity: OK" in out
    cyc_per_step = total // _b.T_FW if total > 0 else -1
    return text + data, data + bss, cyc_per_step, parity


def run(sizes):
    results = []
    for units, density in sizes:
        rc, exe, X, Y = _b._train(units, density)
        cfg = calibrate_from_data(rc, exe, X[:900], storage_bits=8)
        qm = quantize_model_affine(rc, exe, cfg)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        x_seq = X[900:900 + _b.T_FW]

        variants = {}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            driver = _build_driver(td)
            for label, strat in VARIANTS:
                fl, ram, cyc, par = _build_and_run(
                    qm, x_seq, strat, driver, td / label)
                variants[label] = dict(
                    flash=fl, ram=ram, wres_bytes=_b._wres_bytes(qm, strat),
                    cycles_per_step=cyc, parity=par)
        results.append(dict(N=N, density=density, nnz=nnz, variants=variants))
    return results


def _fmt_table(results):
    hdr = (f"{'N':>4} {'dens':>5} {'nnz':>6} {'variant':>7} {'Flash B':>8} "
           f"{'RAM B':>6} {'Wres B':>7} {'cyc/step':>10} {'speedup':>8} "
           f"{'parity':>7}")
    lines = [hdr, "-" * len(hdr)]
    for r in results:
        base = r["variants"]["dense"]["cycles_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            cyc = v["cycles_per_step"]
            sp = (f"{base / cyc:.2f}x" if label != "dense" and cyc > 0 else "-")
            lines.append(
                f"{r['N']:>4} {r['density']:>5.2f} {r['nnz']:>6} {label:>7} "
                f"{v['flash']:>8} {v['ram']:>6} {v['wres_bytes']:>7} "
                f"{cyc:>10} {sp:>8} {'OK' if v['parity'] else 'FAIL':>7}")
    return "\n".join(lines)


def _fmt_md(results):
    lines = [
        "### Arduino Uno (ATmega328P) — dense vs CSR W_res (affine i8)",
        "",
        "`cyc/step` = AVR cycles via **simavr** (cycle-accurate, "
        "**deterministic**). `speedup` = dense / variant. Flash = text+data, "
        "RAM = data+bss (avr-size); Wres B = W_res table bytes. The C kernel "
        "has no value-specialized unroll (LLVM-only), so AVR compares "
        "dense vs CSR.",
        "",
        "| N | density | nnz | variant | Flash B | RAM B | Wres B | "
        "cyc/step | speedup | parity |",
        "|--:|--:|--:|:--|--:|--:|--:|--:|--:|:--:|",
    ]
    for r in results:
        base = r["variants"]["dense"]["cycles_per_step"]
        for label, _ in VARIANTS:
            v = r["variants"][label]
            cyc = v["cycles_per_step"]
            sp = (f"{base / cyc:.2f}×" if label != "dense" and cyc > 0
                  else "–")
            lines.append(
                f"| {r['N']} | {r['density']:.2f} | {r['nnz']} | "
                f"{'**' + label + '**' if label == 'csr' else label} | "
                f"{v['flash']} | {v['ram']} | {v['wres_bytes']} | {cyc} | "
                f"{sp} | {'✅' if v['parity'] else '❌'} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not _have_tools():
        print("Need avr-gcc + avr-libc and host gcc + libsimavr-dev. Aborting.")
        return 1

    sizes = [(32, 0.15), (64, 0.15)] if args.quick else [
        (32, 0.15), (48, 0.15), (64, 0.15)]

    print("Arduino Uno (ATmega328P) firmware — dense vs CSR W_res, affine i8\n")
    results = run(sizes)
    print(_fmt_table(results))
    print("\ncyc/step = simavr cycle-accurate, deterministic; "
          "speedup = dense/variant.")
    print("Flash = text+data, RAM = data+bss (avr-size). Wres B = W_res "
          "table bytes (dense N*N vs CSR val+col+rowptr).")

    all_vars = [v for r in results for v in r["variants"].values()]
    all_ok = all(v["parity"] for v in all_vars)
    measured = all(v["cycles_per_step"] > 0 for v in all_vars)

    if args.md:
        args.md.write_text(_fmt_md(results))
        print(f"\nwrote {args.md}")
    if not all_ok:
        print("\nERROR: a variant failed on-device parity (PARITY_FAIL).")
    if not measured:
        print("\nERROR: a variant did not produce a cycle count.")
    if args.json:
        args.json.write_text(json.dumps(
            dict(target="atmega328p", path="emit_affine_kernel_c",
                 dtype="affine-i8", results=results), indent=2))
        print(f"wrote {args.json}")
    return 0 if (all_ok and measured) else 1


if __name__ == "__main__":
    sys.exit(main())
