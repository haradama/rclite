"""Arduino Uno (ATmega328P) performance bench, unified schema.

The AVR deployment is the affine C kernel (emit_affine_kernel_c). The
**i8/i16 dense/csr** cells are measured; the rest stay blank but keep the
columns common with the Cortex-M0 / WASM benches (benchmarks/_perf_schema.py):
  - We quantize with a linear-interp LUT. The default DIRECT LUT has
    2**storage_bits entries — 128 KB for i16, which overflows the Uno's 32 KB
    Flash and reads back garbage; the small interp LUT fits and stays
    bit-exact on the device (works on the stock avr-gcc 7.x — it is a LUT-size
    issue, not a compiler bug).
  - i32 is blank: affine quantization targets i8/i16; an i32 affine model
    overflows the i64 requantize/accumulator (the Python executor itself
    raises OverflowError). i32 uses the *symmetric* path instead, measured on
    M0/WASM.
  - no float path and no value-specialized unroll (the C "unroll" → CSR).
Speed = AVR cycles per step via simavr (cycle-accurate, deterministic).
Size = Flash/RAM from avr-size.

Requires avr-gcc + avr-libc and host gcc + libsimavr-dev.

    python benchmarks/avr_mcu/bench_avr.py [--json o.json] [--md o.md]
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
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "sparse_mcu"))

import numpy as np

from rclite.quant import LUTStrategy
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.targets.arduino import emit_affine_kernel_c
import bench as _b               # sparse_mcu/bench.py — data + wres helpers
import _perf_schema as S

_b.T_FW = 64                     # short embedded sequence (AVR SRAM, sim time)

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "main_bench.c"
DRIVER_SRC = HERE / "sim_driver.c"
AVR_GCC, AVR_SIZE, MMCU = "avr-gcc", "avr-size", "atmega328p"
CFLAGS = [f"-mmcu={MMCU}", "-Os", "-std=c99", "-ffunction-sections",
          "-fdata-sections", "-Wl,--gc-sections"]
BITS = {"i8": 8, "i16": 16, "i32": 32}


def _have_tools():
    return (shutil.which(AVR_GCC) and shutil.which(AVR_SIZE)
            and shutil.which("gcc"))


def _build_driver(workdir):
    drv = workdir / "sim_driver"
    cp = subprocess.run(["gcc", str(DRIVER_SRC), "-lsimavr", "-o", str(drv)],
                        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError("simavr driver build failed:\n" + cp.stderr)
    return drv


def _build_and_run(qm, x_seq, sparse, driver, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "rc_kernel.c").write_text(emit_affine_kernel_c(qm, sparse=sparse))
    (workdir / "rc_data.h").write_text(_b._emit_data_h(qm, x_seq))
    shutil.copy(HARNESS, workdir / "main_bench.c")
    elf = workdir / "fw.elf"
    cp = subprocess.run(
        [AVR_GCC, *CFLAGS, "-I", str(workdir), str(workdir / "rc_kernel.c"),
         str(workdir / "main_bench.c"), "-o", str(elf)],
        capture_output=True, text=True)
    if cp.returncode != 0:
        raise RuntimeError(f"avr-gcc failed:\n{cp.stderr}")
    sz = subprocess.run([AVR_SIZE, str(elf)], check=True, capture_output=True,
                        text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sz[0]), int(sz[1]), int(sz[2])
    cp = subprocess.run([str(driver), str(elf)], capture_output=True,
                        text=True, timeout=300)
    out = cp.stdout + cp.stderr
    m = re.search(r"avr_cycles:\s*(\d+)", out)
    total = int(m.group(1)) if m else -1
    parity = "parity: OK" in out
    cyc = total // _b.T_FW if total > 0 else None
    return text + data, data + bss, cyc, parity


def run(sizes):
    rows = []
    for units, density in sizes:
        rc, exe, X, Y = _b._train(units, density)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        x_seq = X[900:900 + _b.T_FW]
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            driver = _build_driver(td)
            for dtype in S.DTYPES:
                for kernel in S.KERNELS:
                    r = S.row(N=N, density=density, nnz=nnz, dtype=dtype,
                              kernel=kernel)
                    # AVR (affine C kernel): i8/i16 dense/csr are measurable.
                    #  - We use a linear-interp LUT: the default DIRECT LUT has
                    #    2**storage_bits entries, which for i16 is 128 KB and
                    #    overflows the Uno's 32 KB Flash (pgm_read then wraps to
                    #    garbage). The small LUT fits and is bit-exact on AVR.
                    #  - i32 is blank: affine targets i8/i16; an i32 affine
                    #    model overflows the i64 requantize/accumulator (the
                    #    Python executor itself raises) — i32 uses the symmetric
                    #    path instead (measured on M0/WASM).
                    #  - no float / value-spec-unroll C path.
                    if dtype in ("i8", "i16") and kernel in ("dense", "csr"):
                        try:
                            cfg = calibrate_from_data(rc, exe, X[:900],
                                                      storage_bits=BITS[dtype])
                            qm = quantize_model_affine(
                                rc, exe, cfg,
                                lut_strategy=LUTStrategy.linear_interp(64))
                            strat = None if kernel == "dense" else "csr"
                            fl, ram, cyc, par = _build_and_run(
                                qm, x_seq, strat, driver,
                                td / f"{dtype}_{kernel}")
                            r.update(flash_B=fl, ram_B=ram, ops_per_step=cyc,
                                     parity=par,
                                     wres_B=_b._wres_bytes(qm, strat))
                        except Exception as e:
                            print(f"  (blank {dtype}/{kernel}: "
                                  f"{type(e).__name__})")
                    rows.append(r)
    return rows


TARGET = "Arduino Uno (ATmega328P) — affine i8/i16 (C kernel)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not _have_tools():
        print("Need avr-gcc + avr-libc and host gcc + libsimavr-dev. Aborting.")
        return 1

    sizes = [(32, 0.15)] if args.quick else [(32, 0.15), (64, 0.15)]
    rows = run(sizes)
    print(S.fmt_text(TARGET, rows, unit="AVR cycles (simavr)"))

    if args.md:
        args.md.write_text(S.fmt_md(
            TARGET, rows, unit="AVR cycles (simavr)",
            note="AVR uses the affine C kernel with a linear-interp LUT "
                 "(the DIRECT LUT is 128 KB at i16, overflowing the Uno's "
                 "32 KB Flash). i8/i16 dense/csr are measured; i32 affine "
                 "overflows the i64 requantize (use the symmetric path), and "
                 "there is no float / value-spec-unroll C path — blank."))
        print(f"\nwrote {args.md}")
    ok = S.all_parity_ok(rows)
    if not ok:
        print("\nERROR: a variant failed on-device parity.")
    if args.json:
        args.json.write_text(json.dumps(
            dict(target="atmega328p", rows=rows), indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
