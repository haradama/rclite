"""Cortex-M0 (nRF51 / micro:bit) performance bench, unified schema.

Matrix: dtype in {float (f32), i8, i16, i32} x kernel in {dense, csr,
value-spec unroll}, all via the LLVM cross-compile path. Speed = SysTick
ticks per step under `qemu -icount shift=0` — a DETERMINISTIC op-count proxy
(ticks proportional to executed instructions; NOT silicon cycles). Columns
are shared with the AVR and WASM benches (benchmarks/_perf_schema.py).

Requires arm-none-eabi-gcc + qemu-system-arm.

    python benchmarks/sparse_mcu/bench_llvm.py [--json o.json] [--md o.md]
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

import numpy as np

from rclite.targets.cortex_m0.target import _AEABI_ALIASES, _SUPPORT_DIR
import _perf_kernels as K
import _perf_schema as S

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "main_bench.c"
CC, SIZE, QEMU = "arm-none-eabi-gcc", "arm-none-eabi-size", "qemu-system-arm"
TRIPLE, CPU = "thumbv6m-none-eabi", "cortex-m0"
ARCH = ["-mcpu=cortex-m0", "-mthumb"]
CFLAGS = ARCH + ["-O2", "-std=c99", "-ffunction-sections", "-fdata-sections"]
T_SEQ = 200
_CTYPE = {np.float32: "float", np.int8: "int8_t",
          np.int16: "int16_t", np.int32: "int32_t"}


def _have_tools():
    return all(shutil.which(t) for t in (CC, SIZE, QEMU))


def _emit_data_h(dtype, src, x_seq):
    X, Y, eps, npd, Kk, M, T = K.reference_data(dtype, src, x_seq)
    ct = _CTYPE[npd if dtype != "float" else np.float32]
    if dtype == "float":
        fmt = lambda a: ", ".join(f"{float(v)!r}f" for v in a.ravel())
    else:
        fmt = lambda a: ", ".join(str(int(v)) for v in a.ravel())
    return "\n".join([
        f"typedef {ct} rc_fw_storage_t;",
        f"#define RC_FW_EPS {float(eps)!r}",
        f"#define RC_FW_T {T}", f"#define RC_FW_K {Kk}", f"#define RC_FW_M {M}",
        f"static const rc_fw_storage_t g_x[{T * Kk}] = {{{fmt(X)}}};",
        f"static const rc_fw_storage_t g_y_ref[{T * M}] = {{{fmt(Y)}}};",
        "",
    ]), T


def _build_and_run(dtype, src, x_seq, sparse, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    rc_o = K.build_object(dtype, src, sparse, triple=TRIPLE, cpu=CPU,
                          out_path=workdir / "rc_predict.o")
    data_h, _ = _emit_data_h(dtype, src, x_seq)
    (workdir / "rc_data.h").write_text(data_h)
    shutil.copy(_SUPPORT_DIR / "startup.c", workdir / "startup.c")
    shutil.copy(_SUPPORT_DIR / "nrf51.ld", workdir / "nrf51.ld")
    shutil.copy(HARNESS, workdir / "main_bench.c")
    objs = []
    for s in ("startup.c", "main_bench.c"):
        o = workdir / (s[:-2] + ".o")
        subprocess.run([CC, "-c", *CFLAGS, "-I", str(workdir),
                        str(workdir / s), "-o", str(o)],
                       check=True, capture_output=True, text=True)
        objs.append(str(o))
    elf = workdir / "bench.elf"
    # -lm: the f32 kernel pulls tanhf; harmless for the integer kernels.
    subprocess.run(
        [CC, *ARCH, "-T", str(workdir / "nrf51.ld"), "-nostartfiles",
         "-Wl,--gc-sections", "--specs=nosys.specs", *_AEABI_ALIASES,
         *objs, str(rc_o), "-o", str(elf), "-lgcc", "-lc", "-lm", "-lnosys"],
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
    ticks = int(m.group(1)) if m else -1
    parity = "PARITY_OK" in out
    return text + data, data + bss, ticks, parity


def run(sizes):
    rows = []
    for units, density in sizes:
        rc, exe, x_seq = K.train_model(units, density, T_SEQ)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        qms = {b: K.sym_qmodel(rc, exe, K._BITS[b]) for b in ("i8", "i16", "i32")}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for dtype in S.DTYPES:
                src = (rc, exe) if dtype == "float" else qms[dtype]
                for kernel in S.KERNELS:
                    fl, ram, ticks, par = _build_and_run(
                        dtype, src, x_seq, K.KERNEL_SPARSE[kernel],
                        td / f"{dtype}_{kernel}")
                    rows.append(S.row(
                        N=N, density=density, nnz=nnz, dtype=dtype,
                        kernel=kernel,
                        ops_per_step=(ticks if ticks > 0 else None),
                        parity=par, flash_B=fl, ram_B=ram,
                        wres_B=K.wres_bytes(dtype, src,
                                            K.KERNEL_SPARSE[kernel], N)))
    return rows


TARGET = "Cortex-M0 (nRF51) — float + symmetric i8/i16/i32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not _have_tools():
        print("Need arm-none-eabi-gcc + qemu-system-arm. Aborting.")
        return 1

    sizes = [(64, 0.1)] if args.quick else [(64, 0.1), (128, 0.1)]
    rows = run(sizes)
    print(S.fmt_text(TARGET, rows, unit="SysTick ticks"))

    if args.md:
        args.md.write_text(S.fmt_md(TARGET, rows, unit="SysTick ticks"))
        print(f"\nwrote {args.md}")
    ok = S.all_parity_ok(rows)
    if not ok:
        print("\nERROR: a variant failed on-device parity.")
    if args.json:
        args.json.write_text(json.dumps(
            dict(target="cortex-m0", rows=rows), indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
