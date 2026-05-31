"""Game Boy Advance (ARM7TDMI / thumbv4t) performance bench, unified schema.

Same LLVM cross-compile path as Cortex-M0, so the full matrix applies:
dtype in {float (f32), i8, i16, i32} x kernel in {dense, csr, value-spec
unroll}. Speed = GBA timer ticks per step under mGBA — a DETERMINISTIC
op-count proxy (timers TM0+TM1 cascaded at the system clock; NOT silicon
cycles). Columns are shared with the Cortex-M0 / AVR / WASM benches
(benchmarks/_perf_schema.py). Quant scheme: i8/i16 affine, i32 symmetric.

Requires arm-none-eabi-gcc (+ binutils) and mGBA (`mgba` / `mgba-sdl`).

    python benchmarks/gba_arm7tdmi/bench_gba.py [--json o.json] [--md o.md]
"""
from __future__ import annotations
import argparse
import json
import fcntl
import os
import pathlib
import re
import shutil
import signal
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

import numpy as np

from rclite.targets.gba.target import GbaTarget, _SUPPORT_DIR
import _perf_kernels as K
import _perf_schema as S

HERE = pathlib.Path(__file__).resolve().parent
HARNESS = HERE / "main_bench.c"
SIZE = "arm-none-eabi-size"
TRIPLE, CPU = "thumbv4t-none-eabi", "arm7tdmi"
T_SEQ = 64
_MGBA = shutil.which("mgba") or shutil.which("mgba-sdl") or "/usr/games/mgba"


def _have_tools():
    return (shutil.which("arm-none-eabi-gcc") and shutil.which(SIZE)
            and (shutil.which("mgba") or shutil.which("mgba-sdl")
                 or os.path.exists("/usr/games/mgba")))


def _run_mgba(gba_path, timeout=10.0):
    """Run headless mGBA and return its log. Reads the pipe with non-blocking
    raw os.read (no readline/select buffering races) and stops as soon as both
    the ticks line and a parity verdict have appeared (the GBA otherwise spins
    forever); a hard deadline bounds the wait if the verdict never shows."""
    # mGBA block-buffers its debug log; stdbuf -oL line-buffers it so the
    # firmware's lines reach us before we kill the (otherwise-spinning) ROM.
    cmd = (["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else [])
    cmd += [_MGBA, "-l", "15", str(gba_path)]
    env = dict(os.environ)
    env.setdefault("SDL_VIDEODRIVER", "dummy")
    env.setdefault("SDL_AUDIODRIVER", "dummy")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env=env)
    fd = proc.stdout.fileno()
    fcntl.fcntl(fd, fcntl.F_SETFL,
                fcntl.fcntl(fd, fcntl.F_GETFL) | os.O_NONBLOCK)
    buf, deadline = b"", time.monotonic() + timeout
    try:
        while time.monotonic() < deadline:
            try:
                chunk = os.read(fd, 65536)
            except BlockingIOError:
                chunk = b""
            if chunk:
                buf += chunk
                if (b"ticks_per_step:" in buf
                        and (b"PARITY_OK" in buf or b"PARITY_FAIL" in buf)):
                    break
            elif proc.poll() is not None:
                break
            else:
                time.sleep(0.05)
    finally:
        try:
            os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        except ProcessLookupError:
            pass
        proc.wait()
    return buf.decode(errors="replace")


def _build_and_run(gba, dtype, src, x_seq, sparse, workdir):
    workdir.mkdir(parents=True, exist_ok=True)
    rc_o = K.build_object(dtype, src, sparse, triple=TRIPLE, cpu=CPU,
                          out_path=workdir / "rc_predict.o")
    data_h, _ = K.emit_c_data_h(dtype, src, x_seq)
    (workdir / "rc_data.h").write_text(data_h)
    shutil.copy(_SUPPORT_DIR / "mgba_log.h", workdir / "mgba_log.h")
    shutil.copy(HARNESS, workdir / "main_bench.c")
    gba._build_rom(workdir, rc_o, workdir / "main_bench.c",
                   with_float=(dtype == "float"))

    elf = workdir / "rc.elf"
    sz = subprocess.run([SIZE, str(elf)], check=True, capture_output=True,
                        text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sz[0]), int(sz[1]), int(sz[2])
    log = _run_mgba(workdir / "rc.gba")
    m = re.search(r"ticks_per_step:\s*(-?\d+)", log)
    ticks = int(m.group(1)) if m else -1
    parity = "PARITY_OK" in log and "PARITY_FAIL" not in log
    return text + data, data + bss, ticks, parity


def run(sizes):
    gba = GbaTarget()
    gba._require_cc()
    rows = []
    for units, density in sizes:
        rc, exe, x_seq, y_true, x_cal = K.train_model(units, density, T_SEQ)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        qms = {b: K.quant_model(b, rc, exe, x_cal) for b in ("i8", "i16", "i32")}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for dtype in S.DTYPES:
                src = (rc, exe) if dtype == "float" else qms[dtype]
                mse = K.accuracy_mse(dtype, src, x_seq, y_true)
                for kernel in S.KERNELS:
                    try:
                        fl, ram, ticks, par = _build_and_run(
                            gba, dtype, src, x_seq, K.KERNEL_SPARSE[kernel],
                            td / f"{dtype}_{kernel}")
                    except Exception as e:
                        print(f"  (blank {dtype}/{kernel}: {type(e).__name__})")
                        rows.append(S.row(N=N, density=density, nnz=nnz,
                                          dtype=dtype, kernel=kernel, mse=mse))
                        continue
                    rows.append(S.row(
                        N=N, density=density, nnz=nnz, dtype=dtype,
                        kernel=kernel,
                        ops_per_step=(ticks if ticks > 0 else None),
                        parity=par, flash_B=fl, ram_B=ram, mse=mse,
                        wres_B=K.wres_bytes(dtype, src,
                                            K.KERNEL_SPARSE[kernel], N)))
    return rows


TARGET = "Game Boy Advance (ARM7TDMI) — float, affine i8/i16, symmetric i32"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", type=pathlib.Path, default=None)
    ap.add_argument("--md", type=pathlib.Path, default=None)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()
    if not _have_tools():
        print("Need arm-none-eabi-gcc + mGBA. Aborting.")
        return 1

    sizes = [(32, 0.15)] if args.quick else [(32, 0.15), (64, 0.15)]
    rows = run(sizes)
    print(S.fmt_text(TARGET, rows, unit="GBA timer ticks (mGBA)"))

    if args.md:
        args.md.write_text(S.fmt_md(
            TARGET, rows, unit="GBA timer ticks (mGBA)",
            note="Quant scheme: i8/i16 = affine, i32 = symmetric. Code runs "
                 "from cartridge ROM (with GBA waitstates)."))
        print(f"\nwrote {args.md}")
    ok = S.all_parity_ok(rows)
    if not ok:
        print("\nERROR: a variant failed on-device parity.")
    if args.json:
        args.json.write_text(json.dumps(
            dict(target="gba-arm7tdmi", rows=rows), indent=2))
        print(f"wrote {args.json}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
