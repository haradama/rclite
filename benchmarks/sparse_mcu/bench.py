"""Cortex-M0 firmware benchmark: dense vs CSR-sparse W_res (affine i8).

Quantifies the on-device impact of `SparsifyReservoir` (CSR strategy) on the
three metrics that matter for an MCU deployment:

  * ACCURACY — quantized output vs the float reference (MSE / max-abs). CSR is
    bit-exact with the dense integer kernel, so accuracy is *identical*; the
    benchmark verifies max|dense - csr| == 0 and that the on-device output
    matches the host AffineQuantizedExecutor (PARITY_OK).
  * SPEED    — instructions per inference step under `qemu -icount shift=0`
    (a deterministic, like-for-like op-count proxy; not silicon cycles).
  * SIZE     — Flash (text+data) and static RAM (data+bss) from
    arm-none-eabi-size, plus the W_res table-byte breakdown.

Builds a real Cortex-M0 (nRF51/micro:bit) firmware for each variant, reusing
the self-contained startup / linker / semihosting harness from
benchmarks/tflm_vs_rclite/firmware. Requires arm-none-eabi-gcc + qemu-system-arm.

    .venv/bin/python benchmarks/sparse_mcu/bench.py
"""
from __future__ import annotations
import pathlib
import re
import shutil
import statistics
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
)
from rclite.targets.arduino import emit_affine_kernel_c
from rclite.ir.passes.sparsify import build_csr

ROOT = pathlib.Path(__file__).resolve().parents[2]
FW = ROOT / "benchmarks" / "tflm_vs_rclite" / "firmware"  # reuse support files
CC = "arm-none-eabi-gcc"
SIZE = "arm-none-eabi-size"
QEMU = "qemu-system-arm"
ARCH = ["-mcpu=cortex-m0", "-mthumb"]
CFLAGS = ARCH + ["-Os", "-std=c99", "-ffunction-sections", "-fdata-sections"]
T_FW = 200


def _have_tools():
    return all(shutil.which(t) for t in (CC, SIZE, QEMU))


def _train(units, density, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.35, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=100,
                            include_bias=True, include_input=False, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    series = np.sin(np.arange(1200) * 0.05) + 0.1 * rng.standard_normal(1200)
    X = series[:-1, None]
    Y = series[1:, None]
    exe.fit(X[:900], Y[:900])
    return rc, exe, X, Y


def _emit_data_h(qm, x_seq):
    """Embedded int test sequence + bit-exact host reference (rc_data.h)."""
    cfg = qm.config
    sb = qm.storage_bits
    np_t = {8: np.int8, 16: np.int16}[sb]
    ct = {8: "signed char", 16: "short"}[sb]
    Xq = cfg.input.quantize_array(x_seq).astype(np_t)
    qexe = AffineQuantizedExecutor(qm)
    qexe.reset()
    Yref = np.zeros((T_FW, qm.M), dtype=np_t)
    for t in range(T_FW):
        x_raw_q = qexe._quantize_raw_input(x_seq[t])
        u_pre_q = qexe._quantize_u_pre(x_seq[t])
        qexe.step_q(u_pre_q)
        Yref[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(np_t)
    return "\n".join([
        "#ifndef RC_DATA_H_", "#define RC_DATA_H_",
        f"typedef {ct} rc_fw_storage_t;",
        f"#define RC_FW_T {T_FW}", f"#define RC_FW_K {qm.K}",
        f"#define RC_FW_M {qm.M}",
        f"static const rc_fw_storage_t g_x[{T_FW * qm.K}] = {{"
        + ",".join(str(int(v)) for v in Xq.ravel()) + "};",
        f"static const rc_fw_storage_t g_y_ref[{T_FW * qm.M}] = {{"
        + ",".join(str(int(v)) for v in Yref.ravel()) + "};",
        "#endif", "",
    ])


def _build_and_run(qm, x_seq, sparse, workdir: pathlib.Path):
    """Compile a firmware variant, return (flash, ram, instr_per_step, parity)."""
    workdir.mkdir(parents=True, exist_ok=True)
    (workdir / "rc_kernel.c").write_text(
        emit_affine_kernel_c(qm, sparse=sparse))
    (workdir / "rc_data.h").write_text(_emit_data_h(qm, x_seq))
    elf = workdir / "fw.elf"
    objs = []
    for src, inc in [(workdir / "rc_kernel.c", workdir),
                     (FW / "main_rc.c", workdir),
                     (FW / "startup.c", None)]:
        o = workdir / (src.stem + ".o")
        cmd = [CC, "-c", *CFLAGS, "-I", str(FW)]
        if inc:
            cmd += ["-I", str(inc)]
        cmd += [str(src), "-o", str(o)]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        objs.append(str(o))
    subprocess.run(
        [CC, *ARCH, "-T", str(FW / "nrf51.ld"), "-nostartfiles",
         "-Wl,--gc-sections", "--specs=nano.specs", "--specs=nosys.specs",
         *objs, "-lc", "-lgcc", "-lnosys", "-o", str(elf)],
        check=True, capture_output=True, text=True)

    # size: Berkeley format → text/data/bss columns
    sz = subprocess.run([SIZE, str(elf)], check=True, capture_output=True,
                        text=True).stdout.splitlines()[1].split()
    text, data, bss = int(sz[0]), int(sz[1]), int(sz[2])
    flash, ram = text + data, data + bss

    # run qemu -icount, parse instr_per_step + parity
    cp = subprocess.run(
        [QEMU, "-M", "microbit", "-nographic", "-semihosting",
         "-icount", "shift=0", "-kernel", str(elf)],
        capture_output=True, text=True, timeout=120)
    out = cp.stdout + cp.stderr
    m = re.search(r"instr_per_step:\s*(\d+)", out)
    instr = int(m.group(1)) if m else -1
    parity = "PARITY_OK" in out
    return flash, ram, instr, parity


def _wres_bytes(qm, sparse):
    """Flash bytes occupied by the W_res representation."""
    N = qm.N
    sb_bytes = qm.storage_bits // 8
    if not sparse:
        return N * N * sb_bytes
    val, col, rowptr = build_csr(np.asarray(qm.W_res_q))
    col_bytes = 2 if N <= 32767 else 4
    return len(val) * sb_bytes + len(col) * col_bytes + len(rowptr) * 4


def _accuracy(qm, exe, X, Y):
    """Quantized (== dense == csr) MSE vs the float reference output."""
    qexe = AffineQuantizedExecutor(qm)
    yq = qexe.predict(X[900:1100])
    yf = exe.predict(X[900:1100])
    yt = Y[900:1100]
    mse_q = float(np.mean((yq - yt) ** 2))
    mse_f = float(np.mean((yf - yt) ** 2))
    return mse_f, mse_q


def main():
    if not _have_tools():
        print("Need arm-none-eabi-gcc + qemu-system-arm on PATH. Aborting.")
        return
    print("Cortex-M0 (nRF51) firmware — dense vs CSR W_res, affine i8\n")
    header = (f"{'N':>4} {'dens':>5} {'nnz':>6} {'variant':>7} "
              f"{'Flash B':>8} {'RAM B':>6} {'Wres B':>7} "
              f"{'instr/step':>11} {'speedup':>8} {'parity':>7}")
    print(header)
    print("-" * len(header))
    for units, density in [(64, 0.1), (96, 0.1), (128, 0.1)]:
        rc, exe, X, Y = _train(units, density)
        cfg = calibrate_from_data(rc, exe, X[:900], storage_bits=8)
        qm = quantize_model_affine(rc, exe, cfg)
        N = rc.reservoir.units
        nnz = int(np.count_nonzero(exe.W_res))
        mse_f, mse_q = _accuracy(qm, exe, X, Y)
        x_seq = X[900:900 + T_FW]

        rows = {}
        with tempfile.TemporaryDirectory() as td:
            td = pathlib.Path(td)
            for label, sp in [("dense", None), ("csr", "csr")]:
                fl, ram, instr, par = _build_and_run(
                    qm, x_seq, sp, td / label)
                rows[label] = (fl, ram, instr, par)

        d_instr = rows["dense"][2]
        for label in ("dense", "csr"):
            fl, ram, instr, par = rows[label]
            wb = _wres_bytes(qm, label == "csr")
            sp_ratio = (d_instr / instr) if instr > 0 else float("nan")
            sp_str = f"{sp_ratio:.2f}x" if label == "csr" else "-"
            print(f"{N:>4} {density:>5.2f} {nnz:>6} {label:>7} "
                  f"{fl:>8} {ram:>6} {wb:>7} {instr:>11} {sp_str:>8} "
                  f"{'OK' if par else 'FAIL':>7}")
        print(f"     accuracy: float MSE={mse_f:.3e}  quant(dense==csr) "
              f"MSE={mse_q:.3e}  (CSR bit-exact → identical accuracy)")
    print("\nFlash = text+data, RAM = data+bss (arm-none-eabi-size).")
    print("Wres B = bytes for the W_res representation (dense N*N vs CSR "
          "val+col+rowptr).")
    print("instr/step = qemu -icount op-count proxy (not silicon cycles); "
          "speedup = dense/csr.")


if __name__ == "__main__":
    main()
