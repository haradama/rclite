#!/usr/bin/env python3
"""Build (and screenshot) the NES Echo State Network demo.

Trains an Echo State Network on the Mackey-Glass series with rclite, affine-
quantizes it to i16 (pure integer math + LUT tanh — no FPU, no hardware
multiply, which is what the MOS 6502 needs), and lowers the reservoir kernel to
a Nintendo Entertainment System cartridge (`.nes`, NROM mapper) with llvm-mos.

The cartridge's front-end (main.c) runs the kernel on the embedded eval inputs
and plots its prediction (green) against the ground truth (white) on the NES
background via a CHR-RAM bitmap. This script then runs the ROM in FCEUX and
saves a screenshot of the console actually rendering its own output — the NES
counterpart of the GBA demo's framebuffer grab.

    python examples/nes_esn_demo/build.py
"""
from __future__ import annotations
import os
import pathlib
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from rclite import (InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
                    Activation, Topology, Trainer)
from rclite.runtime import RCExecutor
from rclite.quant import (calibrate_from_data, quantize_model_affine,
                          LUTStrategy, AffineQuantizedExecutor)
from rclite.targets.arduino.emit_c import emit_affine_kernel_c
from font import build_font, FONT_FIRST, FONT_COUNT

HERE = pathlib.Path(__file__).resolve().parent
BUILD = HERE / "build"

N = 48
EVAL_STEPS = 120
STORAGE_BITS = 16   # i16: near-float accuracy (one-step RMSE ~0.005 vs ~0.002
                    # float; i8 would cost ~37x to visible quantization noise).
                    # The labels render immediately; the plot appears once the
                    # 6502 finishes the integer kernel (tens of seconds at the
                    # NES's realtime clock — our build screenshots it at turbo).


def mackey_glass(n: int, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n_init: int = 500) -> np.ndarray:
    rng = np.random.default_rng(0)
    L = n + n_init
    x = np.zeros(L)
    x[:tau + 1] = 1.2 + 0.05 * rng.standard_normal(tau + 1)
    for t in range(tau + 1, L):
        xt = x[t - tau]
        x[t] = x[t - 1] + beta * xt / (1.0 + xt ** 10) - gamma * x[t - 1]
    return x[n_init:]


def train():
    series = mackey_glass(n=4000)
    X, Y = series[:-1, None], series[1:, None]      # one-step-ahead pairs
    n_train = 3000
    in_off = float(X[:n_train].mean())

    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=in_off, name="in"),
        reservoir=ReservoirNode(units=N, activation=Activation.TANH,
                                spectral_radius=0.9, leak_rate=0.3, density=1.0,
                                topology=Topology.SCR, chain_weight=0.5,
                                seed=42, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                            trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=200, include_bias=True,
                            include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])

    eval_start = n_train + 100
    eval_u = series[eval_start:eval_start + EVAL_STEPS]
    eval_truth = series[eval_start + 1:eval_start + EVAL_STEPS + 1]

    cfg = calibrate_from_data(rc, exe, X[:n_train], storage_bits=STORAGE_BITS)
    qm = quantize_model_affine(rc, exe, cfg,
                               lut_strategy=LUTStrategy.linear_interp(64))

    y_q = AffineQuantizedExecutor(qm).predict(eval_u[:, None]).ravel()
    tail = slice(min(8, EVAL_STEPS // 3), EVAL_STEPS)   # drop the transient
    rmse = float(np.sqrt(np.mean((y_q[tail] - eval_truth[tail]) ** 2)))
    print(f"[train] SCR N={N} affine-i{STORAGE_BITS}: one-step RMSE (after "
          f"transient) = {rmse:.4f}  (truth range "
          f"{eval_truth.min():.3f}..{eval_truth.max():.3f})")
    return qm, eval_u, eval_truth, y_q


def emit_data_header(qm, eval_u, eval_truth, y_pred, path: pathlib.Path):
    cfg = qm.config
    x_q = cfg.input.quantize_array(eval_u[:, None]).astype(np.int16).ravel()
    truth_q = cfg.output.quantize_array(eval_truth[:, None]).astype(np.int16).ravel()

    # Plot y-range: cover both curves with a small margin, then take the
    # output-scale quant values of the endpoints as the row-mapping anchors.
    lo = float(min(eval_truth.min(), y_pred.min()))
    hi = float(max(eval_truth.max(), y_pred.max()))
    pad = 0.05 * (hi - lo or 1.0)
    q_lo = int(cfg.output.quantize_array(np.array([[lo - pad]]))[0, 0])
    q_hi = int(cfg.output.quantize_array(np.array([[hi + pad]]))[0, 0])

    txt = [
        "#ifndef ESN_DATA_H\n#define ESN_DATA_H\n#include <stdint.h>\n",
        f"#define EVAL_STEPS {len(eval_u)}\n",
        f"#define Y_Q_LO {q_lo}\n",
        f"#define Y_Q_HI {q_hi}\n",
        "static const int16_t EVAL_X_Q[EVAL_STEPS] = { "
        + ", ".join(str(int(v)) for v in x_q) + " };\n",
        "static const int16_t TRUTH_Q[EVAL_STEPS] = { "
        + ", ".join(str(int(v)) for v in truth_q) + " };\n",
        "#endif\n",
    ]
    path.write_text("".join(txt))


def emit_font_header(path: pathlib.Path):
    font = build_font()
    txt = [
        "#ifndef FONT_H\n#define FONT_H\n#include <stdint.h>\n",
        f"#define FONT_FIRST {FONT_FIRST}\n",
        f"#define FONT_COUNT {FONT_COUNT}\n",
        "static const uint8_t FONT[" + str(len(font)) + "] = { "
        + ", ".join(str(b) for b in font) + " };\n",
        "#endif\n",
    ]
    path.write_text("".join(txt))


def build_rom(qm, eval_u, eval_truth, y_pred):
    BUILD.mkdir(parents=True, exist_ok=True)
    cc = "mos-nes-nrom-clang"
    if shutil.which(cc) is None:
        print(f"[build] {cc} not found — install the llvm-mos SDK and put its "
              "bin/ on PATH. Emitting sources only.")
    (BUILD / "rc_kernel.c").write_text(
        emit_affine_kernel_c(qm, allow_i32_accum=True))
    emit_data_header(qm, eval_u, eval_truth, y_pred, BUILD / "esn_data.h")
    emit_font_header(BUILD / "font.h")
    shutil.copy(HERE / "main.c", BUILD / "main.c")
    if shutil.which(cc) is None:
        return None

    rom = BUILD / "rc.nes"
    cmd = [cc, "-Os", "-flto", f"-I{BUILD}",
           str(BUILD / "main.c"), str(BUILD / "rc_kernel.c"),
           "-lneslib", "-o", str(rom)]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    if cp.returncode != 0:
        raise SystemExit(f"link failed:\n{' '.join(cmd)}\n{cp.stdout}\n{cp.stderr}")
    print(f"[build] {rom}  ({rom.stat().st_size} bytes)")
    return rom


def screenshot(rom: pathlib.Path, out_png: pathlib.Path, timeout=180):
    fceux = shutil.which("fceux") or shutil.which("/usr/games/fceux")
    if fceux is None:
        print("[shot] fceux not found — install it (`apt install fceux`) to "
              "capture the screenshot")
        return
    lua = BUILD / "_shot.lua"
    lua.write_text(
        'if emu.speedmode then emu.speedmode("maximum") end\n'
        'local n = 0\n'
        'while memory.readbyte(0x7000) ~= 0xA5 and n < 100000 do\n'
        '  emu.frameadvance(); n = n + 1\n'
        'end\n'
        'for i=1,3 do emu.frameadvance() end\n'
        'gui.savescreenshotas("%s")\n'
        'emu.frameadvance()\n'
        'os.exit(0)\n' % str(out_png)
    )
    cmd = [fceux, "--no-config", "1", "--loadlua", str(lua), str(rom)]
    if shutil.which("xvfb-run") is not None:
        cmd = ["xvfb-run", "-a"] + cmd
    subprocess.run(cmd, capture_output=True, text=True,
                   timeout=timeout, env=dict(os.environ))
    if out_png.exists():
        print(f"[shot] NES screen -> {out_png}")
    else:
        print("[shot] screenshot not produced")


def main():
    qm, eval_u, eval_truth, y_pred = train()
    rom = build_rom(qm, eval_u, eval_truth, y_pred)
    if rom is None:
        return
    screenshot(rom, HERE / "screen.png")


if __name__ == "__main__":
    main()
