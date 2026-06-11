#!/usr/bin/env python3
"""Build (and verify) the GBA Echo State Network demo.

Trains a dense-readout ESN on Mackey-Glass with rclite, affine-quantizes it to
i16 (integer math + LUT tanh — no soft-float, so it runs fast on the ARM7TDMI),
cross-compiles the reservoir kernel for the Game Boy Advance (thumbv4t), emits
the eval series + quant params as headers, links the Mode-3 plotting front-end
(main.c) into a .gba cartridge, runs it in mGBA, and reconstructs a PNG of the
screen from the framebuffer the GBA streams back over the debug log.

    python examples/gba_esn_demo/build.py
"""

from __future__ import annotations
import os
import pathlib
import shutil
import signal
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    calibrate_from_data,
    quantize_model_affine,
    LUTStrategy,
    AffineQuantizedExecutor,
)
from rclite.codegen.llvm import emit_quantized_affine_module
from rclite.targets.gba import GbaTarget

HERE = pathlib.Path(__file__).resolve().parent
SUPPORT = HERE.parents[1] / "rclite" / "targets" / "gba" / "support"
BUILD = HERE / "build"

N = 64
EVAL_STEPS = 240
STORAGE_BITS = 16


def mackey_glass(
    n: int,
    tau: int = 17,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_init: int = 500,
) -> np.ndarray:
    rng = np.random.default_rng(0)
    L = n + n_init
    x = np.zeros(L)
    x[: tau + 1] = 1.2 + 0.05 * rng.standard_normal(tau + 1)
    for t in range(tau + 1, L):
        xt = x[t - tau]
        x[t] = x[t - 1] + beta * xt / (1.0 + xt**10) - gamma * x[t - 1]
    return x[n_init:]


def train():
    series = mackey_glass(n=4000)
    X, Y = series[:-1, None], series[1:, None]  # one-step-ahead pairs
    n_train = 3000
    in_off = float(X[:n_train].mean())

    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=1.0,
            input_offset=in_off,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=N,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=1.0,
            topology=Topology.SCR,
            chain_weight=0.5,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])

    eval_start = n_train + 100
    eval_u = series[eval_start : eval_start + EVAL_STEPS]
    eval_truth = series[eval_start + 1 : eval_start + EVAL_STEPS + 1]

    # affine-i16 quantize; LUT tanh
    cfg = calibrate_from_data(rc, exe, X[:n_train], storage_bits=STORAGE_BITS)
    qm = quantize_model_affine(
        rc, exe, cfg, lut_strategy=LUTStrategy.linear_interp(256)
    )

    y_q = AffineQuantizedExecutor(qm).predict(eval_u[:, None]).ravel()
    tail = slice(40, EVAL_STEPS)
    rmse = float(np.sqrt(np.mean((y_q[tail] - eval_truth[tail]) ** 2)))
    rng = (float(eval_truth.min()), float(eval_truth.max()))
    print(
        f"[train] SCR N={N} affine-i{STORAGE_BITS}: one-step RMSE (after "
        f"transient) = {rmse:.4f}  (truth range {rng[0]:.3f}..{rng[1]:.3f})"
    )
    return qm, eval_u, eval_truth


def emit_data_header(qm, eval_u, eval_truth, path: pathlib.Path):
    cfg = qm.config
    x_q = cfg.input.quantize_array(eval_u[:, None]).astype(np.int16).ravel()
    truth = np.asarray(eval_truth, dtype=np.float32)
    txt = [
        "#ifndef ESN_DATA_H\n#define ESN_DATA_H\n",
        f"#define EVAL_STEPS {len(eval_u)}\n",
        f"#define OUT_SCALE {cfg.output.scale:.8e}f\n",
        f"#define OUT_ZP {int(cfg.output.zero_point)}\n",
        "static const int16_t EVAL_U_Q[EVAL_STEPS] = { "
        + ", ".join(str(int(v)) for v in x_q)
        + " };\n",
        "static const float EVAL_TRUTH[EVAL_STEPS] = { "
        + ", ".join(f"{float(v):.8e}f" for v in truth)
        + " };\n",
        "#endif\n",
    ]
    path.write_text("".join(txt))


def build_rom(qm, eval_u, eval_truth):
    BUILD.mkdir(parents=True, exist_ok=True)
    cc = "arm-none-eabi-gcc"
    if shutil.which(cc) is None:
        raise SystemExit(f"{cc} not found — install gcc-arm-none-eabi")

    # 1. affine-i16 reservoir kernel object for thumbv4t (reuse target lowering)
    rc_o = GbaTarget()._cross_object(emit_quantized_affine_module(qm), BUILD)

    # 2. data + support + front-end
    emit_data_header(qm, eval_u, eval_truth, BUILD / "esn_data.h")
    for f in ("crt0.s", "gba.ld", "mgba_log.h"):
        shutil.copy(SUPPORT / f, BUILD / f)
    shutil.copy(HERE / "main.c", BUILD / "main.c")

    cpu = ["-mcpu=arm7tdmi", "-mthumb-interwork"]
    subprocess.run(
        [
            cc,
            "-c",
            *cpu,
            "-marm",
            str(BUILD / "crt0.s"),
            "-o",
            str(BUILD / "crt0.o"),
        ],
        check=True,
    )
    subprocess.run(
        [
            cc,
            "-c",
            *cpu,
            "-mthumb",
            "-O2",
            f"-I{BUILD}",
            str(BUILD / "main.c"),
            "-o",
            str(BUILD / "main.o"),
        ],
        check=True,
    )
    elf = BUILD / "esn_demo.elf"
    subprocess.run(
        [
            cc,
            *cpu,
            "-mthumb",
            "-T",
            str(BUILD / "gba.ld"),
            "-nostartfiles",
            "-Wl,--gc-sections",
            "--specs=nosys.specs",
            str(BUILD / "crt0.o"),
            str(BUILD / "main.o"),
            str(rc_o),
            "-o",
            str(elf),
            "-lgcc",
            "-lc",
            "-lnosys",
        ],
        check=True,
    )
    rom = BUILD / "esn_demo.gba"
    subprocess.run(
        [cc.replace("gcc", "objcopy"), "-O", "binary", str(elf), str(rom)],
        check=True,
    )
    print(f"[build] {rom}  ({rom.stat().st_size} bytes)")
    return rom


def run_and_capture(rom: pathlib.Path, timeout=20):
    mgba = shutil.which("mgba") or shutil.which("/usr/games/mgba")
    if mgba is None:
        print("[run] mgba not found — skipping emulator run")
        return None
    env = dict(os.environ, SDL_VIDEODRIVER="dummy", SDL_AUDIODRIVER="dummy")
    cmd = (["stdbuf", "-oL", "-eL"] if shutil.which("stdbuf") else []) + [
        mgba,
        "-l",
        "31",
        str(rom),
    ]
    proc = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        start_new_session=True,
        env=env,
    )
    try:
        out, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(os.getpgid(proc.pid), signal.SIGKILL)
        out, _ = proc.communicate()
    return out


def _write_png(rgb: np.ndarray, path: pathlib.Path):
    """Minimal RGB PNG writer (stdlib zlib only)."""
    import struct, zlib

    h, w, _ = rgb.shape
    raw = bytearray()
    for y in range(h):
        raw.append(0)  # filter type 0
        raw.extend(rgb[y].tobytes())

    def chunk(tag, data):
        c = struct.pack(">I", len(data)) + tag + data
        return c + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)
    png = (
        b"\x89PNG\r\n\x1a\n"
        + chunk(b"IHDR", ihdr)
        + chunk(b"IDAT", zlib.compress(bytes(raw), 9))
        + chunk(b"IEND", b"")
    )
    path.write_bytes(png)


def reconstruct_png(out: str, path: pathlib.Path):
    lines = [
        l.split("GBA Debug: ", 1)[1]
        for l in out.splitlines()
        if "GBA Debug:" in l
    ]
    if "RENDER_DONE" not in lines:
        print("[verify] RENDER_DONE not seen — render did not finish")
        return False
    b, e = lines.index("FB_BEGIN"), lines.index("FB_END")
    data = bytes.fromhex("".join(lines[b + 1 : e]))
    W, H = 240, 160
    pal = np.array(
        [[0, 0, 0], [255, 255, 255], [0, 220, 0], [230, 0, 0]], dtype=np.uint8
    )
    px = np.zeros(W * H, dtype=np.uint8)
    for i in range(min(len(data), (W * H + 3) // 4)):
        for k in range(4):
            idx = i * 4 + k
            if idx < W * H:
                px[idx] = (data[i] >> (k * 2)) & 0x3
    img = pal[px].reshape(H, W, 3)
    img = np.repeat(np.repeat(img, 3, axis=0), 3, axis=1)  # 3x nearest upscale
    _write_png(img, path)
    print(
        f"[verify] reconstructed screen -> {path}  "
        f"(green px={(px == 2).sum()}, red px={(px == 3).sum()})"
    )
    return True


def main():
    qm, eval_u, eval_truth = train()
    rom = build_rom(qm, eval_u, eval_truth)
    out = run_and_capture(rom)
    if out is None:
        return
    ok = reconstruct_png(out, HERE / "screen.png")
    print("[done] OK" if ok else "[done] render incomplete")


if __name__ == "__main__":
    main()
