"""Build all three firmwares, run them on qemu (microbit / Cortex-M0), and
collect Flash/SRAM/parity/latency into out/fw_result.json.

Firmwares (same board, startup, linker, toolchain, semihosting harness):
  * tflm   — TFLM int8 Dense-MLP (LiteRT for Microcontrollers interpreter)
  * rc_i8  — rclite reservoir, affine i8  (portable C kernel)
  * rc_i16 — rclite reservoir, affine i16 (portable C kernel)

Flash = text+data, static RAM = data+bss (both report-comparable because all
working memory is static in both firmwares). Latency is a qemu -icount
instruction estimate (NOT silicon cycles) — median of several runs.

Run after train_tf_esn.py + gen_rclite_fw.py:
    .venv/bin/python benchmarks/tflm_vs_rclite/measure_firmware.py
"""
from __future__ import annotations
import json
import pathlib
import re
import subprocess
import statistics

HERE = pathlib.Path(__file__).resolve().parent
FW = HERE / "firmware"
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
GCCBIN = pathlib.Path(
    "/tmp/tflite-micro/tensorflow/lite/micro/tools/make/downloads/gcc_embedded/bin")
SIZE = str(GCCBIN / "arm-none-eabi-size")
TFLM_ESN_ARENA = 3584        # ESN cell: minimum that allocates (arena_used = 2688)
N_LAT = 5


def sh(cmd, **kw):
    return subprocess.run(cmd, capture_output=True, text=True, **kw)


def elf_sizes(elf: pathlib.Path) -> dict:
    out = sh([SIZE, str(elf)]).stdout.strip().splitlines()[-1].split()
    text, data, bss = int(out[0]), int(out[1]), int(out[2])
    return {"text": text, "data": data, "bss": bss,
            "flash": text + data, "ram": data + bss}


def qemu_run(elf: pathlib.Path) -> str:
    cp = sh(["qemu-system-arm", "-M", "microbit", "-nographic", "-semihosting",
             "-icount", "shift=0", "-kernel", str(elf)], timeout=90)
    return cp.stdout + cp.stderr


def _grep_int(text: str, key: str):
    m = re.search(rf"{key}:\s*(-?\d+)", text)
    return int(m.group(1)) if m else None


def latency_median(elf: pathlib.Path, key: str) -> int:
    vals = []
    for _ in range(N_LAT):
        v = _grep_int(qemu_run(elf), key)
        if v is not None:
            vals.append(v)
    return int(statistics.median(vals)) if vals else -1


def measure_tflm_esn() -> dict:
    """The SAME reservoir as rclite, deployed as a TFLM single-step cell."""
    sh(["bash", str(FW / "build_tflm_esn.sh")],
       env={**__import__("os").environ, "ARENA_SIZE": str(TFLM_ESN_ARENA)})
    elf = FW / "build" / "tflm_esn.elf"
    out = qemu_run(elf)
    d = elf_sizes(elf)
    d.update({
        "name": "TFLM ESN cell (same reservoir)",
        "arena_size": TFLM_ESN_ARENA,
        "arena_used": _grep_int(out, "arena_used_bytes"),
        "functional_match": "FUNCTIONAL_MATCH" in out,
        "max_abs_diff_scaled": _grep_int(out, "max_abs_diff_scaled"),
        "instr_per_inference": latency_median(elf, "instr_per_step"),
    })
    return d


def measure_rclite(variant: str) -> dict:
    sh(["bash", str(FW / "build_rclite.sh"), variant])
    elf = FW / "build" / f"rclite_{variant}.elf"
    out = qemu_run(elf)
    d = elf_sizes(elf)
    d.update({
        "name": f"rclite reservoir affine {variant}",
        "parity_ok": "PARITY_OK" in out,
        "instr_per_inference": latency_median(elf, "instr_per_step"),
    })
    return d


def main() -> int:
    res = {
        "board": "BBC micro:bit v1 (nRF51822, Cortex-M0, 256KB flash / 16KB SRAM)",
        "qemu_machine": "microbit",
        "toolchain": "arm-none-eabi-gcc (TFLM-pinned 14.3.1), -Os, --gc-sections",
        "latency_note": "qemu -icount instruction estimate, not silicon cycles",
        "tflm_esn": measure_tflm_esn(),
        "rc_i8": measure_rclite("i8"),
        "rc_i16": measure_rclite("i16"),
    }
    (OUT / "fw_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
