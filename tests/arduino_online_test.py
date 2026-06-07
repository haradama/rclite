"""Arduino Uno on-device online-learning build path (symmetric LMS / NLMS).

`ArduinoUnoTarget.compile_symmetric_online` wires the symmetric online kernel
(`rc_train_step` / `rc_infer_step` / `rc_export_W_out`) into an Uno sketch that
streams an embedded (input, target) sequence and self-checks, on-device, that
the per-step predictions AND the final learned readout are bit-exact with the
host reference `IntegerLMSLearner`.

Three tiers, each gated on the tools present:
  * always: render the sketch, then compile the kernel with avr-gcc (AVR ABI).
  * arduino-cli + core: full sketch build → Flash/SRAM fit inside the Uno.
  * + libsimavr: run the *exact* ELF under simavr and assert PARITY_OK
    (bit-exact on real 8-bit AVR, where the kernel's i64 math is emulated).
"""
from __future__ import annotations
import pathlib
import re
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I16FixedPoint, quantize_model,
)
from rclite.targets import ArduinoUnoTarget


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

HAVE_AVR_GCC = shutil.which("avr-gcc") is not None
HAVE_ARDUINO = shutil.which("arduino-cli") is not None
_UART_SIM = (pathlib.Path(__file__).resolve().parent.parent /
             "examples" / "arduino_esn_demo" / "sim" / "uart_sim.c")


def _have_libsimavr() -> bool:
    if shutil.which("gcc") is None or not _UART_SIM.exists():
        return False
    return pathlib.Path("/usr/include/simavr/sim_avr.h").exists()


def _model(units=40, normalized=False):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.SCR,
                                chain_weight=0.9, leak_rate=0.3, seed=7,
                                name="r"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=10,
                            include_bias=True, include_input=True, name="o"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(7)
    X = rng.standard_normal((60, 1)) * 0.15
    Y = np.sin(np.arange(60) * 0.1)[:, None]
    exe.fit(X[:50], Y[:50])
    qm = quantize_model(rc, exe, QuantConfig(state_frac=10, input_frac=8,
                                             weight_frac=8),
                        target=I16FixedPoint(), lut=TanhLUTSpec(n=64))
    return qm, X, Y


def _build_and_verify(normalized=False, lr=1e-2, steps=40, warmup=10):
    """Render → avr-gcc kernel → (arduino-cli build) → (simavr PARITY_OK)."""
    qm, X, Y = _model(normalized=normalized)
    X, Y = X[:steps], Y[:steps]
    tgt = ArduinoUnoTarget()
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "ard"
        # Stage 1: render only (no arduino-cli), then avr-gcc the kernel.
        # (collect_training_stream mutates qm.W_out_q; qm is not reused after.)
        art = tgt.compile_symmetric_online(
            qm, output_dir=out, X=X, Y=Y, learning_rate=lr,
            normalized=normalized, warmup=warmup, build=False)
        kernel = out / "sketch" / "rc_kernel.c"
        assert kernel.exists() and (out / "sketch" / "sketch.ino").exists()
        r = subprocess.run(
            ["avr-gcc", "-mmcu=atmega328p", "-Os", "-std=c99", "-c",
             str(kernel), "-o", str(out / "rc_kernel.o")],
            capture_output=True, text=True)
        assert r.returncode == 0, f"avr-gcc kernel failed:\n{r.stderr}"

        if not HAVE_ARDUINO:
            return "rendered+avr-gcc (no arduino-cli)"

        # Stage 2: full Uno build (must fit Flash 32K / SRAM 2K).
        out2 = pathlib.Path(td) / "ard2"
        qm2, X2, Y2 = _model(normalized=normalized)
        art = tgt.compile_symmetric_online(
            qm2, output_dir=out2, X=X2[:steps], Y=Y2[:steps],
            learning_rate=lr, normalized=normalized, warmup=warmup, build=True)
        flash = art.metadata.get("flash_bytes")
        sram = art.metadata.get("sram_bytes")
        assert flash and flash < 32256, f"Flash {flash} overflows Uno"
        assert sram and sram < 2048, f"SRAM {sram} overflows Uno"
        elf = list((out2 / "build").glob("*.elf"))
        if not elf or not _have_libsimavr():
            return f"built (Flash={flash} SRAM={sram}); sim skipped"

        # Stage 3: run the exact ELF under simavr, assert PARITY_OK.
        drv = pathlib.Path(td) / "uart_sim"
        rb = subprocess.run(["gcc", str(_UART_SIM), "-lsimavr", "-o", str(drv)],
                            capture_output=True, text=True)
        assert rb.returncode == 0, f"sim driver build failed:\n{rb.stderr}"
        rs = subprocess.run([str(drv), str(elf[0])], capture_output=True,
                            text=True, timeout=600)
        out_txt = rs.stdout + rs.stderr
        assert "PARITY_OK" in out_txt, f"sim parity failed:\n{out_txt[-800:]}"
        m = re.search(r"us_per_step=(\d+)", out_txt)
        ups = m.group(1) if m else "?"
        return f"PARITY_OK on simavr (Flash={flash} SRAM={sram} us/step={ups})"


def test_arduino_online_lms_build_and_parity():
    if not HAVE_AVR_GCC:
        print("  (skip: avr-gcc not on PATH)")
        return
    print("    " + _build_and_verify(normalized=False))


def test_arduino_online_nlms_build_and_parity():
    if not HAVE_AVR_GCC:
        print("  (skip: avr-gcc not on PATH)")
        return
    print("    " + _build_and_verify(normalized=True, lr=0.5))


def test_arduino_online_rejects_i8():
    qm, X, Y = _model()
    tgt = ArduinoUnoTarget()
    with tempfile.TemporaryDirectory() as td:
        # i32/i16 only — fabricate an i8-claiming target view.
        class _Fake:
            storage_bits = 8
        saved = qm.target
        try:
            qm.target = _Fake()
            try:
                tgt.compile_symmetric_online(qm, output_dir=pathlib.Path(td),
                                             X=X, Y=Y, build=False)
                raise AssertionError("expected NotImplementedError for i8")
            except NotImplementedError:
                pass
        finally:
            qm.target = saved


TESTS = [
    test_arduino_online_lms_build_and_parity,
    test_arduino_online_nlms_build_and_parity,
    test_arduino_online_rejects_i8,
]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
        except Exception:
            failures += 1
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
