"""Tests for the Target abstraction (host + cortex-m0)."""
from __future__ import annotations
import pathlib
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.targets import (
    Target, CompiledArtifact, RunResult,
    HostTarget, CortexM0Target, MicrobitV1, Microbit,
    RaspberryPiPico, Pico,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build():
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.5,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=40, topology=Topology.SCR,
                                chain_weight=0.85, leak_rate=0.3,
                                seed=42, name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=80,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1)) * 0.3 + 0.5
    Y = np.sin(np.arange(300) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X[200:210]


def test_microbit_class_inherits_cortex_m0():
    assert issubclass(Microbit, CortexM0Target)
    mb = Microbit()
    assert mb.triple == "thumbv6m-none-eabi"
    assert mb.cpu == "cortex-m0"
    assert mb.dtype == "f32"
    assert mb.board.qemu_machine == "microbit"


def test_target_run_default_raises():
    """Subclasses must override run() to provide an emulator path."""
    class Stub(Target):
        name = "stub"
        def compile(self, rc, exe, *, output_dir, **_):
            return CompiledArtifact(target_name=self.name,
                                     output_dir=pathlib.Path(output_dir))
    expect_raises(NotImplementedError, Stub().run,
                  CompiledArtifact(target_name="stub",
                                    output_dir=pathlib.Path("/tmp")))


def test_host_target_emits_artifact():
    rc, exe, _ = _build()
    with tempfile.TemporaryDirectory() as td:
        artifact = HostTarget().compile(rc, exe, output_dir=td)
        assert artifact.target_name == "host-native"
        assert artifact.binary is not None
        assert artifact.binary.exists()
        assert artifact.binary.suffix == ".so"
        assert any(s.name == "rc_predict.h" for s in artifact.sources)
        # The held JIT can predict in-process
        Y = artifact.metadata["jit"].predict(np.zeros((10, 1)))
        assert Y.shape == (10, 1)


def test_cortex_m0_target_requires_test_inputs():
    rc, exe, _ = _build()
    with tempfile.TemporaryDirectory() as td:
        expect_raises(ValueError, Microbit().compile, rc, exe,
                      output_dir=td)


def test_microbit_full_pipeline():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip
    if shutil.which("qemu-system-arm") is None:
        return  # skip
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Microbit()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        assert artifact.binary.exists()
        assert artifact.metadata["board"].name == "microbit-v1"
        assert artifact.metadata["triple"] == "thumbv6m-none-eabi"
        result = target.run(artifact)
        assert result.success, f"QEMU output:\n{result.output}"
        assert "EMULATOR_EXIT" in result.output


def test_microbit_v1_board_constants():
    board = MicrobitV1()
    assert board.flash_kb == 256
    assert board.ram_kb == 16
    assert board.qemu_machine == "microbit"
    assert board.linker_script == "nrf51.ld"
    assert board.cpu == "cortex-m0"


def test_pico_class_inherits_cortex_m0():
    assert issubclass(Pico, CortexM0Target)
    p = Pico()
    assert p.triple == "thumbv6m-none-eabi"
    assert p.cpu == "cortex-m0plus"
    assert p.dtype == "f32"
    assert p.board.qemu_machine == ""


def test_pico_board_constants():
    board = RaspberryPiPico()
    assert board.soc == "RP2040"
    assert board.flash_kb == 2048
    assert board.ram_kb == 264
    assert board.cpu == "cortex-m0plus"
    assert board.linker_script == "rp2040.ld"
    assert board.wokwi_part == "wokwi-pi-pico"
    assert "boot2_rp2040.S" in board.extra_asm
    assert board.qemu_machine == ""


def test_pico_full_build():
    """Pico can be cross-compiled; no on-target run (no QEMU model)."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Pico()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        assert artifact.binary.exists()
        assert artifact.target_name == "cortex-m0plus/raspberry-pi-pico"
        assert artifact.metadata["cpu"] == "cortex-m0plus"
        assert artifact.metadata["board"].soc == "RP2040"


def test_pico_run_refuses_without_qemu_machine():
    """Pico has no QEMU machine. run() must raise — either with on-device
    SWD guidance (if no Wokwi available) or with a WOKWI_CLI_TOKEN hint
    (if wokwi-cli is on PATH but the token isn't set)."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return
    import os
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Pico()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        old_token = os.environ.pop("WOKWI_CLI_TOKEN", None)
        try:
            target.run(artifact)
        except RuntimeError as e:
            msg = str(e)
            assert (
                ("openocd" in msg and "SWD" in msg)
                or "WOKWI_CLI_TOKEN" in msg
            ), f"unexpected error: {msg}"
            return
        finally:
            if old_token is not None:
                os.environ["WOKWI_CLI_TOKEN"] = old_token
        raise AssertionError("Pico.run() should raise; nothing was raised")


def test_pico_stages_wokwi_files():
    """The build pipeline drops a diagram.json + wokwi.toml next to rc.elf,
    so a developer with WOKWI_CLI_TOKEN can `wokwi-cli <output_dir>` directly."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Pico()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        diagram = artifact.output_dir / "diagram.json"
        wokwi_toml = artifact.output_dir / "wokwi.toml"
        assert diagram.exists()
        assert wokwi_toml.exists()
        assert "wokwi-pi-pico" in diagram.read_text()
        assert artifact.binary.name in wokwi_toml.read_text()


def test_pico_wokwi_simulation():
    """End-to-end Wokwi simulation: the ELF boots successfully on Wokwi
    (boot2 CRC validates, bootrom branches to __flash_entry, Reset_Handler
    runs main). The output channel is what's brittle: Wokwi's Pi Pico model
    captures USB-CDC or wired-UART output, NOT ARM semihosting (BKPT #0xAB).

    rclite's main_template.c uses semihosting (the same path that works on
    QEMU micro:bit and openocd-attached real hardware). So `--expect-text`
    won't see EMULATOR_EXIT under Wokwi without further runtime work (full
    TinyUSB stack for USB-CDC, or a UART init plus a wired serial-monitor
    part in diagram.json).

    Skipped by default. To revisit, build a UART-printing variant of the
    main template + wire it in diagram.json."""
    return  # see docstring


def test_target_run_result_failure_path():
    """When the binary doesn't exit cleanly, success=False."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip
    if shutil.which("qemu-system-arm") is None:
        return  # skip
    # Build a known-good artifact then truncate the ELF to force a QEMU error
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Microbit()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        artifact.binary.write_bytes(b"\x00" * 64)  # invalid ELF
        result = target.run(artifact, timeout=10)
        assert not result.success


TESTS = [v for k, v in list(globals().items())
         if k.startswith("test_") and callable(v)]


def main() -> int:
    n_pass = n_fail = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
            n_pass += 1
        except Exception:
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {len(TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
