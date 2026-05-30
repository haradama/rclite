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
    WasmTarget, Wasmtime,
    GbaTarget, Gba,
    NesTarget, Nes,
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


def test_wasmtime_class_inherits_wasm_target():
    assert issubclass(Wasmtime, WasmTarget)
    wt = Wasmtime()
    assert wt.triple == "wasm32-wasip1"
    assert wt.rust_target == "wasm32-wasip1"
    assert wt.dtype == "f32"


def test_wasm_target_rejects_non_f32():
    expect_raises(ValueError, WasmTarget, dtype="f64")


def test_wasm_target_requires_test_inputs():
    rc, exe, _ = _build()
    with tempfile.TemporaryDirectory() as td:
        expect_raises(ValueError, Wasmtime().compile, rc, exe,
                      output_dir=td)


def test_wasmtime_full_pipeline():
    if shutil.which("rustc") is None:
        return  # skip — rustc not on PATH
    if shutil.which("wasmtime") is None:
        return  # skip — wasmtime not on PATH
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = Wasmtime()
        artifact = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        assert artifact.binary is not None
        assert artifact.binary.exists()
        assert artifact.binary.suffix == ".wasm"
        assert artifact.metadata["triple"] == "wasm32-wasip1"
        assert artifact.metadata["dtype"] == "f32"
        assert artifact.metadata["T"] == sample.shape[0]
        result = target.run(artifact)
        assert result.success, f"wasmtime output:\n{result.output}"
        assert "EMULATOR_EXIT" in result.output


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


def _build_affine_gba(units=24, T=200, seed=0):
    """A small affine-quantized model for the GBA target tests."""
    from rclite.quant import (calibrate_from_data, quantize_model_affine,
                              LUTStrategy)
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.SCR,
                                 chain_weight=0.9, chain_feedback=0.1,
                                 leak_rate=0.3, seed=42, name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=30,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.2
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[:T - 50], Y[:T - 50])
    cfg = calibrate_from_data(rc, exe, X[:T - 50], storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg,
                                lut_strategy=LUTStrategy.linear_interp(64))
    return qm, X[T - 50:T - 40]


def test_gba_class_attributes():
    assert issubclass(Gba, GbaTarget)
    g = Gba()
    assert g.triple == "thumbv4t-none-eabi"
    assert g.cpu == "arm7tdmi"
    assert g.name == "gba/arm7tdmi"


def test_gba_compile_affine_emits_rom():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip — no ARM toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Gba().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample)
        assert art.binary is not None and art.binary.exists()
        assert art.binary.suffix == ".gba"
        assert art.metadata["triple"] == "thumbv4t-none-eabi"
        assert art.metadata["affine"] is True


def test_gba_full_pipeline_mgba():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip
    if shutil.which("mgba") is None and shutil.which("/usr/games/mgba") is None:
        return  # skip — no emulator
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        target = Gba()
        art = target.compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample)
        result = target.run(art, timeout=6)
        assert result.success, f"mGBA output:\n{result.output}"
        assert "TEST_PASS" in result.output
        assert "TEST_FAIL" not in result.output


def test_nes_class_attributes():
    assert issubclass(Nes, NesTarget)
    n = Nes()
    assert n.name == "nes/6502"
    assert n.mapper == "nrom"
    assert n.cc == "mos-nes-nrom-clang"


def test_nes_compile_emits_sources_without_toolchain():
    # The C kernel + harness are emitted even when llvm-mos is absent;
    # build=False skips the link step so this runs everywhere.
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Nes().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample, build=False)
        assert art.binary is None
        assert art.metadata["cpu"] == "6502"
        assert art.metadata["affine"] is True
        srcs = {p.name for p in art.sources}
        assert {"main.c", "rc_kernel.c"} <= srcs
        # harness embeds the blargg $6000 protocol signature
        main_txt = (pathlib.Path(td) / "main.c").read_text()
        assert "0x6000" in main_txt and "TEST_PASS" in main_txt


def test_nes_compile_emits_rom():
    if shutil.which("mos-nes-nrom-clang") is None:
        return  # skip — no llvm-mos toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Nes().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample)
        assert art.binary is not None and art.binary.exists()
        assert art.binary.suffix == ".nes"


def test_nes_full_pipeline_emulator():
    if shutil.which("mos-nes-nrom-clang") is None:
        return  # skip
    has_mesen = any(shutil.which(b) for b in ("Mesen", "mesen", "Mesen2"))
    has_fceux = shutil.which("fceux") or shutil.which("/usr/games/fceux")
    if not has_mesen and not has_fceux:
        return  # skip — no NES emulator (Mesen --testrunner or fceux+Lua)
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        target = Nes()
        art = target.compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample)
        result = target.run(art)  # auto: Mesen if present, else FCEUX
        assert result.success, f"emulator output:\n{result.output}"
        assert "TEST_PASS" in result.output
        assert "TEST_FAIL" not in result.output


def test_nes_compile_rejects_non_affine():
    rc, exe, sample = _build()
    expect_raises(NotImplementedError, Nes().compile, rc, exe,
                  output_dir="/tmp/unused_nes")


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
