"""Tests for the Target abstraction (host + cortex-m0)."""

from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.targets import (
    Target,
    CompiledArtifact,
    HostTarget,
    CortexM0Target,
    MicrobitV1,
    Microbit,
    WasmTarget,
    Wasmtime,
    BrowserWasm,
    GbaTarget,
    Gba,
    NesTarget,
    Nes,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _rust_target_available(target="wasm32-wasip1", rustc="rustc"):
    """Whether rustc can actually link for ``target``.

    ``rustc`` being on PATH is necessary but not sufficient: CI runners often
    have the host toolchain without the wasm32 ``rust-std`` component, so
    linking ``main.rs`` fails with E0463 ("can't find crate for `std`").
    Probe the target libdir and confirm a std rlib is present before running
    the wasm pipeline tests, otherwise skip them.
    """
    if shutil.which(rustc) is None:
        return False
    try:
        cp = subprocess.run(
            [rustc, "--print", "target-libdir", "--target", target],
            capture_output=True,
            text=True,
        )
    except OSError:
        return False
    if cp.returncode != 0:
        return False
    libdir = pathlib.Path(cp.stdout.strip())
    return libdir.is_dir() and any(libdir.glob("libstd-*.rlib"))


def _build():
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=0.5,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=40,
            topology=Topology.SCR,
            chain_weight=0.85,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=80,
            include_bias=True,
            include_input=True,
            name="out",
        ),
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
            return CompiledArtifact(
                target_name=self.name, output_dir=pathlib.Path(output_dir)
            )

    expect_raises(
        NotImplementedError,
        Stub().run,
        CompiledArtifact(target_name="stub", output_dir=pathlib.Path("/tmp")),
    )


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
        expect_raises(ValueError, Microbit().compile, rc, exe, output_dir=td)


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


def test_wasm_simd_defaults_on_and_names():
    on = Wasmtime()
    off = Wasmtime(simd=False)
    assert on.simd is True
    assert off.simd is False
    assert on.name.endswith("+simd128")
    assert not off.name.endswith("+simd128")
    # `+simd128` feature flag only when SIMD is on.
    assert on._features() == "+simd128"
    assert off._features() == ""


def test_wasm_simd_emits_v128_instructions():
    """With SIMD on, the matmul inner loops lower to packed v128 ops;
    with SIMD off they stay scalar."""
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
    import re

    rc, exe, sample = _build()
    counts = {}
    for simd in (True, False):
        with tempfile.TemporaryDirectory() as td:
            Wasmtime(simd=simd).compile(
                rc, exe, output_dir=td, test_inputs=sample
            )
            asm = pathlib.Path(td, "rc_predict.s").read_text()
            counts[simd] = len(re.findall(r"\bf32x4\.|\bv128\.", asm))
    assert counts[True] > 0, "expected v128 ops in the SIMD build"
    assert counts[False] == 0, "scalar build should have no v128 ops"


def _build_quantized_wasm(storage_bits, state_frac):
    """A small symmetric-quantized model for the WASM integer path."""
    from rclite.quant import (
        QuantConfig,
        TanhLUTSpec,
        quantize_model,
        I8Symmetric,
        I16FixedPoint,
        I32FixedPoint,
    )

    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=0.0,
            input_scaling=1.0,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=24,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=40,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1)) * 0.2
    Y = np.sin(np.arange(300) * 0.1)[:, None]
    exe.fit(X, Y)
    target = {8: I8Symmetric, 16: I16FixedPoint, 32: I32FixedPoint}[
        storage_bits
    ]()
    cfg = QuantConfig(state_frac=state_frac, input_frac=4, weight_frac=4)
    qm = quantize_model(rc, exe, cfg, target=target, lut=TanhLUTSpec(n=64))
    return qm, X[250:258]


def test_wasm_compile_quantized_rejects_bad_storage():
    rc, exe, sample = _build()

    class _FakeTarget:
        storage_bits = 4

    class _FakeQModel:
        rc = None
        config = None
        target = _FakeTarget()
        M = 1

    with tempfile.TemporaryDirectory() as td:
        expect_raises(
            NotImplementedError,
            Wasmtime().compile_quantized,
            _FakeQModel(),
            output_dir=td,
            test_inputs=sample,
        )


def test_wasm_quantized_bit_exact():
    """i8 / i16 / i32 quantized kernels cross-compiled to wasm32 reproduce
    the host quantized kernel bit-for-bit under wasmtime."""
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
    if shutil.which("wasmtime") is None:
        return  # skip — wasmtime not on PATH
    for storage_bits, state_frac in [(32, 16), (16, 10), (8, 5)]:
        qm, sample = _build_quantized_wasm(storage_bits, state_frac)
        with tempfile.TemporaryDirectory() as td:
            target = Wasmtime()
            art = target.compile_quantized(
                qm, output_dir=td, test_inputs=sample
            )
            assert art.binary.exists()
            assert art.metadata["quantized"] is True
            assert art.metadata["dtype"] == f"i{storage_bits}"
            result = target.run(art)
            assert result.success, f"wasmtime output:\n{result.output}"
            assert "BIT_EXACT: yes" in result.output, (
                f"i{storage_bits} not bit-exact:\n{result.output}"
            )


def test_browser_class_and_inheritance():
    assert issubclass(BrowserWasm, WasmTarget)
    b = BrowserWasm()
    assert b.triple == "wasm32-wasip1"
    assert b.name.endswith("browser+simd128")
    assert BrowserWasm(simd=False).name == "wasm32/browser"


def test_browser_f32_reactor():
    """The f32 browser build is a reactor: exports rc_predict/memory, imports
    only env.tanhf, and ships a JS loader + HTML demo with no leftover
    template placeholders."""
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        target = BrowserWasm()
        art = target.compile(rc, exe, output_dir=td, test_inputs=sample)
        assert art.metadata["browser"] is True
        assert art.metadata["reactor"] is True
        assert art.metadata["imports"] == ["env.tanhf"]
        for sym in ("rc_predict", "memory", "__heap_base"):
            assert sym in art.metadata["exports"]
        outdir = pathlib.Path(td)
        loader = outdir / "rclite.js"
        html = outdir / "index.html"
        assert loader.exists() and html.exists()
        assert "@@" not in loader.read_text(), "unfilled placeholder in loader"
        assert "@@" not in html.read_text(), "unfilled placeholder in html"
        # structure-only smoke (f32 needs a JS host for tanhf)
        assert target.run(art).success


def test_browser_quantized_zero_imports():
    """The quantized browser build has ZERO imports and instantiates/runs in
    a non-WASI host (wasmtime --invoke)."""
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
    qm, sample = _build_quantized_wasm(16, 10)
    with tempfile.TemporaryDirectory() as td:
        target = BrowserWasm()
        art = target.compile_quantized(qm, output_dir=td, test_inputs=sample)
        assert art.metadata["imports"] == [], (
            f"quantized reactor should have no imports, got "
            f"{art.metadata['imports']}"
        )
        assert "rc_predict" in art.metadata["exports"]
        assert art.metadata["dtype"] == "i16"
        if shutil.which("wasmtime") is not None:
            assert target.run(art).success


def test_wasm_inspect_parses_imports_exports():
    """The minimal wasm parser agrees with the metadata the linker reports."""
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
    from rclite.targets.wasm._wasm_inspect import inspect_wasm

    rc, exe, sample = _build()
    with tempfile.TemporaryDirectory() as td:
        art = BrowserWasm().compile(rc, exe, output_dir=td, test_inputs=sample)
        info = inspect_wasm(str(art.binary))
        assert "rc_predict" in info.exports
        assert "memory" in info.exports
        assert info.imports == ["env.tanhf"]


def test_wasm_target_rejects_non_f32():
    expect_raises(ValueError, WasmTarget, dtype="f64")


def test_wasm_target_requires_test_inputs():
    rc, exe, _ = _build()
    with tempfile.TemporaryDirectory() as td:
        expect_raises(ValueError, Wasmtime().compile, rc, exe, output_dir=td)


def test_wasmtime_full_pipeline():
    if not _rust_target_available():
        return  # skip — rustc / wasm32 rust-std not available
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
    from rclite.quant import (
        calibrate_from_data,
        quantize_model_affine,
        LUTStrategy,
    )

    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=0.0,
            input_scaling=1.0,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            topology=Topology.SCR,
            chain_weight=0.9,
            chain_feedback=0.1,
            leak_rate=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=30,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.2
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[: T - 50], Y[: T - 50])
    cfg = calibrate_from_data(rc, exe, X[: T - 50], storage_bits=8)
    qm = quantize_model_affine(
        rc, exe, cfg, lut_strategy=LUTStrategy.linear_interp(64)
    )
    return qm, X[T - 50 : T - 40]


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
            qm, output_dir=pathlib.Path(td), test_inputs=sample
        )
        assert art.binary is not None and art.binary.exists()
        assert art.binary.suffix == ".gba"
        assert art.metadata["triple"] == "thumbv4t-none-eabi"
        assert art.metadata["affine"] is True


def test_gba_full_pipeline_mgba():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip
    if (
        shutil.which("mgba") is None
        and shutil.which("/usr/games/mgba") is None
    ):
        return  # skip — no emulator
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        target = Gba()
        art = target.compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample
        )
        result = target.run(art, timeout=6)
        assert result.success, f"mGBA output:\n{result.output}"
        assert "TEST_PASS" in result.output
        assert "TEST_FAIL" not in result.output


def test_microbit_compile_affine_emits_llvm_kernel_artifacts():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip — no ARM toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Microbit().compile_affine_quantized(
            qm,
            output_dir=pathlib.Path(td),
            test_inputs=sample,
            kernel_backend="llvm",
        )
        assert art.binary is not None and art.binary.exists()
        assert art.metadata["kernel_backend"] == "llvm_ir"
        srcs = {p.name for p in art.sources}
        assert {"main.c", "rc_kernel.ll"} <= srcs


def test_microbit_compile_affine_emits_c_kernel_artifacts():
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # skip — no ARM toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Microbit().compile_affine_quantized(
            qm,
            output_dir=pathlib.Path(td),
            test_inputs=sample,
            kernel_backend="c",
        )
        assert art.binary is not None and art.binary.exists()
        assert art.metadata["kernel_backend"] == "portable_c"
        srcs = {p.name for p in art.sources}
        assert {"main.c", "rc_kernel.c", "rc_wrapper.c"} <= srcs


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
            qm, output_dir=pathlib.Path(td), test_inputs=sample, build=False
        )
        assert art.binary is None
        assert art.metadata["cpu"] == "6502"
        assert art.metadata["affine"] is True
        srcs = {p.name for p in art.sources}
        assert {"main.c", "rc_kernel.c"} <= srcs
        # harness embeds the blargg $6000 protocol signature
        main_txt = (pathlib.Path(td) / "main.c").read_text()
        assert "0x6000" in main_txt and "TEST_PASS" in main_txt


def test_nes_compile_emits_llvm_kernel_sources_without_toolchain():
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Nes().compile_affine_quantized(
            qm,
            output_dir=pathlib.Path(td),
            test_inputs=sample,
            build=False,
            kernel_backend="llvm",
        )
        assert art.binary is None
        assert art.metadata["kernel_backend"] == "llvm_ir"
        srcs = {p.name for p in art.sources}
        assert {"main.c", "rc_kernel.ll"} <= srcs


def test_nes_compile_emits_rom():
    if shutil.which("mos-nes-nrom-clang") is None:
        return  # skip — no llvm-mos toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Nes().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=sample
        )
        assert art.binary is not None and art.binary.exists()
        assert art.binary.suffix == ".nes"


def test_nes_compile_emits_rom_with_llvm_kernel():
    if shutil.which("mos-nes-nrom-clang") is None:
        return  # skip — no llvm-mos toolchain
    qm, sample = _build_affine_gba()
    with tempfile.TemporaryDirectory() as td:
        art = Nes().compile_affine_quantized(
            qm,
            output_dir=pathlib.Path(td),
            test_inputs=sample,
            kernel_backend="llvm",
        )
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
            qm, output_dir=pathlib.Path(td), test_inputs=sample
        )
        result = target.run(art)  # auto: Mesen if present, else FCEUX
        assert result.success, f"emulator output:\n{result.output}"
        assert "TEST_PASS" in result.output
        assert "TEST_FAIL" not in result.output


def test_nes_compile_rejects_non_affine():
    rc, exe, sample = _build()
    expect_raises(
        NotImplementedError,
        Nes().compile,
        rc,
        exe,
        output_dir="/tmp/unused_nes",
    )


TESTS = [
    v
    for k, v in list(globals().items())
    if k.startswith("test_") and callable(v)
]


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
