"""Tests for the selectable LUT strategy (Phase 2c).

Covers:
  - `LUTStrategy` validation
  - Per-strategy artifact build (direct / linear_interp / polynomial)
  - Python executor correctness for each strategy
  - JIT vs Python bit-exact parity for each strategy
  - Cortex-M0 cross-compile smoke (skipped when toolchain absent)
"""
from __future__ import annotations
import pathlib
import shutil
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    LUTStrategy, LUTKind, build_lut_artifacts,
)
from rclite.codegen.llvm import CompiledAffineRC


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_and_quant(storage_bits=8, lut_strategy=None, units=20, T=300, seed=0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.SCR,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=40,
                             include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.15
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[:T - 40], Y[:T - 40])
    cfg = calibrate_from_data(rc, exe, X[:T - 40], storage_bits=storage_bits)
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=lut_strategy)
    return rc, exe, qm, X


# ---------------------------------------------------------------- LUTStrategy validation


def test_strategy_default_is_direct():
    s = LUTStrategy()
    assert s.kind == LUTKind.DIRECT


def test_strategy_factories():
    assert LUTStrategy.direct().kind == LUTKind.DIRECT
    s = LUTStrategy.linear_interp(n_entries=128, interp_frac_bits=10)
    assert s.kind == LUTKind.LINEAR_INTERP
    assert s.n_entries == 128
    assert s.interp_frac_bits == 10
    p = LUTStrategy.polynomial(poly_qf_bits=14, poly_clip=3.0)
    assert p.kind == LUTKind.POLYNOMIAL
    assert p.poly_qf_bits == 14
    assert p.poly_clip == 3.0


def test_strategy_rejects_bad_params():
    expect_raises(ValueError, LUTStrategy.linear_interp, n_entries=2)
    expect_raises(ValueError, LUTStrategy.linear_interp, interp_frac_bits=0)
    expect_raises(ValueError, LUTStrategy.linear_interp, interp_frac_bits=16)
    expect_raises(ValueError, LUTStrategy.polynomial, poly_qf_bits=4)
    expect_raises(ValueError, LUTStrategy.polynomial, poly_clip=-1.0)


# ---------------------------------------------------------------- artifact builder


def test_build_direct_artifacts_full_table():
    _, _, qm, _ = _build_and_quant(storage_bits=8, lut_strategy=LUTStrategy.direct())
    art = qm.lut_artifacts
    assert qm.lut_q.shape == (256,)
    assert art.offset == 128


def test_build_linear_interp_n_entries_and_idx_multiplier():
    _, _, qm, _ = _build_and_quant(
        storage_bits=8,
        lut_strategy=LUTStrategy.linear_interp(n_entries=64),
    )
    assert qm.lut_q.shape == (64,)
    assert qm.lut_artifacts.idx_M0 != 0
    assert qm.lut_artifacts.idx_n > 0


def test_build_polynomial_table_is_empty():
    _, _, qm, _ = _build_and_quant(
        storage_bits=8, lut_strategy=LUTStrategy.polynomial(),
    )
    assert qm.lut_q.shape == (0,)
    art = qm.lut_artifacts
    assert art.x_to_qf_M0 != 0
    assert art.qf_to_state_M0 != 0
    assert art.x_clip_qf > 0


def test_build_linear_interp_i16_table_size_matches_n_entries():
    _, _, qm, _ = _build_and_quant(
        storage_bits=16,
        lut_strategy=LUTStrategy.linear_interp(n_entries=512),
    )
    assert qm.lut_q.shape == (512,)
    assert qm.lut_q.dtype == np.int16


# ---------------------------------------------------------------- Python executor


def _check_executor_runs(strategy, sb):
    _, _, qm, X = _build_and_quant(storage_bits=sb, lut_strategy=strategy)
    qexe = AffineQuantizedExecutor(qm)
    Y = qexe.predict(X[200:230])
    assert Y.shape == (30, 1)
    assert np.all(np.isfinite(Y))


def test_executor_direct_i8():    _check_executor_runs(LUTStrategy.direct(), 8)
def test_executor_direct_i16():   _check_executor_runs(LUTStrategy.direct(), 16)
def test_executor_interp_i8():    _check_executor_runs(LUTStrategy.linear_interp(64), 8)
def test_executor_interp_i16():   _check_executor_runs(LUTStrategy.linear_interp(256), 16)
def test_executor_poly_i8():      _check_executor_runs(LUTStrategy.polynomial(), 8)
def test_executor_poly_i16():     _check_executor_runs(LUTStrategy.polynomial(), 16)


# ---------------------------------------------------------------- JIT parity per strategy


def _check_parity(strategy, sb, topology=Topology.SCR):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=20, topology=topology, chain_weight=0.9,
                                 leak_rate=0.3, seed=42),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=40,
                             include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1)) * 0.15
    Y = np.sin(np.arange(300) * 0.1)[:, None]
    exe.fit(X[:260], Y[:260])
    cfg = calibrate_from_data(rc, exe, X[:260], storage_bits=sb)
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=strategy)
    Y_jit = CompiledAffineRC(qm).predict(X[200:230])
    Y_py  = AffineQuantizedExecutor(qm).predict(X[200:230])
    diff = float(np.max(np.abs(Y_jit - Y_py)))
    assert diff == 0.0, f"strategy={strategy.kind}, sb={sb}: JIT vs Py diff = {diff}"


def test_parity_direct_i8():            _check_parity(LUTStrategy.direct(), 8)
def test_parity_direct_i16():           _check_parity(LUTStrategy.direct(), 16)
def test_parity_interp_i8():            _check_parity(LUTStrategy.linear_interp(64), 8)
def test_parity_interp_i16():           _check_parity(LUTStrategy.linear_interp(256), 16)
def test_parity_polynomial_i8():        _check_parity(LUTStrategy.polynomial(), 8)
def test_parity_polynomial_i16():       _check_parity(LUTStrategy.polynomial(), 16)


def test_parity_interp_dense():
    _check_parity(LUTStrategy.linear_interp(128), 8, topology=Topology.ESN_STANDARD)


def test_parity_interp_varied_n_entries():
    for n in (8, 32, 64, 256, 1024):
        _check_parity(LUTStrategy.linear_interp(n_entries=n), 8)


def test_parity_polynomial_varied_qf():
    for qf in (10, 14, 18, 22):
        _check_parity(LUTStrategy.polynomial(poly_qf_bits=qf), 8)


# ---------------------------------------------------------------- IR builder dispatch


def test_lut_table_global_present_for_direct_and_interp():
    from rclite.quant import build_ir_from_quantized_affine
    _, _, qm_d, _ = _build_and_quant(lut_strategy=LUTStrategy.direct())
    _, _, qm_i, _ = _build_and_quant(lut_strategy=LUTStrategy.linear_interp(64))
    _, _, qm_p, _ = _build_and_quant(lut_strategy=LUTStrategy.polynomial())
    mod_d = build_ir_from_quantized_affine(qm_d)
    mod_i = build_ir_from_quantized_affine(qm_i)
    mod_p = build_ir_from_quantized_affine(qm_p)
    assert "lut_table" in mod_d.weights
    assert "lut_table" in mod_i.weights
    assert "lut_table" not in mod_p.weights


def test_metadata_carries_strategy_specific_keys():
    from rclite.quant import build_ir_from_quantized_affine
    _, _, qm_i, _ = _build_and_quant(lut_strategy=LUTStrategy.linear_interp(64))
    md_i = build_ir_from_quantized_affine(qm_i).metadata
    assert md_i["lut_kind"] == "linear_interp"
    assert md_i["lut_n_entries"] == 64
    assert md_i["lut_idx_M0"] != 0

    _, _, qm_p, _ = _build_and_quant(lut_strategy=LUTStrategy.polynomial())
    md_p = build_ir_from_quantized_affine(qm_p).metadata
    assert md_p["lut_kind"] == "polynomial"
    assert md_p["poly_x_M0"] != 0
    assert md_p["poly_clip_qf"] > 0


# ---------------------------------------------------------------- MCU cross-compile smoke


def test_microbit_cross_compile_each_strategy_smoke():
    """Cross-compile + arm-size check for each strategy. Skip if toolchain missing."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return  # toolchain not installed; nothing to validate
    from rclite.targets import Microbit
    import tempfile

    for strategy in (
        LUTStrategy.direct(),
        LUTStrategy.linear_interp(n_entries=64),
        LUTStrategy.polynomial(),
    ):
        _, _, qm, X = _build_and_quant(storage_bits=8, lut_strategy=strategy)
        with tempfile.TemporaryDirectory() as tmp:
            artifact = Microbit().compile_affine_quantized(
                qm, output_dir=pathlib.Path(tmp), test_inputs=X[200:210],
            )
            assert artifact.binary.exists(), \
                f"strategy {strategy.kind}: ELF not produced"
            assert artifact.metadata["lut_kind"] == strategy.kind.value


def test_microbit_cross_compile_i16_interp_size_smaller_than_direct():
    """LUT compression should make linear_interp ~10x smaller than direct (i16)."""
    if shutil.which("arm-none-eabi-gcc") is None:
        return
    if shutil.which("arm-none-eabi-size") is None:
        return
    from rclite.targets import Microbit
    import tempfile
    import subprocess

    def total_bytes(elf, cc):
        cp = subprocess.run([cc.replace("gcc", "size"), str(elf)],
                             capture_output=True, text=True, check=True)
        return int(cp.stdout.strip().splitlines()[1].split()[3])

    microbit = Microbit()
    with tempfile.TemporaryDirectory() as tmp:
        tmp = pathlib.Path(tmp)
        _, _, qm_d, X = _build_and_quant(
            storage_bits=16, lut_strategy=LUTStrategy.direct(),
        )
        a_d = microbit.compile_affine_quantized(
            qm_d, output_dir=tmp / "direct", test_inputs=X[200:210],
        )
        _, _, qm_i, _ = _build_and_quant(
            storage_bits=16, lut_strategy=LUTStrategy.linear_interp(n_entries=256),
        )
        a_i = microbit.compile_affine_quantized(
            qm_i, output_dir=tmp / "interp", test_inputs=X[200:210],
        )
        sz_d = total_bytes(a_d.binary, microbit.cc)
        sz_i = total_bytes(a_i.binary, microbit.cc)
        # LUT-dominated i16 direct should be ~10x bigger than interp_n256
        assert sz_d > 5 * sz_i, \
            f"interp ({sz_i}B) didn't beat direct ({sz_d}B) by 5x"


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
