"""Tests for the affine LLVM lowering (Phase 2b).

Verifies bit-exact parity between the Python `AffineQuantizedExecutor`
and the JIT kernel emitted by `_AffineLowerer`, across storage widths,
topologies, and readout configurations.
"""
from __future__ import annotations
import pathlib
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
    quantize_multiplier, build_ir_from_quantized_affine,
)
from rclite.quant.affine.multiplier import (
    apply_multiplier_scalar, apply_multiplier_array,
)
from rclite.codegen.llvm import CompiledAffineRC, emit_quantized_affine_module


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_and_quant(storage_bits=8, units=20, topology=Topology.SCR,
                       include_bias=True, include_input=True, T=300, seed=0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=40,
                             include_bias=include_bias,
                             include_input=include_input, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.15
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[:T - 40], Y[:T - 40])
    cfg = calibrate_from_data(rc, exe, X[:T - 40], storage_bits=storage_bits)
    qm = quantize_model_affine(rc, exe, cfg)
    return rc, exe, qm, X


def _assert_jit_python_parity(qm, X_test):
    """JIT and Python executor must agree bit-exactly (max |diff| == 0)."""
    Y_jit = CompiledAffineRC(qm).predict(X_test)
    qexe = AffineQuantizedExecutor(qm)
    Y_py = qexe.predict(X_test)
    diff = float(np.max(np.abs(Y_jit - Y_py)))
    assert diff == 0.0, f"JIT vs Python diff = {diff}"


# ---------------------------------------------------------------- multiplier helpers


def test_quantize_multiplier_zero():
    M0, n = quantize_multiplier(0.0)
    assert (M0, n) == (0, 0)


def test_quantize_multiplier_M0_in_normalized_range():
    """For any positive M, M0 should land in [2^30, 2^31)."""
    for M in (1e-6, 1e-3, 0.3, 0.5, 1.0, 1.5, 100.0, 1e6):
        M0, n = quantize_multiplier(M)
        assert (1 << 30) <= M0 < (1 << 31), \
            f"M={M}: M0={M0} not in [2^30, 2^31)"


def test_quantize_multiplier_reproduces_value():
    """M ≈ M0 * 2^-n within ~1e-9 relative error."""
    for M in (0.001, 0.1, 0.3, 1.0, 10.0):
        M0, n = quantize_multiplier(M)
        approx = M0 / (1 << n)
        rel_err = abs(approx - M) / M
        assert rel_err < 1e-8, f"M={M}: approx={approx}, rel_err={rel_err}"


def test_apply_multiplier_scalar_and_array_match():
    """Vectorized apply_multiplier_array should agree with the scalar version."""
    M0, n = quantize_multiplier(0.3)
    xs = np.array([-1000, -1, 0, 1, 1000, 12345], dtype=np.int64)
    arr_res = apply_multiplier_array(xs.astype(np.int32), M0, n)
    for i, x in enumerate(xs.tolist()):
        scalar = apply_multiplier_scalar(int(x), M0, n)
        assert int(arr_res[i]) == scalar, \
            f"x={x}: array={arr_res[i]} scalar={scalar}"


# ---------------------------------------------------------------- IR builder


def test_build_ir_metadata_present():
    _, _, qm, _ = _build_and_quant()
    mod = build_ir_from_quantized_affine(qm)
    md = mod.metadata
    assert md["quantization"] == "affine"
    assert md["dtype"] == "i8"
    assert md["storage_bits"] == 8
    # Affine-specific metadata
    for key in ("zp_input", "zp_state", "zp_pre", "zp_output",
                  "lut_offset", "bias_pre",
                  "M_in_M0", "M_in_n", "M_res_M0", "M_res_n",
                  "leak_M0", "leak_n",
                  "M_out_state_M0", "M_out_state_n"):
        assert key in md, f"missing metadata key: {key}"


def test_build_ir_weights_include_row_sums():
    _, _, qm, _ = _build_and_quant(include_bias=True, include_input=True)
    mod = build_ir_from_quantized_affine(qm)
    for name in ("W_in", "W_res", "W_out", "lut_table",
                  "row_sum_W_in", "row_sum_W_res",
                  "row_sum_Wout_state", "row_sum_Wout_input"):
        assert name in mod.weights, f"missing weight: {name}"


def test_build_ir_rejects_nontrivial_preprocess():
    """offset != 0 or scaling != 1 must raise NotImplementedError in MVP."""
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.5, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=10, topology=Topology.SCR,
                                 chain_weight=0.9, leak_rate=0.3),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=20,
                             include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((150, 1)) * 0.15
    Y = np.sin(np.arange(150) * 0.1)[:, None]
    exe.fit(X, Y)
    cfg = calibrate_from_data(rc, exe, X, storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg)
    expect_raises(NotImplementedError, build_ir_from_quantized_affine, qm)


# ---------------------------------------------------------------- emit / compile


def test_emit_affine_module_ir_has_i8_signature():
    _, _, qm, _ = _build_and_quant(storage_bits=8)
    ir_text = str(emit_quantized_affine_module(qm))
    assert "i8*" in ir_text
    assert "@rc_predict" in ir_text or '@"rc_predict"' in ir_text


def test_emit_affine_module_ir_has_i16_signature():
    _, _, qm, _ = _build_and_quant(storage_bits=16)
    ir_text = str(emit_quantized_affine_module(qm))
    assert "i16*" in ir_text


def test_compiled_affine_predict_finite():
    _, _, qm, X = _build_and_quant()
    Y = CompiledAffineRC(qm).predict(X[200:230])
    assert Y.shape == (30, 1)
    assert np.all(np.isfinite(Y))


# ---------------------------------------------------------------- bit-exact parity


def test_parity_i8_scr():
    _, _, qm, X = _build_and_quant(storage_bits=8, topology=Topology.SCR)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_i8_dense():
    _, _, qm, X = _build_and_quant(storage_bits=8, topology=Topology.ESN_STANDARD)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_i8_dlr():
    _, _, qm, X = _build_and_quant(storage_bits=8, topology=Topology.DLR)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_i8_dlrb():
    _, _, qm, X = _build_and_quant(storage_bits=8, topology=Topology.DLRB)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_i16_scr():
    _, _, qm, X = _build_and_quant(storage_bits=16, topology=Topology.SCR)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_i16_dense():
    _, _, qm, X = _build_and_quant(storage_bits=16, topology=Topology.ESN_STANDARD)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_no_bias_no_input():
    """Readout-only path: just the state column block."""
    _, _, qm, X = _build_and_quant(include_bias=False, include_input=False)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_bias_only():
    """include_bias but not include_input."""
    _, _, qm, X = _build_and_quant(include_bias=True, include_input=False)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_input_only():
    """include_input but not include_bias."""
    _, _, qm, X = _build_and_quant(include_bias=False, include_input=True)
    _assert_jit_python_parity(qm, X[200:230])


def test_parity_across_seeds_i8():
    for seed in (0, 1, 2, 3):
        _, _, qm, X = _build_and_quant(seed=seed, units=30, T=400)
        _assert_jit_python_parity(qm, X[300:330])


def test_parity_across_seeds_i16():
    for seed in (0, 1, 2, 3):
        _, _, qm, X = _build_and_quant(seed=seed, units=30, T=400,
                                          storage_bits=16)
        _assert_jit_python_parity(qm, X[300:330])


# ---------------------------------------------------------------- meta tests


def test_compiled_affine_ir_text_property():
    _, _, qm, _ = _build_and_quant()
    c = CompiledAffineRC(qm)
    assert "lut_table" in c.llvm_ir
    assert "row_sum_W_in" in c.llvm_ir


def test_storage_bits_8_uses_i32_accumulator():
    """For i8 storage, the accumulator type should be i32 (not i64)."""
    _, _, qm, _ = _build_and_quant(storage_bits=8)
    ir_text = str(emit_quantized_affine_module(qm))
    # acc_in / acc_res allocas should be i32 for i8 path
    assert "%acc_in = alloca i32" in ir_text or "alloca i32" in ir_text


def test_storage_bits_16_uses_i64_accumulator():
    """For i16 storage, the matmul accumulator should widen to i64."""
    _, _, qm, _ = _build_and_quant(storage_bits=16)
    ir_text = str(emit_quantized_affine_module(qm))
    # Should see i64 acc allocations
    assert "alloca i64" in ir_text


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
