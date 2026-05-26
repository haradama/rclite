"""Tests for the rclite.quant package — config, LUT, QuantizedExecutor, QAT search."""
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
    QuantConfig, TanhLUTSpec, I32FixedPoint, I16FixedPoint,
    quantize_model, QuantizedExecutor,
    search_quantization, derive_frac_bits,
)
from rclite.quant._intops import (
    fixed_mul_i32, fixed_mul_scalar_i32, tanh_lut_lookup,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_esn(units=80, topology=Topology.SCR, include_input=True,
                input_offset=0.0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=input_offset, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=80,
                             include_bias=True, include_input=include_input,
                             name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((600, 1)) * 0.2
    Y = np.sin(np.arange(600) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X, Y


# ---------------------------------------------------------------- config / target


def test_quant_config_roundtrip():
    cfg = QuantConfig(state_frac=16, input_frac=12, weight_frac=8)
    assert cfg.state_scale == 1 << 16
    assert cfg.input_scale == 1 << 12
    assert cfg.weight_scale == 1 << 8


def test_quant_config_rejects_invalid_frac():
    expect_raises(ValueError, QuantConfig, state_frac=-1)
    expect_raises(ValueError, QuantConfig, state_frac=40)


def test_i32_target_range():
    t = I32FixedPoint()
    assert t.storage_dtype == np.dtype("int32")
    assert t.accum_dtype == np.dtype("int64")
    assert t.llvm_storage_type == "i32"
    cfg = QuantConfig(state_frac=16)
    assert t.quantize_state(1.0, cfg) == 1 << 16
    assert t.quantize_state(-1.0, cfg) == -(1 << 16)


def test_i32_target_saturates():
    t = I32FixedPoint()
    cfg = QuantConfig(state_frac=20)
    huge = 1e12
    q = t.quantize_state(huge, cfg)
    assert q == np.iinfo(np.int32).max


def test_i16_target_metadata():
    t = I16FixedPoint()
    assert t.llvm_storage_type == "i16"
    assert t.llvm_accum_type == "i32"


# ---------------------------------------------------------------- LUT


def test_lut_spec_table_length():
    spec = TanhLUTSpec(xmin=-4, xmax=4, n=64)
    table = spec.build_table_f32()
    assert table.shape == (64,)
    assert abs(table[0] - np.tanh(-4)) < 1e-6
    assert abs(table[-1] - np.tanh(4)) < 1e-6


def test_lut_spec_rejects_invalid():
    expect_raises(ValueError, TanhLUTSpec, n=1)
    expect_raises(ValueError, TanhLUTSpec, xmin=4, xmax=-4)


def test_lut_lookup_matches_libm_within_resolution():
    spec = TanhLUTSpec(n=256)
    table_q = spec.build_table_int(1 << 16)
    xs = np.linspace(-3, 3, 100)
    xs_q = (xs * (1 << 16)).astype(np.int32)
    xmin_q = int(spec.xmin * (1 << 16))
    xmax_q = int(spec.xmax * (1 << 16))
    y_q = tanh_lut_lookup(xs_q, table_q, xmin_q, xmax_q, 16)
    y_lut = y_q.astype(np.float64) / (1 << 16)
    y_true = np.tanh(xs)
    err = float(np.max(np.abs(y_lut - y_true)))
    # n=256 over 8-unit range → step ~ 0.03; LUT linear-interp err ~ 1e-3
    assert err < 5e-3, f"LUT max err {err}"


def test_lut_clamps_outside_domain():
    spec = TanhLUTSpec(xmin=-2, xmax=2, n=32)
    table_q = spec.build_table_int(1 << 16)
    xmin_q = int(spec.xmin * (1 << 16))
    xmax_q = int(spec.xmax * (1 << 16))
    very_neg = np.array([-100 * (1 << 16)], dtype=np.int32)
    very_pos = np.array([100 * (1 << 16)], dtype=np.int32)
    y_lo = tanh_lut_lookup(very_neg, table_q, xmin_q, xmax_q, 16)
    y_hi = tanh_lut_lookup(very_pos, table_q, xmin_q, xmax_q, 16)
    assert y_lo[0] == table_q[0]
    assert y_hi[0] == table_q[-1]


# ---------------------------------------------------------------- fixed-point ops


def test_fixed_mul_matches_float_within_quantization():
    rng = np.random.default_rng(0)
    a_f = rng.standard_normal(100) * 0.5
    b_f = rng.standard_normal(100) * 0.5
    a_q = (a_f * (1 << 16)).astype(np.int32)
    b_q = (b_f * (1 << 16)).astype(np.int32)
    prod_q = fixed_mul_i32(a_q, b_q, 16)
    prod_f = prod_q.astype(np.float64) / (1 << 16)
    err = float(np.max(np.abs(prod_f - a_f * b_f)))
    assert err < 1e-4, f"fixed_mul max err {err}"


def test_fixed_mul_scalar():
    a = (0.5 * (1 << 16))
    b = (0.25 * (1 << 16))
    result = fixed_mul_scalar_i32(int(a), int(b), 16)
    assert abs(result / (1 << 16) - 0.125) < 1e-5


# ---------------------------------------------------------------- quantize_model


def test_quantize_model_shapes():
    rc, exe, _, _ = _build_esn(units=40)
    cfg = QuantConfig(state_frac=16, input_frac=12, weight_frac=12)
    qm = quantize_model(rc, exe, cfg)
    assert qm.W_in_q.shape == exe.W_in.shape
    assert qm.W_res_q.shape == exe.W_res.shape
    assert qm.W_out_q.shape == exe.W_out.shape
    assert qm.lut_table_q is not None
    assert qm.N == 40 and qm.K == 1 and qm.M == 1
    assert qm.W_in_q.dtype == np.int32


def test_quantize_model_requires_trained_readout():
    rc, exe, _, _ = _build_esn(units=10)
    # Replace exe with a fresh one that hasn't been fit
    exe2 = RCExecutor(rc)
    expect_raises(ValueError, quantize_model, rc, exe2, QuantConfig())


def test_quantize_model_W_out_decoded_matches_float():
    rc, exe, _, _ = _build_esn(units=20)
    cfg = QuantConfig(state_frac=18, input_frac=12, weight_frac=12)
    qm = quantize_model(rc, exe, cfg)
    K, N = rc.input.units, rc.reservoir.units
    # Bias column at state_scale
    bias_decoded = qm.W_out_q[:, 0].astype(np.float64) / cfg.state_scale
    assert np.allclose(bias_decoded, exe.W_out[:, 0], atol=1e-4)
    # State columns at state_scale
    state_decoded = qm.W_out_q[:, 1 + K:].astype(np.float64) / cfg.state_scale
    assert np.allclose(state_decoded, exe.W_out[:, 1 + K:], atol=1e-3)


# ---------------------------------------------------------------- executor


def test_executor_requires_lut():
    rc, exe, _, _ = _build_esn(units=10)
    cfg = QuantConfig()
    from rclite.quant.model import QuantizedModel
    qm = quantize_model(rc, exe, cfg)
    # Strip the LUT and expect failure
    qm_bad = QuantizedModel(
        rc=qm.rc, target=qm.target, config=qm.config, lut=None,
        W_in_q=qm.W_in_q, W_res_q=qm.W_res_q, W_out_q=qm.W_out_q,
        lut_table_q=None, state_init_q=qm.state_init_q,
    )
    expect_raises(ValueError, QuantizedExecutor, qm_bad)


def test_executor_reset():
    rc, exe, X, _ = _build_esn(units=20)
    qm = quantize_model(rc, exe, QuantConfig(state_frac=18, input_frac=12,
                                                weight_frac=12))
    qexe = QuantizedExecutor(qm)
    qexe.predict(X[:10])
    s_after = qexe.state_q.copy()
    qexe.reset()
    assert np.all(qexe.state_q == 0)
    qexe.predict(X[:10])
    # Re-running from reset should give the same state
    assert np.allclose(qexe.state_q, s_after)


def test_executor_state_trajectory_close_to_float():
    rc, exe, X, _ = _build_esn(units=40)
    qm = quantize_model(rc, exe, QuantConfig(state_frac=20, input_frac=14,
                                                weight_frac=14))
    qexe = QuantizedExecutor(qm)
    H_q = qexe.collect_states(X[:200])
    H_f = exe.collect_states(X[:200])
    diff = float(np.max(np.abs(H_q - H_f)))
    assert diff < 0.05, f"state trajectory diff {diff}"


# ---------------------------------------------------------------- QAT search


def test_search_converges_below_float_baseline_or_close():
    rc, exe, X, Y = _build_esn(units=80)
    X_tr, Y_tr = X[:400], Y[:400]
    X_ev, Y_ev = X[400:500], Y[400:500]

    Y_f32 = exe.predict(X_ev)
    mse_f32 = float(np.mean((Y_f32 - Y_ev) ** 2))

    result = search_quantization(
        rc, exe, X_tr, Y_tr, X_ev, Y_ev,
        state_frac_range=(12, 22),
        lut=TanhLUTSpec(n=128),
    )
    # With QAT refit, quantized should reach within 10x of float
    assert result.best_mse < 10 * mse_f32, \
        f"qmse {result.best_mse} too far from float {mse_f32}"


def test_search_history_records_all_tried():
    rc, exe, X, Y = _build_esn(units=30)
    result = search_quantization(
        rc, exe,
        X[:200], Y[:200], X[200:250], Y[200:250],
        state_frac_range=(12, 18),
    )
    assert len(result.history) == 7
    # Best should match an entry in history
    finite = [(c, m) for c, m in result.history if np.isfinite(m)]
    assert (result.best_config, result.best_mse) in finite


def test_derive_frac_bits_handles_zero_data():
    assert derive_frac_bits(np.zeros(10)) == 24
    # Mackey-Glass-ish range
    assert derive_frac_bits(np.linspace(-1, 1, 100)) >= 20


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
