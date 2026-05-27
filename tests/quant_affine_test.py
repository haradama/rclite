"""Tests for the asymmetric per-tensor affine quantization path.

Phase 2a scope: Python reference only. The LLVM-emit + on-device kernel
is Phase 2b.
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
    AffineParams, AffineQuantConfig,
    calibrate_from_data,
    AffineQuantizedModel, quantize_model_affine,
    AffineQuantizedExecutor,
    search_quantization_affine, AffineSearchResult,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_esn(units=30, topology=Topology.SCR, include_input=True,
                include_bias=True, T=300, seed=0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=50,
                             include_bias=include_bias,
                             include_input=include_input, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.15
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X, Y


# ---------------------------------------------------------------- AffineParams


def test_affine_params_validates_scale():
    expect_raises(ValueError, AffineParams, scale=0.0, zero_point=0)
    expect_raises(ValueError, AffineParams, scale=-1.0, zero_point=0)


def test_affine_params_validates_zp_range():
    # i8: zp must be in [-128, 127]
    expect_raises(ValueError, AffineParams, scale=1.0, zero_point=200)
    expect_raises(ValueError, AffineParams, scale=1.0, zero_point=-200)


def test_affine_params_roundtrip_scalar():
    p = AffineParams(scale=0.01, zero_point=5)
    for v in (0.0, 0.5, -0.5, 1.0, -1.27):
        q = p.quantize(v)
        r = p.dequantize(q)
        # within 1 quantum
        assert abs(r - v) <= p.scale, f"v={v} q={q} r={r}"


def test_affine_params_quantize_saturates():
    p = AffineParams(scale=0.01, zero_point=0)
    # 10.0 / 0.01 = 1000 → far above i8 max 127
    q = p.quantize(10.0)
    assert q == 127
    q = p.quantize(-10.0)
    assert q == -128


def test_affine_params_symmetric_absmax():
    arr = np.array([-1.0, -0.5, 0.0, 0.5, 0.8])
    p = AffineParams.symmetric_absmax(arr, storage_bits=8)
    assert p.zero_point == 0
    # absmax = 1.0; scale = 1.0 / 127
    assert abs(p.scale - 1.0 / 127) < 1e-9


def test_affine_params_asymmetric_minmax_includes_zero_and_uses_top_end():
    """TFLM convention: range always covers 0. Max value hits qmax."""
    arr = np.array([0.4, 0.5, 1.4])  # all positive
    p = AffineParams.asymmetric_minmax(arr, storage_bits=8)
    # 0 must be exactly representable (this is the TFLM invariant)
    assert p.quantize(0.0) == p.zero_point
    # max value should hit qmax (full upper-end utilization)
    assert p.quantize(arr.max()) == 127
    # For all-positive data, zp should sit at qmin (so positives have full range)
    assert p.zero_point == -128


def test_affine_params_quantize_array_dtype():
    p = AffineParams(scale=0.01, zero_point=10, storage_bits=8)
    q = p.quantize_array(np.array([0.0, 1.0, -1.0]))
    assert q.dtype == np.int8


# ---------------------------------------------------------------- config


def test_config_rejects_nonzero_zp_for_weights():
    p_act = AffineParams(scale=0.01, zero_point=5)
    p_w   = AffineParams(scale=0.01, zero_point=0)
    p_bad = AffineParams(scale=0.01, zero_point=5)
    # W_in with zp != 0 should be rejected
    expect_raises(ValueError, AffineQuantConfig,
                   input=p_act, u_pre=p_act, state=p_w, pre=p_act,
                   W_in=p_bad, W_res=p_w, W_out_state=p_w, output=p_act)


def test_config_rejects_mixed_storage_bits():
    p8  = AffineParams(scale=0.01, zero_point=0, storage_bits=8)
    p16 = AffineParams(scale=0.01, zero_point=0, storage_bits=16)
    expect_raises(ValueError, AffineQuantConfig,
                   input=p8, u_pre=p16, state=p8, pre=p8,
                   W_in=p8, W_res=p8, W_out_state=p8, output=p8)


# ---------------------------------------------------------------- calibration


def test_calibrate_uses_full_range():
    rc, exe, X, _ = _build_esn()
    cfg = calibrate_from_data(rc, exe, X)
    # All params populated
    assert cfg.W_out_bias is not None
    assert cfg.W_out_input is not None
    assert cfg.W_out_state is not None
    # Weights are symmetric (zp=0)
    for name in ("W_in", "W_res", "W_out_bias", "W_out_input", "W_out_state"):
        assert getattr(cfg, name).zero_point == 0, name


def test_calibrate_omits_W_out_bias_if_no_bias():
    rc, exe, X, _ = _build_esn(include_bias=False)
    cfg = calibrate_from_data(rc, exe, X)
    assert cfg.W_out_bias is None
    assert cfg.W_out_input is not None


def test_calibrate_omits_W_out_input_if_no_input_passthrough():
    rc, exe, X, _ = _build_esn(include_input=False)
    cfg = calibrate_from_data(rc, exe, X)
    assert cfg.W_out_input is None
    assert cfg.W_out_bias is not None


# ---------------------------------------------------------------- quantize_model_affine


def test_quantize_model_shapes_and_dtypes():
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    assert qm.W_in_q.shape == exe.W_in.shape
    assert qm.W_res_q.shape == exe.W_res.shape
    assert qm.W_out_q.shape == exe.W_out.shape
    assert qm.W_in_q.dtype == np.int8
    assert qm.lut_q.shape == (256,)
    assert qm.row_sum_W_in.shape == (rc.reservoir.units,)


def test_quantize_model_requires_trained_exe():
    rc, _, X, _ = _build_esn()
    exe = RCExecutor(rc)
    cfg = AffineQuantConfig(
        input=AffineParams(scale=0.01),
        u_pre=AffineParams(scale=0.01),
        state=AffineParams(scale=0.01),
        pre=AffineParams(scale=0.01),
        W_in=AffineParams(scale=0.01),
        W_res=AffineParams(scale=0.01),
        W_out_state=AffineParams(scale=0.01),
        output=AffineParams(scale=0.01),
    )
    expect_raises(ValueError, quantize_model_affine, rc, exe, cfg)


def test_quantize_model_lut_endpoints_are_tanh_extremes():
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    # LUT[0] corresponds to q_pre = qmin (most negative pre); tanh of a very
    # negative value is ~-1, so q_state ≈ quantize(-1)
    state_for_minus1 = cfg.state.quantize(-1.0)
    state_for_plus1 = cfg.state.quantize(1.0)
    # Endpoints should be close (within rounding) to ±1 in tanh saturation
    assert abs(int(qm.lut_q[0]) - state_for_minus1) <= 1
    assert abs(int(qm.lut_q[-1]) - state_for_plus1) <= 1


# ---------------------------------------------------------------- executor


def test_executor_runs_finite():
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    Y = qexe.predict(X[200:230])
    assert Y.shape == (30, 1)
    assert np.all(np.isfinite(Y))


def test_executor_reset_clears_state():
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    qexe.predict(X[:50])
    s_after = qexe.state_q.copy()
    qexe.reset()
    # After reset, state should equal the initial state (zp_state for each elem)
    assert np.all(qexe.state_q == cfg.state.zero_point)
    qexe.predict(X[:50])
    # Same input → same final state
    assert np.array_equal(qexe.state_q, s_after)


def test_executor_state_trajectory_close_to_float():
    """The affine quantized state should track the float state within ~i8 quant noise."""
    rc, exe, X, _ = _build_esn(units=30)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    H_q = qexe.collect_states(X[:200])
    H_f = exe.collect_states(X[:200])
    # State scale is ~1/127. Per-step quant noise compounds through the
    # recurrent matmul; over 200 steps we tolerate accumulated drift up to
    # ~25× the per-element state quantum.
    err = float(np.abs(H_q - H_f).max())
    assert err < 25 * cfg.state.scale, f"state err {err} >> 25*{cfg.state.scale}"


def test_executor_output_in_same_ballpark_as_float():
    """Affine quantized output MSE should be within a small multiple of float MSE."""
    rc, exe, X, Y = _build_esn(units=40, T=600)
    cfg = calibrate_from_data(rc, exe, X[:400])
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    Y_q = qexe.predict(X[400:500])
    Y_f = exe.predict(X[400:500])
    Y_ref = Y[400:500]
    mse_f = float(np.mean((Y_f - Y_ref) ** 2))
    mse_q = float(np.mean((Y_q - Y_ref) ** 2))
    # For i8 with auto-calibration, allow within 10x of float MSE
    assert mse_q < 10 * mse_f + 1e-3, \
        f"affine MSE {mse_q:.4e} too far from float {mse_f:.4e}"


def test_executor_dequantize_matches_int_state_q():
    """state_q should sit inside the i8 range after every step."""
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    for t in range(30):
        u = qexe._quantize_u_pre(X[t])
        qexe.step_q(u)
        assert int(qexe.state_q.min()) >= -128
        assert int(qexe.state_q.max()) <= 127


def test_executor_uses_lut_offset_correctly():
    """The LUT lookup index should never go out of bounds."""
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    # Run a long enough trace that pre_q sees its full range
    for t in range(100):
        u = qexe._quantize_u_pre(X[t])
        qexe.step_q(u)
    # If we got here without an IndexError, the offset is correct


# ---------------------------------------------------------------- per-block W_out


def test_per_block_W_out_keeps_tiny_bias_coefficient():
    """Per-block scaling should preserve W_out bias coefficient even when
    state-column coefficients are orders of magnitude larger.

    With single per-tensor scale, a bias coef of O(0.1) would quantize to 0
    when state coefs are O(100). Per-block keeps it usable.
    """
    rc, exe, X, _ = _build_esn(units=30)
    cfg = calibrate_from_data(rc, exe, X)
    qm = quantize_model_affine(rc, exe, cfg)
    # Bias column is W_out_q[:, 0]; should not be all zeros (assuming bias coef
    # is non-negligible after fitting on the synthetic target).
    bias_col = qm.W_out_q[:, 0]
    assert not np.all(bias_col == 0), "bias column collapsed to zero — per-block scales not working"


# ---------------------------------------------------------------- i16 storage path


def test_affine_params_i16_range():
    """AffineParams with storage_bits=16 should use the full [-32768, 32767] range."""
    p = AffineParams(scale=0.001, zero_point=0, storage_bits=16)
    assert p.storage_dtype == np.dtype("int16")
    # zp at the i16 boundary should be accepted
    AffineParams(scale=1.0, zero_point=32767, storage_bits=16)
    AffineParams(scale=1.0, zero_point=-32768, storage_bits=16)
    # one past the boundary should fail
    expect_raises(ValueError, AffineParams, scale=1.0, zero_point=32768,
                   storage_bits=16)


def test_calibrate_storage_bits_propagates():
    """All AffineParams in the config should carry the requested storage width."""
    rc, exe, X, _ = _build_esn(units=20)
    cfg16 = calibrate_from_data(rc, exe, X, storage_bits=16)
    assert cfg16.storage_bits == 16
    for name in ("input", "u_pre", "state", "pre", "W_in", "W_res",
                  "W_out_state", "output"):
        assert getattr(cfg16, name).storage_bits == 16, name


def test_quantize_model_affine_i16_dtypes_and_lut_size():
    """i16 affine model holds int16 weights and a 65536-entry LUT."""
    rc, exe, X, _ = _build_esn(units=20)
    cfg = calibrate_from_data(rc, exe, X, storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)
    assert qm.W_in_q.dtype == np.int16
    assert qm.W_res_q.dtype == np.int16
    assert qm.W_out_q.dtype == np.int16
    # Direct LUT spans the full i16 input range — 2^16 entries
    assert qm.lut_q.shape == (65536,)
    assert qm.lut_q.dtype == np.int16
    # lut_offset = -qmin = 32768 for i16
    assert qm.lut_offset == 32768


def test_i16_affine_executor_runs_and_stays_in_range():
    rc, exe, X, _ = _build_esn(units=30)
    cfg = calibrate_from_data(rc, exe, X, storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    Y = qexe.predict(X[200:230])
    assert np.all(np.isfinite(Y))
    # State should stay within signed i16
    for t in range(20):
        u = qexe._quantize_u_pre(X[t])
        qexe.step_q(u)
        assert int(qexe.state_q.min()) >= -32768
        assert int(qexe.state_q.max()) <= 32767


def test_i16_affine_accuracy_close_to_float():
    """i16 affine should track the float model very closely (within ~5x of float MSE).

    This is the key value of i16 affine: with 16 bits the per-tensor arbitrary
    scales recover essentially all the precision the float model has, unlike
    symmetric Q-format which is locked to power-of-2 scales.
    """
    rc, exe, X, Y = _build_esn(units=40, T=600)
    cfg = calibrate_from_data(rc, exe, X[:400], storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    Y_q = qexe.predict(X[400:500])
    Y_f = exe.predict(X[400:500])
    mse_f_target = float(np.mean((Y_f - Y[400:500]) ** 2))
    mse_q_target = float(np.mean((Y_q - Y[400:500]) ** 2))
    # i16 affine should be within 5x of float MSE on this synthetic task
    assert mse_q_target < 5 * mse_f_target + 1e-4, \
        f"i16 affine MSE {mse_q_target:.4e} too far from float {mse_f_target:.4e}"


def test_i16_affine_state_trajectory_bounded():
    """With i16 storage the accumulated drift should stay well below 1.

    The per-step state quantum is ~s_state (~1e-5 for tanh-bounded states),
    but error compounds through the recurrent loop. The absolute drift after
    200 steps should still be a small fraction of the [-1, 1] tanh range —
    nowhere near the wild divergence seen with i8 or symmetric i16.
    """
    rc, exe, X, _ = _build_esn(units=30)
    cfg = calibrate_from_data(rc, exe, X, storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)
    qexe = AffineQuantizedExecutor(qm)
    H_q = qexe.collect_states(X[:200])
    H_f = exe.collect_states(X[:200])
    err = float(np.abs(H_q - H_f).max())
    assert err < 0.2, f"i16 state drift {err} >= 0.2 (something is broken)"


# ---------------------------------------------------------------- QAT search


def test_qat_returns_affine_search_result():
    rc, exe, X, Y = _build_esn(units=20, T=400)
    result = search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=8, n_iterations=1,
    )
    assert isinstance(result, AffineSearchResult)
    assert isinstance(result.best_qmodel, AffineQuantizedModel)
    assert isinstance(result.best_config, AffineQuantConfig)
    assert result.best_iteration in (0, 1)
    assert len(result.history) == 2  # iterations 0 and 1


def test_qat_zero_iterations_matches_calibrate():
    """n_iterations=0 should produce the same model as plain calibration."""
    rc, exe, X, Y = _build_esn(units=20, T=400)
    result = search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=8, n_iterations=0,
    )
    # Compare W_out_q to a single-pass calibration
    cfg = calibrate_from_data(rc, exe, X[:300], storage_bits=8)
    qm_ref = quantize_model_affine(rc, exe, cfg)
    assert np.array_equal(result.best_qmodel.W_out_q, qm_ref.W_out_q)
    assert len(result.history) == 1


def test_qat_improves_or_matches_baseline():
    """One refit round should not be worse than the single-pass baseline.

    Best-of-N selection guarantees this even if a particular round
    overshoots — the function returns the argmin over history.
    """
    rc, exe, X, Y = _build_esn(units=30, T=500)
    result = search_quantization_affine(
        rc, exe, X[:400], Y[:400], X[400:500], Y[400:500],
        storage_bits=8, n_iterations=2,
    )
    baseline_mse = result.history[0][1]
    assert result.best_mse <= baseline_mse + 1e-12, \
        f"QAT best {result.best_mse:.4e} worse than baseline {baseline_mse:.4e}"


def test_qat_strictly_improves_on_mackey_glass_i8():
    """On a real-ish task (Mackey-Glass) QAT should comfortably beat baseline.

    MG one-step-ahead is a quant-sensitive prediction task where the refit
    has clear room to compensate for state-quantization noise. We require
    a strict win to validate that the QAT loop is actually doing work.
    """
    from rclite import (InputNode, ReservoirNode, ReadoutNode,
                         ReservoirComputer, Activation,
                         Distribution, Topology, Trainer)
    from examples.mackey_glass_esn import mackey_glass
    series = mackey_glass(n=1500)
    X = series[:-1, None]; Y = series[1:, None]
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=60, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=42),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=200,
                             include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:1000], Y[:1000])
    result = search_quantization_affine(
        rc, exe, X[:1000], Y[:1000], X[1000:1100], Y[1000:1100],
        storage_bits=8, n_iterations=2,
    )
    baseline_mse = result.history[0][1]
    assert result.best_mse < baseline_mse, \
        f"QAT best {result.best_mse:.4e} did not beat baseline {baseline_mse:.4e}"
    # And the improvement should be meaningful, not just numerical
    improvement = baseline_mse / result.best_mse
    assert improvement >= 1.5, \
        f"QAT improvement only {improvement:.2f}x — expected >= 1.5x for MG i8"


def test_qat_best_iteration_matches_history_argmin():
    rc, exe, X, Y = _build_esn(units=20, T=400)
    result = search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=8, n_iterations=2,
    )
    argmin_it, argmin_mse = min(result.history, key=lambda t: t[1])
    assert result.best_iteration == argmin_it
    assert abs(result.best_mse - argmin_mse) < 1e-12


def test_qat_best_qmodel_is_callable():
    rc, exe, X, Y = _build_esn(units=20, T=400)
    result = search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=8, n_iterations=1,
    )
    qexe = AffineQuantizedExecutor(result.best_qmodel)
    Y_q = qexe.predict(X[300:330])
    assert Y_q.shape == (30, 1)
    assert np.all(np.isfinite(Y_q))


def test_qat_supports_i16_storage():
    """QAT search should work end-to-end with storage_bits=16 too."""
    rc, exe, X, Y = _build_esn(units=20, T=400)
    result = search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=16, n_iterations=1,
    )
    assert result.best_qmodel.storage_bits == 16
    assert result.best_qmodel.W_in_q.dtype == np.int16


def test_qat_accepts_1d_arrays():
    """1D X/Y should be auto-promoted to 2D inside the search."""
    rc, exe, X, Y = _build_esn(units=20, T=400)
    # Pass 1D for the K=1 / M=1 case
    result = search_quantization_affine(
        rc, exe,
        X[:300].ravel(), Y[:300].ravel(),
        X[300:400].ravel(), Y[300:400].ravel(),
        storage_bits=8, n_iterations=1,
    )
    assert isinstance(result, AffineSearchResult)


def test_qat_rejects_untrained_exe():
    from rclite import (InputNode, ReservoirNode, ReadoutNode,
                         ReservoirComputer, Distribution, Topology, Trainer)
    rc = ReservoirComputer(
        input=InputNode(units=1, input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=10, topology=Topology.SCR,
                                 chain_weight=0.9, leak_rate=0.3),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             include_bias=True),
    )
    exe = RCExecutor(rc)
    X = np.zeros((100, 1)); Y = np.zeros((100, 1))
    expect_raises(ValueError, search_quantization_affine,
                   rc, exe, X[:80], Y[:80], X[80:], Y[80:])


def test_qat_does_not_mutate_exe_W_out():
    rc, exe, X, Y = _build_esn(units=20, T=400)
    W_out_before = exe.W_out.copy()
    search_quantization_affine(
        rc, exe, X[:300], Y[:300], X[300:400], Y[300:400],
        storage_bits=8, n_iterations=2,
    )
    assert np.array_equal(W_out_before, exe.W_out), \
        "search_quantization_affine mutated exe.W_out"


def test_i16_affine_beats_i8_affine_on_same_model():
    """The whole point of i16 over i8: tighter scales → better fidelity.

    Same model and calibration data, only storage width differs. i16 affine
    should produce noticeably smaller output MSE (vs float baseline) than
    i8 affine — state-trajectory L∞ error is too noisy a metric to test on,
    output MSE captures the practically-relevant degradation cleanly.
    """
    rc, exe, X, _ = _build_esn(units=60, T=600)
    cfg8  = calibrate_from_data(rc, exe, X[:500], storage_bits=8)
    cfg16 = calibrate_from_data(rc, exe, X[:500], storage_bits=16)
    qm8  = quantize_model_affine(rc, exe, cfg8)
    qm16 = quantize_model_affine(rc, exe, cfg16)
    Y_f  = exe.predict(X)
    Y_8  = AffineQuantizedExecutor(qm8).predict(X)
    Y_16 = AffineQuantizedExecutor(qm16).predict(X)
    mse_8  = float(np.mean((Y_8 [500:600] - Y_f[500:600]) ** 2))
    mse_16 = float(np.mean((Y_16[500:600] - Y_f[500:600]) ** 2))
    assert mse_16 < mse_8, \
        f"i16 MSE {mse_16:.4e} should beat i8 MSE {mse_8:.4e} on same model"


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
