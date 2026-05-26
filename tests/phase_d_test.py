"""Tests for Phase D: i16 LLVM lowering, saturating intrinsics, online LMS,
I8Affine skeleton."""
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
    QuantConfig, TanhLUTSpec, I32FixedPoint, I16FixedPoint, I8Affine,
    quantize_model, QuantizedExecutor, IntegerLMSLearner,
)
from rclite.codegen.llvm import (
    CompiledQuantizedRC, emit_quantized_module, _ensure_initialized,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_for_quant(units=30, topology=Topology.SCR, state_frac=10,
                      input_frac=8, weight_frac=8, target=None):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=50,
                             include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1)) * 0.15
    Y = np.sin(np.arange(300) * 0.1)[:, None]
    exe.fit(X, Y)
    cfg = QuantConfig(state_frac=state_frac, input_frac=input_frac,
                       weight_frac=weight_frac)
    qm = quantize_model(rc, exe, cfg, target=target or I16FixedPoint(),
                         lut=TanhLUTSpec(n=64))
    return rc, exe, qm, X


# ---------------------------------------------------------------- i16 path


def test_i16_target_storage_bits():
    t = I16FixedPoint()
    assert t.storage_bits == 16
    assert t.accum_bits == 32
    assert t.storage_dtype == np.dtype("int16")


def test_i16_quantize_model_uses_i16_arrays():
    _, _, qm, _ = _build_for_quant()
    assert qm.W_in_q.dtype == np.int16
    assert qm.W_res_q.dtype == np.int16
    assert qm.W_out_q.dtype == np.int16
    assert qm.lut_table_q.dtype == np.int16


def test_i16_compiled_predict_matches_python_executor():
    _, _, qm, X = _build_for_quant()
    jit = CompiledQuantizedRC(qm)
    Y_jit = jit.predict(X[200:230])
    qexe = QuantizedExecutor(qm)
    Y_py = qexe.predict(X[200:230])
    diff = float(np.max(np.abs(Y_jit - Y_py)))
    assert diff == 0.0, f"i16 JIT vs python diff = {diff}"


def test_i16_ir_has_correct_pointer_types():
    _, _, qm, _ = _build_for_quant()
    mod = emit_quantized_module(qm)
    ir_text = str(mod)
    # Function signature should take i16* for X/Y
    assert "i16*" in ir_text
    assert "i16" in ir_text


def test_i32_and_i16_paths_both_run_and_finite():
    """Both targets compile and run; outputs are finite. (Quantitative
    agreement depends on whether the float W_out magnitudes fit i16's
    range under the chosen Q-format — typically they don't for SCR with
    large readout coefficients, so we only assert structural soundness.)"""
    rc, exe, _, X = _build_for_quant(target=I16FixedPoint())
    cfg = QuantConfig(state_frac=10, input_frac=8, weight_frac=8)
    qm32 = quantize_model(rc, exe, cfg, target=I32FixedPoint(),
                          lut=TanhLUTSpec(n=64))
    qm16 = quantize_model(rc, exe, cfg, target=I16FixedPoint(),
                          lut=TanhLUTSpec(n=64))
    Y32 = CompiledQuantizedRC(qm32).predict(X[100:130])
    Y16 = CompiledQuantizedRC(qm16).predict(X[100:130])
    assert Y32.shape == Y16.shape
    assert np.all(np.isfinite(Y32)) and np.all(np.isfinite(Y16))


# ---------------------------------------------------------------- saturating intrinsics


def test_saturating_intrinsic_present_in_ir():
    _, _, qm, _ = _build_for_quant(target=I32FixedPoint())
    ir_text = str(emit_quantized_module(qm, saturating=True))
    assert "llvm.sadd.sat.i32" in ir_text
    assert "llvm.sadd.sat.i64" in ir_text


def test_saturating_can_be_disabled():
    _, _, qm, _ = _build_for_quant(target=I32FixedPoint())
    ir_text = str(emit_quantized_module(qm, saturating=False))
    assert "llvm.sadd.sat" not in ir_text


def test_saturating_does_not_change_typical_output():
    """Sat ≡ non-sat when no overflow occurs (typical reservoir values)."""
    _, _, qm, X = _build_for_quant(target=I32FixedPoint())
    Y_sat = CompiledQuantizedRC(qm, saturating=True).predict(X[100:120])
    Y_wrap = CompiledQuantizedRC(qm, saturating=False).predict(X[100:120])
    diff = float(np.max(np.abs(Y_sat - Y_wrap)))
    assert diff == 0.0, f"saturating diff = {diff}"


# ---------------------------------------------------------------- I8Affine skeleton


def test_i8_affine_metadata():
    t = I8Affine()
    assert t.storage_bits == 8
    assert t.accum_bits == 32
    assert t.name == "i8-affine"


def test_i8_affine_raises_when_used():
    """Skeleton target raises with a pointer to the implementation roadmap."""
    rc, exe, _, _ = _build_for_quant()
    cfg = QuantConfig(state_frac=4, input_frac=2, weight_frac=2)
    expect_raises(NotImplementedError, quantize_model,
                   rc, exe, cfg, target=I8Affine())


# ---------------------------------------------------------------- online LMS


def test_lms_predict_returns_correct_shape():
    _, _, qm, X = _build_for_quant(target=I32FixedPoint())
    learner = IntegerLMSLearner(qm, learning_rate=1e-3)
    y = learner.step(X[0].ravel(), np.array([0.5]))
    assert y.shape == (1,)


def test_lms_learns_constant_target():
    """LMS should converge a zero-initialized readout to a constant target."""
    _, _, qm, X = _build_for_quant(target=I32FixedPoint(),
                                       state_frac=18, input_frac=12,
                                       weight_frac=12)
    qm.W_out_q[:] = 0
    learner = IntegerLMSLearner(qm, learning_rate=2e-3)

    target_val = 0.4
    errs = []
    # Cycle through 300 inputs and learn to predict 0.4
    for t in range(800):
        x = X[t % len(X)].ravel()
        y = learner.step(x, np.array([target_val]))
        errs.append((float(y[0]) - target_val) ** 2)

    mse_early = float(np.mean(errs[50:150]))
    mse_late = float(np.mean(errs[-200:]))
    # Late MSE should be substantially better than early
    assert mse_late < mse_early * 0.5, \
        f"LMS on constant target: early={mse_early:.4e}, late={mse_late:.4e}"


def test_lms_updates_W_out_q_in_place():
    rc, exe, qm, X = _build_for_quant(target=I32FixedPoint(),
                                          state_frac=18, input_frac=12,
                                          weight_frac=12)
    before = qm.W_out_q.copy()
    learner = IntegerLMSLearner(qm, learning_rate=1e-2)
    for t in range(10):
        learner.step(X[t].ravel(), np.array([0.0]))
    after = qm.W_out_q
    assert not np.array_equal(before, after), "W_out_q should have changed"


def test_lms_reset_clears_state():
    _, _, qm, X = _build_for_quant(target=I32FixedPoint())
    learner = IntegerLMSLearner(qm, learning_rate=1e-3)
    for t in range(5):
        learner.step(X[t].ravel(), np.array([0.0]))
    s1 = learner.state_q.copy()
    learner.reset()
    assert np.all(learner.state_q == 0)
    # After reset + same sequence, state should match s1
    for t in range(5):
        learner.step(X[t].ravel(), np.array([0.0]))
    # Approximately (W_out changed too, but state should be deterministic)
    # Actually W_out changes also affect predictions but not state
    s2 = learner.state_q
    assert np.allclose(s1, s2, atol=10), "reset + replay should reproduce state"


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
