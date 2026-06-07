"""Tests for the symmetric i8 quantization path (Phase E, step 1).

Verifies target metadata, quantize_model storage dtype, IR dtype mapping,
and bit-exact parity between the Python `QuantizedExecutor` and the LLVM
JIT-compiled i8 kernel.
"""

from __future__ import annotations
import pathlib
import sys
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
from rclite.quant import (
    QuantConfig,
    TanhLUTSpec,
    I8Symmetric,
    quantize_model,
    QuantizedExecutor,
    build_ir_from_quantized,
)
from rclite.quant._intops import wrap_to_storage
from rclite.codegen.llvm import CompiledQuantizedRC, emit_quantized_module


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_for_i8(
    units=24,
    topology=Topology.SCR,
    state_frac=5,
    input_frac=4,
    weight_frac=4,
    input_offset=0.0,
    input_scaling=1.0,
):
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=input_offset,
            input_scaling=input_scaling,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            topology=topology,
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
    X = rng.standard_normal((250, 1)) * 0.15
    Y = np.sin(np.arange(250) * 0.1)[:, None]
    exe.fit(X, Y)
    cfg = QuantConfig(
        state_frac=state_frac, input_frac=input_frac, weight_frac=weight_frac
    )
    qm = quantize_model(
        rc, exe, cfg, target=I8Symmetric(), lut=TanhLUTSpec(n=32)
    )
    return rc, exe, qm, X


# ---------------------------------------------------------------- target metadata


def test_i8_symmetric_metadata():
    t = I8Symmetric()
    assert t.storage_bits == 8
    assert t.accum_bits == 32
    assert t.name == "i8"
    assert t.storage_dtype == np.dtype("int8")
    assert t.accum_dtype == np.dtype("int32")
    assert t.llvm_storage_type == "i8"
    assert t.llvm_accum_type == "i32"


def test_i8_symmetric_saturation():
    t = I8Symmetric()
    cfg = QuantConfig(state_frac=4, input_frac=4, weight_frac=4)
    # 1.0 at state_scale=16 → 16, fits i8
    assert t.quantize_state(1.0, cfg) == 16
    # Huge value saturates to i8 max
    assert t.quantize_state(1e9, cfg) == np.iinfo(np.int8).max
    assert t.quantize_state(-1e9, cfg) == np.iinfo(np.int8).min


def test_i8_rejects_too_large_state_frac():
    """I8Symmetric rejects state_frac > 6 since (1<<state_frac) must fit in i8."""
    cfg_bad = QuantConfig(state_frac=7, input_frac=4, weight_frac=4)
    rc = ReservoirComputer(
        input=InputNode(units=1, input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(
            units=8, topology=Topology.SCR, chain_weight=0.9, leak_rate=0.3
        ),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE, include_bias=True),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((100, 1)) * 0.2
    Y = np.zeros((100, 1))
    exe.fit(X, Y)
    expect_raises(
        ValueError, quantize_model, rc, exe, cfg_bad, target=I8Symmetric()
    )


def test_i8_accepts_state_frac_up_to_6():
    for sf in (3, 4, 5, 6):
        cfg = QuantConfig(state_frac=sf, input_frac=4, weight_frac=4)
        I8Symmetric().validate_config(cfg)  # should not raise


# ---------------------------------------------------------------- wrap_to_storage


def test_wrap_to_storage_i32_identity():
    x = np.array([0, 1, -1, 1 << 30], dtype=np.int64)
    assert np.array_equal(wrap_to_storage(x, 32), x.astype(np.int32))


def test_wrap_to_storage_i16_overflow_wraps():
    # 32768 wraps to -32768; -32769 wraps to 32767
    x = np.array([32767, 32768, -32768, -32769, 0], dtype=np.int64)
    out = wrap_to_storage(x, 16)
    assert out.tolist() == [32767, -32768, -32768, 32767, 0]
    assert out.dtype == np.int32


def test_wrap_to_storage_i8_overflow_wraps():
    # 128 wraps to -128; -129 wraps to 127
    x = np.array([127, 128, -128, -129, 256, 0], dtype=np.int64)
    out = wrap_to_storage(x, 8)
    assert out.tolist() == [127, -128, -128, 127, 0, 0]
    assert out.dtype == np.int32


# ---------------------------------------------------------------- quantize_model


def test_i8_quantize_model_uses_i8_arrays():
    _, _, qm, _ = _build_for_i8()
    assert qm.W_in_q.dtype == np.int8
    assert qm.W_res_q.dtype == np.int8
    assert qm.W_out_q.dtype == np.int8
    assert qm.lut_table_q.dtype == np.int8


def test_i8_ir_dtype_metadata():
    _, _, qm, _ = _build_for_i8()
    m = build_ir_from_quantized(qm)
    assert m.metadata["dtype"] == "i8"


def test_i8_ir_has_i8_pointer_types():
    _, _, qm, _ = _build_for_i8()
    ir_text = str(emit_quantized_module(qm))
    assert "i8*" in ir_text


# ---------------------------------------------------------------- executor runs


def test_i8_python_executor_runs():
    _, _, qm, X = _build_for_i8(topology=Topology.SCR)
    qexe = QuantizedExecutor(qm)
    Y = qexe.predict(X[150:170])
    assert Y.shape == (20, 1)
    assert np.all(np.isfinite(Y))


def test_i8_compiled_predict_finite():
    _, _, qm, X = _build_for_i8(topology=Topology.SCR)
    Y = CompiledQuantizedRC(qm).predict(X[150:170])
    assert Y.shape == (20, 1)
    assert np.all(np.isfinite(Y))


# ---------------------------------------------------------------- JIT parity


def _assert_jit_parity(qm, X):
    """JIT and Python executor must agree bit-exactly on i8 storage."""
    Y_jit = CompiledQuantizedRC(qm).predict(X)
    qexe = QuantizedExecutor(qm)
    Y_py = qexe.predict(X)
    diff = float(np.max(np.abs(Y_jit - Y_py)))
    assert diff == 0.0, f"i8 JIT vs python diff = {diff}"


def test_i8_parity_scr():
    _, _, qm, X = _build_for_i8(topology=Topology.SCR)
    _assert_jit_parity(qm, X[150:180])


def test_i8_parity_dense():
    _, _, qm, X = _build_for_i8(topology=Topology.ESN_STANDARD)
    _assert_jit_parity(qm, X[150:180])


def test_i8_parity_dlr():
    _, _, qm, X = _build_for_i8(topology=Topology.DLR)
    _assert_jit_parity(qm, X[150:180])


def test_i8_parity_dlrb():
    _, _, qm, X = _build_for_i8(topology=Topology.DLRB)
    _assert_jit_parity(qm, X[150:180])


def test_i8_parity_with_input_offset_and_scaling():
    _, _, qm, X = _build_for_i8(
        topology=Topology.SCR,
        input_offset=0.3,
        input_scaling=1.5,
    )
    _assert_jit_parity(qm, X[150:180])


def test_i8_parity_across_state_frac():
    for sf in (3, 4, 5, 6):
        _, _, qm, X = _build_for_i8(state_frac=sf)
        _assert_jit_parity(qm, X[150:170])


# ---------------------------------------------------------------- weight footprint


def test_i8_weight_footprint_quarter_of_i32():
    """i8 weight globals should occupy 1/4 the bytes of the equivalent i32."""
    from rclite.quant import I32FixedPoint

    rc, exe, qm_i8, _ = _build_for_i8()
    qm_i32 = quantize_model(
        rc, exe, qm_i8.config, target=I32FixedPoint(), lut=TanhLUTSpec(n=32)
    )
    bytes_i8 = (
        qm_i8.W_in_q.nbytes + qm_i8.W_res_q.nbytes + qm_i8.W_out_q.nbytes
    )
    bytes_i32 = (
        qm_i32.W_in_q.nbytes + qm_i32.W_res_q.nbytes + qm_i32.W_out_q.nbytes
    )
    assert bytes_i8 * 4 == bytes_i32, (
        f"i8 bytes={bytes_i8} i32 bytes={bytes_i32}"
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
