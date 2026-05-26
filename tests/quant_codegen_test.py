"""Tests for the integer LLVM codegen path (Phase C).

Verifies bit-exact parity between the on-host Python `QuantizedExecutor`
and the LLVM-emitted i32 kernel running via JIT.
"""
from __future__ import annotations
import ctypes
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import llvmlite.binding as llvm

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, quantize_model, QuantizedExecutor,
    build_ir_from_quantized,
)
from rclite.codegen.llvm import (
    emit_quantized_module, CompiledQuantizedRC, _ensure_initialized,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_and_quantize(units=40, topology=Topology.SCR,
                         state_frac=18, input_frac=12, weight_frac=12,
                         input_offset=0.0, input_scaling=1.0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=input_offset,
                        input_scaling=input_scaling,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                 chain_weight=0.9, leak_rate=0.3, seed=42,
                                 name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                             regularization=1e-6, washout=80,
                             include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 1)) * 0.2
    Y = np.sin(np.arange(400) * 0.1)[:, None]
    exe.fit(X, Y)
    cfg = QuantConfig(state_frac=state_frac, input_frac=input_frac,
                       weight_frac=weight_frac)
    qm = quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128))
    return rc, exe, qm, X[300:330]


def _jit(qm):
    _ensure_initialized()
    mod = llvm.parse_assembly(str(emit_quantized_module(qm)))
    mod.verify()
    target = llvm.Target.from_triple(llvm.get_default_triple())
    tm = target.create_target_machine(opt=3)
    engine = llvm.create_mcjit_compiler(mod, tm)
    engine.finalize_object()
    addr = engine.get_function_address("rc_predict")
    cfn = ctypes.CFUNCTYPE(
        None, ctypes.c_int64,
        ctypes.POINTER(ctypes.c_int32),
        ctypes.POINTER(ctypes.c_int32),
    )(addr)
    return engine, cfn


def _run_via_jit(qm, X_q):
    _, cfn = _jit(qm)
    T = X_q.shape[0]
    Y_q = np.zeros((T, qm.M), dtype=np.int32)
    cfn(T, X_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
         Y_q.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)))
    return Y_q


# ---------------------------------------------------------------- IR building


def test_build_ir_from_quantized_metadata():
    # Use a dense topology so W_res is present
    _, _, qm, _ = _build_and_quantize(topology=Topology.ESN_STANDARD)
    m = build_ir_from_quantized(qm)
    assert m.metadata["dtype"] == "i32"
    assert m.metadata["state_frac"] == 18
    assert "W_in" in m.weights and "W_res" in m.weights
    assert "W_out" in m.weights and "lut_table" in m.weights


def test_build_ir_drops_W_res_for_structured():
    """Structured topologies use scalar chain ops in the lowering — no
    dense W_res matrix should land in the module's weights dict."""
    for topology in (Topology.SCR, Topology.DLR, Topology.DLRB):
        _, _, qm, _ = _build_and_quantize(topology=topology)
        m = build_ir_from_quantized(qm)
        assert "W_res" not in m.weights, \
            f"W_res should not be emitted for {topology.name}"
        # ReservoirStep should also have W_res_name=None
        from rclite.ir.ops import ReservoirStep, TimeLoop
        for op in m.ops:
            if isinstance(op, TimeLoop):
                for body_op in op.body:
                    if isinstance(body_op, ReservoirStep):
                        assert body_op.W_res_name is None


def test_build_ir_requires_lut():
    _, _, qm, _ = _build_and_quantize()
    qm.lut = None
    qm.lut_table_q = None
    expect_raises(ValueError, build_ir_from_quantized, qm)


# ---------------------------------------------------------------- JIT parity


def _parity(qm, X, atol=0):
    cfg = qm.config
    X_q = qm.target.quantize_input_array(X, cfg).astype(np.int32)
    X_q = np.ascontiguousarray(X_q)
    Y_jit_q = _run_via_jit(qm, X_q)

    qexe = QuantizedExecutor(qm)
    Y_py = qexe.predict(X)
    Y_py_q = (Y_py * cfg.state_scale).round().astype(np.int32)

    # The Python executor returns dequantized float; let's compare on
    # the int values directly for bit-exact check. The manual loop must
    # mirror the kernel: preprocess raw quantized input → u_pre_q, step,
    # then readout with the *raw* quantized input.
    qexe.reset()
    for t in range(X.shape[0]):
        u_pre_q = qexe._preprocess_q(X_q[t])
        qexe.step_q(u_pre_q)
        y_i = qexe.predict_one_q(X_q[t], qexe.state_q)
        diff = int(Y_jit_q[t, 0]) - int(y_i[0])
        if abs(diff) > atol:
            raise AssertionError(
                f"step {t}: jit={Y_jit_q[t,0]} python={y_i[0]} diff={diff}"
            )


def test_parity_scr():
    _, _, qm, sample = _build_and_quantize(topology=Topology.SCR)
    _parity(qm, sample)


def test_parity_random():
    _, _, qm, sample = _build_and_quantize(topology=Topology.ESN_STANDARD)
    _parity(qm, sample)


def test_parity_dlr():
    _, _, qm, sample = _build_and_quantize(topology=Topology.DLR)
    _parity(qm, sample)


def test_parity_dlrb():
    _, _, qm, sample = _build_and_quantize(topology=Topology.DLRB)
    _parity(qm, sample)


def test_parity_across_frac_widths():
    for sf in (14, 16, 20):
        _, _, qm, sample = _build_and_quantize(state_frac=sf)
        _parity(qm, sample)


def test_parity_with_input_offset():
    """With input_offset != 0, the kernel must internally preprocess and
    the readout's include_input branch must use the *raw* (pre-preprocess)
    input — see [[input_preprocess_kernel]]."""
    for topology in (Topology.SCR, Topology.ESN_STANDARD, Topology.DLRB):
        _, _, qm, sample = _build_and_quantize(
            topology=topology, input_offset=0.5,
        )
        _parity(qm, sample)


def test_parity_with_input_scaling():
    """With input_scaling != 1, the kernel must internally scale and the
    readout's include_input branch must still use the *raw* input."""
    for topology in (Topology.SCR, Topology.ESN_STANDARD, Topology.DLR):
        _, _, qm, sample = _build_and_quantize(
            topology=topology, input_scaling=1.5,
        )
        _parity(qm, sample)


def test_parity_with_offset_and_scaling():
    """Joint test: non-zero offset AND non-unit scaling."""
    _, _, qm, sample = _build_and_quantize(
        topology=Topology.SCR, input_offset=-0.3, input_scaling=2.0,
    )
    _parity(qm, sample)


# ---------------------------------------------------------------- CompiledQuantizedRC


def test_compiled_quantized_predict():
    _, _, qm, sample = _build_and_quantize()
    compiled = CompiledQuantizedRC(qm)
    Y = compiled.predict(sample)
    assert Y.shape == (sample.shape[0], 1)
    assert np.all(np.isfinite(Y))


def test_compiled_quantized_matches_python_executor():
    _, _, qm, sample = _build_and_quantize()
    compiled = CompiledQuantizedRC(qm)
    Y_c = compiled.predict(sample)
    qexe = QuantizedExecutor(qm)
    Y_p = qexe.predict(sample)
    diff = float(np.max(np.abs(Y_c - Y_p)))
    assert diff == 0.0, f"compiled vs python differ by {diff}"


def test_compiled_quantized_ir_contains_lut():
    _, _, qm, _ = _build_and_quantize()
    c = CompiledQuantizedRC(qm)
    assert "lut_table" in c.llvm_ir
    assert "@\"rc_predict\"" in c.llvm_ir or "@rc_predict" in c.llvm_ir


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
