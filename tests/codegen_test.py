"""LLVM JIT codegen parity tests."""

from __future__ import annotations
import sys
import pathlib
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"Expected {exc_type.__name__}, no exception raised")


def _build_and_train(**overrides):
    cfg = dict(
        units=80,
        topology=Topology.ESN_STANDARD,
        spectral_radius=0.9,
        leak_rate=0.3,
        density=0.2,
        include_bias=True,
        include_input=True,
        input_distribution=Distribution.NORMAL,
        chain_weight=0.7,
        chain_feedback=0.05,
        activation=Activation.TANH,
    )
    cfg.update(overrides)
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=0.5,
            input_scaling=1.0,
            input_distribution=cfg["input_distribution"],
            name="in",
        ),
        reservoir=ReservoirNode(
            units=cfg["units"],
            activation=cfg["activation"],
            spectral_radius=cfg["spectral_radius"],
            leak_rate=cfg["leak_rate"],
            density=cfg["density"],
            topology=cfg["topology"],
            chain_weight=cfg["chain_weight"],
            chain_feedback=cfg["chain_feedback"],
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=100,
            include_bias=cfg["include_bias"],
            include_input=cfg["include_input"],
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((500, 1)) * 0.3 + 0.5
    Y = np.sin(np.arange(500) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X


def _assert_close(
    a: np.ndarray, b: np.ndarray, atol: float = 1e-10, rtol: float = 1e-10
):
    diff = float(np.max(np.abs(a - b)))
    scale = float(np.max(np.abs(b))) + 1e-30
    rel = diff / scale
    if not (diff < atol or rel < rtol):
        raise AssertionError(
            f"arrays differ: abs={diff:.3e}, rel={rel:.3e}, atol={atol}, rtol={rtol}"
        )


def test_parity_default_random_topology():
    rc, exe, X = _build_and_train()
    Y_np = exe.predict(X)
    jit = compile_rc(rc, exe)
    Y_jit = jit.predict(X)
    _assert_close(Y_jit, Y_np, atol=1e-10)


def test_parity_no_bias_no_input():
    rc, exe, X = _build_and_train(include_bias=False, include_input=False)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_bias_only():
    rc, exe, X = _build_and_train(include_bias=True, include_input=False)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_input_only():
    rc, exe, X = _build_and_train(include_bias=False, include_input=True)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_leak_one():
    rc, exe, X = _build_and_train(leak_rate=1.0)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_scr_structured():
    rc, exe, X = _build_and_train(
        topology=Topology.SCR,
        chain_weight=0.9,
        input_distribution=Distribution.BERNOULLI,
    )
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_dlr_structured():
    rc, exe, X = _build_and_train(
        topology=Topology.DLR,
        chain_weight=0.8,
        input_distribution=Distribution.BERNOULLI,
    )
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_dlrb_structured():
    rc, exe, X = _build_and_train(
        topology=Topology.DLRB,
        chain_weight=0.7,
        chain_feedback=0.1,
        input_distribution=Distribution.BERNOULLI,
    )
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_multi_dim_output():
    rng = np.random.default_rng(0)
    rc = ReservoirComputer(
        input=InputNode(
            units=2, input_offset=0.0, input_scaling=1.0, name="in"
        ),
        reservoir=ReservoirNode(
            units=60,
            spectral_radius=0.9,
            leak_rate=0.5,
            density=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=3,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=80,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = rng.standard_normal((400, 2)) * 0.5
    Y = np.column_stack(
        [
            np.sin(np.arange(400) * 0.1),
            np.cos(np.arange(400) * 0.07),
            np.sin(np.arange(400) * 0.15) * 0.5,
        ]
    )
    exe.fit(X, Y)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_rejects_untrained_readout():
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=10, name="res"),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    expect_raises(ValueError, compile_rc, rc, exe)


def test_parity_relu_activation():
    rc, exe, X = _build_and_train(activation=Activation.RELU)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_sigmoid_activation():
    rc, exe, X = _build_and_train(activation=Activation.SIGMOID)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_identity_activation():
    rc, exe, X = _build_and_train(activation=Activation.IDENTITY)
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_sigmoid_sparse_fused():
    # Guards that the activation threads through SparsifyReservoir + FuseStepReadout
    # (both reconstruct the step op) and stays bit-exact with the dense path.
    from rclite.ir.passes import StructuralSpecialize
    from rclite.ir.passes.sparsify import SparsifyReservoir
    from rclite.ir.passes.fuse import FuseStepReadout

    rc, exe, X = _build_and_train(activation=Activation.SIGMOID, density=0.1)
    Y_np = exe.predict(X)
    passes = [
        StructuralSpecialize(),
        SparsifyReservoir(strategy="csr"),
        FuseStepReadout(),
    ]
    Y_jit = compile_rc(rc, exe, passes=passes).predict(X)
    _assert_close(Y_jit, Y_np)


def test_parity_relu_structured():
    # RELU on a structured topology (no W_res matvec, O(N) chain kernel).
    rc, exe, X = _build_and_train(
        activation=Activation.RELU,
        topology=Topology.SCR,
        chain_weight=0.9,
        input_distribution=Distribution.BERNOULLI,
    )
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    _assert_close(Y_jit, Y_np)


def test_rejects_unsupported_activation():
    # LEAKY_INTEGRATOR / SPIKING are enum members with no runtime/codegen
    # implementation — codegen must reject them rather than emit wrong code.
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=10,
            activation=Activation.SPIKING,
            spectral_radius=0.9,
            density=0.5,
            name="res",
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    # emit_module checks the activation first, before touching trained weights,
    # so an untrained executor is enough to exercise the guard.
    exe = RCExecutor(rc)
    expect_raises(NotImplementedError, compile_rc, rc, exe)


def test_emit_produces_valid_ir():
    rc, exe, _ = _build_and_train()
    jit = compile_rc(rc, exe)
    ir = jit.llvm_ir
    assert '@"rc_predict"' in ir or "@rc_predict" in ir
    assert '@"tanh"' in ir or "@tanh" in ir
    assert "W_in" in ir and "W_res" in ir and "W_out" in ir


def test_assembly_is_emitted():
    rc, exe, _ = _build_and_train(units=20)
    jit = compile_rc(rc, exe)
    asm = jit.assembly
    assert len(asm) > 0
    # Some indication of a function epilogue/prologue
    assert "rc_predict" in asm


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
