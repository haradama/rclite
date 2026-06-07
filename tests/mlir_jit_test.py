"""MLIR -> LLVM IR -> llvmlite MCJIT bridge is bit-exact with the executor.

Validates `mlir_jit`: the xDSL-built MLIR, lowered+translated via the CLI and
executed by llvmlite's MCJIT (the production backend), reproduces the integer
executor exactly. This is the architecture's execution path — llvmlite stays the
single execution substrate, MLIR is the opt-in xDSL-built representation.

Skipped unless xdsl is installed AND mlir-opt/mlir-translate are on PATH
(use the LLVM-20 nix devShell). Execution itself is llvmlite (a core dep).
"""

from __future__ import annotations

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig,
    TanhLUTSpec,
    I8Symmetric,
    I16FixedPoint,
    quantize_model,
    QuantizedExecutor,
)
from rclite.quant.affine import (
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
)
from rclite.codegen import mlir_jit

try:
    import xdsl  # noqa: F401

    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

PASS = "\033[32m[PASS]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"


def _model(M=3, K=2, units=18, topology=Topology.ESN_STANDARD, seed=4):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=topology,
            leak_rate=0.3,
            density=0.3,
            seed=seed,
            chain_weight=0.5,
            chain_feedback=0.1,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
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
    X = rng.standard_normal((300, K)) * 0.3
    Y = np.stack(
        [np.sin(np.arange(300) * 0.03 * (k + 1)) for k in range(M)], axis=1
    )
    exe.fit(X[:240], Y[:240])
    return rc, exe, X


def _sym_ref(qm, Xt):
    qe = QuantizedExecutor(qm)
    qe.reset()
    out = np.zeros((Xt.shape[0], qm.M), np.int64)
    for t in range(Xt.shape[0]):
        u = qm.target.quantize_input_array(Xt[t], qm.config)
        qe.step_q(u)
        out[t] = qe.predict_one_q(u, qe.state_q)
    return out


def _aff_ref(qm, Xt):
    qe = AffineQuantizedExecutor(qm)
    qe.reset()
    out = np.zeros((Xt.shape[0], qm.M), np.int64)
    for t in range(Xt.shape[0]):
        xr = qe._quantize_raw_input(Xt[t])
        up = qe._quantize_u_pre(Xt[t])
        qe.step_q(up)
        out[t] = qe.predict_one_q(xr, qe.state_q)
    return out


def _guard():
    if not _HAVE_XDSL:
        print(f"{SKIP} xdsl not installed")
        return False
    if not mlir_jit.tools_available():
        print(
            f"{SKIP} mlir-opt/mlir-translate not on PATH (use the nix devShell)"
        )
        return False
    return True


def test_bridge_symmetric():
    if not _guard():
        return
    for label, target, topo in [
        ("dense i16", I16FixedPoint(), Topology.ESN_STANDARD),
        ("dense i8", I8Symmetric(), Topology.ESN_STANDARD),
        ("SCR i16", I16FixedPoint(), Topology.SCR),
    ]:
        rc, exe, X = _model(topology=topo)
        sf, inf, wf = (5, 6, 6) if target.storage_bits == 8 else (12, 10, 10)
        qm = quantize_model(
            rc,
            exe,
            QuantConfig(state_frac=sf, input_frac=inf, weight_frac=wf),
            lut=TanhLUTSpec(n=128),
            target=target,
        )
        Xt = X[240:268]
        qx = qm.target.quantize_input_array(Xt, qm.config)
        got = mlir_jit.jit_symmetric(qm).predict_q(qx).astype(np.int64)
        assert np.array_equal(got, _sym_ref(qm, Xt)), (
            f"symmetric {label} != executor"
        )
        print(
            f"{PASS} symmetric {label}: xDSL MLIR -> llvmlite MCJIT == executor"
        )


def test_bridge_affine():
    if not _guard():
        return
    rc, exe, X = _model()
    cfg = calibrate_from_data(rc, exe, X[:240], storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg)
    Xt = X[240:268]
    qx = qm.config.input.quantize_array(Xt).astype(np.int8)
    got = mlir_jit.jit_affine(qm).predict_q(qx).astype(np.int64)
    assert np.array_equal(got, _aff_ref(qm, Xt)), "affine != executor"
    print(f"{PASS} affine dense i8: xDSL MLIR -> llvmlite MCJIT == executor")


def test_bridge_classify_head():
    if not _guard():
        return
    rc, exe, X = _model(M=4, seed=7)
    qm = quantize_model(
        rc,
        exe,
        QuantConfig(state_frac=12, input_frac=10, weight_frac=10),
        lut=TanhLUTSpec(n=128),
        target=I16FixedPoint(),
    )
    Xt = X[240:268]
    qx = qm.target.quantize_input_array(Xt, qm.config)
    got = mlir_jit.jit_symmetric(qm, head="classify").predict_q(qx)
    ref = np.argmax(_sym_ref(qm, Xt), axis=1).astype(np.int64)
    assert np.array_equal(got.astype(np.int64), ref), (
        "classify head != executor argmax"
    )
    print(f"{PASS} symmetric classify head: JIT class ids == executor argmax")


if __name__ == "__main__":
    test_bridge_symmetric()
    test_bridge_affine()
    test_bridge_classify_head()
