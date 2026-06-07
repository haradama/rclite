"""xDSL-built affine MLIR == text-emitted affine MLIR (== executor).

Validates the stage-(1) "IR construction" migration of the affine path:
`mlir_affine_xdsl` assembles the same func/arith/memref/scf IR with the xDSL
Python API, prints it, and feeds it to the unchanged mlir-opt -> mlir-translate
-> llc -> gcc pipeline. Result must be bit-exact with the text emitter (itself
bit-exact with the affine executor in mlir_affine_spike_test).

Skipped unless the `mlir` extra (xdsl) is installed AND the MLIR toolchain is on
PATH. Needs an LLVM-20 mlir-opt (the nix devShell) — IR is printed generic form.
"""
from __future__ import annotations

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data, quantize_model_affine,
)
from rclite.quant.affine.lut import LUTStrategy
from rclite.codegen import mlir_affine

try:
    import xdsl  # noqa: F401
    from rclite.codegen.mlir_affine_xdsl import emit_affine_mlir_xdsl
    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

PASS = "\033[32m[PASS]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"


def _model(M=3, K=2, units=22, topology=Topology.ESN_STANDARD,
           include_input=True, off=0.0, sc=1.0, seed=4):
    rc = ReservoirComputer(
        input=InputNode(units=K, input_offset=off, input_scaling=sc, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology, leak_rate=0.3,
                                density=0.3, seed=seed, chain_weight=0.5,
                                chain_feedback=0.1, name="res"),
        readout=ReadoutNode(units=M, trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=30, include_bias=True,
                            include_input=include_input, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((320, K)) * 0.3
    Y = np.stack([np.sin(np.arange(320) * 0.03 * (k + 1)) for k in range(M)], axis=1)
    exe.fit(X[:240], Y[:240])
    return rc, exe, X


def _qm(rc, exe, X, sb=8, lut_strategy=None):
    cfg = calibrate_from_data(rc, exe, X[:240], storage_bits=sb)
    if lut_strategy is not None:
        return quantize_model_affine(rc, exe, cfg, lut_strategy=lut_strategy)
    return quantize_model_affine(rc, exe, cfg)


def _run_via(emitter, qm, Xt, head, sparse):
    orig = mlir_affine.emit_affine_mlir
    mlir_affine.emit_affine_mlir = emitter
    try:
        return mlir_affine.CompiledAffineMLIR(
            qm, head=head, sparse=sparse).predict(Xt).astype(np.int64)
    finally:
        mlir_affine.emit_affine_mlir = orig


def _check(label, qm, Xt, head=None, sparse=None):
    y_text = _run_via(mlir_affine.emit_affine_mlir, qm, Xt, head, sparse)
    y_xdsl = _run_via(emit_affine_mlir_xdsl, qm, Xt, head, sparse)
    assert np.array_equal(y_xdsl, y_text), f"{label}: xDSL != text emitter"
    print(f"{PASS} {label}: xDSL == text emitter {y_xdsl.shape}")


def _guard():
    if not _HAVE_XDSL:
        print(f"{SKIP} xdsl not installed (pip install 'rclite[mlir]')")
        return False
    if not mlir_affine.tools_available():
        print(f"{SKIP} MLIR toolchain not on PATH")
        return False
    return True


def test_xdsl_affine_dense_direct():
    if not _guard():
        return
    for sb in (8, 16):
        rc, exe, X = _model(seed=sb)
        _check(f"dense DIRECT i{sb}", _qm(rc, exe, X, sb), X[240:262])
    rc, exe, X = _model(include_input=False)
    _check("dense no-input i8", _qm(rc, exe, X), X[240:262])


def test_xdsl_affine_structured():
    if not _guard():
        return
    for topo in (Topology.SCR, Topology.DLR, Topology.DLRB):
        rc, exe, X = _model(topology=topo)
        _check(f"{topo.name} i8", _qm(rc, exe, X), X[240:262])


def test_xdsl_affine_sparse():
    if not _guard():
        return
    rc, exe, X = _model(units=28)
    _check("CSR i8", _qm(rc, exe, X), X[240:262], sparse="csr")


def test_xdsl_affine_integer_preprocess():
    if not _guard():
        return
    rc, exe, X = _model(M=2, off=0.5, sc=1.3)
    qm = _qm(rc, exe, X)
    assert qm.has_integer_preprocess
    _check("int-preprocess i8", qm, X[240:262])


def test_xdsl_affine_lut_strategies():
    if not _guard():
        return
    for lut in (LUTStrategy.linear_interp(), LUTStrategy.polynomial()):
        rc, exe, X = _model(M=2)
        _check(f"LUT={lut.kind.name}", _qm(rc, exe, X, lut_strategy=lut), X[240:262])


def test_xdsl_affine_heads():
    if not _guard():
        return
    for sb in (8, 16):
        rc, exe, X = _model(M=4, seed=sb)
        qm = _qm(rc, exe, X, sb)
        _check(f"classify i{sb}", qm, X[240:262], head="classify")
        _check(f"proba i{sb}", qm, X[240:262], head="proba")


if __name__ == "__main__":
    for fn in (test_xdsl_affine_dense_direct, test_xdsl_affine_structured,
               test_xdsl_affine_sparse, test_xdsl_affine_integer_preprocess,
               test_xdsl_affine_lut_strategies, test_xdsl_affine_heads):
        fn()
