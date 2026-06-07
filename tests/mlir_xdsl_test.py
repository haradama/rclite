"""xDSL-built symmetric MLIR == text-emitted symmetric MLIR == executor.

Validates the stage-(1) "IR construction" migration: `mlir_symmetric_xdsl`
assembles the same arith/memref/scf IR with the xDSL Python API, prints it, and
feeds it to the unchanged mlir-opt -> mlir-translate -> llc -> gcc pipeline. The
result must be bit-exact with both the existing text emitter and the numpy
executor.

Skipped unless the `mlir` extra (xdsl) is installed AND the MLIR toolchain is on
PATH. NB: the LLVM-20 toolchain (e.g. the nix devShell) is required — the IR is
printed in generic form and some other mlir-opt builds choke on it.
"""
from __future__ import annotations

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I8Symmetric, I16FixedPoint,
    quantize_model, QuantizedExecutor,
)
from rclite.codegen import mlir_symmetric

try:
    import xdsl  # noqa: F401
    from rclite.codegen.mlir_symmetric_xdsl import emit_symmetric_mlir_xdsl
    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

PASS = "\033[32m[PASS]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"


def _model(M=3, K=2, units=22, seed=4, inc_b=True, inc_i=True,
           topology=Topology.ESN_STANDARD):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                leak_rate=0.3, density=0.3, seed=seed,
                                chain_weight=0.5, chain_feedback=0.1, name="res"),
        readout=ReadoutNode(units=M, trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=30, include_bias=inc_b, include_input=inc_i,
                            name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((320, K)) * 0.3
    Y = np.stack([np.sin(np.arange(320) * 0.03 * (k + 1)) for k in range(M)], axis=1)
    exe.fit(X[:240], Y[:240])
    return rc, exe, X


def _qm(rc, exe, target):
    sf, inf, wf = (5, 6, 6) if target.storage_bits == 8 else (12, 10, 10)
    return quantize_model(rc, exe, QuantConfig(state_frac=sf, input_frac=inf,
                                               weight_frac=wf),
                          lut=TanhLUTSpec(n=128), target=target)


def _executor_ref(qm, Xt):
    qe = QuantizedExecutor(qm)
    qe.reset()
    out = np.zeros((Xt.shape[0], qm.M), dtype=np.int64)
    for t in range(Xt.shape[0]):
        u = qm.target.quantize_input_array(Xt[t], qm.config)
        qe.step_q(u)
        out[t] = qe.predict_one_q(u, qe.state_q)
    return out


def _run_via(emitter, qm, qx, head, sparse):
    """Compile+run the symmetric kernel via the given MLIR emitter."""
    orig = mlir_symmetric.emit_symmetric_mlir
    mlir_symmetric.emit_symmetric_mlir = emitter
    try:
        return mlir_symmetric.CompiledSymmetricMLIR(
            qm, head=head, sparse=sparse).predict_q(qx).astype(np.int64)
    finally:
        mlir_symmetric.emit_symmetric_mlir = orig


def _check(label, target, *, topology=Topology.ESN_STANDARD, sparse=None,
           head=None, inc_b=True, inc_i=True):
    rc, exe, X = _model(topology=topology, inc_b=inc_b, inc_i=inc_i)
    qm = _qm(rc, exe, target)
    Xt = X[240:268]
    qx = qm.target.quantize_input_array(Xt, qm.config)
    y_text = _run_via(mlir_symmetric.emit_symmetric_mlir, qm, qx, head, sparse)
    y_xdsl = _run_via(emit_symmetric_mlir_xdsl, qm, qx, head, sparse)
    # The text emitter is bit-exact with the executor (mlir_symmetric_spike_test),
    # so xDSL == text emitter establishes xDSL == executor transitively. For the
    # plain head we also assert the executor reference directly.
    assert np.array_equal(y_xdsl, y_text), f"{label}: xDSL != text emitter"
    if head is None:
        assert np.array_equal(y_xdsl, _executor_ref(qm, Xt)), \
            f"{label}: xDSL != executor"
    print(f"{PASS} {label}: xDSL == text emitter{' == executor' if head is None else ''} {y_xdsl.shape}")


def test_xdsl_symmetric_dense_bit_exact():
    if not _HAVE_XDSL:
        print(f"{SKIP} xdsl not installed (pip install 'rclite[mlir]')")
        return
    if not mlir_symmetric.tools_available():
        print(f"{SKIP} MLIR toolchain (mlir-opt/translate/llc/gcc) not on PATH")
        return
    _check("dense i16", I16FixedPoint())
    _check("dense i8", I8Symmetric())
    _check("dense i16 no-bias/no-input", I16FixedPoint(), inc_b=False, inc_i=False)


def test_xdsl_symmetric_structured_bit_exact():
    if not (_HAVE_XDSL and mlir_symmetric.tools_available()):
        print(f"{SKIP} xdsl / MLIR toolchain unavailable")
        return
    for topo in (Topology.SCR, Topology.DLR, Topology.DLRB):
        _check(f"{topo.name} i16", I16FixedPoint(), topology=topo)
        _check(f"{topo.name} i8", I8Symmetric(), topology=topo)


def test_xdsl_symmetric_sparse_bit_exact():
    if not (_HAVE_XDSL and mlir_symmetric.tools_available()):
        print(f"{SKIP} xdsl / MLIR toolchain unavailable")
        return
    _check("CSR i16", I16FixedPoint(), sparse="csr")
    _check("CSR i8", I8Symmetric(), sparse="csr")


def test_xdsl_symmetric_heads_bit_exact():
    if not (_HAVE_XDSL and mlir_symmetric.tools_available()):
        print(f"{SKIP} xdsl / MLIR toolchain unavailable")
        return
    for h in ("logits", "classify", "proba"):
        _check(f"dense {h} i16", I16FixedPoint(), head=h)
        _check(f"dense {h} i8", I8Symmetric(), head=h)
    # heads also over a structured topology + CSR
    _check("SCR classify i16", I16FixedPoint(), topology=Topology.SCR, head="classify")
    _check("CSR proba i16", I16FixedPoint(), sparse="csr", head="proba")


if __name__ == "__main__":
    test_xdsl_symmetric_dense_bit_exact()
