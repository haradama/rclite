"""MLIR symmetric (Q-format) codegen: bit-exact vs the Python executor.

Emits the symmetric quantized reservoir as MLIR (arith/memref/scf), lowers it
with mlir-opt -> mlir-translate -> llc, links a host .so with gcc, and checks
the integer outputs match `QuantizedExecutor` bit-for-bit (the emitted ops
mirror `_IntLowerer`; the symmetric path uses wrapping arithmetic + an
interpolating tanh LUT + shift-based requantize).

Covers: dense + structured (DLR/SCR/DLRB) + CSR-sparse, i8/i16, logits/
argmax/softmax heads. Skipped when the MLIR toolchain is absent.
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
from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params, softmax_q
from rclite.codegen import mlir_jit


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
try:
    import xdsl  # noqa: F401
    from rclite.codegen.mlir_symmetric_xdsl import (
        emit_symmetric_mlir_xdsl,
    )

    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

HAVE = mlir_jit.tools_available() and _HAVE_XDSL


def _model(M=3, K=2, units=22, topology=Topology.ESN_STANDARD, seed=4):
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
    X = rng.standard_normal((320, K)) * 0.3
    Y = np.stack(
        [np.sin(np.arange(320) * 0.03 * (k + 1)) for k in range(M)], axis=1
    )
    exe.fit(X[:240], Y[:240])
    return rc, exe, X


def _qm(rc, exe, target):
    sf, inf, wf = (5, 6, 6) if target.storage_bits == 8 else (12, 10, 10)
    cfg = QuantConfig(state_frac=sf, input_frac=inf, weight_frac=wf)
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128), target=target)


def _logits_ref(qm, X_float):
    qe = QuantizedExecutor(qm)
    qe.reset()
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        u = qm.target.quantize_input_array(X_float[t], qm.config)
        qe.step_q(u)
        out[t] = qe.predict_one_q(u, qe.state_q)
    return out


def _check(qm, Xt, label, head=None, sparse=None):
    logits = _logits_ref(qm, Xt)
    if head == "classify":
        ref = np.argmax(logits, axis=1).astype(np.int64)
    elif head == "proba":
        ss = 1 << qm.config.state_frac
        sm = build_params(
            SoftmaxLUTSpec(),
            s_diff=1.0 / ss,
            storage_bits=qm.target.storage_bits,
            storage_dtype=np.dtype(f"int{qm.target.storage_bits}"),
        )
        ref = np.stack(
            [softmax_q(logits[t], sm) for t in range(logits.shape[0])]
        )
    else:
        ref = logits
    qx = qm.target.quantize_input_array(Xt, qm.config)
    got = (
        mlir_jit.jit_symmetric(qm, head=head, sparse=sparse)
        .predict_q(qx)
        .astype(np.int64)
    )
    d = int(np.max(np.abs(ref.reshape(got.shape) - got)))
    assert d == 0, f"{label}: MLIR vs executor max|diff|={d}"


def test_dense():
    if not HAVE:
        print("  (skip)")
        return
    for tgt in (I8Symmetric(), I16FixedPoint()):
        for M in (1, 4):
            rc, exe, X = _model(M=M, seed=M + tgt.storage_bits)
            _check(
                _qm(rc, exe, tgt),
                X[240:262],
                f"dense i{tgt.storage_bits} M={M}",
            )
    print("  dense i8/i16, M=1/4 bit-exact vs executor")


def test_structured():
    if not HAVE:
        print("  (skip)")
        return
    for topo in (Topology.DLR, Topology.SCR, Topology.DLRB):
        rc, exe, X = _model(M=2, topology=topo)
        _check(_qm(rc, exe, I16FixedPoint()), X[240:262], topo.name)
    print("  structured DLR/SCR/DLRB bit-exact")


def test_csr_sparse():
    if not HAVE:
        print("  (skip)")
        return
    rc, exe, X = _model(M=2, units=28)
    qm = _qm(rc, exe, I16FixedPoint())
    _check(qm, X[240:262], "dense-ref")
    _check(qm, X[240:262], "csr", sparse="csr")
    print("  CSR-sparse W_res bit-exact (matches dense)")


def test_heads():
    if not HAVE:
        print("  (skip)")
        return
    for tgt in (I8Symmetric(), I16FixedPoint()):
        rc, exe, X = _model(M=4, seed=tgt.storage_bits)
        qm = _qm(rc, exe, tgt)
        _check(qm, X[240:262], f"argmax i{tgt.storage_bits}", head="classify")
        _check(qm, X[240:262], f"softmax i{tgt.storage_bits}", head="proba")
    print("  argmax + softmax heads bit-exact, i8/i16")


def test_i32_unsupported():
    if not _HAVE_XDSL:
        print("  (skip: xdsl not installed)")
        return
    from rclite.quant import I32FixedPoint

    rc, exe, X = _model(M=2)
    cfg = QuantConfig(state_frac=16, input_frac=12, weight_frac=12)
    qm = quantize_model(
        rc, exe, cfg, lut=TanhLUTSpec(n=128), target=I32FixedPoint()
    )
    try:
        emit_symmetric_mlir_xdsl(qm)
        raise AssertionError("expected NotImplementedError for i32")
    except NotImplementedError:
        pass
    print("  i32 storage raises NotImplementedError (i8/i16 only for now)")


TESTS = [
    test_dense,
    test_structured,
    test_csr_sparse,
    test_heads,
    test_i32_unsupported,
]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
        except Exception:
            failures += 1
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
