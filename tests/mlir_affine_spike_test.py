"""MLIR affine codegen: bit-exact vs the Python executor (full feature set).

Emits the affine quantized reservoir as textual MLIR, lowers it with
mlir-opt -> mlir-translate -> llc, links a host .so with gcc, and checks the
integer outputs match `AffineQuantizedExecutor` bit-for-bit. The emitted ops
mirror `_AffineLowerer`, so equality proves the MLIR pipeline reproduces
rclite's exact integer kernel.

Covers: dense + structured (DLR/SCR/DLRB), CSR-sparse W_res, identity +
integer preprocess, DIRECT/LINEAR_INTERP/POLYNOMIAL LUT, logits/argmax/softmax
heads, i8/i16, M=1 and M>1. Skipped when the MLIR toolchain is absent.
"""
from __future__ import annotations
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
)
from rclite.quant.affine.lut import LUTStrategy
from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params, softmax_q
from rclite.codegen import mlir_affine


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE = mlir_affine.tools_available()


def _model(M=3, K=2, units=22, topology=Topology.ESN_STANDARD,
           include_input=True, off=0.0, sc=1.0, seed=4):
    rc = ReservoirComputer(
        input=InputNode(units=K, input_offset=off, input_scaling=sc, name="in"),
        reservoir=ReservoirNode(units=units, topology=topology, leak_rate=0.3,
                                density=0.3, seed=seed, chain_weight=0.5,
                                chain_feedback=0.1, name="res"),
        readout=ReadoutNode(units=M, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=30,
                            include_bias=True, include_input=include_input,
                            name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((320, K)) * 0.3
    Y = np.stack([np.sin(np.arange(320) * 0.03 * (k + 1)) for k in range(M)],
                 axis=1)
    exe.fit(X[:240], Y[:240])
    return rc, exe, X


def _logits_ref(qm, X_float):
    qe = AffineQuantizedExecutor(qm)
    qe.reset()
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        xr = qe._quantize_raw_input(X_float[t])
        up = qe._quantize_u_pre(X_float[t])
        qe.step_q(up)
        out[t] = qe.predict_one_q(xr, qe.state_q)
    return out


def _check(qm, Xt, label, head=None, sparse=None):
    logits = _logits_ref(qm, Xt)
    if head == "classify":
        ref = np.argmax(logits, axis=1).astype(np.int64)
    elif head == "proba":
        sm = build_params(SoftmaxLUTSpec(), s_diff=qm.config.output.scale,
                          storage_bits=qm.storage_bits,
                          storage_dtype=np.dtype(f"int{qm.storage_bits}"))
        ref = np.stack([softmax_q(logits[t], sm) for t in range(logits.shape[0])])
    else:
        ref = logits
    got = mlir_affine.CompiledAffineMLIR(qm, head=head, sparse=sparse).predict(Xt)
    got = got.astype(np.int64)
    d = int(np.max(np.abs(ref.reshape(got.shape) - got)))
    assert d == 0, f"{label}: MLIR vs executor max|diff|={d}"


def _qm(rc, exe, X, sb=8, **kw):
    cfg = calibrate_from_data(rc, exe, X[:240], storage_bits=sb,
                              **{k: kw[k] for k in ("per_channel_W_res",)
                                 if k in kw})
    lut = kw.get("lut_strategy")
    return (quantize_model_affine(rc, exe, cfg, lut_strategy=lut) if lut
            else quantize_model_affine(rc, exe, cfg))


def test_dense_logits_matrix():
    if not HAVE:
        print("  (skip: MLIR toolchain not on PATH)"); return
    for sb in (8, 16):
        for M in (1, 4):
            for inc_i in (True, False):
                rc, exe, X = _model(M=M, include_input=inc_i, seed=M + sb)
                _check(_qm(rc, exe, X, sb), X[240:262],
                       f"i{sb} M={M} inc_input={inc_i}")
    print("  dense/logits: i8/i16 × M=1/4 × include_input{T,F} bit-exact")


def test_structured():
    if not HAVE:
        print("  (skip)"); return
    for topo in (Topology.DLR, Topology.SCR, Topology.DLRB):
        rc, exe, X = _model(M=2, topology=topo)
        _check(_qm(rc, exe, X), X[240:262], topo.name)
    print("  structured DLR/SCR/DLRB bit-exact")


def test_integer_preprocess():
    if not HAVE:
        print("  (skip)"); return
    rc, exe, X = _model(M=2, off=0.5, sc=1.3)
    qm = _qm(rc, exe, X)
    assert qm.has_integer_preprocess
    _check(qm, X[240:262], "int-preprocess")
    print("  integer preprocess (offset/scaling != identity) bit-exact")


def test_lut_strategies():
    if not HAVE:
        print("  (skip)"); return
    for lut in (LUTStrategy.linear_interp(), LUTStrategy.polynomial()):
        rc, exe, X = _model(M=2)
        _check(_qm(rc, exe, X, lut_strategy=lut), X[240:262],
               f"LUT={lut.kind.name}")
    print("  LINEAR_INTERP + POLYNOMIAL LUT bit-exact")


def test_csr_sparse():
    if not HAVE:
        print("  (skip)"); return
    rc, exe, X = _model(M=2, units=28)
    qm = _qm(rc, exe, X)
    _check(qm, X[240:262], "dense-ref")          # dense path
    _check(qm, X[240:262], "csr", sparse="csr")  # csr path, same result
    print("  CSR-sparse W_res bit-exact (matches dense)")


def test_heads():
    if not HAVE:
        print("  (skip)"); return
    for sb in (8, 16):
        rc, exe, X = _model(M=4, seed=sb)
        qm = _qm(rc, exe, X, sb)
        _check(qm, X[240:262], f"argmax i{sb}", head="classify")
        _check(qm, X[240:262], f"softmax i{sb}", head="proba")
    print("  argmax (classify) + softmax (proba) heads bit-exact, i8/i16")


def test_per_channel_unsupported():
    rc, exe, X = _model(M=2)
    cfg = calibrate_from_data(rc, exe, X[:240], storage_bits=8,
                              per_channel_W_res=True)
    qm = quantize_model_affine(rc, exe, cfg)
    try:
        mlir_affine.emit_affine_mlir(qm)
        raise AssertionError("expected NotImplementedError for per-channel")
    except NotImplementedError:
        pass
    print("  per-channel raises NotImplementedError (not yet in MLIR path)")


TESTS = [
    test_dense_logits_matrix,
    test_structured,
    test_integer_preprocess,
    test_lut_strategies,
    test_csr_sparse,
    test_heads,
    test_per_channel_unsupported,
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
