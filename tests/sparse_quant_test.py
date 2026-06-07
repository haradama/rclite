"""Bit-exactness of W_res sparse specialization in the QUANTIZED paths.

`SparsifyReservoir` rewrites the dense RANDOM/ESN_STANDARD recurrent matvec
to skip exact-zero MACs. With threshold=0.0 the sparse integer kernels must
be bit-identical to the dense integer kernel (nonzeros kept in ascending
column order; adding 0 is the identity; the affine zero-point correction
uses the preserved row_sum_W_res). Covers:

  - symmetric fixed-point: i32 (default) and i8 (I8Symmetric)
  - affine: i8 and i16
  - strategies: unroll / csr / auto
  - structured topology no-op
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
    quantize_model,
)
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.codegen.llvm import CompiledQuantizedRC, CompiledAffineRC
from rclite.ir import SparsifyReservoir


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

STRATEGIES = ("unroll", "csr", "auto")


def _model(topology=Topology.ESN_STANDARD, units=48, density=0.15, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=topology,
            leak_rate=0.3,
            density=density,
            seed=seed,
            chain_weight=0.9,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=60,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((400, 1)) * 0.15
    exe.fit(X[:340], np.sin(np.arange(340) * 0.1)[:, None])
    return rc, exe, X, X[340:375]


def _sym_qmodel(rc, exe, target=None):
    cfg = QuantConfig(state_frac=16, input_frac=12, weight_frac=12)
    if target is not None:
        cfg = QuantConfig(state_frac=5, input_frac=6, weight_frac=6)
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128), target=target)


def _check(label, dense, sparse_fn):
    max_d = 0
    for s in STRATEGIES:
        sp = sparse_fn(s)
        d = int(np.max(np.abs(dense.astype(np.int64) - sp.astype(np.int64))))
        assert d == 0, f"{label} [{s}] not bit-exact: max|diff|={d}"
        max_d = max(max_d, d)
    print(f"  {label}: unroll/csr/auto all bit-exact (max|diff|={max_d})")


# ---------------------------------------------------------------------------


def test_symmetric_i32():
    rc, exe, _, Xt = _model()
    qm = _sym_qmodel(rc, exe)
    dense = CompiledQuantizedRC(qm).predict(Xt)
    _check(
        "symmetric i32",
        dense,
        lambda s: CompiledQuantizedRC(
            qm, passes=[SparsifyReservoir(strategy=s)]
        ).predict(Xt),
    )


def test_symmetric_i8():
    rc, exe, _, Xt = _model(units=40)
    qm = _sym_qmodel(rc, exe, target=I8Symmetric())
    dense = CompiledQuantizedRC(qm).predict(Xt)
    _check(
        "symmetric i8",
        dense,
        lambda s: CompiledQuantizedRC(
            qm, passes=[SparsifyReservoir(strategy=s)]
        ).predict(Xt),
    )


def _affine_check(storage_bits):
    rc, exe, Xtrain, Xt = _model(units=48)
    cfg = calibrate_from_data(rc, exe, Xtrain[:340], storage_bits=storage_bits)
    qm = quantize_model_affine(rc, exe, cfg)
    dense = CompiledAffineRC(qm).predict(Xt)
    _check(
        f"affine i{storage_bits}",
        dense,
        lambda s: CompiledAffineRC(
            qm, passes=[SparsifyReservoir(strategy=s)]
        ).predict(Xt),
    )


def test_affine_i8():
    _affine_check(8)


def test_affine_i16():
    _affine_check(16)


def test_structured_noop_quant():
    """DLR carries no W_res; SparsifyReservoir must be a no-op (symmetric)."""
    rc, exe, _, Xt = _model(topology=Topology.DLR, units=40)
    qm = _sym_qmodel(rc, exe)
    dense = CompiledQuantizedRC(qm).predict(Xt)
    sp = CompiledQuantizedRC(qm, passes=[SparsifyReservoir()]).predict(Xt)
    d = int(np.max(np.abs(dense.astype(np.int64) - sp.astype(np.int64))))
    assert d == 0, f"structured quant changed: {d}"
    print("  DLR (symmetric): SparsifyReservoir no-op, bit-exact")


TESTS = [
    test_symmetric_i32,
    test_symmetric_i8,
    test_affine_i8,
    test_affine_i16,
    test_structured_noop_quant,
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
