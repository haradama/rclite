"""Value specialization of baked weights in the "unroll" sparse kernel.

When SparsifyReservoir bakes a nonzero W_res weight as a compile-time
constant and that constant is +-1 or +-2**k, the LLVM lowerers replace the
multiply with a negate / shift (float +-1 -> fadd/fsub). This is bit-exact
(see _pow2_exp / _fixed_const_mul_to_accum / _const_mul_accum in
codegen/llvm.py) and removes a multiply per specialized MAC -- the
FPU-less / multiplier-light win flagged in the roadmap.

These tests pin two things the existing random-weight bit-exact suites do
not guarantee deterministically:
  1. the specialization *fires* (the IR shows shl / fewer fmul), and
  2. it stays bit-exact against the dense kernel on controlled weights.
"""
from __future__ import annotations
import pathlib
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite.codegen.llvm import (
    _pow2_exp, CompiledRC, CompiledQuantizedRC, CompiledAffineRC,
)
from rclite.ir import SparsifyReservoir
from rclite.ir.passes import StructuralSpecialize
from rclite.quant import I8Symmetric
from rclite.quant.affine import calibrate_from_data, quantize_model_affine

from sparse_quant_test import _model, _sym_qmodel

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


# ---------------------------------------------------------------------------

def test_pow2_exp():
    assert _pow2_exp(1) == 0
    assert _pow2_exp(-1) == 0
    assert _pow2_exp(2) == 1
    assert _pow2_exp(-2) == 1
    assert _pow2_exp(4) == 2
    assert _pow2_exp(-8) == 3
    assert _pow2_exp(1024) == 10
    # non powers of two and zero are not specialized
    for v in (0, 3, -3, 5, 6, 7, -6, 100):
        assert _pow2_exp(v) is None, v
    print("  _pow2_exp: +-2**k -> k, others -> None")


def _bit_exact(dense, sparse, label):
    d = int(np.max(np.abs(dense.astype(np.int64) - sparse.astype(np.int64))))
    assert d == 0, f"{label} not bit-exact: max|diff|={d}"


def test_symmetric_int_fires_and_exact():
    """Symmetric i8: baked +-2**k weights emit `shl`; dense uses none."""
    rc, exe, _, Xt = _model(units=40)
    qm = _sym_qmodel(rc, exe, target=I8Symmetric())
    # Controlled recurrent weights: powers of two (specialized) + a 3
    # (kept as a multiply), the rest zero (pruned by sparsify).
    W = np.zeros_like(qm.W_res_q)
    W[0, 1], W[0, 2], W[0, 3], W[0, 4] = 1, 2, 4, 3
    W[2, 0], W[3, 5] = -2, -8
    qm.W_res_q = W

    dense = CompiledQuantizedRC(qm)
    sparse = CompiledQuantizedRC(qm, passes=[SparsifyReservoir("unroll")])
    _bit_exact(dense.predict(Xt), sparse.predict(Xt), "symmetric i8 unroll")

    extra = sparse.llvm_ir.count("shl") - dense.llvm_ir.count("shl")
    assert extra > 0, "symmetric unroll did not add shl from specialization"
    print(f"  symmetric i8: unroll adds {extra} shl, bit-exact")


def test_affine_int_fires_and_exact():
    """Affine i8: baked +-2**k weights fold the multiply into `shl`."""
    rc, exe, Xtrain, Xt = _model(units=40)
    cfg = calibrate_from_data(rc, exe, Xtrain[:340], storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg)
    W = np.zeros_like(qm.W_res_q)
    W[0, 1], W[0, 2], W[0, 3], W[0, 4] = 1, 2, 4, 3
    W[2, 0], W[3, 5] = -2, -8
    qm.W_res_q = W

    dense = CompiledAffineRC(qm)
    sparse = CompiledAffineRC(qm, passes=[SparsifyReservoir("unroll")])
    _bit_exact(dense.predict(Xt), sparse.predict(Xt), "affine i8 unroll")

    extra = sparse.llvm_ir.count("shl") - dense.llvm_ir.count("shl")
    assert extra > 0, "affine unroll did not add shl from specialization"
    print(f"  affine i8: unroll adds {extra} shl, bit-exact")


def _float_fmul_count(exe_W_res, rc, exe, Xt):
    exe.W_res = exe_W_res
    comp = CompiledRC(rc, exe,
                      passes=[StructuralSpecialize(),
                              SparsifyReservoir("unroll")])
    return comp, comp.llvm_ir.count("fmul")


def test_float_pm1_fires_and_exact():
    """Float: baked +-1.0 needs no fmul (fadd/fsub); +-2**k keeps fmul."""
    rc, exe, _, Xt = _model(units=40)
    N = exe.W_res.shape[0]

    # Same sparsity structure, different magnitudes: the +-1 variant must
    # emit exactly two fewer fmul than the all-multiply variant.
    base = np.zeros((N, N))
    pm1 = base.copy(); pm1[0, 1], pm1[0, 2], pm1[0, 3] = 1.0, -1.0, 0.5
    mul = base.copy(); mul[0, 1], mul[0, 2], mul[0, 3] = 0.7, 0.6, 0.5

    comp_pm1, n_pm1 = _float_fmul_count(pm1, rc, exe, Xt)
    dense_pm1 = CompiledRC(rc, exe,
                           passes=[StructuralSpecialize()]).predict(Xt)
    _bit_exact_float = np.max(np.abs(dense_pm1 - comp_pm1.predict(Xt)))
    assert _bit_exact_float == 0.0, f"float unroll not bit-exact: {_bit_exact_float}"

    _, n_mul = _float_fmul_count(mul, rc, exe, Xt)
    assert n_mul - n_pm1 == 2, (
        f"expected 2 fewer fmul for +-1 weights, got {n_mul - n_pm1}")
    print(f"  float: +-1 removes 2 fmul ({n_mul}->{n_pm1}), bit-exact")


TESTS = [
    test_pow2_exp,
    test_symmetric_int_fires_and_exact,
    test_affine_int_fires_and_exact,
    test_float_pm1_fires_and_exact,
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
