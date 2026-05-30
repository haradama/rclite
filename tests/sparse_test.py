"""Bit-exactness tests for the W_res sparse-specialization pass.

`SparsifyReservoir` rewrites the dense RANDOM/ESN_STANDARD recurrent matvec
to skip the exact-zero MACs, in two flavours:

  - unroll: nonzero (col, weight) pairs baked as constants
  - csr:    compressed-sparse-row arrays + per-row nonzero loop

With the default threshold=0.0 both must be **bit-identical** to the dense
LLVM kernel (nonzeros kept in ascending column order, adding 0.0 is the
identity), and match the reference runtime to float tolerance. These tests
cover fused / non-fused lowering, include_input on/off, several densities,
and confirm that structured topologies are left untouched (no-op).
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
from rclite.ir import (
    build_ir, StructuralSpecialize, FuseStepReadout, SparsifyReservoir,
    TimeLoop, ReservoirStep, FusedStepReadout,
)
from rclite.codegen import compile_rc


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def _model(topology=Topology.RANDOM, units=70, density=0.1, seed=3,
           include_input=True):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.4, input_scaling=1.1,
                        name="in"),
        reservoir=ReservoirNode(units=units, activation=Activation.TANH,
                                topology=topology, spectral_radius=0.9,
                                leak_rate=0.35, density=density, seed=seed,
                                chain_weight=0.5, name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=50,
                            include_bias=True, include_input=include_input,
                            name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((260, 1)) * 0.3 + 0.4
    Y = np.sin(np.arange(260) * 0.07)[:, None]
    exe.fit(X, Y)
    return rc, exe, X[200:240]


def _predict(rc, exe, X, passes):
    return compile_rc(rc, exe, passes=passes).predict(X)


def _find_step(passes, rc, exe):
    """Return the (Fused)ReservoirStep op produced by `passes`."""
    mod = build_ir(rc, exe)
    for p in passes:
        mod = p(mod)
    for op in mod.ops:
        if isinstance(op, TimeLoop):
            for o in op.body:
                if isinstance(o, (ReservoirStep, FusedStepReadout)):
                    return o
        if isinstance(op, (ReservoirStep, FusedStepReadout)):
            return op
    return None


# ---------------------------------------------------------------------------

def test_unroll_bit_exact_vs_dense():
    rc, exe, X = _model(density=0.1)
    dense_passes = [StructuralSpecialize()]
    sparse_passes = [StructuralSpecialize(), SparsifyReservoir(strategy="unroll")]
    Y_dense = _predict(rc, exe, X, dense_passes)
    Y_sparse = _predict(rc, exe, X, sparse_passes)
    diff = float(np.max(np.abs(Y_dense - Y_sparse)))
    assert diff == 0.0, f"unroll not bit-exact vs dense: max|diff|={diff}"
    # also matches runtime
    Y_ref = exe.predict(X)
    assert np.allclose(Y_ref, Y_sparse, atol=1e-10)
    print(f"  unroll vs dense bit-exact (diff={diff}), runtime parity ok")


def test_csr_bit_exact_vs_dense():
    rc, exe, X = _model(density=0.1)
    Y_dense = _predict(rc, exe, X, [StructuralSpecialize()])
    Y_csr = _predict(rc, exe, X,
                     [StructuralSpecialize(), SparsifyReservoir(strategy="csr")])
    diff = float(np.max(np.abs(Y_dense - Y_csr)))
    assert diff == 0.0, f"csr not bit-exact vs dense: max|diff|={diff}"
    Y_ref = exe.predict(X)
    assert np.allclose(Y_ref, Y_csr, atol=1e-10)
    print(f"  csr vs dense bit-exact (diff={diff}), runtime parity ok")


def test_fused_sparse_bit_exact():
    rc, exe, X = _model(density=0.15)
    dense = [StructuralSpecialize(), FuseStepReadout()]
    for strat in ("unroll", "csr"):
        sparse = [StructuralSpecialize(), FuseStepReadout(),
                  SparsifyReservoir(strategy=strat)]
        Y_dense = _predict(rc, exe, X, dense)
        Y_sparse = _predict(rc, exe, X, sparse)
        diff = float(np.max(np.abs(Y_dense - Y_sparse)))
        assert diff == 0.0, f"fused {strat} not bit-exact: {diff}"
        # spec survives fusion
        op = _find_step(sparse, rc, exe)
        assert isinstance(op, FusedStepReadout) and op.res_sparse is not None
        assert op.res_sparse.kind == strat
    print("  fused unroll/csr bit-exact, res_sparse preserved through fusion")


def test_sparsify_then_fuse_order():
    """Sparsify before Fuse also works (FuseStepReadout carries res_sparse)."""
    rc, exe, X = _model(density=0.12)
    Y_dense = _predict(rc, exe, X, [StructuralSpecialize(), FuseStepReadout()])
    passes = [StructuralSpecialize(), SparsifyReservoir(strategy="unroll"),
              FuseStepReadout()]
    Y_sparse = _predict(rc, exe, X, passes)
    diff = float(np.max(np.abs(Y_dense - Y_sparse)))
    assert diff == 0.0, f"sparsify-then-fuse not bit-exact: {diff}"
    op = _find_step(passes, rc, exe)
    assert isinstance(op, FusedStepReadout) and op.res_sparse is not None
    print("  sparsify→fuse order bit-exact, spec carried through fusion")


def test_auto_picks_both_kernels():
    rc, exe, X = _model(units=70, density=0.1)
    Y_dense = _predict(rc, exe, X, [StructuralSpecialize()])
    # small cap -> csr ; large cap -> unroll
    csr_passes = [StructuralSpecialize(),
                  SparsifyReservoir(strategy="auto", max_unroll_nnz=1)]
    unroll_passes = [StructuralSpecialize(),
                     SparsifyReservoir(strategy="auto", max_unroll_nnz=10**9)]
    assert _find_step(csr_passes, rc, exe).res_sparse.kind == "csr"
    assert _find_step(unroll_passes, rc, exe).res_sparse.kind == "unroll"
    for p in (csr_passes, unroll_passes):
        diff = float(np.max(np.abs(Y_dense - _predict(rc, exe, X, p))))
        assert diff == 0.0, f"auto kernel not bit-exact: {diff}"
    print("  auto selects csr (small cap) / unroll (large cap), both bit-exact")


def test_variants_density_input():
    for seed, density in enumerate((0.05, 0.1, 0.3)):
        for include_input in (True, False):
            rc, exe, X = _model(units=64, density=density, seed=seed + 1,
                                include_input=include_input)
            Y_dense = _predict(rc, exe, X, [StructuralSpecialize()])
            for strat in ("unroll", "csr"):
                Y_sparse = _predict(
                    rc, exe, X,
                    [StructuralSpecialize(), SparsifyReservoir(strategy=strat)])
                diff = float(np.max(np.abs(Y_dense - Y_sparse)))
                assert diff == 0.0, (
                    f"density={density} input={include_input} {strat} "
                    f"diff={diff}")
    print("  density {0.05,0.1,0.3} × include_input{T,F} × {unroll,csr} bit-exact")


def test_structured_topology_noop():
    """DLR carries no W_res; SparsifyReservoir must be a no-op."""
    rc, exe, X = _model(topology=Topology.DLR, units=60)
    Y_dense = _predict(rc, exe, X, [StructuralSpecialize()])
    Y_sparse = _predict(
        rc, exe, X, [StructuralSpecialize(), SparsifyReservoir()])
    diff = float(np.max(np.abs(Y_dense - Y_sparse)))
    assert diff == 0.0, f"structured topology changed: {diff}"
    op = _find_step([StructuralSpecialize(), SparsifyReservoir()], rc, exe)
    assert op.res_sparse is None
    print("  DLR: SparsifyReservoir is a no-op (no W_res), bit-exact")


def test_nnz_matches_matrix():
    rc, exe, _ = _model(units=64, density=0.2, seed=5)
    op = _find_step([StructuralSpecialize(), SparsifyReservoir(strategy="unroll")],
                    rc, exe)
    nnz_actual = int(np.count_nonzero(exe.W_res))
    assert op.res_sparse.nnz == nnz_actual, (
        f"spec nnz {op.res_sparse.nnz} != matrix nnz {nnz_actual}")
    flat = [w for row in op.res_sparse.rows for (_, w) in row]
    assert len(flat) == nnz_actual
    print(f"  nnz bookkeeping correct ({nnz_actual} nonzeros)")


TESTS = [
    test_unroll_bit_exact_vs_dense,
    test_csr_bit_exact_vs_dense,
    test_fused_sparse_bit_exact,
    test_sparsify_then_fuse_order,
    test_auto_picks_both_kernels,
    test_variants_density_input,
    test_structured_topology_noop,
    test_nnz_matches_matrix,
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
