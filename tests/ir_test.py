"""Tests for the rclite IR layer (ops, passes, printer) and parity with
direct lowering."""

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
    Activation,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.ir import (
    build_ir,
    to_mlir_text,
    StructuralSpecialize,
    FuseStepReadout,
    TimeUnroll,
    NormalizeReservoir,
    VerifyEchoStateConstraint,
    PruneInactiveNodes,
    ProfileReservoir,
    SparsifyReservoir,
    TimeLoop,
    ReservoirStep,
    FusedStepReadout,
)
from rclite.codegen import compile_rc


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build(
    topology=Topology.ESN_STANDARD,
    units=60,
    include_input=True,
    include_bias=True,
):
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_offset=0.5,
            input_scaling=1.0,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            topology=topology,
            chain_weight=0.85,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.2,
            seed=11,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=80,
            include_bias=include_bias,
            include_input=include_input,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((300, 1)) * 0.3 + 0.5
    Y = np.sin(np.arange(300) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X[200:230]


def test_build_ir_default_shape():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    assert m.K == 1 and m.N == 60 and m.M == 1
    assert len(m.ops) == 1
    loop = m.ops[0]
    assert isinstance(loop, TimeLoop)
    assert loop.unroll == 1
    types = [type(o).__name__ for o in loop.body]
    assert types == [
        "PreprocessInput",
        "ReservoirStep",
        "BuildPhi",
        "ReadoutLinear",
    ]
    assert set(m.weights.keys()) == {"W_in", "W_res", "W_out"}


def test_build_ir_structured_omits_W_res():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    assert "W_res" not in m.weights
    step = m.ops[0].body[1]
    assert isinstance(step, ReservoirStep)
    assert step.W_res_name is None


def test_structural_specialize_drops_W_res_when_unused():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD)  # has W_res
    m = build_ir(rc, exe)
    # Manually convert the op's topology to SCR but keep W_res referenced;
    # StructuralSpecialize should clean it up.
    from dataclasses import replace

    step = m.ops[0].body[1]
    forced = TimeLoop(
        body=tuple(
            replace(step, topology=Topology.SCR, chain_weight=0.85)
            if isinstance(o, ReservoirStep)
            else o
            for o in m.ops[0].body
        )
    )
    m2 = type(m)(
        K=m.K,
        N=m.N,
        M=m.M,
        weights=dict(m.weights),
        ops=[forced],
        metadata=dict(m.metadata),
    )
    m3 = StructuralSpecialize()(m2)
    assert "W_res" not in m3.weights
    new_step = m3.ops[0].body[1]
    assert new_step.W_res_name is None


def test_structural_specialize_rejects_unstable_scr():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    from dataclasses import replace

    step = m.ops[0].body[1]
    bad = TimeLoop(
        body=tuple(
            replace(step, chain_weight=1.5)
            if isinstance(o, ReservoirStep)
            else o
            for o in m.ops[0].body
        )
    )
    m_bad = type(m)(
        K=m.K,
        N=m.N,
        M=m.M,
        weights=dict(m.weights),
        ops=[bad],
        metadata=dict(m.metadata),
    )
    expect_raises(ValueError, StructuralSpecialize(), m_bad)


def test_fuse_step_readout_collapses_quadruple():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    m = FuseStepReadout()(m)
    body = m.ops[0].body
    types = [type(o).__name__ for o in body]
    assert types == ["PreprocessInput", "FusedStepReadout"]


def test_fuse_preserves_topology_and_phi_config():
    rc, exe, _ = _build(topology=Topology.DLRB)
    m = build_ir(rc, exe)
    m = FuseStepReadout()(m)
    fused = m.ops[0].body[1]
    assert isinstance(fused, FusedStepReadout)
    assert fused.topology == Topology.DLRB
    assert fused.include_bias_phi
    assert fused.include_input_phi


def test_time_unroll_sets_unroll_attr():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    m = TimeUnroll(K=4)(m)
    assert m.ops[0].unroll == 4


def test_time_unroll_k1_is_identity():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    m2 = TimeUnroll(K=1)(m)
    assert m2 is m or m2.ops[0].unroll == 1


def test_time_unroll_rejects_invalid_k():
    expect_raises(ValueError, TimeUnroll, K=0)


def test_printer_emits_expected_ops():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    text = to_mlir_text(m)
    assert "rc.preprocess_input" in text
    assert "rc.reservoir_step" in text
    assert "rc.build_phi" in text
    assert "rc.readout_linear" in text
    assert 'topology = "SCR"' in text
    assert "scf.for" in text
    assert "@W_in" in text


def test_printer_shows_fused_op():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    m = FuseStepReadout()(m)
    text = to_mlir_text(m)
    assert "rc.fused_step_readout" in text
    assert "rc.build_phi" not in text
    assert "rc.readout_linear" not in text


def test_printer_shows_unroll_hint():
    rc, exe, _ = _build()
    m = build_ir(rc, exe)
    m = TimeUnroll(K=8)(m)
    text = to_mlir_text(m)
    assert "unroll=8" in text


def test_normalize_reservoir_scales_dense_w_res():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD)
    m = build_ir(rc, exe)
    W0 = np.asarray(m.weights["W_res"])
    sr0 = float(np.max(np.abs(np.linalg.eigvals(W0))))
    assert sr0 > 0.0

    m2 = NormalizeReservoir(target_spectral_radius=0.7)(m)
    W1 = np.asarray(m2.weights["W_res"])
    sr1 = float(np.max(np.abs(np.linalg.eigvals(W1))))
    assert abs(sr1 - 0.7) < 1e-6
    assert "normalization_report" in m2.metadata


def test_normalize_reservoir_noop_for_structured():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    m2 = NormalizeReservoir(target_spectral_radius=0.7)(m)
    assert m2.weights.keys() == m.weights.keys()
    assert "normalization_report" not in m2.metadata


def test_verify_echo_state_dense_warning_non_strict():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD)
    m = build_ir(rc, exe)
    m2 = VerifyEchoStateConstraint(strict=False, dense_radius_limit=0.1)(m)
    warns = m2.metadata.get("echo_state_warnings", [])
    assert warns, "expected dense-radius warning"
    assert m2.metadata.get("echo_state_verified") is False


def test_verify_echo_state_dense_raises_strict():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD)
    m = build_ir(rc, exe)
    expect_raises(
        ValueError,
        VerifyEchoStateConstraint(strict=True, dense_radius_limit=0.1),
        m,
    )


def test_verify_echo_state_structured_scr_ok():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    m2 = VerifyEchoStateConstraint(strict=True)(m)
    assert m2.metadata.get("echo_state_verified") is True


def test_verify_echo_state_structured_scr_violation():
    rc, exe, _ = _build(topology=Topology.SCR)
    m = build_ir(rc, exe)
    from dataclasses import replace

    step = m.ops[0].body[1]
    bad = TimeLoop(
        body=tuple(
            replace(step, chain_weight=1.2)
            if isinstance(o, ReservoirStep)
            else o
            for o in m.ops[0].body
        )
    )
    m_bad = type(m)(
        K=m.K,
        N=m.N,
        M=m.M,
        weights=dict(m.weights),
        ops=[bad],
        metadata=dict(m.metadata),
    )
    expect_raises(ValueError, VerifyEchoStateConstraint(strict=True), m_bad)


def test_prune_inactive_nodes_rewrites_dimensions_and_weights():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD, units=60)
    m = build_ir(rc, exe)
    m2 = PruneInactiveNodes(keep_ratio=0.5)(m)
    assert m2.N == 30
    assert m2.weights["W_in"].shape == (30, 1)
    assert m2.weights["W_res"].shape == (30, 30)
    # F = 1(bias) + 1(input) + N
    assert m2.weights["W_out"].shape[1] == 32
    assert m2.metadata.get("pruned_nodes") == 30


def test_prune_inactive_nodes_compiles_and_predicts():
    rc, exe, Xs = _build(topology=Topology.ESN_STANDARD, units=80)
    jit = compile_rc(rc, exe, passes=[PruneInactiveNodes(keep_ratio=0.5)])
    Y = jit.predict(Xs)
    assert Y.shape == (Xs.shape[0], 1)
    assert np.all(np.isfinite(Y))


def test_prune_after_sparsify_is_skipped_with_warning():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD, units=60)
    m = build_ir(rc, exe)
    m_sp = SparsifyReservoir()(m)
    m_pr = PruneInactiveNodes(keep_ratio=0.5)(m_sp)
    warns = m_pr.metadata.get("prune_warnings", [])
    assert warns
    assert "Run prune before SparsifyReservoir" in warns[0]


def test_prune_profile_criterion_uses_variance_and_correlation():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD, units=6)
    m = build_ir(rc, exe)
    # Neutralize readout effect so profile stats dominate selection.
    W_out = np.asarray(m.weights["W_out"]).copy()
    W_out[:, 2:] = 1.0
    m.weights["W_out"] = W_out
    m.metadata["profile_stats"] = {
        "node_variance": [0.9, 0.1, 0.2, 0.8, 0.05, 1.0],
        "correlation_with_other_nodes": [0.1, 0.95, 0.8, 0.2, 0.9, 0.1],
    }

    m2 = PruneInactiveNodes(
        keep_ratio=0.5,
        criterion="low_variance_or_high_corr",
    )(m)
    assert m2.N == 3
    # Expected strongest nodes: indices 0, 3, 5
    assert m2.metadata.get("kept_indices") == [0, 3, 5]
    assert m2.metadata.get("prune_criterion") == "low_variance_or_high_corr"


def test_prune_profile_criterion_fallbacks_without_stats():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD, units=12)
    m = build_ir(rc, exe)
    m2 = PruneInactiveNodes(
        keep_ratio=0.5,
        criterion="low_variance_or_high_corr",
    )(m)
    warns = m2.metadata.get("prune_warnings", [])
    assert warns
    assert "falling back to readout_norm" in warns[0]


def test_prune_profile_criterion_score_weights_affect_selection():
    rc, exe, _ = _build(topology=Topology.ESN_STANDARD, units=6)
    m = build_ir(rc, exe)
    # Make readout term prefer node 1 strongly.
    W_out = np.asarray(m.weights["W_out"]).copy()
    W_out[:, 2:] = 0.0
    W_out[:, 2 + 1] = 10.0
    m.weights["W_out"] = W_out
    m.metadata["profile_stats"] = {
        "node_variance": [0.0, 0.1, 1.0, 0.9, 0.8, 0.7],
        "correlation_with_other_nodes": [0.0, 0.9, 0.1, 0.2, 0.1, 0.2],
    }

    a = PruneInactiveNodes(
        keep_ratio=0.5,
        criterion="low_variance_or_high_corr",
        w_readout=1.0,
        w_variance=0.0,
        w_corr=0.0,
    )(m)
    b = PruneInactiveNodes(
        keep_ratio=0.5,
        criterion="low_variance_or_high_corr",
        w_readout=0.0,
        w_variance=1.0,
        w_corr=1.0,
    )(m)

    assert a.metadata.get("kept_indices") != b.metadata.get("kept_indices")
    assert a.metadata.get("prune_score_weights") == {
        "readout": 1.0,
        "variance": 0.0,
        "corr": 0.0,
    }


def test_profile_reservoir_populates_stats():
    rc, exe, Xs = _build(topology=Topology.ESN_STANDARD, units=20)
    H = exe.collect_states(Xs)
    m = build_ir(rc, exe)
    m2 = ProfileReservoir(H, drop_prefix=3)(m)
    ps = m2.metadata.get("profile_stats")
    assert isinstance(ps, dict)
    for k in (
        "node_variance",
        "mean_activation",
        "correlation_with_other_nodes",
        "n_profile_steps",
    ):
        assert k in ps
    assert len(ps["node_variance"]) == m.N
    assert len(ps["correlation_with_other_nodes"]) == m.N
    assert ps["n_profile_steps"] == max(1, Xs.shape[0] - 3)


def test_profile_then_prune_profile_criterion_no_fallback_warning():
    rc, exe, Xs = _build(topology=Topology.ESN_STANDARD, units=24)
    H = exe.collect_states(Xs)
    m = build_ir(rc, exe)
    m = ProfileReservoir(H)(m)
    m2 = PruneInactiveNodes(
        keep_ratio=0.5,
        criterion="low_variance_or_high_corr",
    )(m)
    warns = m2.metadata.get("prune_warnings", [])
    assert not any("falling back to readout_norm" in w for w in warns)
    assert m2.N == 12


def _parity(passes, topology=Topology.ESN_STANDARD, units=60):
    rc, exe, sample = _build(topology=topology, units=units)
    Y_ref = exe.predict(sample)
    Y_jit = compile_rc(rc, exe, passes=passes).predict(sample)
    diff = float(np.max(np.abs(Y_ref - Y_jit)))
    assert diff < 1e-10, f"parity violated by {diff} with passes={passes}"


def test_parity_default_passes():
    _parity(passes=None)


def test_parity_no_passes():
    _parity(passes=[])


def test_parity_fuse_only():
    _parity(passes=[FuseStepReadout()])


def test_parity_structural_and_fuse():
    _parity(
        passes=[StructuralSpecialize(), FuseStepReadout()],
        topology=Topology.SCR,
    )


def test_parity_with_time_unroll():
    for K in (2, 4, 8):
        _parity(
            passes=[StructuralSpecialize(), FuseStepReadout(), TimeUnroll(K=K)]
        )


def test_parity_unroll_with_dlr_dlrb():
    for topo in (Topology.DLR, Topology.DLRB):
        _parity(
            passes=[
                StructuralSpecialize(),
                FuseStepReadout(),
                TimeUnroll(K=3),
            ],
            topology=topo,
        )


def test_parity_unroll_non_divisible():
    # T=30 not divisible by K=4 — tail loop path must work.
    rc, exe, _ = _build()
    rng = np.random.default_rng(1)
    X = rng.standard_normal((37, 1)) * 0.3 + 0.5
    Y_ref = exe.predict(X)
    jit = compile_rc(
        rc,
        exe,
        passes=[StructuralSpecialize(), FuseStepReadout(), TimeUnroll(K=4)],
    )
    Y_jit = jit.predict(X)
    diff = float(np.max(np.abs(Y_ref - Y_jit)))
    assert diff < 1e-10, f"non-divisible unroll parity diff = {diff}"


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
