"""JIT parity tests for the classification heads (argmax / softmax) and
sequence-to-label aggregation in the float LLVM backend.

Each test trains a classifier in the reference runtime, compiles it with a
classification head, and checks the JIT output matches the runtime
bit-closely (logits / probabilities) or exactly (argmax class id).
"""

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
    Topology,
    Trainer,
    Task,
    Aggregation,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

ATOL = 1e-9


def _reservoir(units=80, seed=1):
    return ReservoirNode(
        units=units,
        activation=Activation.TANH,
        spectral_radius=0.9,
        leak_rate=0.3,
        density=0.1,
        topology=Topology.RANDOM,
        seed=seed,
        name="reservoir",
    )


# ---------------------------------------------------------------------------
# per-step classification


def _per_step_clf(include_input):
    rng = np.random.default_rng(0)
    u = np.zeros(1200)
    for t in range(1, 1200):
        u[t] = 0.9 * u[t - 1] + 0.1 * rng.standard_normal()
    X, y = u[:, None], (u > 0).astype(int)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(seed=3),
        readout=ReadoutNode(
            units=2,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-4,
            washout=100,
            include_input=include_input,
            task=Task.CLASSIFICATION,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:800], y[:800])
    return rc, exe, X[800:]


def test_per_step_logits_parity():
    rc, exe, X = _per_step_clf(include_input=True)
    jit = compile_rc(rc, exe, head="logits").predict(X)
    assert np.allclose(jit, exe.predict(X), atol=ATOL)


def test_per_step_proba_parity():
    rc, exe, X = _per_step_clf(include_input=True)
    jit = compile_rc(rc, exe, head="proba").predict(X)
    ref = exe.predict_proba(X)
    assert jit.shape == ref.shape
    assert np.allclose(jit, ref, atol=ATOL)
    assert np.allclose(jit.sum(axis=1), 1.0, atol=ATOL)


def test_per_step_classify_exact():
    rc, exe, X = _per_step_clf(include_input=False)
    jit = compile_rc(rc, exe, head="classify").predict(X)
    ref = np.argmax(exe.predict(X), axis=1)
    assert jit.dtype == np.int32
    assert jit.shape == (X.shape[0],)
    assert np.array_equal(jit, ref)


# ---------------------------------------------------------------------------
# sequence-to-label classification


def _make_waveform(kind, length, rng):
    t = np.linspace(0.0, 1.0, length)
    if kind == 0:
        s = -1.0 + 2.0 * t
    elif kind == 1:
        s = 1.0 - 2.0 * t
    else:
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    return (s + 0.05 * rng.standard_normal(length))[:, None]


def _sequence_clf(aggregation):
    rng = np.random.default_rng(11)
    seqs, labels = [], []
    for kind in range(3):
        for _ in range(40):
            seqs.append(_make_waveform(kind, 60, rng))
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in idx]
    labels = np.array(labels)[idx]
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(units=120, seed=9),
        readout=ReadoutNode(
            units=3,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=10,
            include_input=False,
            task=Task.CLASSIFICATION,
            aggregation=aggregation,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(seqs[:90], labels[:90])
    return rc, exe, seqs[90:]


def _check_sequence(aggregation):
    rc, exe, test_seqs = _sequence_clf(aggregation)
    jit_logits = compile_rc(rc, exe, head="logits")
    jit_proba = compile_rc(rc, exe, head="proba")
    jit_classify = compile_rc(rc, exe, head="classify")
    for X in test_seqs:
        ref_logits = exe._sequence_features([X]) @ exe.W_out.T  # (1, C)
        jl = jit_logits.predict(X)
        assert jl.shape == (1, 3)
        assert np.allclose(jl, ref_logits, atol=ATOL)

        ref_proba = exe.predict_proba_sequences([X])
        jp = jit_proba.predict(X)
        assert np.allclose(jp, ref_proba, atol=ATOL)
        assert np.allclose(jp.sum(axis=1), 1.0, atol=ATOL)

        jc = jit_classify.predict(X)
        assert jc.dtype == np.int32 and jc.shape == (1,)
        assert jc[0] == int(np.argmax(ref_logits))


def test_sequence_mean_parity():
    _check_sequence(Aggregation.MEAN)


def test_sequence_last_parity():
    _check_sequence(Aggregation.LAST)


def test_sequence_include_input_not_supported():
    """Codegen rejects include_input=True under aggregation (Phase 2 limit)."""
    rng = np.random.default_rng(1)
    seqs = [rng.standard_normal((20, 1)) for _ in range(6)]
    labels = np.array([0, 1, 2, 0, 1, 2])
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(seed=2),
        readout=ReadoutNode(
            units=3,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-2,
            washout=2,
            include_input=True,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.MEAN,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(seqs, labels)
    try:
        compile_rc(rc, exe, head="classify")
    except NotImplementedError:
        return
    raise AssertionError(
        "expected NotImplementedError for include_input + aggregation"
    )


# ---------------------------------------------------------------------------
# header


def test_classify_header_declares_int_output():
    rc, exe, _ = _per_step_clf(include_input=False)
    import tempfile

    jit = compile_rc(rc, exe, head="classify")
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False, mode="w") as f:
        path = f.name
    try:
        jit.emit_header(path)
        text = pathlib.Path(path).read_text()
        assert "int32_t *Y" in text
        assert "RC_NUM_CLASSES 2" in text
        assert "head             = classify" in text
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


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
