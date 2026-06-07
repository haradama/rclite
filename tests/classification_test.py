"""Tests for classification support in the reference runtime.

Covers per-step classification (Task.CLASSIFICATION, Aggregation.NONE) and
sequence-to-label classification (Aggregation.MEAN / LAST), plus the
one-hot / softmax / argmax plumbing and ReadoutNode validation.
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
from rclite.runtime.reference import _softmax, _one_hot

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    except Exception as e:
        raise AssertionError(
            f"Expected {exc_type.__name__}, got {type(e).__name__}: {e}"
        )
    raise AssertionError(f"Expected {exc_type.__name__}, no exception raised")


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
# helpers


def test_softmax_rows_sum_to_one():
    Z = np.array([[1.0, 2.0, 3.0], [-5.0, 0.0, 5.0]])
    P = _softmax(Z)
    assert np.allclose(P.sum(axis=1), 1.0)
    assert np.all(P > 0.0)
    # argmax preserved
    assert np.array_equal(np.argmax(P, axis=1), np.argmax(Z, axis=1))


def test_softmax_is_shift_invariant():
    Z = np.array([[1.0, 2.0, 3.0]])
    assert np.allclose(_softmax(Z), _softmax(Z + 100.0))


def test_one_hot_arbitrary_labels():
    classes = np.array([2, 5, 9])
    Y = _one_hot(np.array([5, 2, 9, 5]), classes)
    expected = np.array(
        [
            [0, 1, 0],
            [1, 0, 0],
            [0, 0, 1],
            [0, 1, 0],
        ],
        dtype=float,
    )
    assert np.array_equal(Y, expected)


# ---------------------------------------------------------------------------
# validation


def test_classification_requires_two_units():
    expect_raises(
        ValueError,
        ReadoutNode,
        units=1,
        task=Task.CLASSIFICATION,
        name="ro",
    )


def test_classification_rejects_online_trainer():
    expect_raises(
        ValueError,
        ReadoutNode,
        units=3,
        task=Task.CLASSIFICATION,
        trainer=Trainer.RLS,
        name="ro",
    )


# ---------------------------------------------------------------------------
# per-step classification


def _per_step_dataset(n=1500, seed=0):
    """Label = sign of a smoothed random walk; two linearly separable classes
    in reservoir-state space."""
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + 0.1 * rng.standard_normal()
    X = u[:, None]
    y = (u > 0).astype(int)
    return X, y


def test_per_step_classification_accuracy():
    X, y = _per_step_dataset()
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(seed=3),
        readout=ReadoutNode(
            units=2,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-4,
            washout=100,
            include_input=True,
            task=Task.CLASSIFICATION,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    n_tr = 1000
    exe.fit(X[:n_tr], y[:n_tr])

    assert exe.classes_ is not None and list(exe.classes_) == [0, 1]

    proba = exe.predict_proba(X[n_tr:])
    assert proba.shape == (X.shape[0] - n_tr, 2)
    assert np.allclose(proba.sum(axis=1), 1.0)

    pred = exe.predict_classes(X[n_tr:])
    acc = float(np.mean(pred == y[n_tr:]))
    assert acc > 0.9, f"per-step accuracy too low: {acc:.3f}"


def test_predict_proba_argmax_matches_predict_classes():
    X, y = _per_step_dataset(seed=7)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(seed=5),
        readout=ReadoutNode(
            units=2,
            activation=Activation.IDENTITY,
            task=Task.CLASSIFICATION,
            regularization=1e-4,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X, y)
    proba = exe.predict_proba(X)
    cls = exe.predict_classes(X)
    assert np.array_equal(exe.classes_[np.argmax(proba, axis=1)], cls)


def test_predict_proba_requires_classification_task():
    X, y = _per_step_dataset(seed=1)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(),
        readout=ReadoutNode(
            units=1, activation=Activation.IDENTITY, name="ro"
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X, X)  # regression
    expect_raises(ValueError, exe.predict_proba, X)


# ---------------------------------------------------------------------------
# sequence-to-label classification


def _make_waveform(kind, length, rng):
    """Three trend classes a leaky reservoir separates by temporal integration:
    rising ramp / falling ramp / triangle (up then down)."""
    t = np.linspace(0.0, 1.0, length)
    if kind == 0:  # rising ramp
        s = -1.0 + 2.0 * t
    elif kind == 1:  # falling ramp
        s = 1.0 - 2.0 * t
    else:  # triangle: rise then fall
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    s = s + 0.05 * rng.standard_normal(length)
    return s[:, None]


def _sequence_dataset(n_per_class=40, length=60, seed=0):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for kind in (0, 1, 2):
        for _ in range(n_per_class):
            seqs.append(_make_waveform(kind, length, rng))
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in idx]
    labels = np.array(labels)[idx]
    return seqs, labels


def _run_sequence_classification(aggregation):
    seqs, labels = _sequence_dataset(seed=11)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(units=120, seed=9),
        readout=ReadoutNode(
            units=3,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=10,
            task=Task.CLASSIFICATION,
            aggregation=aggregation,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    n_tr = 90
    exe.fit_sequences(seqs[:n_tr], labels[:n_tr])

    proba = exe.predict_proba_sequences(seqs[n_tr:])
    assert proba.shape == (len(seqs) - n_tr, 3)
    assert np.allclose(proba.sum(axis=1), 1.0)

    pred = exe.predict_sequences(seqs[n_tr:])
    assert pred.shape == (len(seqs) - n_tr,)
    acc = float(np.mean(pred == labels[n_tr:]))
    return acc


def test_sequence_classification_mean():
    acc = _run_sequence_classification(Aggregation.MEAN)
    assert acc > 0.9, f"MEAN sequence accuracy too low: {acc:.3f}"


def test_sequence_classification_last():
    acc = _run_sequence_classification(Aggregation.LAST)
    assert acc > 0.8, f"LAST sequence accuracy too low: {acc:.3f}"


def test_fit_sequences_requires_aggregation():
    seqs, labels = _sequence_dataset(n_per_class=5, length=20, seed=1)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(),
        readout=ReadoutNode(
            units=3,
            activation=Activation.IDENTITY,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.NONE,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    expect_raises(ValueError, exe.fit_sequences, seqs, labels)


def test_sequence_regression_returns_scores():
    """Aggregation also works for sequence-to-scalar regression."""
    rng = np.random.default_rng(2)
    seqs = [rng.standard_normal((30, 1)) for _ in range(20)]
    targets = np.array([float(s.mean()) for s in seqs])
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=_reservoir(seed=4),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-4,
            washout=5,
            task=Task.REGRESSION,
            aggregation=Aggregation.MEAN,
            name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(seqs, targets)
    out = exe.predict_sequences(seqs)
    assert out.shape == (20, 1)


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
