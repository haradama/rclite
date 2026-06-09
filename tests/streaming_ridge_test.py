"""Streaming ridge training parity tests.

Validates that `RCExecutor.fit(..., materialize_states=False)` matches the
existing materialized-state ridge path numerically, including teacher-forced
feedback mode.
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
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def _make_model(*, has_feedback: bool = False, seed: int = 7):
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=96,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.15,
            topology=Topology.ESN_STANDARD,
            has_feedback=has_feedback,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=80,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    return rc


def _dataset(T=600, seed=1):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.4
    Y = np.sin(np.arange(T) * 0.05)[:, None]
    return X, Y


def _assert_close(a, b, atol=1e-9, rtol=1e-9, msg=""):
    d = float(np.max(np.abs(a - b)))
    s = float(np.max(np.abs(b))) + 1e-30
    rel = d / s
    if not (d < atol or rel < rtol):
        raise AssertionError(
            f"{msg} max|diff|={d:.3e}, rel={rel:.3e}, atol={atol}, rtol={rtol}"
        )


def test_streaming_ridge_matches_materialized_dense():
    X, Y = _dataset()
    rc = _make_model(has_feedback=False)
    exe_mat = RCExecutor(rc)
    exe_str = RCExecutor(rc)

    Wm = exe_mat.fit(X, Y, materialize_states=True)
    Ws = exe_str.fit(X, Y, materialize_states=False)

    _assert_close(Ws, Wm, atol=2e-6, rtol=1e-8, msg="W_out")
    _assert_close(
        exe_str.predict(X),
        exe_mat.predict(X),
        atol=2e-9,
        rtol=2e-9,
        msg="predict",
    )


def test_streaming_ridge_matches_materialized_feedback():
    X, Y = _dataset(seed=11)
    rc = _make_model(has_feedback=True, seed=13)
    exe_mat = RCExecutor(rc)
    exe_str = RCExecutor(rc)

    Wm = exe_mat.fit(X, Y, materialize_states=True)
    Ws = exe_str.fit(X, Y, materialize_states=False)

    _assert_close(Ws, Wm, atol=1e-8, rtol=1e-8, msg="W_out feedback")


def test_streaming_handles_washout_ge_length():
    X, Y = _dataset(T=60, seed=23)
    rc = _make_model(has_feedback=False, seed=29)
    rc.readout.washout = 1000
    exe = RCExecutor(rc)

    W = exe.fit(X, Y, materialize_states=False)
    assert np.allclose(W, 0.0), (
        "With no post-washout samples W_out should be zero"
    )


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
