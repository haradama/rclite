"""Numpy-dependent tests for the runtime + verification modules."""

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
    WellPosedReservoir,
    ConstraintViolation,
    Activation,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import (
    RCExecutor,
)
from rclite.verification import (
    InputDrivenESPCheck,
    maximum_lyapunov_exponent,
    reservoir_singular_value,
)


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


def _sample_series(n=1500):
    rng = np.random.default_rng(0)
    x = np.zeros(n + 100)
    for t in range(18, n + 100):
        x[t] = (
            x[t - 1] + 0.2 * x[t - 17] / (1 + x[t - 17] ** 10) - 0.1 * x[t - 1]
        )
        if t < 30:
            x[t] = 1.2 + 0.05 * rng.standard_normal()
    return x[100:]


def _build(sr: float, leak: float = 0.3, seed: int = 42, units: int = 200):
    return ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=1.0,
            input_offset=0.9,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            spectral_radius=sr,
            leak_rate=leak,
            density=0.05,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )


def test_mle_negative_for_stable_reservoir():
    rc = _build(sr=0.9)
    exe = RCExecutor(rc)
    series = _sample_series()
    mle = maximum_lyapunov_exponent(exe, series[:, None], warmup=200)
    assert mle < 0, f"Expected MLE < 0 for ρ=0.9, got {mle}"


def test_mle_positive_for_unstable_reservoir():
    rc = _build(sr=2.5)
    exe = RCExecutor(rc)
    series = _sample_series()
    mle = maximum_lyapunov_exponent(exe, series[:, None], warmup=200)
    assert mle > 0, f"Expected MLE > 0 for ρ=2.5, got {mle}"


def test_yildiz_check_accepts_sr_above_one_when_contractive():
    rc = _build(sr=1.1, leak=0.5)
    exe = RCExecutor(rc)
    series = _sample_series()
    req = WellPosedReservoir(
        rc.reservoir,
        empirical_check=InputDrivenESPCheck(
            executor=exe,
            sample_input=series[:, None],
        ),
    )
    assert req.satisfied(), (
        f"Should pass for ρ=1.1 with contractive trajectory, violations: {req.violations()}"
    )
    assert any("conservative structural" in w for w in req.warnings())


def test_yildiz_check_rejects_clearly_unstable():
    rc = _build(sr=2.5)
    exe = RCExecutor(rc)
    series = _sample_series()
    req = WellPosedReservoir(
        rc.reservoir,
        empirical_check=InputDrivenESPCheck(
            executor=exe,
            sample_input=series[:, None],
        ),
    )
    assert not req.satisfied()
    expect_raises(ConstraintViolation, req.check)


def test_mle_rejects_short_trajectory():
    rc = _build(sr=0.9)
    exe = RCExecutor(rc)
    short = np.zeros((50, 1))
    expect_raises(
        ValueError, maximum_lyapunov_exponent, exe, short, warmup=200
    )


def test_mle_rejects_non_tanh_activation():
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=50,
            activation=Activation.RELU,
            spectral_radius=0.9,
            density=0.1,
            name="res",
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    expect_raises(
        NotImplementedError,
        maximum_lyapunov_exponent,
        exe,
        _sample_series()[:, None],
    )


def test_singular_value_bound_matches_numpy():
    rc = _build(sr=0.9)
    exe = RCExecutor(rc)
    sv = reservoir_singular_value(exe)
    expected = float(np.linalg.svd(exe.W_res, compute_uv=False)[0])
    assert abs(sv - expected) < 1e-10


def test_fit_predict_shapes():
    rc = _build(sr=0.9)
    exe = RCExecutor(rc)
    series = _sample_series()
    X = series[:-1, None]
    Y = series[1:, None]
    exe.fit(X, Y)
    Yhat = exe.predict(X)
    assert Yhat.shape == Y.shape
    one_step_rmse = float(np.sqrt(np.mean((Yhat - Y) ** 2)))
    assert one_step_rmse < 0.05, f"Expected small RMSE, got {one_step_rmse}"


def test_dlr_matrix_structure():
    N = 8
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=N, topology=Topology.DLR, chain_weight=0.7, name="res"
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    W = exe.W_res
    # Sub-diagonal only
    expected = np.zeros((N, N))
    for i in range(1, N):
        expected[i, i - 1] = 0.7
    assert np.allclose(W, expected)
    # Spectral radius is zero (nilpotent)
    assert max(abs(np.linalg.eigvals(W))) < 1e-9


def test_scr_matrix_structure():
    N = 8
    r = 0.9
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=N, topology=Topology.SCR, chain_weight=r, name="res"
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    W = exe.W_res
    # Cyclic: w[i, (i-1) mod N] = r
    expected = np.zeros((N, N))
    for i in range(N):
        expected[i, (i - 1) % N] = r
    assert np.allclose(W, expected)
    # Spectral radius equals |chain_weight|
    assert abs(max(abs(np.linalg.eigvals(W))) - r) < 1e-9


def test_dlrb_matrix_structure():
    N = 6
    r, b = 0.6, 0.1
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=N,
            topology=Topology.DLRB,
            chain_weight=r,
            chain_feedback=b,
            name="res",
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    W = exe.W_res
    expected = np.zeros((N, N))
    for i in range(1, N):
        expected[i, i - 1] = r
    for i in range(N - 1):
        expected[i, i + 1] = b
    assert np.allclose(W, expected)


def test_structured_esn_end_to_end():
    series = _sample_series()
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            input_distribution=Distribution.BERNOULLI,
            input_scaling=0.5,
            input_offset=0.9,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=200,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            activation=Activation.TANH,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X, Y = series[:-1, None], series[1:, None]
    exe.fit(X, Y)
    yhat = exe.predict(X)
    rmse = float(np.sqrt(np.mean((yhat - Y) ** 2)))
    assert rmse < 0.05, f"SCR ESN should solve Mackey-Glass; RMSE={rmse}"


def test_bernoulli_input_weights_are_plus_minus_one():
    rc = ReservoirComputer(
        input=InputNode(
            units=3, input_distribution=Distribution.BERNOULLI, name="in"
        ),
        reservoir=ReservoirNode(
            units=50, topology=Topology.SCR, chain_weight=0.7, name="res"
        ),
        readout=ReadoutNode(units=1, name="out"),
    )
    exe = RCExecutor(rc)
    assert set(np.unique(exe.W_in)).issubset({-1.0, 1.0})


def _make_online_esn(trainer: Trainer, units: int = 200):
    return ReservoirComputer(
        input=InputNode(
            units=1, input_scaling=1.0, input_offset=0.9, name="in"
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.1,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=trainer,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            learning_rate=1e-2,
            forgetting_factor=0.999,
            init_variance=1e-2,
            name="out",
        ),
    )


def test_rls_converges_close_to_batch_ridge():
    series = _sample_series(n=2000)
    X, Y = series[:-1, None], series[1:, None]
    # Batch ridge baseline
    rc_b = _make_online_esn(Trainer.RIDGE)
    rc_b.readout.regularization = (
        1e-2  # match RLS init_variance regularization
    )
    exe_b = RCExecutor(rc_b)
    exe_b.fit(X, Y)
    yhat_b = exe_b.predict(X[1000:])
    rmse_b = float(np.sqrt(np.mean((yhat_b - Y[1000:]) ** 2)))
    # Online RLS
    rc_o = _make_online_esn(Trainer.RLS)
    exe_o = RCExecutor(rc_o)
    exe_o.online_fit(X, Y, warmup_steps=200)
    yhat_o = exe_o.predict(X[1000:])
    rmse_o = float(np.sqrt(np.mean((yhat_o - Y[1000:]) ** 2)))
    # RLS should land within ~3x of batch
    assert rmse_o < max(3 * rmse_b, 0.01), (
        f"RLS RMSE {rmse_o} should be close to ridge {rmse_b}"
    )


def test_lms_makes_progress():
    series = _sample_series(n=3000)
    X, Y = series[:-1, None], series[1:, None]
    rc = _make_online_esn(Trainer.LMS)
    rc.readout.learning_rate = 5e-3
    exe = RCExecutor(rc)
    yhat_online = exe.online_fit(X, Y, warmup_steps=200)
    # LMS predictions should improve over training
    early = float(np.mean((yhat_online[200:700] - Y[200:700]) ** 2))
    late = float(np.mean((yhat_online[-500:] - Y[-500:]) ** 2))
    assert late < early, f"LMS should reduce error: early={early}, late={late}"


def test_force_requires_feedback():
    rc = _make_online_esn(Trainer.FORCE)
    exe = RCExecutor(rc)
    expect_raises(ValueError, exe.make_online_trainer)


def test_force_with_feedback_runs():
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.9, name="in"),
        reservoir=ReservoirNode(
            units=100,
            activation=Activation.TANH,
            spectral_radius=1.2,
            leak_rate=0.5,
            density=0.1,
            has_feedback=True,
            seed=7,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.FORCE,
            washout=100,
            include_bias=True,
            include_input=False,
            forgetting_factor=1.0,
            init_variance=1.0,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    series = _sample_series(n=1500)
    X, Y = series[:-1, None], series[1:, None]
    yhat = exe.online_fit(X, Y, warmup_steps=100)
    assert np.all(np.isfinite(yhat))
    early = float(np.mean((yhat[100:300] - Y[100:300]) ** 2))
    late = float(np.mean((yhat[-300:] - Y[-300:]) ** 2))
    assert late < early, (
        f"FORCE should reduce error: early={early}, late={late}"
    )


def test_ridge_fit_rejects_online_trainer():
    rc = _make_online_esn(Trainer.RLS)
    exe = RCExecutor(rc)
    X = np.zeros((300, 1))
    Y = np.zeros((300, 1))
    expect_raises(ValueError, exe.fit, X, Y)


def test_free_run_shape_and_finite():
    rc = _build(sr=0.9)
    exe = RCExecutor(rc)
    series = _sample_series()
    exe.fit(series[:-1, None], series[1:, None])
    seed = series[-200:, None]
    fr = exe.free_run(seed, n_steps=100)
    assert fr.shape == (100, 1)
    assert np.all(np.isfinite(fr))


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
