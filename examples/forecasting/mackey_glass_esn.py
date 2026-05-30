"""Mackey-Glass time-series prediction with an Echo State Network.

Mirrors the SysML v2 example `Examples::MackeyGlassESN`, augmented with:
  - input centering via InputNode.input_offset (train-mean subtraction)
  - readout bias term via ReadoutNode.include_bias
  - direct input pass-through via ReadoutNode.include_input
  - one-step and free-running (autoregressive) evaluation
"""
from __future__ import annotations
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    WellPosedReservoir,
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor


def mackey_glass(n: int = 3000, tau: int = 17, beta: float = 0.2,
                 gamma: float = 0.1, n_init: int = 500) -> np.ndarray:
    rng = np.random.default_rng(0)
    L = n + n_init
    x = np.zeros(L)
    x[:tau + 1] = 1.2 + 0.05 * rng.standard_normal(tau + 1)
    for t in range(tau + 1, L):
        x_tau = x[t - tau]
        x[t] = x[t - 1] + beta * x_tau / (1.0 + x_tau ** 10) - gamma * x[t - 1]
    return x[n_init:]


def build_esn(input_offset: float, input_scaling: float = 1.0) -> ReservoirComputer:
    """SysML2: part esn : ReservoirComputer { ... }"""
    return ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=input_scaling,
            input_offset=input_offset,
            name="input",
        ),
        reservoir=ReservoirNode(
            units=500,
            activation=Activation.TANH,
            spectral_radius=0.95,
            leak_rate=0.3,
            density=0.05,
            topology=Topology.ESN_STANDARD,
            seed=42,
            name="reservoir",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="readout",
        ),
    )


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.sqrt(np.mean((a - b) ** 2)))


def nrmse(a: np.ndarray, b: np.ndarray) -> float:
    return rmse(a, b) / float(np.std(b))


def main() -> None:
    series = mackey_glass()
    X, Y = series[:-1, None], series[1:, None]

    n_train = 2000
    X_tr, Y_tr = X[:n_train], Y[:n_train]
    X_te, Y_te = X[n_train:], Y[n_train:]

    # IDL: centering parameter derived from training data, written into InputNode.
    input_offset = float(X_tr.mean())

    esn = build_esn(input_offset=input_offset, input_scaling=1.0)
    WellPosedReservoir(esn.reservoir).check()
    print(f"[ok] WellPosedReservoir satisfied")
    print(f"     input_offset = {input_offset:.4f} (train mean)")
    print(f"     readout features: bias={esn.readout.include_bias}, "
          f"input={esn.readout.include_input}, state(N={esn.reservoir.units})")

    exe = RCExecutor(esn)
    exe.fit(X_tr, Y_tr)

    Y_hat = exe.predict(X_te)
    one_step_rmse = rmse(Y_hat, Y_te)
    one_step_nrmse = nrmse(Y_hat, Y_te)
    print(f"\n[one-step] RMSE  = {one_step_rmse:.5f}")
    print(f"[one-step] NRMSE = {one_step_nrmse:.5f}")

    seed_len = 200
    horizon = 200
    seed = X[n_train - seed_len: n_train]
    target = series[n_train: n_train + horizon][:, None]
    free = exe.free_run(seed, n_steps=horizon)
    fr_rmse = rmse(free, target)
    fr_nrmse = nrmse(free, target)
    print(f"\n[free-run, horizon={horizon}]")
    print(f"           RMSE  = {fr_rmse:.5f}")
    print(f"           NRMSE = {fr_nrmse:.5f}")


if __name__ == "__main__":
    main()
