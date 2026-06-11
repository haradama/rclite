"""Shared task definition for the TFLM-vs-rclite benchmark.

Pure NumPy so it imports in *both* virtualenvs (the isolated TensorFlow env
and the rclite env). Both frameworks must see byte-identical data and predict
the **same** held-out targets, so accuracy is compared on the same ground.

Task: Mackey-Glass one-step-ahead prediction.
  * target at step t is  series[t + 1]  (predict the next value)
  * MLP  input for target t: the window series[t-W+1 .. t]   (no recurrence)
  * RC   input for target t: series[t] fed sequentially      (recurrent memory)
Both predict the same target indices in the test region.
"""

from __future__ import annotations
import numpy as np

# ---- task constants (identical for both frameworks) --------------------------
MG_N = 3000  # series length after the init transient is dropped
MG_TAU = 17
WINDOW = 16  # MLP look-back window
TRAIN_END = 2000  # targets [.. TRAIN_END-1] train, [TRAIN_END ..] test
RC_WASHOUT = 200  # reservoir settle-in before RC training targets count


def mackey_glass(
    n: int = MG_N,
    tau: int = MG_TAU,
    beta: float = 0.2,
    gamma: float = 0.1,
    n_init: int = 500,
) -> np.ndarray:
    """Deterministic Mackey-Glass series (matches examples/forecasting/mackey_glass_esn)."""
    rng = np.random.default_rng(0)
    L = n + n_init
    x = np.zeros(L)
    x[: tau + 1] = 1.2 + 0.05 * rng.standard_normal(tau + 1)
    for t in range(tau + 1, L):
        x_tau = x[t - tau]
        x[t] = x[t - 1] + beta * x_tau / (1.0 + x_tau**10) - gamma * x[t - 1]
    return x[n_init:]


def series() -> np.ndarray:
    return mackey_glass()


def target_indices():
    """(train_t, test_t): arrays of step t for which we predict series[t+1].

    Both regions require t >= WINDOW-1 (MLP) and the test region starts well
    past RC_WASHOUT, so RC and MLP predict exactly the same target set.
    """
    s = series()
    L = len(s)
    start = max(WINDOW - 1, RC_WASHOUT)
    train_t = np.arange(start, TRAIN_END)
    test_t = np.arange(TRAIN_END, L - 1)
    return train_t, test_t


def windowed(s: np.ndarray, ts: np.ndarray):
    """For each t in ts return (window series[t-W+1..t], target series[t+1])."""
    X = np.stack([s[t - WINDOW + 1 : t + 1] for t in ts]).astype(np.float32)
    y = s[ts + 1].astype(np.float32)
    return X, y


def nrmse(pred: np.ndarray, true: np.ndarray) -> float:
    """Normalized RMSE (by the std of the truth) — scale-free, %."""
    pred = np.asarray(pred, dtype=np.float64).ravel()
    true = np.asarray(true, dtype=np.float64).ravel()
    rmse = np.sqrt(np.mean((pred - true) ** 2))
    return float(rmse / (np.std(true) + 1e-12))
