"""Shared ridge-regression helpers for the QAT search paths.

Both the symmetric (`rclite.quant.search`) and affine
(`rclite.quant.affine.search`) refit loops build the same feature matrix
and solve the same regularized least-squares problem. Keeping that logic in
one place means a fix to the washout / feature-augmentation contract applies
to both quantization schemes.
"""

from __future__ import annotations

import numpy as np

from rclite.core.composite import ReservoirComputer


def augment_phi(
    rc: ReservoirComputer, X: np.ndarray, H: np.ndarray
) -> np.ndarray:
    """Build phi = [1?] ++ [u?] ++ h for a (T, K) input and (T, N) state."""
    T = H.shape[0]
    parts = []
    if rc.readout.include_bias:
        parts.append(np.ones((T, 1)))
    if rc.readout.include_input:
        parts.append(X)
    parts.append(H)
    return np.concatenate(parts, axis=1)


def fit_ridge(
    phi: np.ndarray, Y: np.ndarray, ridge_lambda: float, washout: int
) -> np.ndarray:
    """Ridge regression on (T, F) features, (T, M) targets. Returns (M, F)."""
    phi_w = phi[washout:]
    Y_w = Y[washout:]
    F = phi_w.shape[1]
    A = phi_w.T @ phi_w + ridge_lambda * np.eye(F)
    B = phi_w.T @ Y_w
    return np.linalg.solve(A, B).T
