"""Input-driven verification of the echo state property.

Implements an empirical ESP check based on the sign of the maximum local
Lyapunov exponent (MLE) computed along an input-driven trajectory. This
condition can accept reservoirs with spectral_radius >= 1 that are
nevertheless contractive in their actual operating regime, which the
conservative `spectral_radius < 1` bound would reject.

Reference:
    Yildiz, I. B., Jaeger, H., & Kiebel, S. J. (2012).
    Re-visiting the echo state property. Neural Networks, 35, 1-9.
    https://doi.org/10.1016/j.neunet.2012.07.005
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "rc_idl.verification requires numpy. Install with `pip install numpy`."
    ) from e

from rclite.core.profile import Activation
from rclite.runtime.reference import RCExecutor


def maximum_lyapunov_exponent(
    executor: RCExecutor,
    X: "np.ndarray",
    *,
    warmup: int = 200,
    seed: int = 0,
) -> float:
    """Estimate the largest local Lyapunov exponent along the input trajectory.

    Single-vector estimator: propagate a unit perturbation through the
    Jacobian product, re-normalize each step, and accumulate the log-stretch.
    Negative values indicate contractive (input-driven) dynamics.

    Currently supports tanh activation only.
    """
    rc = executor.rc
    if rc.reservoir.activation != Activation.TANH:
        raise NotImplementedError(
            f"maximum_lyapunov_exponent currently only supports tanh; "
            f"got {rc.reservoir.activation.name}"
        )
    if X.ndim == 1:
        X = X[:, None]

    W = executor.W_res
    W_in = executor.W_in
    N = rc.reservoir.units
    bias = rc.reservoir.bias
    leak = rc.reservoir.leak_rate

    Xp = executor._preprocess(X)
    T = Xp.shape[0]
    if T <= warmup:
        raise ValueError(
            f"Trajectory too short for MLE estimate: T={T} <= warmup={warmup}"
        )

    rng = np.random.default_rng(seed)
    h = np.zeros(N)
    v = rng.standard_normal(N)
    v /= np.linalg.norm(v)

    log_stretch_sum = 0.0
    counted = 0

    for t in range(T):
        z = W @ h + W_in @ Xp[t] + bias
        h_new = (1.0 - leak) * h + leak * np.tanh(z)

        D = 1.0 - np.tanh(z) ** 2
        # Jacobian-vector product: J v = (1-leak) v + leak (D ⊙ (W v))
        Jv = (1.0 - leak) * v + leak * (D * (W @ v))

        norm = float(np.linalg.norm(Jv))
        if norm > 0.0:
            if t >= warmup:
                log_stretch_sum += float(np.log(norm))
                counted += 1
            v = Jv / norm
        h = h_new

    if counted == 0:
        raise ValueError("No post-warmup samples accumulated for MLE estimate")
    return log_stretch_sum / counted


def reservoir_singular_value(executor: RCExecutor) -> float:
    """σ_max(W_res). Sufficient (worst-case) ESP if < 1, regardless of input."""
    return float(np.linalg.svd(executor.W_res, compute_uv=False)[0])


@dataclass
class InputDrivenESPCheck:
    """Yildiz et al. (2012) empirical ESP verifier.

    Computes the maximum local Lyapunov exponent along `sample_input` and
    declares ESP satisfied when it is below `threshold` (typically 0).
    """
    executor: RCExecutor
    sample_input: "np.ndarray"
    threshold: float = 0.0
    warmup: int = 200
    seed: int = 0

    last_mle: Optional[float] = field(default=None, init=False, repr=False)

    def violations(self) -> List[str]:
        mle = maximum_lyapunov_exponent(
            self.executor, self.sample_input,
            warmup=self.warmup, seed=self.seed,
        )
        self.last_mle = mle
        if mle >= self.threshold:
            return [
                f"InputDrivenESP (Yildiz et al. 2012) violated: "
                f"max local Lyapunov exponent = {mle:.4f} >= {self.threshold} — "
                f"reservoir is not contractive along the supplied input trajectory"
            ]
        return []

    def satisfied(self) -> bool:
        return not self.violations()
