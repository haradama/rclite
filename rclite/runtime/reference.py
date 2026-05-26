"""Reference executor for the RC IDL (optional, NumPy-based).

Not part of the IDL itself. Compiles a `ReservoirComputer` description
into runnable weights and provides fit / predict / free_run / online_fit
against ndarray time series.

Requires numpy. Importing this module fails gracefully if numpy is missing.
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

try:
    import numpy as np
except ImportError as e:
    raise ImportError(
        "rc_idl.runtime requires numpy. Install with `pip install numpy`."
    ) from e

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Activation, Distribution, Topology, Trainer


_ACTIVATIONS = {
    Activation.TANH: np.tanh,
    Activation.SIGMOID: lambda x: 1.0 / (1.0 + np.exp(-x)),
    Activation.RELU: lambda x: np.maximum(0.0, x),
    Activation.IDENTITY: lambda x: x,
}


def _activation_fn(a: Activation):
    try:
        return _ACTIVATIONS[a]
    except KeyError:
        raise NotImplementedError(
            f"Activation {a.name} is not implemented in the reference runtime"
        )


def _sample(rng, shape, distribution: Distribution):
    if distribution == Distribution.NORMAL:
        return rng.standard_normal(shape)
    if distribution == Distribution.UNIFORM:
        return rng.uniform(-1.0, 1.0, size=shape)
    if distribution == Distribution.BERNOULLI:
        return rng.choice([-1.0, 1.0], size=shape).astype(float)
    raise NotImplementedError(f"Distribution {distribution.name} not implemented")


@dataclass
class RCExecutor:
    """Materializes a ReservoirComputer IDL description into runnable weights."""
    rc: ReservoirComputer

    W_in: "np.ndarray" = field(init=False)
    W_res: "np.ndarray" = field(init=False)
    W_fb: Optional["np.ndarray"] = field(init=False, default=None)
    W_out: Optional["np.ndarray"] = field(init=False, default=None)

    def __post_init__(self):
        rng = np.random.default_rng(self.rc.reservoir.seed)
        self._build_input_weights(rng)
        self._build_reservoir_weights(rng)
        if self.rc.W_fb is not None:
            self._build_feedback_weights(rng)

    def _build_input_weights(self, rng):
        N = self.rc.reservoir.units
        K = self.rc.input.units
        self.W_in = _sample(rng, (N, K), self.rc.W_in.spec.distribution)

    def _build_reservoir_weights(self, rng):
        topo = self.rc.reservoir.topology
        if topo == Topology.DLR:
            self.W_res = self._dlr_matrix()
        elif topo == Topology.DLRB:
            self.W_res = self._dlrb_matrix()
        elif topo == Topology.SCR:
            self.W_res = self._scr_matrix()
        else:
            self.W_res = self._random_matrix(rng)

    def _random_matrix(self, rng):
        N = self.rc.reservoir.units
        W = _sample(rng, (N, N), self.rc.W_res.spec.distribution)
        density = self.rc.reservoir.density
        if density < 1.0:
            mask = rng.random((N, N)) < density
            W = W * mask
        eigs = np.linalg.eigvals(W)
        sr = float(np.max(np.abs(eigs)))
        if sr > 0:
            W *= self.rc.reservoir.spectral_radius / sr
        return W

    def _dlr_matrix(self) -> "np.ndarray":
        """Rodan-Tino DLR: w[i, i-1] = chain_weight, zero elsewhere."""
        N = self.rc.reservoir.units
        r = self.rc.reservoir.chain_weight
        W = np.zeros((N, N))
        idx = np.arange(1, N)
        W[idx, idx - 1] = r
        return W

    def _dlrb_matrix(self) -> "np.ndarray":
        """Rodan-Tino DLRB: DLR plus w[i, i+1] = chain_feedback."""
        N = self.rc.reservoir.units
        r = self.rc.reservoir.chain_weight
        b = self.rc.reservoir.chain_feedback
        W = np.zeros((N, N))
        idx = np.arange(1, N)
        W[idx, idx - 1] = r
        idx2 = np.arange(0, N - 1)
        W[idx2, idx2 + 1] = b
        return W

    def _scr_matrix(self) -> "np.ndarray":
        """Rodan-Tino SCR: cyclic, w[i, (i-1) mod N] = chain_weight."""
        N = self.rc.reservoir.units
        r = self.rc.reservoir.chain_weight
        W = np.zeros((N, N))
        i = np.arange(N)
        W[i, (i - 1) % N] = r
        return W

    def _build_feedback_weights(self, rng):
        N = self.rc.reservoir.units
        M = self.rc.readout.units
        self.W_fb = _sample(rng, (N, M), self.rc.W_fb.spec.distribution)

    def _preprocess(self, X: "np.ndarray") -> "np.ndarray":
        offset = self.rc.input.input_offset
        scale = self.rc.input.input_scaling
        return (X - offset) * scale

    def _augment(self, X_raw: "np.ndarray", H: "np.ndarray") -> "np.ndarray":
        T = H.shape[0]
        parts = []
        if self.rc.readout.include_bias:
            parts.append(np.ones((T, 1)))
        if self.rc.readout.include_input:
            parts.append(X_raw)
        parts.append(H)
        return np.concatenate(parts, axis=1)

    def _augment_one(self, x_raw: "np.ndarray", h: "np.ndarray") -> "np.ndarray":
        parts = []
        if self.rc.readout.include_bias:
            parts.append(np.ones(1))
        if self.rc.readout.include_input:
            parts.append(np.atleast_1d(x_raw))
        parts.append(h)
        return np.concatenate(parts)

    def _feature_dim(self) -> int:
        F = self.rc.reservoir.units
        if self.rc.readout.include_bias:
            F += 1
        if self.rc.readout.include_input:
            F += self.rc.input.units
        return F

    def collect_states(
        self,
        X: "np.ndarray",
        Y_teach: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        T = X.shape[0]
        N = self.rc.reservoir.units
        act = _activation_fn(self.rc.reservoir.activation)
        leak = self.rc.reservoir.leak_rate
        bias = self.rc.reservoir.bias
        use_fb = self.W_fb is not None
        if use_fb and Y_teach is None:
            raise ValueError(
                "Feedback is enabled — collect_states requires Y_teach"
            )

        Xp = self._preprocess(X)
        H = np.zeros((T, N))
        h = np.zeros(N)
        for t in range(T):
            pre = self.W_in @ Xp[t] + self.W_res @ h + bias
            if use_fb:
                y_prev = Y_teach[t - 1] if t > 0 else np.zeros(self.rc.readout.units)
                pre = pre + self.W_fb @ y_prev
            h = (1.0 - leak) * h + leak * act(pre)
            H[t] = h
        return H

    def fit(self, X: "np.ndarray", Y: "np.ndarray") -> "np.ndarray":
        """Batch training (RIDGE / PINV). Use `online_fit` for online trainers."""
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        use_fb = self.W_fb is not None
        H = self.collect_states(X, Y if use_fb else None)
        Phi = self._augment(X, H)
        w = self.rc.readout.washout
        Phi_w, Y_w = Phi[w:], Y[w:]
        trainer = self.rc.readout.trainer
        if trainer == Trainer.RIDGE:
            lam = self.rc.readout.regularization
            A = Phi_w.T @ Phi_w + lam * np.eye(Phi_w.shape[1])
            B = Phi_w.T @ Y_w
            self.W_out = np.linalg.solve(A, B).T
        elif trainer == Trainer.PINV:
            self.W_out = (np.linalg.pinv(Phi_w) @ Y_w).T
        elif trainer in (Trainer.RLS, Trainer.LMS, Trainer.FORCE):
            raise ValueError(
                f"Trainer {trainer.name} is an online trainer; use online_fit()"
            )
        else:
            raise NotImplementedError(
                f"Trainer {trainer.name} is not implemented"
            )
        return self.W_out

    def predict(self, X: "np.ndarray") -> "np.ndarray":
        if self.W_out is None:
            raise RuntimeError("Readout has not been trained — call fit() first")
        if X.ndim == 1:
            X = X[:, None]
        H = self.collect_states(X)
        Phi = self._augment(X, H)
        return Phi @ self.W_out.T

    def free_run(self, X_seed: "np.ndarray", n_steps: int) -> "np.ndarray":
        if self.W_out is None:
            raise RuntimeError("Readout has not been trained — call fit() first")
        if X_seed.ndim == 1:
            X_seed = X_seed[:, None]
        K = self.rc.input.units
        M = self.rc.readout.units
        if K != M:
            raise ValueError(
                f"free_run requires input dim ({K}) == output dim ({M})"
            )

        act = _activation_fn(self.rc.reservoir.activation)
        leak = self.rc.reservoir.leak_rate
        bias = self.rc.reservoir.bias

        Xp_seed = self._preprocess(X_seed)
        h = np.zeros(self.rc.reservoir.units)
        for t in range(Xp_seed.shape[0]):
            pre = self.W_in @ Xp_seed[t] + self.W_res @ h + bias
            h = (1.0 - leak) * h + leak * act(pre)

        last_x = X_seed[-1].copy()
        preds = np.zeros((n_steps, M))
        for t in range(n_steps):
            phi = self._augment_one(last_x, h)
            y_hat = self.W_out @ phi
            preds[t] = y_hat
            xp = self._preprocess(y_hat)
            pre = self.W_in @ xp + self.W_res @ h + bias
            h = (1.0 - leak) * h + leak * act(pre)
            last_x = y_hat
        return preds

    def make_online_trainer(self) -> "OnlineTrainer":
        t = self.rc.readout.trainer
        if t == Trainer.RLS:
            return RLSTrainer(self)
        if t == Trainer.LMS:
            return LMSTrainer(self)
        if t == Trainer.FORCE:
            return FORCETrainer(self)
        raise ValueError(
            f"Trainer {t.name} is not an online trainer; use RLS / LMS / FORCE"
        )

    def online_fit(
        self,
        X: "np.ndarray",
        Y: "np.ndarray",
        *,
        warmup_steps: Optional[int] = None,
        trainer: Optional["OnlineTrainer"] = None,
    ) -> "np.ndarray":
        """Train readout online, one sample at a time. Returns predictions per step."""
        if X.ndim == 1:
            X = X[:, None]
        if Y.ndim == 1:
            Y = Y[:, None]
        if trainer is None:
            trainer = self.make_online_trainer()
        if warmup_steps is None:
            warmup_steps = self.rc.readout.washout
        T = X.shape[0]
        M = self.rc.readout.units
        Y_hat = np.zeros((T, M))
        for t in range(T):
            if t < warmup_steps:
                Y_hat[t] = trainer.step_no_update(X[t])
            else:
                Y_hat[t] = trainer.step(X[t], Y[t])
        return Y_hat


# ----------------------------------------------------------------------------
# Online learning


class OnlineTrainer:
    """Base class for sample-by-sample readout training.

    The reservoir state is owned by the trainer (so multiple online sessions
    can run independently against the same executor). The executor owns W_out
    and the trainer mutates it in place.
    """

    def __init__(self, executor: RCExecutor):
        self.executor = executor
        self.h = np.zeros(executor.rc.reservoir.units)
        if executor.W_out is None:
            M = executor.rc.readout.units
            F = executor._feature_dim()
            executor.W_out = np.zeros((M, F))
        self._init_state()

    def _init_state(self) -> None:
        pass

    def _step_reservoir(self, x: "np.ndarray") -> None:
        rc = self.executor.rc
        act = _activation_fn(rc.reservoir.activation)
        leak = rc.reservoir.leak_rate
        bias = rc.reservoir.bias
        xp = self.executor._preprocess(x)
        z = self.executor.W_in @ xp + self.executor.W_res @ self.h + bias
        self.h = (1.0 - leak) * self.h + leak * act(z)

    def step(self, x: "np.ndarray", y: "np.ndarray") -> "np.ndarray":
        x = np.atleast_1d(np.asarray(x, dtype=float))
        y = np.atleast_1d(np.asarray(y, dtype=float))
        self._step_reservoir(x)
        phi = self.executor._augment_one(x, self.h)
        y_hat = self.executor.W_out @ phi
        self._update(phi, y - y_hat)
        return y_hat

    def step_no_update(self, x: "np.ndarray") -> "np.ndarray":
        x = np.atleast_1d(np.asarray(x, dtype=float))
        self._step_reservoir(x)
        phi = self.executor._augment_one(x, self.h)
        return self.executor.W_out @ phi

    def _update(self, phi: "np.ndarray", error: "np.ndarray") -> None:
        raise NotImplementedError


class RLSTrainer(OnlineTrainer):
    """Recursive Least Squares with optional forgetting factor."""

    def _init_state(self) -> None:
        F = self.executor._feature_dim()
        delta = self.executor.rc.readout.init_variance
        self.P = np.eye(F) / delta

    def _update(self, phi, error):
        lam = self.executor.rc.readout.forgetting_factor
        Pphi = self.P @ phi
        denom = lam + phi @ Pphi
        k = Pphi / denom
        self.executor.W_out += np.outer(error, k)
        self.P = (self.P - np.outer(k, Pphi)) / lam


class LMSTrainer(OnlineTrainer):
    """Least Mean Squares gradient step."""

    def _update(self, phi, error):
        eta = self.executor.rc.readout.learning_rate
        self.executor.W_out += eta * np.outer(error, phi)


class FORCETrainer(RLSTrainer):
    """FORCE learning (Sussillo & Abbott 2009).

    RLS variant where the reservoir receives the *predicted* output as
    feedback (closed-loop), rather than a teacher signal. Requires the
    reservoir to have feedback enabled.
    """

    def _init_state(self) -> None:
        super()._init_state()
        if not self.executor.rc.reservoir.has_feedback:
            raise ValueError("FORCE requires reservoir.has_feedback=True")
        self.last_pred: Optional["np.ndarray"] = None

    def _step_reservoir(self, x):
        rc = self.executor.rc
        act = _activation_fn(rc.reservoir.activation)
        leak = rc.reservoir.leak_rate
        bias = rc.reservoir.bias
        xp = self.executor._preprocess(x)
        if self.last_pred is None:
            y_prev = np.zeros(rc.readout.units)
        else:
            y_prev = self.last_pred
        z = (self.executor.W_in @ xp
             + self.executor.W_res @ self.h
             + self.executor.W_fb @ y_prev
             + bias)
        self.h = (1.0 - leak) * self.h + leak * act(z)

    def step(self, x, y):
        y_hat = super().step(x, y)
        self.last_pred = y_hat
        return y_hat

    def step_no_update(self, x):
        y_hat = super().step_no_update(x)
        self.last_pred = y_hat
        return y_hat
