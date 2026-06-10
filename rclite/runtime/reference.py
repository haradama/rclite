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
from rclite.core.profile import (
    Activation,
    Aggregation,
    Distribution,
    Task,
    Topology,
    Trainer,
)


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


def _softmax(Z: "np.ndarray") -> "np.ndarray":
    """Numerically stable softmax over the last axis."""
    Z = Z - np.max(Z, axis=-1, keepdims=True)
    E = np.exp(Z)
    return E / np.sum(E, axis=-1, keepdims=True)


def _one_hot(y: "np.ndarray", classes: "np.ndarray") -> "np.ndarray":
    """Encode integer/label vector y (n,) into a one-hot matrix (n, C).

    `classes` is the sorted array of unique labels; column j corresponds to
    classes[j].
    """
    idx = np.searchsorted(classes, y)
    Y = np.zeros((len(y), len(classes)))
    Y[np.arange(len(y)), idx] = 1.0
    return Y


def _sample(rng, shape, distribution: Distribution):
    if distribution == Distribution.NORMAL:
        return rng.standard_normal(shape)
    if distribution == Distribution.UNIFORM:
        return rng.uniform(-1.0, 1.0, size=shape)
    if distribution == Distribution.BERNOULLI:
        return rng.choice([-1.0, 1.0], size=shape).astype(float)
    raise NotImplementedError(
        f"Distribution {distribution.name} not implemented"
    )


@dataclass
class RCExecutor:
    """Materializes a ReservoirComputer IDL description into runnable weights."""

    rc: ReservoirComputer

    W_in: "np.ndarray" = field(init=False)
    W_res: "np.ndarray" = field(init=False)
    W_fb: Optional["np.ndarray"] = field(init=False, default=None)
    W_out: Optional["np.ndarray"] = field(init=False, default=None)
    # Sorted unique labels seen at fit time (classification only); column j of
    # the readout corresponds to classes_[j]. None for regression / untrained.
    classes_: Optional["np.ndarray"] = field(init=False, default=None)

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

    def _reservoir_step(
        self, h: "np.ndarray", xp: "np.ndarray", fb=None
    ) -> "np.ndarray":
        """One leaky-integrator update from a preprocessed input row `xp`.

        Returns ``(1-leak)*h + leak*act(W_in@xp + W_res@h + bias [+ fb])``.
        `fb` is an optional additive pre-activation term (e.g. ``W_fb@y_prev``)
        added after the bias — matching the teacher-forced feedback path in
        `collect_states` / `_fit_ridge_streaming`. (FORCE keeps its own step:
        it folds feedback in before the bias, a different summation order.)
        """
        r = self.rc.reservoir
        act = _activation_fn(r.activation)
        pre = self.W_in @ xp + self.W_res @ h + r.bias
        if fb is not None:
            pre = pre + fb
        return (1.0 - r.leak_rate) * h + r.leak_rate * act(pre)

    def _augment(self, X_raw: "np.ndarray", H: "np.ndarray") -> "np.ndarray":
        T = H.shape[0]
        parts = []
        if self.rc.readout.include_bias:
            parts.append(np.ones((T, 1)))
        if self.rc.readout.include_input:
            parts.append(X_raw)
        parts.append(H)
        return np.concatenate(parts, axis=1)

    def _augment_one(
        self, x_raw: "np.ndarray", h: "np.ndarray"
    ) -> "np.ndarray":
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

    def _solve_ridge(
        self, A: "np.ndarray", B: "np.ndarray", n_outputs: int
    ) -> "np.ndarray":
        if B.shape[0] == 0:
            self.W_out = np.zeros((n_outputs, A.shape[0]))
        else:
            self.W_out = np.linalg.solve(A, B).T
        return self.W_out

    def _fit_ridge_streaming(
        self, X: "np.ndarray", Y: "np.ndarray"
    ) -> "np.ndarray":
        Xp = self._preprocess(X)
        T = X.shape[0]
        N = self.rc.reservoir.units
        M = Y.shape[1]
        F = self._feature_dim()
        washout = max(int(self.rc.readout.washout), 0)
        lam = self.rc.readout.regularization
        use_fb = self.W_fb is not None

        A = lam * np.eye(F)
        B = np.zeros((F, M))
        h = np.zeros(N)
        samples = 0

        for t in range(T):
            fb = None
            if use_fb:
                y_prev = Y[t - 1] if t > 0 else np.zeros(M)
                fb = self.W_fb @ y_prev
            h = self._reservoir_step(h, Xp[t], fb)

            if t < washout:
                continue

            phi = self._augment_one(X[t], h)
            A += np.outer(phi, phi)
            B += np.outer(phi, Y[t])
            samples += 1

        if samples == 0:
            self.W_out = np.zeros((M, F))
            return self.W_out
        return self._solve_ridge(A, B, M)

    def collect_states(
        self,
        X: "np.ndarray",
        Y_teach: Optional["np.ndarray"] = None,
    ) -> "np.ndarray":
        T = X.shape[0]
        N = self.rc.reservoir.units
        use_fb = self.W_fb is not None
        if use_fb and Y_teach is None:
            raise ValueError(
                "Feedback is enabled — collect_states requires Y_teach"
            )

        Xp = self._preprocess(X)
        H = np.zeros((T, N))
        h = np.zeros(N)
        for t in range(T):
            fb = None
            if use_fb:
                y_prev = (
                    Y_teach[t - 1]
                    if t > 0
                    else np.zeros(self.rc.readout.units)
                )
                fb = self.W_fb @ y_prev
            h = self._reservoir_step(h, Xp[t], fb)
            H[t] = h
        return H

    def _encode_targets(self, Y: "np.ndarray") -> "np.ndarray":
        """For classification, turn a label vector into a one-hot matrix.

        A 1-D `Y` is treated as integer/label targets and one-hot encoded
        (recording `classes_`). An already 2-D `Y` is assumed to be one-hot
        and passed through, with `classes_` defaulting to 0..C-1.
        """
        if self.rc.readout.task != Task.CLASSIFICATION:
            return Y
        if Y.ndim == 1:
            self.classes_ = np.unique(Y)
            return _one_hot(Y, self.classes_)
        self.classes_ = np.arange(Y.shape[1])
        return Y

    def fit(
        self,
        X: "np.ndarray",
        Y: "np.ndarray",
        *,
        materialize_states: bool = True,
    ) -> "np.ndarray":
        """Batch training (RIDGE / PINV). Use `online_fit` for online trainers.

        `materialize_states` is accepted for backward compatibility with the
        previous API. Both modes currently use the same numerically stable
        batch solve path.
        """
        _ = materialize_states
        if X.ndim == 1:
            X = X[:, None]
        Y = np.asarray(Y)
        Y = self._encode_targets(Y)
        if Y.ndim == 1:
            Y = Y[:, None]
        trainer = self.rc.readout.trainer
        if trainer == Trainer.RIDGE:
            if not materialize_states:
                return self._fit_ridge_streaming(X, Y)
            use_fb = self.W_fb is not None
            H = self.collect_states(X, Y if use_fb else None)
            Phi = self._augment(X, H)
            w = self.rc.readout.washout
            Phi_w, Y_w = Phi[w:], Y[w:]
            lam = self.rc.readout.regularization
            A = Phi_w.T @ Phi_w + lam * np.eye(Phi_w.shape[1])
            B = Phi_w.T @ Y_w
            return self._solve_ridge(A, B, Y.shape[1])
        elif trainer == Trainer.PINV:
            use_fb = self.W_fb is not None
            H = self.collect_states(X, Y if use_fb else None)
            Phi = self._augment(X, H)
            w = self.rc.readout.washout
            Phi_w, Y_w = Phi[w:], Y[w:]
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
            raise RuntimeError(
                "Readout has not been trained — call fit() first"
            )
        if X.ndim == 1:
            X = X[:, None]
        H = self.collect_states(X)
        Phi = self._augment(X, H)
        return Phi @ self.W_out.T

    def _check_classification(self) -> None:
        if self.rc.readout.task != Task.CLASSIFICATION:
            raise ValueError(
                "predict_proba / predict_classes require "
                "readout.task == Task.CLASSIFICATION"
            )
        if self.classes_ is None:
            raise RuntimeError(
                "Classifier has not been trained — call fit() first"
            )

    def predict_proba(self, X: "np.ndarray") -> "np.ndarray":
        """Per-step class probabilities (T, C) via softmax of the readout."""
        self._check_classification()
        return _softmax(self.predict(X))

    def predict_classes(self, X: "np.ndarray") -> "np.ndarray":
        """Per-step predicted labels (T,) via argmax, mapped back to classes_."""
        self._check_classification()
        return self.classes_[np.argmax(self.predict(X), axis=1)]

    # ------------------------------------------------------------------
    # Sequence-to-label (state aggregation over time)

    def _aggregate_states(self, H: "np.ndarray", X_raw: "np.ndarray"):
        """Pool a sequence's states (and raw input) into one feature vector.

        Returns (h_bar, u_bar) where u_bar is None unless include_input.
        MEAN averages post-washout steps; LAST takes the final step.
        """
        agg = self.rc.readout.aggregation
        want_input = self.rc.readout.include_input
        if agg == Aggregation.LAST:
            return H[-1], (X_raw[-1] if want_input else None)
        if agg == Aggregation.MEAN:
            w = min(self.rc.readout.washout, H.shape[0] - 1)
            w = max(w, 0)
            h_bar = H[w:].mean(axis=0)
            u_bar = X_raw[w:].mean(axis=0) if want_input else None
            return h_bar, u_bar
        raise ValueError(
            "Sequence methods require readout.aggregation in {MEAN, LAST}; "
            f"got {agg.name}"
        )

    def _augment_agg(self, u_bar, h_bar: "np.ndarray") -> "np.ndarray":
        parts = []
        if self.rc.readout.include_bias:
            parts.append(np.ones(1))
        if self.rc.readout.include_input:
            parts.append(np.atleast_1d(u_bar))
        parts.append(h_bar)
        return np.concatenate(parts)

    def _sequence_features(self, seqs) -> "np.ndarray":
        feats = []
        for X in seqs:
            X = np.asarray(X, dtype=float)
            if X.ndim == 1:
                X = X[:, None]
            H = self.collect_states(X)
            h_bar, u_bar = self._aggregate_states(H, X)
            feats.append(self._augment_agg(u_bar, h_bar))
        return np.stack(feats)

    def fit_sequences(self, seqs, labels: "np.ndarray") -> "np.ndarray":
        """Train a sequence-to-label readout (one feature vector per sequence).

        `seqs` is a list of (T_i, K) arrays; `labels` is (S,). For
        classification, labels are one-hot encoded; for regression they are
        used as (S,) or (S, M) targets. Requires aggregation in {MEAN, LAST}.
        """
        if self.rc.readout.aggregation == Aggregation.NONE:
            raise ValueError(
                "fit_sequences requires readout.aggregation in {MEAN, LAST}; "
                "use fit() for per-step training"
            )
        Phi = self._sequence_features(seqs)
        Y = self._encode_targets(np.asarray(labels))
        if Y.ndim == 1:
            Y = Y[:, None]
        trainer = self.rc.readout.trainer
        if trainer == Trainer.RIDGE:
            lam = self.rc.readout.regularization
            A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
            self.W_out = np.linalg.solve(A, Phi.T @ Y).T
        elif trainer == Trainer.PINV:
            self.W_out = (np.linalg.pinv(Phi) @ Y).T
        else:
            raise NotImplementedError(
                f"Trainer {trainer.name} is not supported for sequence "
                "training; use RIDGE or PINV"
            )
        return self.W_out

    def predict_sequences(self, seqs) -> "np.ndarray":
        """Per-sequence output. Regression: (S, M). Classification: labels (S,)."""
        if self.W_out is None:
            raise RuntimeError(
                "Readout has not been trained — call fit_sequences() first"
            )
        Z = self._sequence_features(seqs) @ self.W_out.T
        if self.rc.readout.task == Task.CLASSIFICATION:
            return self.classes_[np.argmax(Z, axis=1)]
        return Z

    def predict_proba_sequences(self, seqs) -> "np.ndarray":
        """Per-sequence class probabilities (S, C) via softmax."""
        self._check_classification()
        Z = self._sequence_features(seqs) @ self.W_out.T
        return _softmax(Z)

    def free_run(self, X_seed: "np.ndarray", n_steps: int) -> "np.ndarray":
        if self.W_out is None:
            raise RuntimeError(
                "Readout has not been trained — call fit() first"
            )
        if X_seed.ndim == 1:
            X_seed = X_seed[:, None]
        K = self.rc.input.units
        M = self.rc.readout.units
        if K != M:
            raise ValueError(
                f"free_run requires input dim ({K}) == output dim ({M})"
            )

        Xp_seed = self._preprocess(X_seed)
        h = np.zeros(self.rc.reservoir.units)
        for t in range(Xp_seed.shape[0]):
            h = self._reservoir_step(h, Xp_seed[t])

        last_x = X_seed[-1].copy()
        preds = np.zeros((n_steps, M))
        for t in range(n_steps):
            phi = self._augment_one(last_x, h)
            y_hat = self.W_out @ phi
            preds[t] = y_hat
            h = self._reservoir_step(h, self._preprocess(y_hat))
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
        xp = self.executor._preprocess(x)
        self.h = self.executor._reservoir_step(self.h, xp)

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
        z = (
            self.executor.W_in @ xp
            + self.executor.W_res @ self.h
            + self.executor.W_fb @ y_prev
            + bias
        )
        self.h = (1.0 - leak) * self.h + leak * act(z)

    def step(self, x, y):
        y_hat = super().step(x, y)
        self.last_pred = y_hat
        return y_hat

    def step_no_update(self, x):
        y_hat = super().step_no_update(x)
        self.last_pred = y_hat
        return y_hat
