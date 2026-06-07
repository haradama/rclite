"""Online integer-LMS readout learner for quantized models.

The algorithm mirrors the float LMS in `rclite.runtime.reference` but
operates entirely on the QuantizedModel's i32 weights. Per sample:

    1. Forward step the reservoir (via QuantizedExecutor).
    2. Predict via the mixed-scale W_out_q dot product.
    3. Compute prediction error e_q = y_target_q - y_pred_q  (i32, state scale).
    4. Update W_out_q in place with column-specific shifts:
         bias  col :  dW = (lr_q * e_q)          >> state_frac
         input col :  dW = (lr_q * e_q * u_q)    >> (2*input_frac)
         state col :  dW = (lr_q * e_q * s_q)    >> (2*state_frac)
       Each column's shift compensates for its individual W_out scaling.

Saturating add at each update prevents wrap-around when the learning
rate is too high. An LLVM emit for the same update kernel is sketched at
the bottom — defer integration until needed.
"""

from __future__ import annotations

import numpy as np

from .model import QuantizedModel
from .executor import QuantizedExecutor


def _sadd_sat(a: int, b: int, storage_bits: int) -> int:
    """Saturating add into the **storage** range [-2^(sb-1), 2^(sb-1)-1].

    Saturating to the storage width (not int32) is what actually prevents
    wrap-around: W_out_q is stored as the target's storage dtype, so an i32
    saturation would still wrap on assignment to an i16 array. For i32 storage
    this is identical to a plain int32 saturate.
    """
    lo = -(1 << (storage_bits - 1))
    hi = (1 << (storage_bits - 1)) - 1
    s = int(a) + int(b)
    if s > hi:
        return hi
    if s < lo:
        return lo
    return s


def _tdiv(a: int, b: int) -> int:
    """Truncate-toward-zero integer division, matching C99 `/` (b > 0).

    Python's `//` floors toward -inf; C truncates toward zero. NLMS divides by
    the (positive) squared-norm, so reproducing C's truncation keeps Python and
    the generated kernel bit-exact for negative numerators.
    """
    q = abs(int(a)) // int(b)
    return -q if a < 0 else q


class IntegerLMSLearner:
    """Online integer LMS / NLMS with column-specific shifts.

    `learning_rate` is the conventional float rate η; internally it is
    quantized at state_scale.

    With ``normalized=True`` the step is divided by the squared norm of the
    readout feature vector φ = [1, u, h] (normalized LMS). This removes LMS's
    sensitivity to the feature scale — the effective rate adapts per step — so
    η near 1.0 is usable. The squared norm is accumulated at state-scale fixed
    point and the per-step division is truncate-toward-zero (C-compatible).
    ``delta`` is the NLMS regularizer δ (in float feature units); δ·‖φ‖²-scale
    is added to the denominator and also guards against divide-by-zero.

    Updates are applied to `qmodel.W_out_q` in place, so the underlying
    QuantizedModel evolves as samples stream in. Use `predict` to get the
    current model's output; use `step` to feed (x, y_target) pairs.
    """

    def __init__(
        self,
        qmodel: QuantizedModel,
        learning_rate: float = 1e-2,
        *,
        normalized: bool = False,
        delta: float = 1.0,
    ):
        self.qmodel = qmodel
        self.cfg = qmodel.config
        self.target = qmodel.target
        self.learning_rate = float(learning_rate)
        self.normalized = bool(normalized)
        self.delta = float(delta)
        # quantize η at state_scale (state_scale plays the role of unit)
        self.lr_q = self.target.quantize_state(learning_rate, self.cfg)
        if self.normalized:
            sf, inp_f = self.cfg.state_frac, self.cfg.input_frac
            if 2 * inp_f < sf:
                raise NotImplementedError(
                    "NLMS needs 2*input_frac >= state_frac for the squared-"
                    f"norm fixed point (got input_frac={inp_f}, state_frac={sf})"
                )
            self.delta_q = int(self.delta * (1 << sf))
        self._executor = QuantizedExecutor(qmodel)

    def _norm_q(self, u_q, state_q) -> int:
        """δ·Q + Q·‖φ‖² at state-scale fixed point Q=2^state_frac (i64)."""
        cfg = self.cfg
        rc = self.qmodel.rc
        sf, inp_f = cfg.state_frac, cfg.input_frac
        shift_u = 2 * inp_f - sf
        norm = int(self.delta_q)
        if rc.readout.include_bias:
            norm += 1 << sf
        if rc.readout.include_input:
            for k in range(self.qmodel.K):
                norm += (int(u_q[k]) ** 2) >> shift_u
        for j in range(self.qmodel.N):
            norm += (int(state_q[j]) ** 2) >> sf
        return norm if norm >= 1 else 1

    @property
    def state_q(self) -> np.ndarray:
        return self._executor.state_q

    def reset(self) -> None:
        self._executor.reset()

    def _quantize_input(self, u: np.ndarray) -> np.ndarray:
        rc = self.qmodel.rc
        u_pre = (u - rc.input.input_offset) * rc.input.input_scaling
        return self.target.quantize_input_array(u_pre, self.cfg).astype(
            np.int32
        )

    def step(self, x_f: np.ndarray, y_target_f: np.ndarray) -> np.ndarray:
        """Process one streaming sample. Returns the (dequantized) prediction.

        x_f       : (K,) float input
        y_target_f: (M,) float target
        """
        x_f = np.atleast_1d(x_f).astype(np.float64)
        y_target_f = np.atleast_1d(y_target_f).astype(np.float64)

        u_pre_q = self._quantize_input(x_f)
        # Forward reservoir
        self._executor.step_q(u_pre_q)
        state_q = self._executor.state_q

        # Predict (mixed-scale i64 acc → state_scale i32)
        y_pred_q = self._executor.predict_one_q(u_pre_q, state_q)

        # Quantize target at state_scale
        y_target_q = np.array(
            [
                self.target.quantize_state(float(v), self.cfg)
                for v in y_target_f
            ],
            dtype=np.int32,
        )

        error_q = y_target_q.astype(np.int64) - y_pred_q.astype(np.int64)

        # Update W_out_q with column-specific shifts
        self._apply_lms_update(error_q, u_pre_q, state_q)

        return y_pred_q.astype(np.float64) / self.cfg.state_scale

    def _apply_lms_update(self, error_q, u_q, state_q) -> None:
        """In-place update of W_out_q. Saturating add prevents wrap-around.

        Per column, the raw fixed-point product is ``prod = lr_q*e*feature_q``
        with a column-specific scale shift (bias→state_frac, input→2*input_frac,
        state→2*state_frac). LMS shifts ``prod`` down by that amount. NLMS first
        divides ``prod`` by the squared norm (state-scale Q), which costs one
        extra ``- state_frac`` on the shift.
        """
        qm = self.qmodel
        cfg = self.cfg
        rc = qm.rc
        M = qm.M
        K = qm.K
        N = qm.N
        sf = cfg.state_frac
        inp_f = cfg.input_frac
        sb = self.target.storage_bits

        lr_q = self.lr_q
        Wout = qm.W_out_q  # int dtype storage

        norm_q = self._norm_q(u_q, state_q) if self.normalized else None

        def _dw(prod: int, shift_col: int) -> int:
            if norm_q is None:
                return prod >> shift_col
            return _tdiv(prod, norm_q) >> (shift_col - sf)

        off = 0
        if rc.readout.include_bias:
            for m in range(M):
                dw = _dw(int(lr_q) * int(error_q[m]), sf)
                Wout[m, 0] = _sadd_sat(int(Wout[m, 0]), dw, sb)
            off = 1
        if rc.readout.include_input:
            for m in range(M):
                for k in range(K):
                    prod = int(lr_q) * int(error_q[m]) * int(u_q[k])
                    dw = _dw(prod, 2 * inp_f)
                    Wout[m, off + k] = _sadd_sat(int(Wout[m, off + k]), dw, sb)
            off += K
        for m in range(M):
            for i in range(N):
                prod = int(lr_q) * int(error_q[m]) * int(state_q[i])
                dw = _dw(prod, 2 * sf)
                Wout[m, off + i] = _sadd_sat(int(Wout[m, off + i]), dw, sb)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the current model on a sequence (no learning); returns float."""
        # Fresh executor so we don't disturb online state
        ro = QuantizedExecutor(self.qmodel)
        ro.state_q = self.state_q.copy()
        return ro.predict(X)


def collect_training_stream(
    qmodel: QuantizedModel,
    X: np.ndarray,
    Y: np.ndarray,
    *,
    learning_rate: float,
    normalized: bool = False,
    delta: float = 1.0,
    warmup: int = 0,
) -> dict:
    """Drive `IntegerLMSLearner` over (X, Y); return the integer I/O streams a
    deployed kernel must reproduce, plus the final learned readout.

    This is the single source of truth for "what the on-device online kernel
    should compute": both the bit-exact tests and the Arduino build path embed
    these streams and assert the device reproduces them. The per-step records
    are the *pre-quantized* integer inputs/targets (so the device stays pure
    integer) and the reference predictions / final W_out.

    NOTE: this MUTATES ``qmodel.W_out_q`` in place (online learning evolves the
    readout). Emit any device kernel from the model BEFORE calling this, so the
    kernel bakes the *initial* weights while the returned ``final_W_out`` is the
    *post-training* checkpoint.

    Returns a dict with keys: ``u_q`` (T, K) int32, ``y_target_q`` (T, M) int32,
    ``y_pred_q`` (T, M) int32, ``warm`` (T,) uint8 (1 = inference-only step),
    ``final_W_out`` (M, F) storage-dtype, ``lr_q`` int.
    """
    X = np.asarray(X, dtype=np.float64)
    Y = np.asarray(Y, dtype=np.float64)
    if X.ndim == 1:
        X = X[:, None]
    if Y.ndim == 1:
        Y = Y[:, None]
    learner = IntegerLMSLearner(
        qmodel, learning_rate=learning_rate, normalized=normalized, delta=delta
    )
    exe = learner._executor
    cfg = learner.cfg
    target = learner.target
    T, M = X.shape[0], qmodel.M
    u_stream = np.zeros((T, qmodel.K), dtype=np.int32)
    yt_stream = np.zeros((T, M), dtype=np.int32)
    yp_stream = np.zeros((T, M), dtype=np.int32)
    warm = np.zeros(T, dtype=np.uint8)
    for t in range(T):
        u_q = learner._quantize_input(X[t]).astype(np.int32)
        exe.step_q(u_q)
        state_q = exe.state_q
        y_pred_q = exe.predict_one_q(u_q, state_q).astype(np.int32)
        if t < warmup:
            warm[t] = 1
        else:
            y_target_q = np.array(
                [target.quantize_state(float(v), cfg) for v in Y[t]],
                dtype=np.int32,
            )
            error_q = y_target_q.astype(np.int64) - y_pred_q.astype(np.int64)
            learner._apply_lms_update(error_q, u_q, state_q)
            yt_stream[t] = y_target_q
        u_stream[t] = u_q
        yp_stream[t] = y_pred_q
    return {
        "u_q": u_stream,
        "y_target_q": yt_stream,
        "y_pred_q": yp_stream,
        "warm": warm,
        "final_W_out": np.asarray(qmodel.W_out_q).copy(),
        "lr_q": int(learner.lr_q),
    }
