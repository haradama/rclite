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
from typing import Optional

import numpy as np

from .model import QuantizedModel
from .executor import QuantizedExecutor
from ._intops import fixed_mul_scalar_i32, trunc_i32


_INT32_MIN = -(1 << 31)
_INT32_MAX = (1 << 31) - 1


def _sadd_sat_i32(a: int, b: int) -> int:
    s = int(a) + int(b)
    if s > _INT32_MAX:
        return _INT32_MAX
    if s < _INT32_MIN:
        return _INT32_MIN
    return s


class IntegerLMSLearner:
    """Online integer LMS with column-specific shifts.

    `learning_rate` is the conventional float rate η; internally it is
    quantized at state_scale.

    Updates are applied to `qmodel.W_out_q` in place, so the underlying
    QuantizedModel evolves as samples stream in. Use `predict` to get the
    current model's output; use `step` to feed (x, y_target) pairs.
    """

    def __init__(self, qmodel: QuantizedModel, learning_rate: float = 1e-2):
        self.qmodel = qmodel
        self.cfg = qmodel.config
        self.target = qmodel.target
        self.learning_rate = float(learning_rate)
        # quantize η at state_scale (state_scale plays the role of unit)
        self.lr_q = self.target.quantize_state(learning_rate, self.cfg)
        self._executor = QuantizedExecutor(qmodel)

    @property
    def state_q(self) -> np.ndarray:
        return self._executor.state_q

    def reset(self) -> None:
        self._executor.reset()

    def _quantize_input(self, u: np.ndarray) -> np.ndarray:
        rc = self.qmodel.rc
        u_pre = (u - rc.input.input_offset) * rc.input.input_scaling
        return self.target.quantize_input_array(u_pre, self.cfg).astype(np.int32)

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
            [self.target.quantize_state(float(v), self.cfg) for v in y_target_f],
            dtype=np.int32,
        )

        error_q = (y_target_q.astype(np.int64) - y_pred_q.astype(np.int64))

        # Update W_out_q with column-specific shifts
        self._apply_lms_update(error_q, u_pre_q, state_q)

        return y_pred_q.astype(np.float64) / self.cfg.state_scale

    def _apply_lms_update(self, error_q, u_q, state_q) -> None:
        """In-place update of W_out_q. Saturating add prevents wrap-around."""
        qm = self.qmodel
        cfg = self.cfg
        rc = qm.rc
        M = qm.M
        K = qm.K
        N = qm.N
        sf = cfg.state_frac
        inp_f = cfg.input_frac

        lr_q = self.lr_q
        Wout = qm.W_out_q  # int dtype storage

        off = 0
        if rc.readout.include_bias:
            # dW = (lr_q * error_q) >> state_frac
            for m in range(M):
                dw = (int(lr_q) * int(error_q[m])) >> sf
                Wout[m, 0] = _sadd_sat_i32(int(Wout[m, 0]), dw)
            off = 1
        if rc.readout.include_input:
            for m in range(M):
                for k in range(K):
                    prod = int(lr_q) * int(error_q[m]) * int(u_q[k])
                    dw = prod >> (2 * inp_f)
                    Wout[m, off + k] = _sadd_sat_i32(int(Wout[m, off + k]), dw)
            off += K
        for m in range(M):
            for i in range(N):
                prod = int(lr_q) * int(error_q[m]) * int(state_q[i])
                dw = prod >> (2 * sf)
                Wout[m, off + i] = _sadd_sat_i32(int(Wout[m, off + i]), dw)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the current model on a sequence (no learning); returns float."""
        # Fresh executor so we don't disturb online state
        ro = QuantizedExecutor(self.qmodel)
        ro.state_q = self.state_q.copy()
        return ro.predict(X)
