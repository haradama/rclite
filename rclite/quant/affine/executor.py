"""Integer Python reference executor for the affine i8 path.

The per-step computation mirrors a TFLM-style int8 kernel:

  # Preprocess input (float helper here; on-device this is also integer)
  u_pre_q[k] = quantize(u_pre[k], s_upre, zp_upre)

  # Reservoir matmul + bias, in i32 accumulators, with zp folding
  acc_in[i]  = sum_k  q_W_in[i,k]  * q_upre[k] - zp_upre * R_in[i]
  acc_res[i] = sum_j  q_W_res[i,j] * q_h[j]    - zp_h    * R_res[i]
  q_pre[i]   = zp_pre + bias_pre
             + saturate( round(M_in  * acc_in[i])  )
             + saturate( round(M_res * acc_res[i]) )

  # Activation by direct LUT (256 entries — no interpolation)
  q_act[i]   = LUT[q_pre[i] - qmin]

  # Leaky integration (state and activated share params, so all in one scale)
  q_h_new[i] = zp_state + round(
                  (1 - leak) * (q_h[i] - zp_state)
                + leak       * (q_act[i] - zp_state)
              )

The multipliers `M_in`, `M_res`, ... are kept as Python floats — converting
them to integer `(M0, n)` form is the LLVM-emit problem in Phase 2b. The
Python results are still well-defined because all ints are wrapped, only
the multiplier is float.

Saturation happens at each requantize boundary and is **not** a wrap;
i.e. values that overflow the storage range clamp to ±qmax, matching
what a TFLM kernel does.
"""
from __future__ import annotations

import numpy as np

from .quantize import AffineQuantizedModel
from .multiplier import apply_multiplier_array


def _saturate(arr: np.ndarray, storage_bits: int) -> np.ndarray:
    """Clamp values to the signed storage range (saturating)."""
    qmin = -(1 << (storage_bits - 1))
    qmax = (1 << (storage_bits - 1)) - 1
    return np.clip(arr, qmin, qmax)


def _saturate_scalar(v: int, storage_bits: int) -> int:
    qmin = -(1 << (storage_bits - 1))
    qmax = (1 << (storage_bits - 1)) - 1
    return int(max(qmin, min(qmax, int(v))))


class AffineQuantizedExecutor:
    """Python reference for the asymmetric per-tensor affine kernel."""

    def __init__(self, qmodel: AffineQuantizedModel):
        self.qmodel = qmodel
        self.cfg = qmodel.config
        self.storage_bits = self.cfg.storage_bits
        self.storage_dtype = self.cfg.state.storage_dtype
        self.reset()

    def reset(self) -> None:
        # State carries q_h in storage dtype; promoted to i32 inside the loop.
        self.state_q = self.qmodel.state_init_q.astype(np.int32).copy()

    # ------------------------------------------------------------------
    # Input quantization helpers

    def _quantize_raw_input(self, x_float: np.ndarray) -> np.ndarray:
        """Float raw input → q_x (i32, at input scale/zp)."""
        return self.cfg.input.quantize_array(x_float).astype(np.int32)

    def _quantize_u_pre(self, x_float: np.ndarray) -> np.ndarray:
        """Float raw input → q_upre (i32, at u_pre scale/zp).

        We compute `(x - off) * scale` in float, then quantize at the
        u_pre params. (On-device this preprocess would also be integer,
        but the float helper keeps the Python reference simple.)
        """
        rc = self.qmodel.rc
        u_pre = (np.asarray(x_float, dtype=np.float64) - rc.input.input_offset) \
                * rc.input.input_scaling
        return self.cfg.u_pre.quantize_array(u_pre).astype(np.int32)

    # ------------------------------------------------------------------
    # Reservoir step

    def step_q(self, u_pre_q: np.ndarray) -> np.ndarray:
        """One reservoir step; returns the new q_h (i32, at state scale/zp)."""
        q = self.qmodel
        N = q.N
        cfg = self.cfg
        sb = self.storage_bits

        zp_upre = cfg.u_pre.zero_point
        zp_state = cfg.state.zero_point
        zp_pre = cfg.pre.zero_point

        # acc_in[i] = sum_k q_W_in[i,k] * q_upre[k] - zp_upre * R_in[i]
        if q.K > 0:
            acc_in = (q.W_in_q.astype(np.int32)
                       @ u_pre_q.astype(np.int32)
                       - zp_upre * q.row_sum_W_in)
        else:
            acc_in = np.zeros(N, dtype=np.int32)

        # acc_res[i] = sum_j q_W_res[i,j] * q_h[j] - zp_state * R_res[i]
        acc_res = (q.W_res_q.astype(np.int32) @ self.state_q
                    - zp_state * q.row_sum_W_res)

        # Requantize each contribution to pre-scale, then sum + add zp + bias.
        # Uses integer (M0, n) multipliers so the JIT and this Python ref are
        # bit-exact on the requantize step.
        rq_in  = apply_multiplier_array(acc_in,  q.M_in_M0,  q.M_in_n)
        rq_res = apply_multiplier_array(acc_res, q.M_res_M0, q.M_res_n)
        pre_q = zp_pre + q.bias_pre + rq_in + rq_res
        pre_q = _saturate(pre_q, sb).astype(np.int32)

        # Direct LUT lookup (no interpolation): lut[q_pre - qmin]
        idx = pre_q + q.lut_offset
        activated_q = q.lut_q[idx].astype(np.int32)

        # Leaky integration: new_h = h + leak * (activated - h), all centered
        # at zp_state. Integer (M0, n) leak multiplier keeps JIT parity.
        h_centered = self.state_q - zp_state
        a_centered = activated_q - zp_state
        diff = a_centered - h_centered
        delta = apply_multiplier_array(diff, q.leak_M0, q.leak_n)
        new_h_centered = h_centered.astype(np.int64) + delta
        self.state_q = _saturate(new_h_centered + zp_state, sb).astype(np.int32)
        return self.state_q.copy()

    # ------------------------------------------------------------------
    # Readout

    def predict_one_q(self, x_raw_q: np.ndarray,
                       state_q: np.ndarray) -> np.ndarray:
        """Mixed-scale W_out matmul → y_q (i32 at output scale/zp).

        Per-column-block requantize:
          y_q[m] = zp_y + round(M_bias  * acc_bias)
                       + round(M_input * (acc_input_dot - zp_input * R_in[m]))
                       + round(M_state * (acc_state_dot - zp_state * R_state[m]))
        """
        q = self.qmodel
        cfg = self.cfg
        rc = q.rc
        K = q.K
        N = q.N
        M = q.M
        Wo = q.W_out_q.astype(np.int32)
        zp_y = cfg.output.zero_point
        zp_input = cfg.input.zero_point
        zp_state = cfg.state.zero_point

        y_acc = np.full(M, zp_y, dtype=np.int64)
        off = 0
        if rc.readout.include_bias:
            bias_col = Wo[:, 0]  # (M,)
            y_acc += apply_multiplier_array(bias_col, q.M_out_bias_M0,
                                              q.M_out_bias_n)
            off += 1
        if rc.readout.include_input:
            Wi = Wo[:, off:off + K]  # (M, K)
            dot_in = Wi @ x_raw_q.astype(np.int32)  # (M,)
            adj_in = dot_in - zp_input * q.row_sum_Wout_input
            y_acc += apply_multiplier_array(adj_in, q.M_out_input_M0,
                                              q.M_out_input_n)
            off += K
        Ws = Wo[:, off:off + N]  # (M, N)
        dot_st = Ws @ state_q.astype(np.int32)
        adj_st = dot_st - zp_state * q.row_sum_Wout_state
        y_acc += apply_multiplier_array(adj_st, q.M_out_state_M0,
                                          q.M_out_state_n)

        return _saturate(y_acc, self.storage_bits).astype(np.int32)

    # ------------------------------------------------------------------
    # Convenience: trajectory predict over float input, returning float output

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run inference over a length-T sequence; returns dequantized y."""
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        Y_q = np.zeros((T, self.qmodel.M), dtype=np.int32)
        for t in range(T):
            x_raw_q = self._quantize_raw_input(X[t])
            u_pre_q = self._quantize_u_pre(X[t])
            self.step_q(u_pre_q)
            Y_q[t] = self.predict_one_q(x_raw_q, self.state_q)
        return self.cfg.output.dequantize_array(Y_q)

    def collect_states(self, X: np.ndarray) -> np.ndarray:
        """Run the reservoir over X; return the dequantized state trajectory."""
        if X.ndim == 1:
            X = X[:, None]
        T = X.shape[0]
        H_q = np.zeros((T, self.qmodel.N), dtype=np.int32)
        for t in range(T):
            u_pre_q = self._quantize_u_pre(X[t])
            self.step_q(u_pre_q)
            H_q[t] = self.state_q
        return self.cfg.state.dequantize_array(H_q)
