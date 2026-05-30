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
from .multiplier import apply_multiplier_array, apply_multiplier_perrow
from .lut import LUTKind


def _saturate(arr: np.ndarray, storage_bits: int) -> np.ndarray:
    """Clamp values to the signed storage range (saturating)."""
    qmin = -(1 << (storage_bits - 1))
    qmax = (1 << (storage_bits - 1)) - 1
    return np.clip(arr, qmin, qmax)


def _saturate_scalar(v: int, storage_bits: int) -> int:
    qmin = -(1 << (storage_bits - 1))
    qmax = (1 << (storage_bits - 1)) - 1
    return int(max(qmin, min(qmax, int(v))))


def _clamp_i32(arr: np.ndarray) -> np.ndarray:
    """Clamp to signed i32 range, return i32. Mirrors the JIT's
    `_clamp_to_i32` before each requantize so Python and JIT agree even
    when an accumulator would overflow i32 (large N or mixed precision)."""
    return np.clip(arr.astype(np.int64), -(1 << 31), (1 << 31) - 1).astype(np.int32)


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

        When `qmodel.has_integer_preprocess` is True the kernel does the
        preprocess in pure integer (`q_upre = pre_const + apply_mult(q_x −
        zp_x, pre_M0, pre_n)`); the Python ref mirrors that exactly so it
        stays bit-exact with the JIT. Otherwise (identity preprocess, the
        common case) we just quantize the raw input.
        """
        q = self.qmodel
        if q.has_integer_preprocess:
            x_raw_q = self._quantize_raw_input(x_float)
            centered = (x_raw_q.astype(np.int32)
                        - self.cfg.input.zero_point).astype(np.int32)
            delta = apply_multiplier_array(centered, q.pre_M0, q.pre_n)
            total = (q.pre_const + delta).astype(np.int64)
            return _saturate(total, self.storage_bits).astype(np.int32)
        # Identity preprocess: input and u_pre share scale/zp (by calibration).
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
        # Accumulate in i64 to avoid overflow, then clamp to i32 (matching
        # the JIT) before the requantize.
        if q.K > 0:
            acc_in = (q.W_in_q.astype(np.int64)
                       @ u_pre_q.astype(np.int64)
                       - zp_upre * q.row_sum_W_in.astype(np.int64))
        else:
            acc_in = np.zeros(N, dtype=np.int64)

        # acc_res[i] = sum_j q_W_res[i,j] * q_h[j] - zp_state * R_res[i]
        acc_res = (q.W_res_q.astype(np.int64) @ self.state_q.astype(np.int64)
                    - zp_state * q.row_sum_W_res.astype(np.int64))

        # Requantize each contribution to pre-scale, then sum + add zp + bias.
        # Uses integer (M0, n) multipliers so the JIT and this Python ref are
        # bit-exact on the requantize step.
        rq_in  = apply_multiplier_array(_clamp_i32(acc_in),  q.M_in_M0,  q.M_in_n)
        if q.M_res_M0_arr is not None:
            # per-channel: each reservoir row uses its own (M0[i], n[i]).
            rq_res = apply_multiplier_perrow(
                _clamp_i32(acc_res), q.M_res_M0_arr, q.M_res_n_arr)
        else:
            rq_res = apply_multiplier_array(
                _clamp_i32(acc_res), q.M_res_M0, q.M_res_n)
        pre_q = zp_pre + q.bias_pre + rq_in + rq_res
        pre_q = _saturate(pre_q, sb).astype(np.int32)

        # Activation: dispatch on the chosen strategy.
        activated_q = self._activate(pre_q)

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
    # Activation (one of three strategies)

    def _activate(self, pre_q: np.ndarray) -> np.ndarray:
        """Compute q_act from q_pre using the model's LUTStrategy."""
        q = self.qmodel
        kind = q.lut_strategy.kind
        if kind == LUTKind.DIRECT:
            return self._activate_direct(pre_q)
        if kind == LUTKind.LINEAR_INTERP:
            return self._activate_linear_interp(pre_q)
        if kind == LUTKind.POLYNOMIAL:
            return self._activate_polynomial(pre_q)
        raise ValueError(f"unknown LUTKind: {kind}")

    def _activate_direct(self, pre_q: np.ndarray) -> np.ndarray:
        q = self.qmodel
        idx = pre_q + q.lut_offset
        return q.lut_q[idx].astype(np.int32)

    def _activate_linear_interp(self, pre_q: np.ndarray) -> np.ndarray:
        """Subsampled table + linear interpolation, matching the JIT emit."""
        q = self.qmodel
        art = q.lut_artifacts
        f = q.lut_strategy.interp_frac_bits
        n = q.lut_strategy.n_entries
        # t_q = (q_pre - qmin) * idx_M0 >> idx_n, in Q.f
        # (apply_multiplier_array implements the M0,n requantize bit-exactly)
        normalized = (pre_q.astype(np.int64) + art.offset)
        t_q = apply_multiplier_array(normalized.astype(np.int32),
                                       art.idx_M0, art.idx_n)
        # Split into integer index and fractional remainder.
        idx = (t_q >> f).astype(np.int64)
        idx = np.clip(idx, 0, n - 2)
        frac_q = t_q - (idx << f)
        # Lerp between adjacent table entries, in i64 to be safe.
        y0 = q.lut_q[idx].astype(np.int64)
        y1 = q.lut_q[idx + 1].astype(np.int64)
        dy = y1 - y0
        interp = y0 + ((dy * frac_q) >> f)
        return _saturate(interp, self.storage_bits).astype(np.int32)

    def _activate_polynomial(self, pre_q: np.ndarray) -> np.ndarray:
        """Odd-only minimax polynomial in Q.qf_bits, clamped to ±1.

        Evaluates  tanh(x) ≈ a1·x + a3·x³ + a5·x⁵   (a5=0 if degree==3),
        clamped to |x| ≤ poly_clip and the result clamped to ±1. The same
        integer ops the JIT will emit (Q.qf intermediates) keep Python
        and JIT bit-exact.
        """
        q = self.qmodel
        art = q.lut_artifacts
        qf = q.lut_strategy.poly_qf_bits
        cfg = self.cfg
        zp_pre = cfg.pre.zero_point
        zp_state = cfg.state.zero_point

        # 1) Convert q_pre to x in Q.qf
        centered = (pre_q.astype(np.int64) - zp_pre).astype(np.int32)
        x_qf = apply_multiplier_array(centered, art.x_to_qf_M0,
                                        art.x_to_qf_n).astype(np.int64)
        # 2) Clamp |x| ≤ x_clip_qf
        x_qf = np.clip(x_qf, -art.x_clip_qf, art.x_clip_qf)
        # 3) Horner-style poly in x²:
        #       u² = x²
        #       y/x = a1 + u²·(a3 + u²·a5)
        #       y   = x · (a1 + u²·(a3 + u²·a5))
        #    All in Q.qf; each multiply shifts back by qf.
        x2_qf = (x_qf * x_qf) >> qf
        inner = ((x2_qf * art.poly_a5_qf) >> qf) + art.poly_a3_qf
        outer = ((x2_qf * inner) >> qf) + art.poly_a1_qf
        y_qf = (x_qf * outer) >> qf
        # 4) Clamp tanh value to ±1 (= ±one_qf)
        y_qf = np.clip(y_qf, -art.one_qf, art.one_qf)
        # 5) Map y_qf → Δq_state via back multiplier
        delta = apply_multiplier_array(y_qf.astype(np.int32),
                                         art.qf_to_state_M0,
                                         art.qf_to_state_n)
        out = (delta + zp_state).astype(np.int64)
        return _saturate(out, self.storage_bits).astype(np.int32)

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
        # W_out may be wider than the base storage (mixed precision); widen
        # to i64 for the matmul so it never overflows, then clamp to i32
        # before each requantize (matching the JIT).
        Wo = q.W_out_q.astype(np.int64)
        zp_y = cfg.output.zero_point
        zp_input = cfg.input.zero_point
        zp_state = cfg.state.zero_point

        per_channel = q.M_out_state_M0_arr is not None

        def _rq(x, M0_s, n_s, M0_a, n_a):
            if per_channel:
                return apply_multiplier_perrow(x, M0_a, n_a)
            return apply_multiplier_array(x, M0_s, n_s)

        y_acc = np.full(M, zp_y, dtype=np.int64)
        off = 0
        if rc.readout.include_bias:
            bias_col = _clamp_i32(Wo[:, 0])  # (M,)
            y_acc += _rq(bias_col, q.M_out_bias_M0, q.M_out_bias_n,
                         q.M_out_bias_M0_arr, q.M_out_bias_n_arr)
            off += 1
        if rc.readout.include_input:
            Wi = Wo[:, off:off + K]  # (M, K)
            dot_in = Wi @ x_raw_q.astype(np.int64)  # (M,)
            adj_in = dot_in - zp_input * q.row_sum_Wout_input.astype(np.int64)
            y_acc += _rq(_clamp_i32(adj_in), q.M_out_input_M0, q.M_out_input_n,
                         q.M_out_input_M0_arr, q.M_out_input_n_arr)
            off += K
        Ws = Wo[:, off:off + N]  # (M, N)
        dot_st = Ws @ state_q.astype(np.int64)
        adj_st = dot_st - zp_state * q.row_sum_Wout_state.astype(np.int64)
        y_acc += _rq(_clamp_i32(adj_st), q.M_out_state_M0, q.M_out_state_n,
                     q.M_out_state_M0_arr, q.M_out_state_n_arr)

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
