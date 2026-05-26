"""Bit-exact Python reference executor for the quantized kernel.

The QAT search drives this executor on candidate `QuantConfig`s and
selects the configuration with the lowest MSE on a held-out window. The
arithmetic here is identical (modulo floating-point preprocessing) to
what the LLVM integer lowering emits, so a config that scores well here
will score the same on the deployed binary.
"""
from __future__ import annotations
from typing import Optional

import numpy as np

from rclite.core.profile import Topology
from .model import QuantizedModel
from ._intops import fixed_mul_i32, trunc_i32, tanh_lut_lookup


class QuantizedExecutor:
    """Python reference for quantized reservoir inference."""

    def __init__(self, qmodel: QuantizedModel):
        self.qmodel = qmodel
        cfg = qmodel.config
        self.config = cfg
        self.target = qmodel.target
        self.lut = qmodel.lut

        if self.lut is None or qmodel.lut_table_q is None:
            raise ValueError(
                "QuantizedExecutor requires a TanhLUT; libm tanhf is "
                "not available in the integer path"
            )

        self.shift_input = cfg.weight_frac + cfg.input_frac - cfg.state_frac
        self.shift_recurrent = cfg.weight_frac
        self.leak_q = self.target.quantize_state(qmodel.rc.reservoir.leak_rate, cfg)
        self.one_minus_leak_q = (1 << cfg.state_frac) - self.leak_q
        self.bias_q = self.target.quantize_state(qmodel.rc.reservoir.bias, cfg)

        self.xmin_q = self.target.quantize_state(self.lut.xmin, cfg)
        self.xmax_q = self.target.quantize_state(self.lut.xmax, cfg)

        # CSR sparsity for the recurrent matrix (skip zeros). Always built
        # from the full dense W_res_q, since structured topologies (DLR/SCR/DLRB)
        # also produce a sparse W_res after quantization.
        self._W_res_csr = self._build_csr(qmodel.W_res_q)

        self.reset()

    @staticmethod
    def _build_csr(W: np.ndarray):
        rows = []
        for i in range(W.shape[0]):
            nz = np.flatnonzero(W[i])
            rows.append((nz, W[i, nz]))
        return rows

    def reset(self) -> None:
        self.state_q = self.qmodel.state_init_q.astype(np.int32).copy()

    # ------------------------------------------------------------------
    # Core step

    def step_q(self, u_pre_q: np.ndarray) -> np.ndarray:
        """One reservoir step with already-quantized input. Returns new state_q (i32)."""
        q = self.qmodel
        cfg = self.config
        N = q.N

        # pre[i] = bias + sum_k (W_in[i,k] * u[k]) >> shift_in
        #              + sum_j (W_res[i,j] * state[j]) >> shift_res
        pre_q = np.full(N, self.bias_q, dtype=np.int32)

        if q.K > 0:
            # Vectorized: pre += (W_in @ u_pre) >> shift_in
            #   per element: fixed_mul_i32(w, u, shift_in)
            #   accumulator wraps in i32 across the K terms — mirror that
            W_in = q.W_in_q.astype(np.int64)
            u = u_pre_q.astype(np.int64)
            # term[i, k] = (W_in[i, k] * u[k]) >> shift_in, trunc to i32
            terms = trunc_i32((W_in * u[np.newaxis, :]) >> self.shift_input)
            for k in range(q.K):
                pre_q = trunc_i32(pre_q.astype(np.int64) + terms[:, k].astype(np.int64))

        for i in range(N):
            nz_idx, w_row = self._W_res_csr[i]
            if nz_idx.size == 0:
                continue
            terms = trunc_i32((w_row.astype(np.int64)
                                * self.state_q[nz_idx].astype(np.int64))
                               >> self.shift_recurrent)
            acc = int(pre_q[i])
            for t in terms:
                acc = int(trunc_i32(np.int64(acc + int(t))))
            pre_q[i] = acc

        # Activation via LUT
        activated_q = tanh_lut_lookup(
            pre_q,
            self.qmodel.lut_table_q,
            self.xmin_q,
            self.xmax_q,
            cfg.state_frac,
        )

        # Leaky integration: state = (1-leak)*state + leak*activated  (state scale)
        t1 = fixed_mul_i32(
            self.state_q,
            np.full_like(self.state_q, self.one_minus_leak_q),
            cfg.state_frac,
        )
        t2 = fixed_mul_i32(
            activated_q,
            np.full_like(activated_q, self.leak_q),
            cfg.state_frac,
        )
        self.state_q = trunc_i32(t1.astype(np.int64) + t2.astype(np.int64))
        return self.state_q.copy()

    # ------------------------------------------------------------------
    # Readout

    def predict_one_q(self, u_raw_q: np.ndarray, state_q: np.ndarray) -> np.ndarray:
        """mirage-style readout with mixed-scale W_out_q. Returns y_q (state scale, i32)."""
        cfg = self.config
        q = self.qmodel
        M = q.M
        F = q.F

        # acc starts with bias contribution if include_bias
        # mirage: acc starts at 0; bias contribution is bias_input * w_out[0]
        #         where bias_input = 1 << state_frac
        out = np.zeros(M, dtype=np.int64)
        off = 0
        if q.rc.readout.include_bias:
            bias_scaled = np.int64(1) << cfg.state_frac
            out += bias_scaled * q.W_out_q[:, 0].astype(np.int64)
            off = 1
        if q.rc.readout.include_input:
            out += (q.W_out_q[:, off:off + q.K].astype(np.int64)
                     @ u_raw_q.astype(np.int64))
            off += q.K
        out += (q.W_out_q[:, off:off + q.N].astype(np.int64)
                 @ state_q.astype(np.int64))
        # Shift back to state scale and truncate
        return trunc_i32(out >> cfg.state_frac)

    # ------------------------------------------------------------------
    # Convenience wrappers operating on float arrays

    def _preprocess_q(self, u_raw_q: np.ndarray) -> np.ndarray:
        """Fixed-point ((u_raw_q - offset_q) * scaling_q) >> weight_frac.

        Mirrors `_IntLowerer._lower_preprocess` so the Python reference and
        the LLVM kernel agree bit-exactly when `input_offset != 0` or
        `input_scaling != 1`.
        """
        cfg = self.config
        rc = self.qmodel.rc
        offset_q = int(round(rc.input.input_offset * cfg.input_scale))
        scaling_q = int(round(rc.input.input_scaling * cfg.weight_scale))
        diff = u_raw_q.astype(np.int64) - offset_q
        u_pre_64 = (diff * scaling_q) >> cfg.weight_frac
        return trunc_i32(u_pre_64)

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Run the entire trajectory; X is a (T, K) float array."""
        if X.ndim == 1:
            X = X[:, None]
        cfg = self.config

        T = X.shape[0]
        Y_q = np.zeros((T, self.qmodel.M), dtype=np.int32)

        for t in range(T):
            u_raw_q = self.target.quantize_input_array(X[t], cfg)
            u_pre_q = self._preprocess_q(u_raw_q)
            self.step_q(u_pre_q)
            Y_q[t] = self.predict_one_q(u_raw_q, self.state_q)

        # Dequantize Y to float (state scale → float)
        return Y_q.astype(np.float64) / cfg.state_scale

    def collect_states(self, X: np.ndarray) -> np.ndarray:
        """Run forward and return (T, N) state trajectory as floats."""
        if X.ndim == 1:
            X = X[:, None]
        cfg = self.config

        T = X.shape[0]
        H_q = np.zeros((T, self.qmodel.N), dtype=np.int32)
        for t in range(T):
            u_raw_q = self.target.quantize_input_array(X[t], cfg)
            u_pre_q = self._preprocess_q(u_raw_q)
            self.step_q(u_pre_q)
            H_q[t] = self.state_q
        return H_q.astype(np.float64) / cfg.state_scale
