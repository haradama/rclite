"""Calibration: run the float reservoir on representative inputs and
derive an `AffineQuantConfig` from the observed tensor ranges.

The float trace gives us:
  - input X            min/max → input params (asymmetric)
  - u_pre = (X-off)*sc min/max → u_pre params (asymmetric)
  - state h            absmax  → state params (symmetric, since h ≈ tanh)
  - pre = bias + ...   min/max → pre params (asymmetric, bias shifts mean)
  - output y           min/max → output params (asymmetric)

Weights get symmetric absmax (TFLM convention, zero_point=0).
"""

from __future__ import annotations
from typing import Optional

import numpy as np

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from .types import AffineParams, AffineQuantConfig


def _collect_float_traces(
    rc: ReservoirComputer, exe: RCExecutor, X: np.ndarray
) -> dict:
    """Replay the float reservoir; return per-step intermediates.

    Returns dict with arrays:
      - "X"     : raw input,           shape (T, K)
      - "U_pre" : preprocessed input,  shape (T, K)
      - "H"     : reservoir state,     shape (T, N)
      - "PRE"   : pre-activation,      shape (T, N)
      - "Y"     : readout output,      shape (T, M)
    """
    if X.ndim == 1:
        X = X[:, None]
    T, K = X.shape
    N = rc.reservoir.units
    M = rc.readout.units

    U_pre = (
        X.astype(np.float64) - rc.input.input_offset
    ) * rc.input.input_scaling

    H = np.zeros((T, N), dtype=np.float64)
    PRE = np.zeros((T, N), dtype=np.float64)
    Y = np.zeros((T, M), dtype=np.float64)

    leak = float(rc.reservoir.leak_rate)
    bias = float(rc.reservoir.bias)
    state = np.zeros(N, dtype=np.float64)
    W_in = exe.W_in.astype(np.float64)
    W_res = exe.W_res.astype(np.float64)
    W_out = exe.W_out.astype(np.float64) if exe.W_out is not None else None

    for t in range(T):
        pre = bias + W_in @ U_pre[t] + W_res @ state
        PRE[t] = pre
        activated = np.tanh(pre)
        state = (1.0 - leak) * state + leak * activated
        H[t] = state
        if W_out is not None:
            phi_parts = []
            if rc.readout.include_bias:
                phi_parts.append(np.array([1.0]))
            if rc.readout.include_input:
                phi_parts.append(X[t])
            phi_parts.append(state)
            phi = np.concatenate(phi_parts)
            Y[t] = W_out @ phi

    return {"X": X, "U_pre": U_pre, "H": H, "PRE": PRE, "Y": Y}


def calibrate_from_data(
    rc: ReservoirComputer,
    exe: RCExecutor,
    X: np.ndarray,
    *,
    storage_bits: int = 8,
    w_out_storage_bits: Optional[int] = None,
    washout: Optional[int] = None,
    per_channel_W_res: bool = False,
    per_channel_W_out: bool = False,
) -> AffineQuantConfig:
    """Build an `AffineQuantConfig` from float traces on calibration data.

    `washout` defaults to the readout's washout (so transient values don't
    skew the activation/state ranges).

    `w_out_storage_bits` selects a (typically wider) storage width for the
    readout weights — the mixed-precision path. Defaults to `storage_bits`.
    Setting it to 16 while `storage_bits=8` keeps the reservoir at i8 but
    represents W_out at i16, which on single-output reservoirs recovers most
    of the accuracy lost to readout-coefficient quantization (the measured
    i8 bottleneck) at a tiny footprint cost.
    """
    if w_out_storage_bits is None:
        w_out_storage_bits = storage_bits
    if exe.W_out is None:
        raise ValueError(
            "calibration needs a trained readout; call exe.fit() first"
        )
    if washout is None:
        washout = int(rc.readout.washout)

    traces = _collect_float_traces(rc, exe, X)

    # Skip washout for activation stats
    sl = slice(washout, None)
    X_eff = traces["X"][sl]
    U_pre_eff = traces["U_pre"][sl]
    H_eff = traces["H"][sl]
    PRE_eff = traces["PRE"][sl]
    Y_eff = traces["Y"][sl]

    # Split W_out per column block (mirage layout) to avoid one tiny
    # bias coefficient being crushed by a huge state coefficient.
    K = rc.input.units
    N = rc.reservoir.units
    off = 0
    W_out_bias_p = None
    W_out_input_p = None
    if rc.readout.include_bias:
        W_out_bias_p = AffineParams.symmetric_absmax(
            exe.W_out[:, 0:1], w_out_storage_bits
        )
        off = 1
    if rc.readout.include_input:
        W_out_input_p = AffineParams.symmetric_absmax(
            exe.W_out[:, off : off + K], w_out_storage_bits
        )
        off += K
    W_out_state_p = AffineParams.symmetric_absmax(
        exe.W_out[:, off : off + N], w_out_storage_bits
    )

    # Per-channel W_res: one symmetric scale per reservoir row (output axis).
    # `W_res` param stays a valid per-tensor representative (unused on the
    # per-channel path); `W_res_scales` carries the per-row scales.
    W_res_scales = (
        AffineParams.symmetric_absmax_peraxis(exe.W_res, storage_bits)
        if per_channel_W_res
        else None
    )

    # Per-channel W_out: one symmetric scale per output row (output axis),
    # computed per column block (mirror of the per-tensor block layout).
    W_out_bias_scales = W_out_input_scales = W_out_state_scales = None
    if per_channel_W_out:
        offc = 0
        if rc.readout.include_bias:
            W_out_bias_scales = AffineParams.symmetric_absmax_peraxis(
                exe.W_out[:, 0:1], w_out_storage_bits
            )
            offc = 1
        if rc.readout.include_input:
            W_out_input_scales = AffineParams.symmetric_absmax_peraxis(
                exe.W_out[:, offc : offc + K], w_out_storage_bits
            )
            offc += K
        W_out_state_scales = AffineParams.symmetric_absmax_peraxis(
            exe.W_out[:, offc : offc + N], w_out_storage_bits
        )

    return AffineQuantConfig(
        input=AffineParams.asymmetric_minmax(X_eff, storage_bits),
        u_pre=AffineParams.asymmetric_minmax(U_pre_eff, storage_bits),
        # state covers both stored h and post-tanh activated values; absmax
        # over the actual trajectory keeps both bounded
        state=AffineParams.symmetric_absmax(H_eff, storage_bits),
        pre=AffineParams.asymmetric_minmax(PRE_eff, storage_bits),
        W_in=AffineParams.symmetric_absmax(exe.W_in, storage_bits),
        W_res=AffineParams.symmetric_absmax(exe.W_res, storage_bits),
        W_out_bias=W_out_bias_p,
        W_out_input=W_out_input_p,
        W_out_state=W_out_state_p,
        output=AffineParams.asymmetric_minmax(Y_eff, storage_bits),
        W_res_scales=W_res_scales,
        W_out_bias_scales=W_out_bias_scales,
        W_out_input_scales=W_out_input_scales,
        W_out_state_scales=W_out_state_scales,
    )
