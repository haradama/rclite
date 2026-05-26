"""Convert a trained float reservoir computer into a QuantizedModel."""
from __future__ import annotations
from typing import Optional

import numpy as np

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from .config import QuantConfig
from .target import QuantTarget, I32FixedPoint
from .tanh_lut import TanhLUTSpec
from .model import QuantizedModel


def quantize_model(
    rc: ReservoirComputer,
    exe: RCExecutor,
    config: QuantConfig,
    *,
    target: Optional[QuantTarget] = None,
    lut: Optional[TanhLUTSpec] = None,
) -> QuantizedModel:
    """Quantize a trained `RCExecutor` according to `config`.

    Output weight encoding follows the mirage scheme so the readout can
    accumulate all three contributions (bias, input pass-through, state)
    at one Q.state_frac scale.
    """
    if exe.W_out is None:
        raise ValueError("Readout has not been trained — call exe.fit() first")
    if target is None:
        target = I32FixedPoint()
    if lut is None:
        lut = TanhLUTSpec()

    W_in_q = target.quantize_weight_array(exe.W_in, config)
    W_res_q = target.quantize_weight_array(exe.W_res, config)
    W_out_q = quantize_W_out(exe.W_out, rc, config, target)

    lut_table_q = lut.build_table_int(config.state_scale,
                                        dtype=target.storage_dtype)

    return QuantizedModel(
        rc=rc, target=target, config=config, lut=lut,
        W_in_q=W_in_q, W_res_q=W_res_q, W_out_q=W_out_q,
        lut_table_q=lut_table_q,
    )


def quantize_W_out(W_out: np.ndarray, rc: ReservoirComputer,
                    cfg: QuantConfig, target: QuantTarget) -> np.ndarray:
    """mirage-style mixed-scale W_out quantization.

      col 0           — bias  (if include_bias)        at state_scale
      cols [1, 1+K)   — input pass-through (if any)    at state_scale^2 / input_scale
      cols [1+K, F)   — reservoir state                at state_scale
    """
    M, F = W_out.shape
    out = np.zeros((M, F), dtype=target.storage_dtype)
    K = rc.input.units
    N = rc.reservoir.units
    off = 0
    if rc.readout.include_bias:
        out[:, off] = target.quantize_state_array(W_out[:, off], cfg)
        off += 1
    if rc.readout.include_input:
        # Input-weight scale: state_scale^2 / input_scale
        # = 2^(2*state_frac - input_frac)
        shift = 2 * cfg.state_frac - cfg.input_frac
        scale = float(1 << abs(shift))
        if shift >= 0:
            scaled = W_out[:, off:off + K].astype(np.float64) * scale
        else:
            scaled = W_out[:, off:off + K].astype(np.float64) / scale
        out[:, off:off + K] = target._saturate_array(scaled)
        off += K
    # State weights (the trailing N columns)
    out[:, off:off + N] = target.quantize_state_array(W_out[:, off:off + N], cfg)
    return out
