"""Container for a trained, quantized reservoir computer.

Holds the integer-encoded weights, the quantization config and target
(so codegen can pick the right LLVM types and shift amounts), and the
optional tanh LUT.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from rclite.core.composite import ReservoirComputer
from .config import QuantConfig
from .target import QuantTarget
from .tanh_lut import TanhLUTSpec


@dataclass
class QuantizedModel:
    """A quantized reservoir computer ready for execution or export.

    `W_out_q` follows the mirage scheme:
      [0]      bias term       at state_scale
      [1..1+I) input weights   at state_scale^2 / input_scale
      [1+I..)  state weights   at state_scale
    so the readout can sum all contributions at the same Q.state_frac scale.
    """

    rc: ReservoirComputer
    target: QuantTarget
    config: QuantConfig
    lut: Optional[TanhLUTSpec]

    W_in_q: np.ndarray  # (N, K), storage dtype, weight scale
    W_res_q: np.ndarray  # (N, N), storage dtype, weight scale
    W_out_q: np.ndarray  # (M, F), storage dtype, mixed scales per column block
    lut_table_q: Optional[np.ndarray] = None  # (n,), state scale

    state_init_q: Optional[np.ndarray] = field(default=None)

    def __post_init__(self):
        N = self.rc.reservoir.units
        K = self.rc.input.units
        M = self.rc.readout.units
        F = (
            (1 if self.rc.readout.include_bias else 0)
            + (K if self.rc.readout.include_input else 0)
            + N
        )
        if self.W_in_q.shape != (N, K):
            raise ValueError(f"W_in_q shape {self.W_in_q.shape} != ({N}, {K})")
        if self.W_res_q.shape != (N, N):
            raise ValueError(
                f"W_res_q shape {self.W_res_q.shape} != ({N}, {N})"
            )
        if self.W_out_q.shape != (M, F):
            raise ValueError(
                f"W_out_q shape {self.W_out_q.shape} != ({M}, {F})"
            )
        if self.state_init_q is None:
            self.state_init_q = np.zeros(N, dtype=self.target.storage_dtype)

    @property
    def N(self) -> int:
        return self.rc.reservoir.units

    @property
    def K(self) -> int:
        return self.rc.input.units

    @property
    def M(self) -> int:
        return self.rc.readout.units

    @property
    def F(self) -> int:
        K = self.K
        return (
            (1 if self.rc.readout.include_bias else 0)
            + (K if self.rc.readout.include_input else 0)
            + self.N
        )
