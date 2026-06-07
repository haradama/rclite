"""Quantization configuration.

`QuantConfig` controls the Q-format used by all three quantity classes
in the reservoir computer:
    state   — reservoir state h (Q-format: state_frac fractional bits)
    input   — preprocessed input u_pre
    weight  — W_in, W_res, W_out

Different fractional widths let small dynamic-range quantities trade
precision for headroom independently. mirage-style search optimizes
`state_frac` while pinning the others to data-derived bounds.
"""

from __future__ import annotations
from dataclasses import dataclass


@dataclass(frozen=True)
class QuantConfig:
    state_frac: int = 16
    input_frac: int = 16
    weight_frac: int = 16

    def __post_init__(self):
        for f, name in [
            (self.state_frac, "state"),
            (self.input_frac, "input"),
            (self.weight_frac, "weight"),
        ]:
            if not (0 <= f <= 30):
                raise ValueError(f"{name}_frac must be in [0, 30], got {f}")

    @property
    def state_scale(self) -> int:
        return 1 << self.state_frac

    @property
    def input_scale(self) -> int:
        return 1 << self.input_frac

    @property
    def weight_scale(self) -> int:
        return 1 << self.weight_frac

    def __repr__(self) -> str:
        return (
            f"QuantConfig(state=Q{31 - self.state_frac}.{self.state_frac}, "
            f"input=Q{31 - self.input_frac}.{self.input_frac}, "
            f"weight=Q{31 - self.weight_frac}.{self.weight_frac})"
        )
