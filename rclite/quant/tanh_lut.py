"""Tanh lookup table specification.

A `TanhLUTSpec` defines the domain ([xmin, xmax]) and resolution (n entries)
of a precomputed tanh table. The table itself is built lazily — both
float and quantized variants are available so the IR and the runtime can
share the same spec.
"""
from __future__ import annotations
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TanhLUTSpec:
    xmin: float = -4.0
    xmax: float = 4.0
    n: int = 128

    def __post_init__(self):
        if self.n < 2:
            raise ValueError(f"TanhLUTSpec.n must be >= 2, got {self.n}")
        if self.xmax <= self.xmin:
            raise ValueError(
                f"TanhLUTSpec.xmax ({self.xmax}) must be > xmin ({self.xmin})"
            )

    def build_table_f32(self) -> np.ndarray:
        """Precompute tanh values at n evenly-spaced points in [xmin, xmax]."""
        xs = np.linspace(self.xmin, self.xmax, self.n, dtype=np.float64)
        return np.tanh(xs).astype(np.float32)

    def build_table_int(self, state_scale: int, dtype: np.dtype = np.int32
                         ) -> np.ndarray:
        """Quantized table: tanh values multiplied by `state_scale`."""
        return np.clip(
            np.rint(self.build_table_f32().astype(np.float64) * state_scale),
            np.iinfo(dtype).min, np.iinfo(dtype).max,
        ).astype(dtype)
