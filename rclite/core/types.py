"""SysML v2: package RC::Types"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Tuple

from .profile import DType


@dataclass
class Tensor:
    """SysML2: item def Tensor { rank, dim, dtype }"""

    rank: int
    dim: Tuple[int, ...]
    dtype: DType = DType.FLOAT32

    def __post_init__(self):
        if isinstance(self.dim, list):
            self.dim = tuple(self.dim)
        if not isinstance(self.dim, tuple):
            raise TypeError(
                f"Tensor.dim must be a tuple, got {type(self.dim)}"
            )
        if self.rank != len(self.dim):
            raise ValueError(
                f"Tensor.rank ({self.rank}) must equal len(dim) ({len(self.dim)})"
            )
        if any(d <= 0 for d in self.dim):
            raise ValueError(
                f"Tensor.dim must be all positive, got {self.dim}"
            )


@dataclass
class TimeSeries(Tensor):
    """SysML2: item def TimeSeries :> Tensor { dt }"""

    dt: float = 1.0

    def __post_init__(self):
        super().__post_init__()
        if self.dt <= 0:
            raise ValueError(f"TimeSeries.dt must be positive, got {self.dt}")
