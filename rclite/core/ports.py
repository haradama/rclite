"""SysML v2: package RC::Ports"""
from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Type, Union

from .profile import Distribution
from .types import Tensor, TimeSeries


class Direction(Enum):
    IN = "in"
    OUT = "out"


@dataclass
class SignalIn:
    """SysML2: port def SignalIn { in signal : TimeSeries }"""
    name: str = ""
    item_type: Type = TimeSeries
    direction: Direction = field(default=Direction.IN, init=False)


@dataclass
class SignalOut:
    """SysML2: port def SignalOut { out signal : TimeSeries }"""
    name: str = ""
    item_type: Type = TimeSeries
    direction: Direction = field(default=Direction.OUT, init=False)


Port = Union[SignalIn, SignalOut]


@dataclass
class WeightMatrix:
    """SysML2: metadata def WeightMatrix
    Stereotype attached to Synapse connections to qualify weights.
    """
    sparsity: float = 1.0
    distribution: Distribution = Distribution.NORMAL
    trainable: bool = False

    def __post_init__(self):
        if not (0.0 <= self.sparsity <= 1.0):
            raise ValueError(
                f"WeightMatrix.sparsity must be in [0,1], got {self.sparsity}"
            )


@dataclass
class Synapse:
    """SysML2: interface def Synapse { source : SignalOut; target : SignalIn; weights : Tensor }"""
    source: Port
    target: Port
    spec: WeightMatrix = field(default_factory=WeightMatrix)
    weights: Optional[Tensor] = None

    def __post_init__(self):
        if self.source.direction != Direction.OUT:
            raise ValueError(
                f"Synapse.source must be an out-port, got {self.source.direction}"
            )
        if self.target.direction != Direction.IN:
            raise ValueError(
                f"Synapse.target must be an in-port, got {self.target.direction}"
            )
