"""Core IDL: model description vocabulary (SysML v2 subset)."""
from .profile import (
    Distribution, Activation, Topology, Trainer, DType, Task, Aggregation,
)
from .types import Tensor, TimeSeries
from .ports import Direction, SignalIn, SignalOut, Synapse, WeightMatrix
from .blocks import Layer, InputNode, ReservoirNode, ReadoutNode
from .composite import ReservoirComputer
from .behavior import Mode, RCMode, Train, Infer

__all__ = [
    "Distribution", "Activation", "Topology", "Trainer", "DType",
    "Task", "Aggregation",
    "Tensor", "TimeSeries",
    "Direction", "SignalIn", "SignalOut", "Synapse", "WeightMatrix",
    "Layer", "InputNode", "ReservoirNode", "ReadoutNode",
    "ReservoirComputer",
    "Mode", "RCMode", "Train", "Infer",
]
