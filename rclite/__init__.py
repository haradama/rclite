"""rclite — Reservoir Computing deployment framework for embedded systems.

Mirrors TensorFlow Lite for Microcontrollers in spirit: a model description
(`rclite.core`), reference + native runtimes (`rclite.runtime`, `rclite.codegen`),
verification passes (`rclite.verification`), and per-target deployment glue
(`rclite.targets`).

Top-level re-exports keep the most common IDL identifiers one import away:

    from rclite import ReservoirComputer, InputNode, ReservoirNode, ReadoutNode
    from rclite.runtime import RCExecutor
    from rclite.codegen import compile_rc
    from rclite.targets import HostTarget, Microbit
"""
from rclite.core import (
    Distribution, Activation, Topology, Trainer, DType,
    Tensor, TimeSeries,
    Direction, SignalIn, SignalOut, Synapse, WeightMatrix,
    Layer, InputNode, ReservoirNode, ReadoutNode,
    ReservoirComputer,
    Mode, RCMode, Train, Infer,
)
from rclite.verification import (
    ConstraintViolation, ESPChecker,
    echo_state_property, leak_range, density_range,
    WellPosedReservoir,
)

__all__ = [
    "Distribution", "Activation", "Topology", "Trainer", "DType",
    "Tensor", "TimeSeries",
    "Direction", "SignalIn", "SignalOut", "Synapse", "WeightMatrix",
    "Layer", "InputNode", "ReservoirNode", "ReadoutNode",
    "ReservoirComputer",
    "Mode", "RCMode", "Train", "Infer",
    "ConstraintViolation", "ESPChecker",
    "echo_state_property", "leak_range", "density_range",
    "WellPosedReservoir",
]
