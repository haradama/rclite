"""SysML v2: package RC::Profile

Enumerations corresponding to the SysML v2 enum defs and stereotype tags.
"""
from enum import Enum


class Distribution(Enum):
    """SysML2: enum def Distribution"""
    UNIFORM = "uniform"
    NORMAL = "normal"
    BERNOULLI = "bernoulli"


class Activation(Enum):
    """SysML2: enum def Activation"""
    TANH = "tanh"
    SIGMOID = "sigmoid"
    RELU = "relu"
    IDENTITY = "identity"
    LEAKY_INTEGRATOR = "leakyIntegrator"
    SPIKING = "spiking"


class Topology(Enum):
    """SysML2: enum def Topology

    Random/ESN_STANDARD are the classical Jaeger-style reservoir.
    DLR / DLRB / SCR are the structured, minimum-complexity reservoirs of
    Rodan & Tino (2011) "Minimum complexity echo state network" — fully
    deterministic and competitive with random reservoirs.
    """
    RANDOM = "random"
    SMALL_WORLD = "smallWorld"
    SCALE_FREE = "scaleFree"
    RING = "ring"
    ESN_STANDARD = "ESNStandard"
    DLR = "DLR"      # Delay Line Reservoir (Rodan-Tino 2011)
    DLRB = "DLRB"    # Delay Line with Backward connections
    SCR = "SCR"      # Simple Cycle Reservoir


class Trainer(Enum):
    """SysML2: enum def Trainer"""
    RIDGE = "ridge"
    PINV = "pinv"
    FORCE = "FORCE"
    LMS = "LMS"
    RLS = "RLS"


class DType(Enum):
    """SysML2: enum def DType"""
    FLOAT16 = "float16"
    FLOAT32 = "float32"
    FLOAT64 = "float64"
    INT8 = "int8"
