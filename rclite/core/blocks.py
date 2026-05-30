"""SysML v2: package RC::Blocks"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional

from .profile import Activation, Aggregation, Distribution, Task, Topology, Trainer
from .ports import SignalIn, SignalOut


@dataclass
class Layer:
    """SysML2: part def Layer { units, activation; port in_; port out_ }"""
    units: int
    activation: Activation = Activation.TANH
    name: str = ""

    in_: SignalIn = field(init=False)
    out_: SignalOut = field(init=False)

    def __post_init__(self):
        if self.units <= 0:
            raise ValueError(f"Layer.units must be positive, got {self.units}")
        prefix = f"{self.name}." if self.name else ""
        self.in_ = SignalIn(name=f"{prefix}in_")
        self.out_ = SignalOut(name=f"{prefix}out_")


@dataclass
class InputNode(Layer):
    """SysML2: part def InputNode :> Layer  @InputLayer

    Preprocessing semantics:
        u_pre(t) = (u(t) - input_offset) * input_scaling

    `input_distribution` controls how the input weight matrix W_in is sampled
    in the runtime. Setting it to BERNOULLI yields the deterministic ±v signs
    used by the Rodan-Tino minimum-complexity reservoirs.
    """
    input_scaling: float = 1.0
    input_offset: float = 0.0
    input_distribution: Distribution = Distribution.NORMAL


@dataclass
class ReservoirNode(Layer):
    """SysML2: part def ReservoirNode :> Layer  @Reservoir

    Dynamics (leaky):
        h(t+1) = (1 - leak_rate) h(t) + leak_rate * activation(W h + W_in u + bias)
    With leak_rate=1.0 this reduces to the classical (non-leaky) ESN.

    For structured topologies (DLR/DLRB/SCR, Rodan-Tino 2011):
      - `chain_weight` is the weight along the chain
      - `chain_feedback` is the backward weight (DLRB only)
      - `spectral_radius` is ignored at construction (informational)
    """
    spectral_radius: float = 0.9
    leak_rate: float = 1.0
    density: float = 0.1
    topology: Topology = Topology.RANDOM
    bias: float = 0.0
    seed: int = 0
    has_feedback: bool = False
    chain_weight: float = 0.5
    chain_feedback: float = 0.05

    fb_: Optional[SignalIn] = field(init=False, default=None)

    def __post_init__(self):
        super().__post_init__()
        if self.spectral_radius < 0:
            raise ValueError(
                f"ReservoirNode.spectral_radius must be >= 0, got {self.spectral_radius}"
            )
        if not (0.0 < self.leak_rate <= 1.0):
            raise ValueError(
                f"ReservoirNode.leak_rate must be in (0,1], got {self.leak_rate}"
            )
        if not (0.0 <= self.density <= 1.0):
            raise ValueError(
                f"ReservoirNode.density must be in [0,1], got {self.density}"
            )
        if self.is_structured() and not (0.0 <= abs(self.chain_weight) <= 10.0):
            raise ValueError(
                f"ReservoirNode.chain_weight unreasonable: {self.chain_weight}"
            )
        if self.has_feedback:
            prefix = f"{self.name}." if self.name else ""
            self.fb_ = SignalIn(name=f"{prefix}fb_")

    def is_structured(self) -> bool:
        return self.topology in (Topology.DLR, Topology.DLRB, Topology.SCR)


@dataclass
class ReadoutNode(Layer):
    """SysML2: part def ReadoutNode :> Layer  @Readout

    Batch trainers (RIDGE / PINV) consume `regularization` and `washout`.
    Online trainers (RLS / LMS / FORCE) consume `learning_rate` (LMS),
    `forgetting_factor` (RLS / FORCE), and `init_variance` (initial P
    matrix for RLS-family trainers as `P_0 = init_variance^{-1} I`).

    `task` selects regression (default) vs classification. For
    CLASSIFICATION, `units` is the number of classes C (one linear output per
    class); the readout is trained on one-hot targets via RIDGE / PINV, and
    class id / probabilities are recovered with argmax / softmax.

    `aggregation` pools reservoir states over time. NONE keeps the per-step
    readout; MEAN / LAST collapse a whole sequence to one feature vector so a
    sequence maps to a single label (or scalar).
    """
    trainer: Trainer = Trainer.RIDGE
    regularization: float = 1e-6
    washout: int = 100
    include_bias: bool = True
    include_input: bool = False
    learning_rate: float = 1e-2
    forgetting_factor: float = 1.0
    init_variance: float = 1e-3
    task: Task = Task.REGRESSION
    aggregation: Aggregation = Aggregation.NONE

    def __post_init__(self):
        super().__post_init__()
        if self.task == Task.CLASSIFICATION:
            if self.units < 2:
                raise ValueError(
                    "ReadoutNode.units must be >= 2 for classification "
                    f"(one output per class; got {self.units}). Binary "
                    "classification uses C=2 one-hot outputs."
                )
            if self.trainer not in (Trainer.RIDGE, Trainer.PINV):
                raise ValueError(
                    "Classification supports only batch least-squares "
                    f"trainers (RIDGE / PINV); got {self.trainer.name}. "
                    "Online classification is not implemented."
                )
        if self.washout < 0:
            raise ValueError(
                f"ReadoutNode.washout must be >= 0, got {self.washout}"
            )
        if self.regularization < 0:
            raise ValueError(
                f"ReadoutNode.regularization must be >= 0, got {self.regularization}"
            )
        if not (0.0 < self.forgetting_factor <= 1.0):
            raise ValueError(
                f"ReadoutNode.forgetting_factor must be in (0,1], "
                f"got {self.forgetting_factor}"
            )
        if self.learning_rate <= 0:
            raise ValueError(
                f"ReadoutNode.learning_rate must be > 0, got {self.learning_rate}"
            )
        if self.init_variance <= 0:
            raise ValueError(
                f"ReadoutNode.init_variance must be > 0, got {self.init_variance}"
            )
