"""rclite IR operation definitions.

Each op is an immutable dataclass that captures one logical step of the
reservoir-computer inference. Ops are sequenced inside a `TimeLoop`.
The lowering visitor (`rclite.codegen.llvm`) emits target code per op
class; passes (`rclite.ir.passes`) rewrite the op sequence to encode
RC-specific structural optimizations.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple

from rclite.core.profile import Topology


@dataclass(frozen=True)
class Op:
    """Base class for all rclite IR ops."""


@dataclass(frozen=True)
class PreprocessInput(Op):
    """u_pre := (u_raw - offset) * scale"""
    offset: float
    scale: float
    K: int


@dataclass(frozen=True)
class ReservoirStep(Op):
    """h := (1-leak)*h + leak*tanh(W_res*h + W_in*u_pre + bias)

    Lowered with topology-specific kernels:
      RANDOM / ESN_STANDARD: dense W_res matmul
      DLR / DLRB / SCR:      O(N) scalar chain (W_res unused)
    """
    leak: float
    bias: float
    N: int
    K: int
    topology: Topology
    chain_weight: float = 0.0
    chain_feedback: float = 0.0
    W_in_name: str = "W_in"
    W_res_name: Optional[str] = "W_res"


@dataclass(frozen=True)
class BuildPhi(Op):
    """phi := [1?] ++ [u_raw?] ++ h

    Materializes the readout feature vector in a buffer.
    Eliminated by `FuseStepReadout` when followed by `ReadoutLinear`.
    """
    include_bias: bool
    include_input: bool
    K: int
    N: int


@dataclass(frozen=True)
class ReadoutLinear(Op):
    """y := W_out @ phi  (phi already materialized by BuildPhi)"""
    M: int
    F: int
    W_out_name: str = "W_out"


@dataclass(frozen=True)
class FusedStepReadout(Op):
    """ReservoirStep + BuildPhi + ReadoutLinear collapsed.

    The readout's inner loop indexes W_out columns directly:
      - column 0           = bias term  (if include_bias_phi)
      - columns [1, 1+K)   = input pass-through  (if include_input_phi)
      - columns [1+K, F)   = reservoir state h
    No phi buffer is allocated.
    """
    leak: float
    bias: float
    N: int
    K: int
    M: int
    F: int
    topology: Topology
    chain_weight: float = 0.0
    chain_feedback: float = 0.0
    include_bias_phi: bool = True
    include_input_phi: bool = True
    W_in_name: str = "W_in"
    W_res_name: Optional[str] = "W_res"
    W_out_name: str = "W_out"


@dataclass(frozen=True)
class TimeLoop(Op):
    """`for t in 0..T: <body>`

    `unroll` is a lowering hint: emit `unroll` body copies per iteration
    on the strided range, with a tail loop for the remainder.
    """
    body: Tuple[Op, ...]
    unroll: int = 1
