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
class SparseSpec:
    """Compile-time sparsity plan for a dense W_res matvec.

    Produced by the `SparsifyReservoir` pass when a RANDOM/ESN_STANDARD
    reservoir's recurrent matrix has many exact zeros. The lowering skips
    the zero MACs, preserving bit-exactness with the dense kernel because
    nonzeros are visited in increasing column order and `acc + 0.0 == acc`.

    kind == "unroll":
        Weights are baked as constants. `rows[i]` is the tuple of
        (col_j, weight) nonzeros for output row i, in ascending col_j.
        No W_res global is emitted.

    kind == "csr":
        Compressed sparse row arrays are emitted as module weights and
        referenced by name: `val_name` (float values), `col_name` (i32
        column indices), `rowptr_name` (i32, length N+1).
    """
    kind: str  # "unroll" | "csr"
    nnz: int
    rows: Tuple[Tuple[Tuple[int, float], ...], ...] = ()
    val_name: str = ""
    col_name: str = ""
    rowptr_name: str = ""


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
    res_sparse: Optional[SparseSpec] = None


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
    res_sparse: Optional[SparseSpec] = None


@dataclass(frozen=True)
class Argmax(Op):
    """class_id := argmax_m y[m]   (writes one int32 per output row)

    Classification head. Consumes the M linear scores produced by the
    readout and emits the index of the largest. Monotone, so it is exact
    under any quantization of the readout.
    """
    M: int


@dataclass(frozen=True)
class Softmax(Op):
    """p[m] := exp(y[m] - max) / sum_j exp(y[j] - max)   (M probabilities)

    Classification head producing calibrated class probabilities. The
    max-subtraction keeps exp() in range; the float path calls libm exp.
    """
    M: int


@dataclass(frozen=True)
class AccumulateState(Op):
    """Pool reservoir state over time (sequence-to-label).

    Lives inside the TimeLoop body. mode="mean" adds h into a running sum
    over post-washout steps (t >= min(washout, T-1)); mode="last" is a no-op
    (the final h is already in place). Paired with `FinalizeAggregate`.
    """
    N: int
    mode: str  # "mean" | "last"
    washout: int = 0


@dataclass(frozen=True)
class FinalizeAggregate(Op):
    """Finish time pooling after the TimeLoop: write the pooled state into h.

    mode="mean" divides the running sum by the number of pooled steps
    (T - min(washout, T-1)); mode="last" is a no-op. The following BuildPhi /
    ReadoutLinear then run once on the pooled state, producing a single
    output row for the whole sequence.
    """
    N: int
    mode: str  # "mean" | "last"
    washout: int = 0


@dataclass(frozen=True)
class TimeLoop(Op):
    """`for t in 0..T: <body>`

    `unroll` is a lowering hint: emit `unroll` body copies per iteration
    on the strided range, with a tail loop for the remainder.
    """
    body: Tuple[Op, ...]
    unroll: int = 1
