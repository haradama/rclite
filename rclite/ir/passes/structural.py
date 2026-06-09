"""StructuralSpecialize: encode topology specialization into the IR.

Currently the LLVM lowering branches on `topology` to emit O(N) scalar
chain code for DLR/DLRB/SCR. This pass makes the specialization explicit
at the IR level:

  - Drops the `W_res_name` reference from structured ReservoirStep /
    FusedStepReadout ops (the dense matrix is unused).
  - Removes the W_res weight tensor from the module if no op references
    it anymore — saves it from being emitted as a global constant.

Verifies the chain_weight magnitude bound for structured topologies and
errors out if it is unstable.
"""

from __future__ import annotations
from dataclasses import replace
from rclite.core.profile import Topology

from ..module import Module
from ..ops import (
    Op,
    ReservoirStep,
    FusedStepReadout,
    TimeLoop,
)
from ._ops_utils import iter_reservoir_ops


_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)


class StructuralSpecialize:
    name = "rc-structural-specialize"

    def __call__(self, module: Module) -> Module:
        new_ops = [self._fix(op) for op in module.ops]
        weights = dict(module.weights)
        if "W_res" in weights and not _module_uses_W_res(new_ops):
            del weights["W_res"]
        return Module(
            K=module.K,
            N=module.N,
            M=module.M,
            weights=weights,
            ops=new_ops,
            metadata=dict(module.metadata),
        )

    def _fix(self, op: Op) -> Op:
        if isinstance(op, TimeLoop):
            return replace(op, body=tuple(self._fix(o) for o in op.body))
        if isinstance(op, ReservoirStep):
            if op.topology in _STRUCTURED:
                _validate_chain_bounds(
                    op.topology, op.chain_weight, op.chain_feedback
                )
                if op.W_res_name is not None:
                    return replace(op, W_res_name=None)
        if isinstance(op, FusedStepReadout):
            if op.topology in _STRUCTURED:
                _validate_chain_bounds(
                    op.topology, op.chain_weight, op.chain_feedback
                )
                if op.W_res_name is not None:
                    return replace(op, W_res_name=None)
        return op


def _validate_chain_bounds(topology: Topology, cw: float, cb: float) -> None:
    if topology == Topology.DLR:
        return  # nilpotent — any chain_weight is fine
    if topology == Topology.SCR and abs(cw) >= 1.0:
        raise ValueError(f"SCR chain_weight={cw} violates |chain_weight| < 1")
    if topology == Topology.DLRB and abs(cw) + abs(cb) >= 1.0:
        raise ValueError(
            f"DLRB |chain_weight|+|chain_feedback|={abs(cw) + abs(cb)} "
            f"violates < 1"
        )


def _module_uses_W_res(ops) -> bool:
    return any(op.W_res_name is not None for op in iter_reservoir_ops(ops))
