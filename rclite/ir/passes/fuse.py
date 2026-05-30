"""FuseStepReadout: collapse Step → BuildPhi → ReadoutLinear into one op.

The readout's inner loop indexes W_out columns directly, so the phi
feature vector never needs to be materialized as a separate buffer:
saves a (1 + K + N) f32 allocation per step and removes a write/read
round-trip.
"""
from __future__ import annotations
from dataclasses import replace
from typing import List, Tuple

from ..module import Module
from ..ops import (
    Op, PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear,
    FusedStepReadout, TimeLoop,
)


class FuseStepReadout:
    name = "rc-fuse-step-readout"

    def __call__(self, module: Module) -> Module:
        new_ops = [self._fuse_op(op) for op in module.ops]
        return replace(module, ops=new_ops)

    def _fuse_op(self, op: Op) -> Op:
        if isinstance(op, TimeLoop):
            return replace(op, body=tuple(self._fuse_seq(op.body)))
        return op

    def _fuse_seq(self, body: Tuple[Op, ...]) -> List[Op]:
        out: List[Op] = []
        i = 0
        while i < len(body):
            # Pattern: PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear
            if (i + 3 < len(body)
                    and isinstance(body[i], PreprocessInput)
                    and isinstance(body[i + 1], ReservoirStep)
                    and isinstance(body[i + 2], BuildPhi)
                    and isinstance(body[i + 3], ReadoutLinear)):
                pp = body[i]
                step = body[i + 1]
                phi = body[i + 2]
                ro = body[i + 3]
                out.append(pp)
                out.append(FusedStepReadout(
                    leak=step.leak, bias=step.bias,
                    N=step.N, K=step.K, M=ro.M, F=ro.F,
                    topology=step.topology,
                    chain_weight=step.chain_weight,
                    chain_feedback=step.chain_feedback,
                    include_bias_phi=phi.include_bias,
                    include_input_phi=phi.include_input,
                    W_in_name=step.W_in_name,
                    W_res_name=step.W_res_name,
                    W_out_name=ro.W_out_name,
                    res_sparse=step.res_sparse,
                ))
                i += 4
            else:
                out.append(body[i])
                i += 1
        return out
