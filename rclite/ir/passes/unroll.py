"""TimeUnroll: hint the lowering to unroll the t-loop by `K`.

Concrete pseudo-code emitted by the lowering when K > 1:

    T_unrolled = (T / K) * K
    for t_base in 0..T_unrolled step K:
        body[t = t_base + 0]
        body[t = t_base + 1]
        ...
        body[t = t_base + K-1]
    for t in T_unrolled..T:
        body[t = t]

This is the same shape as rc-bench's `Unroll2ElideCopyConstT` but with a
parameterizable K and no need for buffer ping-pong (we already update h
in place).
"""
from __future__ import annotations
from dataclasses import dataclass, replace

from ..module import Module
from ..ops import TimeLoop


@dataclass
class TimeUnroll:
    K: int = 2
    name: str = "rc-time-unroll"

    def __post_init__(self):
        if self.K < 1:
            raise ValueError(f"TimeUnroll K must be >= 1, got {self.K}")

    def __call__(self, module: Module) -> Module:
        if self.K == 1:
            return module
        new_ops = []
        for op in module.ops:
            if isinstance(op, TimeLoop):
                new_ops.append(replace(op, unroll=self.K))
            else:
                new_ops.append(op)
        return replace(module, ops=new_ops)
