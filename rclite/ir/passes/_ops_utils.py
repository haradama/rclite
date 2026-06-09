from __future__ import annotations

from typing import Iterable, Iterator

from ..ops import Op, ReservoirStep, FusedStepReadout, TimeLoop


def iter_reservoir_ops(
    ops: Iterable[Op],
) -> Iterator[ReservoirStep | FusedStepReadout]:
    """Yield reservoir-step-like ops recursively through nested TimeLoop."""
    for op in ops:
        if isinstance(op, TimeLoop):
            yield from iter_reservoir_ops(op.body)
            continue
        if isinstance(op, (ReservoirStep, FusedStepReadout)):
            yield op
