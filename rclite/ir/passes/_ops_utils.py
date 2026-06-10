from __future__ import annotations

from typing import Iterable, Iterator

from rclite.core.profile import Topology

from ..ops import Op, ReservoirStep, FusedStepReadout, TimeLoop


# Topology groupings shared across the reservoir passes. Structured topologies
# carry their recurrence as scalar chain weights (no dense W_res); dense ones
# reference a full W_res matrix in module.weights.
STRUCTURED_TOPOLOGIES = (Topology.DLR, Topology.DLRB, Topology.SCR)
DENSE_TOPOLOGIES = (Topology.RANDOM, Topology.ESN_STANDARD)


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
