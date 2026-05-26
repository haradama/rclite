"""SysML v2: package RC::Composite"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

from .blocks import InputNode, ReservoirNode, ReadoutNode
from .ports import Synapse, WeightMatrix
from .profile import Distribution


@dataclass
class ReservoirComputer:
    """SysML2: part def ReservoirComputer

    Holds the three layer parts and the four named Synapse connections
    (W_in, W_res, W_out, optional W_fb).
    """
    input: InputNode
    reservoir: ReservoirNode
    readout: ReadoutNode

    W_in: Synapse = field(init=False)
    W_res: Synapse = field(init=False)
    W_out: Synapse = field(init=False)
    W_fb: Optional[Synapse] = field(init=False, default=None)

    def __post_init__(self):
        self.W_in = Synapse(
            source=self.input.out_,
            target=self.reservoir.in_,
            spec=WeightMatrix(
                trainable=False,
                sparsity=1.0,
                distribution=self.input.input_distribution,
            ),
        )
        self.W_res = Synapse(
            source=self.reservoir.out_,
            target=self.reservoir.in_,
            spec=WeightMatrix(
                trainable=False,
                sparsity=self.reservoir.density,
                distribution=Distribution.NORMAL,
            ),
        )
        self.W_out = Synapse(
            source=self.reservoir.out_,
            target=self.readout.in_,
            spec=WeightMatrix(trainable=True),
        )
        if self.reservoir.has_feedback and self.reservoir.fb_ is not None:
            self.W_fb = Synapse(
                source=self.readout.out_,
                target=self.reservoir.fb_,
                spec=WeightMatrix(trainable=False),
            )

    def synapses(self) -> List[Synapse]:
        out = [self.W_in, self.W_res, self.W_out]
        if self.W_fb is not None:
            out.append(self.W_fb)
        return out

    def parts(self):
        return {"input": self.input, "reservoir": self.reservoir, "readout": self.readout}
