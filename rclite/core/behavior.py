"""SysML v2: package RC::Behavior"""
from __future__ import annotations
from dataclasses import dataclass
from enum import Enum
from typing import Optional

from .composite import ReservoirComputer
from .types import TimeSeries


class Mode(Enum):
    IDLE = "idle"
    TRAINING = "training"
    INFERRING = "inferring"


@dataclass
class Train:
    """SysML2: action def Train { in X, in Y, in rc, out W }"""
    X: TimeSeries
    Y: TimeSeries
    rc: ReservoirComputer


@dataclass
class Infer:
    """SysML2: action def Infer { in X, in rc, out Yhat }"""
    X: TimeSeries
    rc: ReservoirComputer


@dataclass
class RCMode:
    """SysML2: state def RCMode

    Transitions:
        idle      --"fit"-->     training
        idle      --"predict"--> inferring
        training  --"done"-->    idle
        inferring --"done"-->    idle
    """
    state: Mode = Mode.IDLE

    def fit(self) -> None:
        self._require(Mode.IDLE, transition="fit")
        self.state = Mode.TRAINING

    def predict(self) -> None:
        self._require(Mode.IDLE, transition="predict")
        self.state = Mode.INFERRING

    def done(self) -> None:
        if self.state == Mode.IDLE:
            raise RuntimeError("Cannot signal 'done' while idle")
        self.state = Mode.IDLE

    def _require(self, expected: Mode, transition: str) -> None:
        if self.state != expected:
            raise RuntimeError(
                f"Invalid transition '{transition}' from state {self.state.name}"
            )
