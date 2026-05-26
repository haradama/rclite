"""Abstract code-generation backend interface.

A backend takes a trained `ReservoirComputer` (and its `RCExecutor` holding
the materialized weight matrices) and produces a callable object exposing
the `.predict(X)` shape-compatible with `RCExecutor.predict`.
"""
from __future__ import annotations
from typing import Protocol, runtime_checkable

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor


@runtime_checkable
class CompiledModel(Protocol):
    def predict(self, X): ...


@runtime_checkable
class Backend(Protocol):
    name: str

    def compile(self, rc: ReservoirComputer, exe: RCExecutor) -> CompiledModel: ...
