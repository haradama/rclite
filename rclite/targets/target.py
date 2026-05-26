"""Abstract deployment target.

A `Target` lowers a trained `ReservoirComputer` into native machine code
suitable for a particular hardware/OS combination, produces the supporting
sources (header, startup, main, linker script, ...) needed to use it,
and optionally provides a `run()` hook that executes the artifact on an
emulator or host process.
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class CompiledArtifact:
    """Files produced by `Target.compile()`."""
    target_name: str
    output_dir: Path
    binary: Optional[Path] = None
    sources: List[Path] = field(default_factory=list)
    objects: List[Path] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class RunResult:
    """Outcome of `Target.run()`."""
    success: bool
    output: str
    returncode: int


class Target(ABC):
    """Abstract deployment target."""

    name: str = "abstract"

    @abstractmethod
    def compile(self, rc, exe, *, output_dir, **kwargs) -> CompiledArtifact:
        """Lower the trained reservoir computer to native artifacts."""

    def run(self, artifact: CompiledArtifact, **kwargs) -> RunResult:
        """Execute the artifact. Override on targets that have an emulator."""
        raise NotImplementedError(
            f"target {self.name!r} provides no runner; subclass `run()` "
            f"if an emulator/host execution path is available."
        )
