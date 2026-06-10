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
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


def affine_reference_outputs(
    qmodel, test_inputs: np.ndarray, np_storage
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quantize inputs and compute bit-exact affine reference outputs.

    Runs the float test sequence through `AffineQuantizedExecutor` — the same
    path the emitted C / LLVM kernel reproduces exactly — and returns
    ``(X_q, Y_ref_q, n_rows)`` cast to `np_storage`. A pooled readout
    (`aggregation != NONE`) yields a single output row; otherwise one row per
    step. Shared by the Arduino and NES affine targets.
    """
    from rclite.core.profile import Aggregation
    from rclite.quant.affine.executor import AffineQuantizedExecutor

    X = test_inputs[:, None] if test_inputs.ndim == 1 else test_inputs
    X_q = qmodel.config.input.quantize_array(X).astype(np_storage)
    qexe = AffineQuantizedExecutor(qmodel)
    T = X.shape[0]

    if qmodel.rc.readout.aggregation != Aggregation.NONE:
        # Sequence-to-label: pool the whole window into one readout row.
        Y_ref_q = qexe.predict_pooled_q(X)[None, :].astype(np_storage)
        return X_q, Y_ref_q, 1

    Y_ref_q = np.zeros((T, qmodel.M), dtype=np_storage)
    for t in range(T):
        x_raw_q = qexe._quantize_raw_input(X[t])
        u_pre_q = qexe._quantize_u_pre(X[t])
        qexe.step_q(u_pre_q)
        Y_ref_q[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(
            np_storage
        )
    return X_q, Y_ref_q, T


def symmetric_reference_outputs(
    qmodel, test_inputs: np.ndarray, np_storage
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Quantize inputs and compute bit-exact symmetric (Q-format) references.

    Pre-processes the float test sequence exactly as
    `CompiledQuantizedRC.predict` does (input offset/scaling then
    `quantize_input_array`), then replays it through `QuantizedExecutor` — the
    path the emitted integer kernel reproduces exactly — and returns
    ``(X_q, Y_ref_q, n_rows)`` cast to `np_storage`, one output row per step.
    Shared by the Cortex-M0 and GBA symmetric targets.
    """
    from rclite.quant.executor import QuantizedExecutor

    cfg = qmodel.config
    rc = qmodel.rc
    u_pre = (test_inputs - rc.input.input_offset) * rc.input.input_scaling
    X_q = qmodel.target.quantize_input_array(u_pre, cfg).astype(np_storage)

    qexe = QuantizedExecutor(qmodel)
    T = test_inputs.shape[0]
    Y_ref_q = np.zeros((T, qmodel.M), dtype=np_storage)
    for t in range(T):
        x_row = (
            X_q[t] if X_q.ndim > 1 else np.array([X_q[t]], dtype=np_storage)
        )
        qexe.step_q(x_row.astype(np.int32))
        Y_ref_q[t] = qexe.predict_one_q(
            x_row.astype(np.int32), qexe.state_q
        ).astype(np_storage)
    return X_q, Y_ref_q, T


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
