"""Stability-aware reservoir passes.

The passes in this module keep reservoir-structure constraints explicit in
the IR, mirroring the role of an MLIR-level `rc-normalize-reservoir` /
`rc-verify-echo-state` stage:

  - NormalizeReservoir rescales dense recurrent weights to a target
    spectral radius.
  - VerifyEchoStateConstraint checks sufficient/heuristic echo-state-style
    stability conditions and records warnings in module metadata.
"""

from __future__ import annotations

from dataclasses import replace
from typing import Dict, Iterable, List, Set

import numpy as np

from rclite.core.profile import Topology

from ..module import Module
from ..ops import Op
from ._ops_utils import DENSE_TOPOLOGIES, iter_reservoir_ops


def _collect_dense_wres_names(ops: Iterable[Op]) -> Set[str]:
    out: Set[str] = set()
    for op in iter_reservoir_ops(ops):
        if op.topology in DENSE_TOPOLOGIES and op.W_res_name:
            out.add(op.W_res_name)
    return out


def _spectral_radius(W: np.ndarray) -> float:
    vals = np.linalg.eigvals(np.asarray(W, dtype=np.float64))
    if vals.size == 0:
        return 0.0
    return float(np.max(np.abs(vals)))


class NormalizeReservoir:
    """Rescale dense W_res to match a target spectral radius.

    Target precedence:
      1. Explicit constructor argument `target_spectral_radius`.
      2. `module.metadata["spectral_radius"]` when present.

    Structured topologies (DLR / DLRB / SCR) are not modified.
    """

    name = "rc-normalize-reservoir"

    def __init__(self, target_spectral_radius: float | None = None):
        if target_spectral_radius is not None and target_spectral_radius < 0:
            raise ValueError(
                "target_spectral_radius must be >= 0, "
                f"got {target_spectral_radius}"
            )
        self.target_spectral_radius = target_spectral_radius

    def __call__(self, module: Module) -> Module:
        target = self._resolve_target(module)
        if target is None:
            return module

        weights = dict(module.weights)
        touched: Dict[str, Dict[str, float]] = {}

        for name in _collect_dense_wres_names(module.ops):
            if name not in weights:
                continue
            W = np.asarray(weights[name], dtype=np.float64)
            sr = _spectral_radius(W)
            if sr <= 0.0:
                touched[name] = {
                    "before": 0.0,
                    "after": 0.0,
                    "scale": 1.0,
                }
                continue
            scale = target / sr
            Wn = (W * scale).astype(weights[name].dtype, copy=False)
            weights[name] = Wn
            touched[name] = {
                "before": sr,
                "after": _spectral_radius(Wn),
                "scale": scale,
            }

        if not touched:
            return module

        md = dict(module.metadata)
        md["normalized_spectral_radius"] = target
        md["normalization_report"] = touched
        return replace(module, weights=weights, metadata=md)

    def _resolve_target(self, module: Module) -> float | None:
        if self.target_spectral_radius is not None:
            return float(self.target_spectral_radius)
        md_target = module.metadata.get("spectral_radius")
        if md_target is None:
            return None
        md_target = float(md_target)
        if md_target < 0:
            raise ValueError(
                "module.metadata['spectral_radius'] must be >= 0, "
                f"got {md_target}"
            )
        return md_target


class VerifyEchoStateConstraint:
    """Check sufficient/heuristic echo-state stability conditions.

    For structured topologies:
      - DLR: always accepted (nilpotent chain).
      - SCR: requires |chain_weight| < 1 (sufficient).
      - DLRB: requires |chain_weight| + |chain_feedback| < 1 (sufficient).

    For dense topologies:
      - Checks spectral_radius(W_res) < dense_radius_limit (heuristic).

    Violations are collected in metadata under "echo_state_warnings".
    If strict=True, the first violation raises ValueError.
    """

    name = "rc-verify-echo-state-constraint"

    def __init__(
        self, *, strict: bool = False, dense_radius_limit: float = 1.0
    ):
        if dense_radius_limit <= 0:
            raise ValueError(
                f"dense_radius_limit must be > 0, got {dense_radius_limit}"
            )
        self.strict = strict
        self.dense_radius_limit = dense_radius_limit

    def __call__(self, module: Module) -> Module:
        issues: List[str] = []
        weights = module.weights

        for op in iter_reservoir_ops(module.ops):
            if op.topology == Topology.DLR:
                continue

            if op.topology == Topology.SCR:
                if abs(op.chain_weight) >= 1.0:
                    issues.append(
                        "SCR sufficient condition failed: "
                        f"|chain_weight|={abs(op.chain_weight):.6g} >= 1"
                    )
                continue

            if op.topology == Topology.DLRB:
                s = abs(op.chain_weight) + abs(op.chain_feedback)
                if s >= 1.0:
                    issues.append(
                        "DLRB sufficient condition failed: "
                        f"|chain_weight|+|chain_feedback|={s:.6g} >= 1"
                    )
                continue

            if op.topology in DENSE_TOPOLOGIES and op.W_res_name:
                if op.W_res_name not in weights:
                    issues.append(
                        f"Dense topology references missing weight {op.W_res_name!r}"
                    )
                    continue
                sr = _spectral_radius(np.asarray(weights[op.W_res_name]))
                if sr >= self.dense_radius_limit:
                    issues.append(
                        "Dense topology heuristic failed: "
                        f"spectral_radius={sr:.6g} >= {self.dense_radius_limit:.6g}"
                    )

        if issues and self.strict:
            raise ValueError(issues[0])

        md = dict(module.metadata)
        md["echo_state_warnings"] = issues
        md["echo_state_verified"] = len(issues) == 0
        return replace(module, metadata=md)
