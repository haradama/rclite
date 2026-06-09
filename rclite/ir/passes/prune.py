"""Prune low-contribution reservoir nodes at IR level.

MVP criteria:
    - "readout_norm": readout contribution score per reservoir node from the
        state block of W_out (L2 norm across outputs).
    - "low_variance_or_high_corr": profile-aware score combining readout norm,
        node variance, and inverse correlation.

The pass rewrites:
  - module dimensions (N and feature_dim)
  - W_in (row prune)
  - W_res (row/col prune when present)
  - W_out (state-feature column prune)
  - IR ops carrying N / F attributes

This pass is intended to run before sparsification.
"""

from __future__ import annotations

from dataclasses import replace
import numpy as np

from ..module import Module
from ..ops import (
    Op,
    TimeLoop,
    ReservoirStep,
    BuildPhi,
    ReadoutLinear,
    FusedStepReadout,
    AccumulateState,
    FinalizeAggregate,
)
from ._ops_utils import iter_reservoir_ops


def _has_sparse_spec(ops) -> bool:
    return any(op.res_sparse is not None for op in iter_reservoir_ops(ops))


class PruneInactiveNodes:
    """Prune low-contribution reservoir nodes.

    Args:
      keep_ratio: Fraction of reservoir nodes to keep (0, 1].
      min_keep: Lower bound for kept nodes.
        criterion: "readout_norm" | "low_variance_or_high_corr".
        w_readout: Coefficient for readout-norm term in the score.
        w_variance: Coefficient for variance term in the score.
        w_corr: Coefficient for correlation penalty term in the score.
      score_threshold: Keep any node with score >= threshold in addition to
          ratio-based top-K.
    """

    name = "rc-prune-inactive-nodes"

    def __init__(
        self,
        *,
        keep_ratio: float = 0.6,
        min_keep: int = 1,
        criterion: str = "readout_norm",
        w_readout: float = 1.0,
        w_variance: float = 1.0,
        w_corr: float = 1.0,
        score_threshold: float = 0.0,
    ):
        if not (0.0 < keep_ratio <= 1.0):
            raise ValueError(f"keep_ratio must be in (0,1], got {keep_ratio}")
        if min_keep < 1:
            raise ValueError(f"min_keep must be >=1, got {min_keep}")
        if score_threshold < 0.0:
            raise ValueError(
                f"score_threshold must be >=0, got {score_threshold}"
            )
        if criterion not in ("readout_norm", "low_variance_or_high_corr"):
            raise ValueError(
                "criterion must be 'readout_norm' or "
                f"'low_variance_or_high_corr', got {criterion!r}"
            )
        if w_readout < 0.0 or w_variance < 0.0 or w_corr < 0.0:
            raise ValueError(
                "w_readout/w_variance/w_corr must be >=0, got "
                f"{w_readout}, {w_variance}, {w_corr}"
            )
        self.keep_ratio = keep_ratio
        self.min_keep = min_keep
        self.criterion = criterion
        self.w_readout = w_readout
        self.w_variance = w_variance
        self.w_corr = w_corr
        self.score_threshold = score_threshold

    def __call__(self, module: Module) -> Module:
        old_N = int(module.N)
        if old_N <= 1 or self.keep_ratio >= 1.0:
            return module
        if _has_sparse_spec(module.ops):
            md = dict(module.metadata)
            warns = list(md.get("prune_warnings", []))
            warns.append(
                "PruneInactiveNodes skipped: sparse reservoir spec already "
                "materialized. Run prune before SparsifyReservoir."
            )
            md["prune_warnings"] = warns
            return replace(module, metadata=md)

        md = dict(module.metadata)
        include_bias = bool(md.get("include_bias", True))
        include_input = bool(md.get("include_input", False))
        K = int(module.K)
        bias_cols = 1 if include_bias else 0
        input_cols = K if include_input else 0
        state_off = bias_cols + input_cols

        W_out = np.asarray(module.weights["W_out"])
        state_block = W_out[:, state_off : state_off + old_N]
        if state_block.shape[1] != old_N:
            raise ValueError(
                "W_out state block width does not match module.N: "
                f"{state_block.shape[1]} vs {old_N}"
            )

        score = self._node_score(md, state_block, old_N)
        keep_by_ratio = max(
            self.min_keep, int(np.ceil(old_N * self.keep_ratio))
        )
        keep_by_ratio = min(keep_by_ratio, old_N)

        order = np.argsort(-score, kind="mergesort")
        keep_top = order[:keep_by_ratio]
        keep_thr = (
            np.flatnonzero(score >= self.score_threshold)
            if self.score_threshold > 0.0
            else np.array([], dtype=np.int64)
        )
        keep_idx = np.union1d(keep_top, keep_thr).astype(np.int64)
        keep_idx.sort()
        if keep_idx.size == 0:
            keep_idx = np.array([int(np.argmax(score))], dtype=np.int64)

        new_N = int(keep_idx.size)
        if new_N == old_N:
            return module

        weights = dict(module.weights)
        if "W_in" in weights:
            weights["W_in"] = np.asarray(weights["W_in"])[keep_idx, :]
        if "W_res" in weights:
            W_res = np.asarray(weights["W_res"])
            weights["W_res"] = W_res[np.ix_(keep_idx, keep_idx)]

        keep_state_cols = state_off + keep_idx
        prefix = np.arange(state_off, dtype=np.int64)
        keep_cols = np.concatenate([prefix, keep_state_cols])
        weights["W_out"] = W_out[:, keep_cols]

        old_F = int(md.get("feature_dim", W_out.shape[1]))
        new_F = old_F - (old_N - new_N)
        new_ops = [
            self._rewrite_op(op, old_N, new_N, old_F, new_F)
            for op in module.ops
        ]

        md["feature_dim"] = new_F
        md["pruned_nodes"] = int(old_N - new_N)
        md["kept_nodes"] = new_N
        md["kept_indices"] = keep_idx.tolist()
        md["prune_criterion"] = self.criterion
        md["prune_score_weights"] = {
            "readout": float(self.w_readout),
            "variance": float(self.w_variance),
            "corr": float(self.w_corr),
        }
        md["prune_keep_ratio_effective"] = float(new_N / old_N)

        return Module(
            K=module.K,
            N=new_N,
            M=module.M,
            weights=weights,
            ops=new_ops,
            metadata=md,
        )

    def _rewrite_op(
        self,
        op: Op,
        old_N: int,
        new_N: int,
        old_F: int,
        new_F: int,
    ) -> Op:
        if isinstance(op, TimeLoop):
            return replace(
                op,
                body=tuple(
                    self._rewrite_op(o, old_N, new_N, old_F, new_F)
                    for o in op.body
                ),
            )
        if isinstance(op, ReservoirStep) and op.N == old_N:
            return replace(op, N=new_N)
        if isinstance(op, BuildPhi) and op.N == old_N:
            return replace(op, N=new_N)
        if isinstance(op, ReadoutLinear) and op.F == old_F:
            return replace(op, F=new_F)
        if isinstance(op, FusedStepReadout):
            nn = new_N if op.N == old_N else op.N
            ff = new_F if op.F == old_F else op.F
            if nn != op.N or ff != op.F:
                return replace(op, N=nn, F=ff)
        if isinstance(op, AccumulateState) and op.N == old_N:
            return replace(op, N=new_N)
        if isinstance(op, FinalizeAggregate) and op.N == old_N:
            return replace(op, N=new_N)
        return op

    def _node_score(self, md, state_block, N: int):
        rw = np.linalg.norm(state_block, axis=0)
        if self.criterion == "readout_norm":
            return self.w_readout * rw

        # Profile-aware criterion:
        # high readout norm + high variance - high correlation.
        prof = md.get("profile_stats")
        if not isinstance(prof, dict):
            warns = list(md.get("prune_warnings", []))
            warns.append(
                "PruneInactiveNodes: profile_stats missing; falling back to "
                "readout_norm criterion."
            )
            md["prune_warnings"] = warns
            return rw

        var = prof.get("node_variance", prof.get("variance"))
        corr = prof.get(
            "correlation_with_other_nodes",
            prof.get("mean_abs_corr", prof.get("corr_with_others")),
        )
        if var is None or corr is None:
            warns = list(md.get("prune_warnings", []))
            warns.append(
                "PruneInactiveNodes: profile_stats missing node_variance/"
                "correlation fields; falling back to readout_norm criterion."
            )
            md["prune_warnings"] = warns
            return rw

        var = np.asarray(var, dtype=np.float64).reshape(-1)
        corr = np.asarray(corr, dtype=np.float64).reshape(-1)
        if var.size != N or corr.size != N:
            warns = list(md.get("prune_warnings", []))
            warns.append(
                "PruneInactiveNodes: profile_stats length mismatch; falling "
                "back to readout_norm criterion."
            )
            md["prune_warnings"] = warns
            return rw

        eps = 1e-12
        rw_n = rw / (np.max(np.abs(rw)) + eps)
        var_n = var / (np.max(np.abs(var)) + eps)
        corr_n = corr / (np.max(np.abs(corr)) + eps)
        return (
            self.w_readout * rw_n
            + self.w_variance * var_n
            - self.w_corr * corr_n
        )
