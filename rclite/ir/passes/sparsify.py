"""SparsifyReservoir: specialize a dense W_res matvec to its nonzeros.

For RANDOM / ESN_STANDARD reservoirs the recurrent matrix W_res is a
compile-time constant that is typically sparse (default density 0.1, so
~90% of the N*N MACs multiply by exactly zero). This pass inspects the
actual W_res values and rewrites `ReservoirStep` / `FusedStepReadout` to
carry a `SparseSpec`, so the LLVM lowering emits only the nonzero MACs.

Two kernel strategies (selected per op):

  - "unroll": bake each row's nonzero (col, weight) pairs as constants.
    No W_res global, no index loads — fastest, code size scales with nnz.
  - "csr": emit compressed-sparse-row arrays as globals and loop over the
    nonzeros per row. Code size is independent of nnz — for large N.

  - "auto": pick "unroll" when nnz <= max_unroll_nnz, else "csr".

Bit-exactness: with the default `threshold=0.0` only exact zeros are
dropped, and nonzeros are kept in ascending column order, so the partial
sums match the dense kernel bit-for-bit (adding 0.0 is the identity for
finite accumulators; ESN states are bounded by tanh, so no 0*inf=nan).
A positive `threshold` prunes near-zero weights and is therefore lossy.

Structured topologies (DLR/DLRB/SCR) are left untouched — they are
already O(N) and carry no W_res.
"""

from __future__ import annotations
from dataclasses import replace
from typing import Iterable

import numpy as np

from ..module import Module
from ..ops import (
    Op,
    ReservoirStep,
    FusedStepReadout,
    SparseSpec,
    TimeLoop,
)
from ._ops_utils import DENSE_TOPOLOGIES, iter_reservoir_ops
from .structural import StructuralSpecialize


def sparse_passes(sparse, *, include_structural: bool):
    """Build a passes list for a target's `sparse` argument.

    `sparse` accepts False/None (no sparsification → return None so callers
    fall through to their default passes), True / "auto" / "unroll" / "csr".
    `include_structural=True` prepends `StructuralSpecialize()` (needed on the
    float cross-compile path, whose default passes include it); the quantized
    paths default to no passes, so they pass include_structural=False.
    """
    if not sparse:
        return None
    strategy = "auto" if sparse is True else sparse
    base = [StructuralSpecialize()] if include_structural else []
    return base + [SparsifyReservoir(strategy=strategy)]


def count_nonzeros(W, threshold: float = 0.0) -> int:
    """Number of |entries| > threshold in the matrix W."""
    return int((np.abs(np.asarray(W)) > threshold).sum())


def pick_kind(nnz: int, strategy: str, max_unroll_nnz: int) -> str:
    """Resolve 'auto' to 'unroll'/'csr' by the nnz threshold."""
    if strategy == "auto":
        return "unroll" if nnz <= max_unroll_nnz else "csr"
    return strategy


def build_unroll_rows(W, threshold: float = 0.0):
    """Per-row nonzeros as a tuple of ((col_j, weight), ...) in ascending j.

    `weight` keeps W's native dtype scalar (Python int for integer W_res_q,
    float for float W_res), so callers can bake exact constants.
    """
    W = np.asarray(W)
    mask = np.abs(W) > threshold
    return tuple(
        tuple((int(j), W[i, j].item()) for j in np.nonzero(mask[i])[0])
        for i in range(W.shape[0])
    )


def build_csr(W, threshold: float = 0.0):
    """Return (val, col, rowptr) CSR arrays in ascending column order per row.

    `val` preserves W's dtype (int storage for quantized W_res_q, float for
    the float path); `col`/`rowptr` are int32.
    """
    W = np.asarray(W)
    N = W.shape[0]
    mask = np.abs(W) > threshold
    val, col, rowptr = [], [], [0]
    for i in range(N):
        cols = np.nonzero(mask[i])[0]
        for j in cols:
            col.append(int(j))
            val.append(W[i, j])
        rowptr.append(len(col))
    return (
        np.asarray(val, dtype=W.dtype),
        np.asarray(col, dtype=np.int32),
        np.asarray(rowptr, dtype=np.int32),
    )


class SparsifyReservoir:
    name = "rc-sparsify-reservoir"

    def __init__(
        self,
        strategy: str = "auto",
        max_unroll_nnz: int = 4096,
        threshold: float = 0.0,
    ):
        if strategy not in ("auto", "unroll", "csr"):
            raise ValueError(
                f"strategy must be 'auto'|'unroll'|'csr', got {strategy!r}"
            )
        if threshold < 0.0:
            raise ValueError(f"threshold must be >= 0, got {threshold}")
        self.strategy = strategy
        self.max_unroll_nnz = max_unroll_nnz
        self.threshold = threshold

    def __call__(self, module: Module) -> Module:
        self._weights = dict(module.weights)
        new_ops = [self._fix(op) for op in module.ops]
        # Drop the dense W_res tensors that no op references anymore.
        for name in [
            n
            for n in self._weights
            if n.startswith("W_res")
            and not n.endswith(("_val", "_col", "_rowptr"))
        ]:
            if not _module_uses(new_ops, name):
                del self._weights[name]
        return Module(
            K=module.K,
            N=module.N,
            M=module.M,
            weights=self._weights,
            ops=new_ops,
            metadata=dict(module.metadata),
        )

    def _fix(self, op: Op) -> Op:
        if isinstance(op, TimeLoop):
            return replace(op, body=tuple(self._fix(o) for o in op.body))
        if isinstance(op, (ReservoirStep, FusedStepReadout)):
            if (
                op.topology in DENSE_TOPOLOGIES
                and op.W_res_name is not None
                and op.res_sparse is None
            ):
                spec = self._build_spec(op)
                return replace(op, res_sparse=spec, W_res_name=None)
        return op

    def _build_spec(self, op) -> SparseSpec:
        W = np.asarray(self._weights[op.W_res_name])
        nnz = count_nonzeros(W, self.threshold)
        kind = pick_kind(nnz, self.strategy, self.max_unroll_nnz)

        if kind == "unroll":
            rows = build_unroll_rows(W, self.threshold)
            return SparseSpec(kind="unroll", nnz=nnz, rows=rows)

        # CSR: arrays preserve W's dtype for val, int32 for indices.
        val, col, rowptr = build_csr(W, self.threshold)
        base = op.W_res_name
        self._weights[f"{base}_val"] = val
        self._weights[f"{base}_col"] = col
        self._weights[f"{base}_rowptr"] = rowptr
        return SparseSpec(
            kind="csr",
            nnz=nnz,
            val_name=f"{base}_val",
            col_name=f"{base}_col",
            rowptr_name=f"{base}_rowptr",
        )


def _module_uses(ops: Iterable[Op], name: str) -> bool:
    return any(op.W_res_name == name for op in iter_reservoir_ops(ops))
