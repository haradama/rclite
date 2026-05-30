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

from rclite.core.profile import Topology

from ..module import Module
from ..ops import (
    Op, ReservoirStep, FusedStepReadout, SparseSpec, TimeLoop,
)


_DENSE = (Topology.RANDOM, Topology.ESN_STANDARD)


class SparsifyReservoir:
    name = "rc-sparsify-reservoir"

    def __init__(self, strategy: str = "auto", max_unroll_nnz: int = 4096,
                 threshold: float = 0.0):
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
        for name in [n for n in self._weights if n.startswith("W_res")
                     and not n.endswith(("_val", "_col", "_rowptr"))]:
            if not _module_uses(new_ops, name):
                del self._weights[name]
        return Module(
            K=module.K, N=module.N, M=module.M,
            weights=self._weights, ops=new_ops,
            metadata=dict(module.metadata),
        )

    def _fix(self, op: Op) -> Op:
        if isinstance(op, TimeLoop):
            return replace(op, body=tuple(self._fix(o) for o in op.body))
        if isinstance(op, (ReservoirStep, FusedStepReadout)):
            if (op.topology in _DENSE and op.W_res_name is not None
                    and op.res_sparse is None):
                spec = self._build_spec(op)
                return replace(op, res_sparse=spec, W_res_name=None)
        return op

    def _build_spec(self, op) -> SparseSpec:
        W = np.asarray(self._weights[op.W_res_name])
        N = W.shape[0]
        mask = np.abs(W) > self.threshold
        nnz = int(mask.sum())

        kind = self.strategy
        if kind == "auto":
            kind = "unroll" if nnz <= self.max_unroll_nnz else "csr"

        if kind == "unroll":
            rows = tuple(
                tuple((int(j), float(W[i, j]))
                      for j in np.nonzero(mask[i])[0])
                for i in range(N)
            )
            return SparseSpec(kind="unroll", nnz=nnz, rows=rows)

        # CSR: build val / col / rowptr in ascending column order per row.
        val, col, rowptr = [], [], [0]
        for i in range(N):
            cols = np.nonzero(mask[i])[0]
            for j in cols:
                col.append(int(j))
                val.append(float(W[i, j]))
            rowptr.append(len(col))
        base = op.W_res_name
        self._weights[f"{base}_val"] = np.asarray(val, dtype=np.float64)
        self._weights[f"{base}_col"] = np.asarray(col, dtype=np.int32)
        self._weights[f"{base}_rowptr"] = np.asarray(rowptr, dtype=np.int32)
        return SparseSpec(
            kind="csr", nnz=nnz,
            val_name=f"{base}_val", col_name=f"{base}_col",
            rowptr_name=f"{base}_rowptr",
        )


def _module_uses(ops: Iterable[Op], name: str) -> bool:
    for op in ops:
        if isinstance(op, TimeLoop):
            if _module_uses(op.body, name):
                return True
        if isinstance(op, (ReservoirStep, FusedStepReadout)):
            if op.W_res_name == name:
                return True
    return False
