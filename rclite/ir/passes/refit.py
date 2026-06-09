"""Readout refit pass.

Refits `W_out` on top of a (possibly pruned) reservoir state trajectory.
This is intended to be used after structural passes such as pruning to recover
accuracy while keeping the optimized reservoir structure.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..module import Module


class RefitReadout:
    """Refit readout weights on provided trajectory data.

    Args:
            states:
                aggregation=NONE:
                    Reservoir states H with shape (T, N_orig) or (T, module.N).
                aggregation in {MEAN, LAST}:
                    Either pooled states with shape (S, N_orig|module.N) or per-sequence
                    trajectories as a list/tuple of (T_i, N_orig|module.N) arrays.
            inputs:
                aggregation=NONE:
                    Raw input sequence X with shape (T, K) or (T,).
                aggregation in {MEAN, LAST}:
                    Either pooled sequence inputs (S, K), per-sequence trajectories as a
                    list/tuple of (T_i, K), or None when include_input=False.
            targets:
                aggregation=NONE: (T, M) or (T,)
                aggregation in {MEAN, LAST}: (S, M) or (S,)
      drop_prefix: Optional number of initial steps to skip. If None,
        uses module.metadata['washout'] when available.
      ridge_lambda: Optional ridge coefficient. If None, uses
        module.metadata['regularization'] when available (default 1e-6).
    """

    name = "rc-refit-readout"

    def __init__(
        self,
        states,
        inputs,
        targets,
        *,
        drop_prefix: int | None = None,
        ridge_lambda: float | None = None,
    ):
        self.states = self._coerce_array_like(states, allow_none=False)
        self.inputs = self._coerce_array_like(inputs, allow_none=True)

        self.targets = np.asarray(targets, dtype=np.float64)
        self.drop_prefix = drop_prefix
        self.ridge_lambda = ridge_lambda

    def _coerce_array_like(self, value, *, allow_none: bool):
        if value is None:
            if allow_none:
                return None
            raise ValueError("value must not be None")
        if isinstance(value, (list, tuple)):
            return [np.asarray(v, dtype=np.float64) for v in value]
        return np.asarray(value, dtype=np.float64)

    def __call__(self, module: Module) -> Module:
        md = dict(module.metadata)
        agg = str(md.get("aggregation", "NONE"))
        include_bias = bool(md.get("include_bias", True))
        include_input = bool(md.get("include_input", False))
        washout = (
            int(md.get("washout", 0))
            if self.drop_prefix is None
            else int(self.drop_prefix)
        )
        lam = (
            float(md.get("regularization", 1e-6))
            if self.ridge_lambda is None
            else float(self.ridge_lambda)
        )

        if agg == "NONE":
            X = self._as_2d(self.inputs, cols=module.K, name="inputs")
            Y = self._as_2d(self.targets, cols=module.M, name="targets")
            if X.shape[0] != Y.shape[0]:
                raise ValueError(
                    "inputs/targets length mismatch: "
                    f"{X.shape[0]} vs {Y.shape[0]}"
                )

            H = self._select_state_matrix_for_module(self.states, module)
            if H.shape[0] != X.shape[0]:
                raise ValueError(
                    "states length "
                    f"({H.shape[0]}) != inputs length ({X.shape[0]})"
                )
            Phi = self._build_phi(
                X,
                H,
                include_bias=include_bias,
                include_input=include_input,
            )
            Y_fit = Y
            samples = Phi.shape[0]
        elif agg in ("MEAN", "LAST"):
            Phi, Y_fit = self._build_sequence_design(
                module,
                mode=agg,
                include_bias=include_bias,
                include_input=include_input,
                washout=washout,
            )
            samples = Phi.shape[0]
        else:
            raise NotImplementedError(
                f"RefitReadout does not support aggregation={agg}"
            )

        if Phi.shape[1] != int(md.get("feature_dim", Phi.shape[1])):
            # Keep this strict: metadata/weights drift should surface loudly.
            raise ValueError(
                "feature_dim mismatch after refit Phi build: "
                f"Phi has {Phi.shape[1]} cols, metadata has {md.get('feature_dim')}"
            )

        A = Phi.T @ Phi + lam * np.eye(Phi.shape[1])
        B = Phi.T @ Y_fit
        W_new = np.linalg.solve(A, B).T

        W_old = np.asarray(module.weights["W_out"])
        W_new = W_new.astype(W_old.dtype, copy=False)
        weights = dict(module.weights)
        weights["W_out"] = W_new

        md["readout_refit"] = {
            "samples": int(samples),
            "washout": int(washout),
            "ridge_lambda": float(lam),
            "aggregation": agg,
        }
        return replace(module, weights=weights, metadata=md)

    def _as_2d(self, arr, *, cols: int, name: str):
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got shape {arr.shape}")
        if arr.shape[1] != cols:
            raise ValueError(
                f"{name} width mismatch: {arr.shape[1]} vs expected {cols}"
            )
        return arr

    def _select_state_matrix_for_module(self, H, module: Module):
        H = np.asarray(H, dtype=np.float64)
        if H.ndim != 2:
            raise ValueError(f"states must be 2D (T,N), got {H.shape}")

        if H.shape[1] == module.N:
            return H

        kept = module.metadata.get("kept_indices")
        if kept is None:
            raise ValueError(
                "states width differs from module.N and kept_indices is missing"
            )
        kept = np.asarray(kept, dtype=np.int64)
        if kept.ndim != 1 or kept.size != module.N:
            raise ValueError(
                "invalid kept_indices in metadata for refit mapping"
            )
        if np.max(kept) >= H.shape[1]:
            raise ValueError(
                "kept_indices refer to columns outside provided states"
            )
        return H[:, kept]

    def _aggregate_rows(self, arr, *, mode: str, washout: int, name: str):
        arr = np.asarray(arr, dtype=np.float64)
        if arr.ndim == 1:
            arr = arr[:, None]
        if arr.ndim != 2:
            raise ValueError(f"{name} must be 2D, got {arr.shape}")
        if arr.shape[0] == 0:
            raise ValueError(f"{name} must have at least 1 row")
        if mode == "LAST":
            return arr[-1]
        if mode == "MEAN":
            w = min(max(int(washout), 0), arr.shape[0] - 1)
            return arr[w:].mean(axis=0)
        raise ValueError(f"unsupported aggregation mode {mode!r}")

    def _build_sequence_design(
        self,
        module: Module,
        *,
        mode: str,
        include_bias: bool,
        include_input: bool,
        washout: int,
    ):
        Y = self._as_2d(self.targets, cols=module.M, name="targets")

        # Path A: states already pooled per sequence: (S, N)
        if isinstance(self.states, np.ndarray) and self.states.ndim == 2:
            H = self._select_state_matrix_for_module(self.states, module)
            if H.shape[0] != Y.shape[0]:
                raise ValueError(
                    "pooled states/targets length mismatch: "
                    f"{H.shape[0]} vs {Y.shape[0]}"
                )
            if include_input:
                X = self._as_2d(self.inputs, cols=module.K, name="inputs")
                if X.shape[0] != H.shape[0]:
                    raise ValueError(
                        "pooled inputs/states length mismatch: "
                        f"{X.shape[0]} vs {H.shape[0]}"
                    )
            else:
                X = np.zeros((H.shape[0], module.K), dtype=np.float64)
            return (
                self._build_phi(
                    X,
                    H,
                    include_bias=include_bias,
                    include_input=include_input,
                ),
                Y,
            )

        # Path B: per-sequence trajectories (list/tuple of arrays)
        if not isinstance(self.states, (list, tuple)):
            raise ValueError(
                "For aggregation=MEAN/LAST, states must be either "
                "a pooled 2D array or a list/tuple of 2D trajectories"
            )

        states_seq = list(self.states)
        if len(states_seq) != Y.shape[0]:
            raise ValueError(
                "states/targets sequence count mismatch: "
                f"{len(states_seq)} vs {Y.shape[0]}"
            )

        if include_input:
            if not isinstance(self.inputs, (list, tuple)):
                raise ValueError(
                    "For aggregation with include_input=True, inputs must be "
                    "a list/tuple of per-sequence trajectories"
                )
            inputs_seq = list(self.inputs)
            if len(inputs_seq) != len(states_seq):
                raise ValueError(
                    "inputs/states sequence count mismatch: "
                    f"{len(inputs_seq)} vs {len(states_seq)}"
                )
        else:
            inputs_seq = [None] * len(states_seq)

        H_rows = []
        X_rows = []
        for idx, H_one in enumerate(states_seq):
            Hm = self._select_state_matrix_for_module(H_one, module)
            H_rows.append(
                self._aggregate_rows(
                    Hm,
                    mode=mode,
                    washout=washout,
                    name=f"states[{idx}]",
                )
            )
            if include_input:
                X_one = self._as_2d(
                    np.asarray(inputs_seq[idx], dtype=np.float64),
                    cols=module.K,
                    name=f"inputs[{idx}]",
                )
                X_rows.append(
                    self._aggregate_rows(
                        X_one,
                        mode=mode,
                        washout=washout,
                        name=f"inputs[{idx}]",
                    )
                )

        H = np.stack(H_rows, axis=0)
        if include_input:
            X = np.stack(X_rows, axis=0)
        else:
            X = np.zeros((H.shape[0], module.K), dtype=np.float64)
        return (
            self._build_phi(
                X, H, include_bias=include_bias, include_input=include_input
            ),
            Y,
        )

    def _build_phi(self, X, H, *, include_bias: bool, include_input: bool):
        parts = []
        if include_bias:
            parts.append(np.ones((X.shape[0], 1), dtype=np.float64))
        if include_input:
            parts.append(X)
        parts.append(H)
        return np.concatenate(parts, axis=1)
