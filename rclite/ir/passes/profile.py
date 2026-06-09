"""Reservoir profiling pass.

Computes simple per-node statistics from a reservoir state trajectory and
stores them under `module.metadata["profile_stats"]`.

Intended use:
    H = exe.collect_states(X)
    m = build_ir(rc, exe)
    m = ProfileReservoir(H)(m)
    m = PruneInactiveNodes(criterion="low_variance_or_high_corr")(m)
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from ..module import Module


class ProfileReservoir:
    """Attach profile stats computed from state trajectories.

    Args:
      states: Reservoir trajectory H with shape (T, N).
      drop_prefix: Number of initial steps to ignore when computing stats.
      max_corr_samples: Optional cap on timesteps used for correlation
        computation (keeps profiling cheap for very long sequences).
    """

    name = "rc-profile-reservoir"

    def __init__(
        self,
        states,
        *,
        drop_prefix: int = 0,
        max_corr_samples: int | None = 4096,
    ):
        self.states = np.asarray(states, dtype=np.float64)
        self.drop_prefix = int(drop_prefix)
        self.max_corr_samples = max_corr_samples

    def __call__(self, module: Module) -> Module:
        H = self._select_states(module.N)

        var = np.var(H, axis=0)
        mean = np.mean(H, axis=0)
        corr_others = self._mean_abs_corr(H)

        rw = None
        if "W_out" in module.weights:
            md = module.metadata
            include_bias = bool(md.get("include_bias", True))
            include_input = bool(md.get("include_input", False))
            state_off = (1 if include_bias else 0) + (
                module.K if include_input else 0
            )
            W_out = np.asarray(module.weights["W_out"], dtype=np.float64)
            block = W_out[:, state_off : state_off + module.N]
            if block.shape[1] == module.N:
                rw = np.linalg.norm(block, axis=0)

        prof = {
            "node_variance": var.tolist(),
            "mean_activation": mean.tolist(),
            "correlation_with_other_nodes": corr_others.tolist(),
            "n_profile_steps": int(H.shape[0]),
        }
        if rw is not None:
            prof["readout_weight_norm"] = rw.tolist()
            # MVP proxy for contribution: readout coupling strength.
            prof["contribution_to_loss"] = rw.tolist()

        md = dict(module.metadata)
        md["profile_stats"] = prof
        return replace(module, metadata=md)

    def _select_states(self, N: int):
        H = self.states
        if H.ndim != 2:
            raise ValueError(f"states must be 2D (T,N), got shape {H.shape}")
        if H.shape[1] != N:
            raise ValueError(
                f"states second dim ({H.shape[1]}) != module.N ({N})"
            )
        s = min(max(self.drop_prefix, 0), max(H.shape[0] - 1, 0))
        H = H[s:]
        if H.shape[0] == 0:
            H = self.states[-1:, :]
        return H

    def _mean_abs_corr(self, H):
        if H.shape[0] < 2 or H.shape[1] <= 1:
            return np.zeros(H.shape[1], dtype=np.float64)
        if (
            self.max_corr_samples is not None
            and H.shape[0] > self.max_corr_samples
        ):
            H = H[-self.max_corr_samples :]
        C = np.corrcoef(H, rowvar=False)
        C = np.nan_to_num(C, nan=0.0, posinf=0.0, neginf=0.0)
        A = np.abs(C)
        np.fill_diagonal(A, 0.0)
        return A.sum(axis=0) / max(H.shape[1] - 1, 1)
