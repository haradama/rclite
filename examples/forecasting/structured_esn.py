"""Compare random vs structured (Rodan-Tino 2011) reservoirs on Mackey-Glass.

Structured reservoirs (DLR / DLRB / SCR) are fully deterministic — no
random weights, no random sparsity — yet often match random reservoirs.
This demo runs one-step prediction and 200-step free-running across the
four topologies with matched dimensionality.
"""
from __future__ import annotations
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    WellPosedReservoir,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor

from examples.forecasting.mackey_glass_esn import mackey_glass, rmse, nrmse


def build(topology: Topology, input_offset: float, *,
          units: int = 500, seed: int = 42) -> ReservoirComputer:
    """Build an ESN for the requested topology, with topology-appropriate
    defaults (binary ±v input weights for the Rodan-Tino reservoirs,
    Gaussian for the random reservoir)."""
    if topology in (Topology.DLR, Topology.DLRB, Topology.SCR):
        input_dist = Distribution.BERNOULLI
    else:
        input_dist = Distribution.NORMAL
    return ReservoirComputer(
        input=InputNode(
            units=1, activation=Activation.IDENTITY,
            input_scaling=0.5, input_offset=input_offset,
            input_distribution=input_dist, name="input",
        ),
        reservoir=ReservoirNode(
            units=units, activation=Activation.TANH,
            spectral_radius=0.95,            # used by RANDOM only
            chain_weight=0.9,                # used by DLR/DLRB/SCR
            chain_feedback=0.05,             # used by DLRB only
            leak_rate=0.3, density=0.05,
            topology=topology, seed=seed, name="reservoir",
        ),
        readout=ReadoutNode(
            units=1, activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE, regularization=1e-6, washout=200,
            include_bias=True, include_input=True, name="readout",
        ),
    )


def main() -> None:
    series = mackey_glass(n=3000)
    n_train = 2000
    horizon = 200
    X_tr, Y_tr = series[:n_train - 1, None], series[1:n_train, None]
    input_offset = float(X_tr.mean())
    X_te, Y_te = series[n_train:-1, None], series[n_train + 1:, None]
    seed_X = series[n_train - 200:n_train, None]
    free_target = series[n_train:n_train + horizon, None]

    print(f"{'topology':<10} {'one-step RMSE':>14} {'free-run NRMSE':>16} "
          f"{'ESP':>5}")
    for topo in [Topology.RANDOM, Topology.DLR, Topology.DLRB, Topology.SCR]:
        rc = build(topo, input_offset)
        esp = WellPosedReservoir(rc.reservoir).satisfied()
        exe = RCExecutor(rc)
        exe.fit(X_tr, Y_tr)
        Yp = exe.predict(X_te)
        os_rmse = rmse(Yp, Y_te)
        fr = exe.free_run(seed_X, n_steps=horizon)
        fr_nrmse = nrmse(fr, free_target)
        print(f"{topo.name:<10} {os_rmse:>14.5f} {fr_nrmse:>16.5f}   "
              f"{'ok' if esp else 'NO':>4}")


if __name__ == "__main__":
    main()
