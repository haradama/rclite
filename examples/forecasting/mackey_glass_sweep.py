"""Hyperparameter sweep for the Mackey-Glass ESN.

Search the (spectral_radius, leak_rate, input_scaling, regularization) grid,
ranking configurations by free-run NRMSE on a validation horizon (unseen by
training). The best configuration is then re-evaluated on a held-out test
region with multiple reservoir seeds for stability.
"""

from __future__ import annotations
import sys
import pathlib
import itertools
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
)
from rclite import WellPosedReservoir, ConstraintViolation
from rclite.runtime import RCExecutor
from rclite.verification import InputDrivenESPCheck

from examples.forecasting.mackey_glass_esn import mackey_glass, rmse, nrmse


GRID = {
    "spectral_radius": [0.8, 0.95, 1.1, 1.25],
    "leak_rate": [0.1, 0.3, 0.5, 0.8],
    "input_scaling": [0.5, 1.0, 2.0],
    "regularization": [1e-4, 1e-6, 1e-8],
}


def build(
    input_offset: float,
    *,
    spectral_radius: float,
    leak_rate: float,
    input_scaling: float,
    regularization: float,
    seed: int = 42,
    units: int = 500,
) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=input_scaling,
            input_offset=input_offset,
            name="input",
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            spectral_radius=spectral_radius,
            leak_rate=leak_rate,
            density=0.05,
            topology=Topology.ESN_STANDARD,
            seed=seed,
            name="reservoir",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=regularization,
            washout=200,
            include_bias=True,
            include_input=True,
            name="readout",
        ),
    )


def evaluate(esn: ReservoirComputer, X_tr, Y_tr, seed_X, val_target):
    exe = RCExecutor(esn)
    exe.fit(X_tr, Y_tr)
    fr = exe.free_run(seed_X, n_steps=len(val_target))
    return nrmse(fr, val_target), exe


def main() -> None:
    series = mackey_glass(n=3000)
    n_train = 2000
    val_horizon = 200
    test_horizon = 300

    X_tr = series[: n_train - 1, None]
    Y_tr = series[1:n_train, None]
    input_offset = float(X_tr.mean())

    val_seed = series[n_train - 200 : n_train, None]
    val_target = series[n_train : n_train + val_horizon, None]

    test_seed_start = n_train + val_horizon + 100  # gap from validation region
    test_seed = series[test_seed_start : test_seed_start + 200, None]
    test_target = series[
        test_seed_start + 200 : test_seed_start + 200 + test_horizon, None
    ]

    one_step_X = series[test_seed_start:-1, None]
    one_step_Y = series[test_seed_start + 1 :, None]

    keys = list(GRID.keys())
    combos = list(itertools.product(*[GRID[k] for k in keys]))
    print(f"Sweeping {len(combos)} configurations on N=500 reservoir, seed=42")
    t0 = time.time()
    results = []
    for vals in combos:
        cfg = dict(zip(keys, vals))
        esn = build(input_offset, **cfg)
        score, _ = evaluate(esn, X_tr, Y_tr, val_seed, val_target)
        results.append((score, cfg))
    dt = time.time() - t0
    results.sort(key=lambda r: r[0])
    print(f"Sweep done in {dt:.1f}s\n")

    print(f"{'rank':>4} {'NRMSE_val':>10}  spec_r   leak   in_sc   reg")
    for i, (score, cfg) in enumerate(results[:10]):
        print(
            f"{i + 1:>4} {score:>10.5f}  "
            f"{cfg['spectral_radius']:>6.2f} {cfg['leak_rate']:>6.2f} "
            f"{cfg['input_scaling']:>6.2f} {cfg['regularization']:>7.0e}"
        )

    best_score, best_cfg = results[0]
    print(f"\nBest config (val NRMSE = {best_score:.5f}):")
    for k, v in best_cfg.items():
        print(f"  {k} = {v}")

    print(
        f"\n=== Final evaluation on held-out test region "
        f"(horizon={test_horizon}, 5 seeds) ==="
    )
    one_rmses, fr_rmses, fr_nrmses, mles = [], [], [], []
    for seed in [1, 7, 17, 42, 99]:
        esn = build(input_offset, **best_cfg, seed=seed)
        exe = RCExecutor(esn)

        # Yildiz et al. (2012) input-driven ESP check. If structural SR<1
        # holds the empirical check is redundant but cheap; otherwise the
        # empirical check is what authorizes this configuration.
        req = WellPosedReservoir(
            esn.reservoir,
            empirical_check=InputDrivenESPCheck(
                executor=exe,
                sample_input=X_tr,
                threshold=0.0,
            ),
        )
        try:
            req.check()
            mles.append(req.empirical_check.last_mle)
        except ConstraintViolation as e:
            print(f"[seed={seed}] ESP check failed: {e}")
            continue

        exe.fit(X_tr, Y_tr)
        Yp = exe.predict(one_step_X)
        one_rmses.append(rmse(Yp, one_step_Y))
        fr = exe.free_run(test_seed, n_steps=test_horizon)
        fr_rmses.append(rmse(fr, test_target))
        fr_nrmses.append(nrmse(fr, test_target))

    one_rmses = np.array(one_rmses)
    fr_rmses = np.array(fr_rmses)
    fr_nrmses = np.array(fr_nrmses)
    mles = np.array(mles)
    print(
        f"one-step RMSE        : mean {one_rmses.mean():.5f}  "
        f"std {one_rmses.std():.5f}  min {one_rmses.min():.5f}"
    )
    print(
        f"free-run({test_horizon}) RMSE   : "
        f"mean {fr_rmses.mean():.5f}  std {fr_rmses.std():.5f}  "
        f"min {fr_rmses.min():.5f}"
    )
    print(
        f"free-run({test_horizon}) NRMSE  : "
        f"mean {fr_nrmses.mean():.5f}  std {fr_nrmses.std():.5f}  "
        f"min {fr_nrmses.min():.5f}"
    )
    print(
        f"Yildiz MLE (5 seeds) : "
        f"mean {mles.mean():+.4f}  std {mles.std():.4f}  max {mles.max():+.4f}"
    )


if __name__ == "__main__":
    main()
