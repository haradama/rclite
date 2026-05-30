"""Online learning for an ESN: RLS / LMS comparison vs batch ridge.

Same Mackey-Glass task, same reservoir, three readout training regimes.
For online trainers we measure both the *per-step* prediction error
(what the deployed model would see in production) and the *post-training*
RMSE (final readout state evaluated on a held-out window).
"""
from __future__ import annotations
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor

from examples.forecasting.mackey_glass_esn import mackey_glass, rmse


def build(trainer: Trainer, *, input_offset: float, seed: int = 42,
          **readout_kwargs) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=input_offset,
                        name="input"),
        reservoir=ReservoirNode(units=400, activation=Activation.TANH,
                                spectral_radius=0.95, leak_rate=0.3,
                                density=0.05, topology=Topology.ESN_STANDARD,
                                seed=seed, name="reservoir"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                            trainer=trainer, washout=200,
                            include_bias=True, include_input=True,
                            name="readout", **readout_kwargs),
    )


def main() -> None:
    series = mackey_glass(n=4000)
    n_train = 3000
    X_tr, Y_tr = series[:n_train - 1, None], series[1:n_train, None]
    X_te, Y_te = series[n_train:-1, None], series[n_train + 1:, None]
    input_offset = float(X_tr.mean())

    print(f"{'trainer':<10} {'per-step RMSE (last 500)':>26} "
          f"{'held-out RMSE':>15}")

    # Batch ridge baseline
    rc_b = build(Trainer.RIDGE, input_offset=input_offset, regularization=1e-6)
    exe_b = RCExecutor(rc_b)
    exe_b.fit(X_tr, Y_tr)
    yhat_te_b = exe_b.predict(X_te)
    print(f"{'RIDGE':<10} {'(batch, n/a)':>26} {rmse(yhat_te_b, Y_te):>15.5f}")

    # Online RLS
    rc_r = build(Trainer.RLS, input_offset=input_offset,
                 forgetting_factor=0.9995, init_variance=1e-2)
    exe_r = RCExecutor(rc_r)
    yhat_tr_r = exe_r.online_fit(X_tr, Y_tr, warmup_steps=200)
    yhat_te_r = exe_r.predict(X_te)
    per_step_r = rmse(yhat_tr_r[-500:], Y_tr[-500:])
    print(f"{'RLS':<10} {per_step_r:>26.5f} {rmse(yhat_te_r, Y_te):>15.5f}")

    # Online LMS — needs a learning rate that's small relative to feature scale
    rc_l = build(Trainer.LMS, input_offset=input_offset, learning_rate=5e-3)
    exe_l = RCExecutor(rc_l)
    yhat_tr_l = exe_l.online_fit(X_tr, Y_tr, warmup_steps=200)
    yhat_te_l = exe_l.predict(X_te)
    per_step_l = rmse(yhat_tr_l[-500:], Y_tr[-500:])
    print(f"{'LMS':<10} {per_step_l:>26.5f} {rmse(yhat_te_l, Y_te):>15.5f}")


if __name__ == "__main__":
    main()
