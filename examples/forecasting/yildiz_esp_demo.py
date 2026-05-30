"""Demo: Yildiz et al. (2012) input-driven ESP check vs structural bound.

Shows two reservoirs:
  (A) spectral_radius = 0.95  — passes the conservative structural bound
  (B) spectral_radius = 1.10  — fails the structural bound, but the
      input-driven Lyapunov-based check accepts it because the actual
      operating trajectory is contractive.

This is the situation that motivated Yildiz et al. 2012 — the conservative
bound is sufficient but unnecessarily strict for input-driven reservoirs.
"""
from __future__ import annotations
import sys
import pathlib

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    WellPosedReservoir, ConstraintViolation,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.verification import (
    InputDrivenESPCheck,
    maximum_lyapunov_exponent,
    reservoir_singular_value,
)

from examples.forecasting.mackey_glass_esn import mackey_glass


def build(spectral_radius: float, leak_rate: float, input_offset: float,
          seed: int = 42) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(
            units=1, activation=Activation.IDENTITY,
            input_scaling=1.0, input_offset=input_offset,
            name="input",
        ),
        reservoir=ReservoirNode(
            units=500, activation=Activation.TANH,
            spectral_radius=spectral_radius, leak_rate=leak_rate,
            density=0.05, topology=Topology.ESN_STANDARD, seed=seed,
            name="reservoir",
        ),
        readout=ReadoutNode(
            units=1, activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE, regularization=1e-6, washout=200,
            include_bias=True, include_input=True, name="readout",
        ),
    )


def main() -> None:
    series = mackey_glass(n=3000)
    X_sample = series[:2000, None]
    input_offset = float(X_sample.mean())

    print("=== (A) spectral_radius = 0.95 — conservative configuration ===")
    rc_a = build(spectral_radius=0.95, leak_rate=0.3, input_offset=input_offset)
    exe_a = RCExecutor(rc_a)
    print(f"  ρ(W) = {rc_a.reservoir.spectral_radius:.3f}")
    print(f"  σ_max(W) = {reservoir_singular_value(exe_a):.3f}")
    mle_a = maximum_lyapunov_exponent(exe_a, X_sample)
    print(f"  MLE      = {mle_a:+.4f}")
    WellPosedReservoir(rc_a.reservoir).check()
    print(f"  [ok] WellPosedReservoir (structural) satisfied\n")

    print("=== (B) spectral_radius = 1.10 — beyond the conservative bound ===")
    rc_b = build(spectral_radius=1.10, leak_rate=0.50, input_offset=input_offset)
    exe_b = RCExecutor(rc_b)
    print(f"  ρ(W) = {rc_b.reservoir.spectral_radius:.3f}")
    print(f"  σ_max(W) = {reservoir_singular_value(exe_b):.3f}")
    mle_b = maximum_lyapunov_exponent(exe_b, X_sample)
    print(f"  MLE      = {mle_b:+.4f}")

    print("\n  Structural check (no empirical):")
    try:
        WellPosedReservoir(rc_b.reservoir).check()
        print("    [ok] satisfied")
    except ConstraintViolation as e:
        print(f"    [reject] {e}\n")

    print("  Input-driven check (Yildiz et al. 2012):")
    req = WellPosedReservoir(
        rc_b.reservoir,
        empirical_check=InputDrivenESPCheck(
            executor=exe_b, sample_input=X_sample, threshold=0.0,
        ),
    )
    req.check()
    print(f"    [ok] satisfied (MLE = {req.empirical_check.last_mle:+.4f})")
    for w in req.warnings():
        print(f"    [warn] {w}")

    print("\n=== (C) Counter-example: unstable reservoir (ρ = 2.0) ===")
    rc_c = build(spectral_radius=2.0, leak_rate=0.5, input_offset=input_offset)
    exe_c = RCExecutor(rc_c)
    mle_c = maximum_lyapunov_exponent(exe_c, X_sample)
    print(f"  ρ(W) = {rc_c.reservoir.spectral_radius:.3f}")
    print(f"  σ_max(W) = {reservoir_singular_value(exe_c):.3f}")
    print(f"  MLE      = {mle_c:+.4f}")
    req_c = WellPosedReservoir(
        rc_c.reservoir,
        empirical_check=InputDrivenESPCheck(executor=exe_c, sample_input=X_sample),
    )
    try:
        req_c.check()
        print("  [ok] satisfied — unexpected for unstable reservoir")
    except ConstraintViolation as e:
        print(f"  [reject] {e}")


if __name__ == "__main__":
    main()
