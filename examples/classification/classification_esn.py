"""Classification with an Echo State Network (Phase 3 of the ROADMAP).

The reservoir / linear readout are unchanged from the regression path; a thin
task layer trains the readout on one-hot targets and recovers class id /
probabilities via argmax / softmax. Two task shapes are demonstrated:

  1. sequence-to-label  (ReadoutNode.aggregation = MEAN / LAST)
       Each whole sequence maps to one class. Reservoir states are pooled over
       time into one feature vector before the readout.

  2. per-step           (ReadoutNode.aggregation = NONE)
       Each timestep gets a class — the existing per-step readout with an
       argmax / softmax head.
"""

from __future__ import annotations
import sys
import pathlib

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
    Task,
    Aggregation,
)
from rclite.runtime import RCExecutor


# ---------------------------------------------------------------------------
# 1. Sequence-to-label: classify a signal's temporal trend
#    (rising ramp / falling ramp / triangle). A leaky reservoir separates
#    these by integrating the trend over time.


def make_waveform(kind: int, length: int, rng) -> np.ndarray:
    t = np.linspace(0.0, 1.0, length)
    if kind == 0:  # rising ramp
        s = -1.0 + 2.0 * t
    elif kind == 1:  # falling ramp
        s = 1.0 - 2.0 * t
    else:  # triangle: rise then fall
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    return (s + 0.05 * rng.standard_normal(length))[:, None]


CLASS_NAMES = ["rising", "falling", "triangle"]


def make_sequence_dataset(n_per_class=60, length=60, seed=0):
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for kind in range(3):
        for _ in range(n_per_class):
            seqs.append(make_waveform(kind, length, rng))
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    return [seqs[i] for i in idx], np.array(labels)[idx]


def run_sequence_classification() -> None:
    seqs, labels = make_sequence_dataset(seed=11)
    n_tr = int(0.7 * len(seqs))

    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="input"),
        reservoir=ReservoirNode(
            units=120,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.1,
            topology=Topology.RANDOM,
            seed=9,
            name="reservoir",
        ),
        readout=ReadoutNode(
            units=3,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=10,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.MEAN,
            name="readout",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(seqs[:n_tr], labels[:n_tr])

    pred = exe.predict_sequences(seqs[n_tr:])
    proba = exe.predict_proba_sequences(seqs[n_tr:])
    y_te = labels[n_tr:]
    acc = float(np.mean(pred == y_te))

    print("=== Sequence-to-label classification (MEAN aggregation) ===")
    print(f"  classes        : {CLASS_NAMES}")
    print(f"  train / test   : {n_tr} / {len(seqs) - n_tr} sequences")
    print(f"  accuracy       : {acc:.3f}")
    print("  sample predictions:")
    for i in range(3):
        probs = ", ".join(
            f"{CLASS_NAMES[c]}={proba[i, c]:.2f}" for c in range(3)
        )
        print(
            f"    true={CLASS_NAMES[y_te[i]]:8s} pred={CLASS_NAMES[pred[i]]:8s}"
            f"  [{probs}]"
        )


# ---------------------------------------------------------------------------
# 2. Per-step: label every timestep by the sign of a smoothed random walk.


def make_per_step_dataset(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + 0.1 * rng.standard_normal()
    return u[:, None], (u > 0).astype(int)


def run_per_step_classification() -> None:
    X, y = make_per_step_dataset(seed=0)
    n_tr = 1200

    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="input"),
        reservoir=ReservoirNode(
            units=80,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.1,
            topology=Topology.RANDOM,
            seed=3,
            name="reservoir",
        ),
        readout=ReadoutNode(
            units=2,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-4,
            washout=100,
            include_input=True,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.NONE,
            name="readout",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_tr], y[:n_tr])

    pred = exe.predict_classes(X[n_tr:])
    acc = float(np.mean(pred == y[n_tr:]))
    print("\n=== Per-step classification (NONE aggregation) ===")
    print(f"  classes        : negative (0) / positive (1)")
    print(f"  train / test   : {n_tr} / {len(X) - n_tr} steps")
    print(f"  per-step acc   : {acc:.3f}")


def main() -> None:
    run_sequence_classification()
    run_per_step_classification()


if __name__ == "__main__":
    main()
