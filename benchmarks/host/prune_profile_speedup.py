"""Benchmark: baseline vs prune (readout/profile-aware) on host LLVM JIT.

Compares three inference pipelines:
  1) baseline (no prune)
  2) prune_readout_norm
  3) prune_low_variance_or_high_corr (with ProfileReservoir)

Reports:
  - effective reservoir size N after passes
  - weight bytes proxy (sum of module weight tensor nbytes)
  - predict wall-clock median (host JIT)
  - one-step RMSE on a held-out split

Usage:
  python benchmarks/host/prune_profile_speedup.py
"""

from __future__ import annotations

import argparse
import pathlib
import statistics
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    Activation,
    InputNode,
    ReadoutNode,
    ReservoirComputer,
    ReservoirNode,
    Topology,
    Trainer,
)
from rclite.codegen import compile_rc
from rclite.ir import build_ir
from rclite.ir.passes import (
    StructuralSpecialize,
    FuseStepReadout,
    PruneInactiveNodes,
    ProfileReservoir,
)
from rclite.runtime import RCExecutor


def _dataset(T: int = 5000):
    t = np.arange(T)
    x = np.sin(0.021 * t) + 0.25 * np.sin(0.003 * t + 0.6)
    X = x[:, None]
    Y = np.roll(x, -1)[:, None]
    return X[:-1], Y[:-1]


def _train(units: int = 220, seed: int = 7):
    X, Y = _dataset()
    n_train = 3200

    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=0.12,
            topology=Topology.ESN_STANDARD,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=120,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )

    exe = RCExecutor(rc)
    exe.fit(X[:n_train], Y[:n_train])

    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_eval = X[n_train : n_train + 1200]
    Y_eval = Y[n_train : n_train + 1200]
    H_train = exe.collect_states(X[:n_train])
    return rc, exe, H_train, X_train, Y_train, X_eval, Y_eval


def _median_predict_ms(compiled, X, repeats: int = 7) -> float:
    compiled.predict(X)
    ts = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        compiled.predict(X)
        ts.append((time.perf_counter() - t0) * 1e3)
    return float(statistics.median(ts))


def _rmse(y_hat, y_true) -> float:
    d = y_hat - y_true
    return float(np.sqrt(np.mean(d * d)))


def _apply_passes(rc, exe, passes):
    m = build_ir(rc, exe)
    for p in passes:
        m = p(m)
    return m


def _weight_bytes(m) -> int:
    return int(sum(np.asarray(w).nbytes for w in m.weights.values()))


def _run_case(name: str, rc, exe, X_eval, Y_eval, passes):
    m = _apply_passes(rc, exe, passes)
    jit = compile_rc(rc, exe, passes=passes)
    y = jit.predict(X_eval)
    return {
        "name": name,
        "N": m.N,
        "weight_bytes": _weight_bytes(m),
        "pred_ms": _median_predict_ms(jit, X_eval),
        "rmse": _rmse(y, Y_eval),
        "module": m,
    }


def _act_fn(a):
    if a == Activation.TANH:
        return np.tanh
    if a == Activation.SIGMOID:
        return lambda x: 1.0 / (1.0 + np.exp(-x))
    if a == Activation.RELU:
        return lambda x: np.maximum(0.0, x)
    if a == Activation.IDENTITY:
        return lambda x: x
    raise NotImplementedError(f"activation {a.name} not supported in refit")


def _collect_states_pruned(rc, m, X):
    W_in = np.asarray(m.weights["W_in"], dtype=np.float64)
    W_res = np.asarray(m.weights["W_res"], dtype=np.float64)
    N = W_res.shape[0]
    T = X.shape[0]
    h = np.zeros(N, dtype=np.float64)
    H = np.zeros((T, N), dtype=np.float64)
    act = _act_fn(rc.reservoir.activation)
    leak = float(rc.reservoir.leak_rate)
    bias = float(rc.reservoir.bias)
    Xp = (X - rc.input.input_offset) * rc.input.input_scaling
    for t in range(T):
        pre = W_in @ Xp[t] + W_res @ h + bias
        h = (1.0 - leak) * h + leak * act(pre)
        H[t] = h
    return H


def _fit_readout_on_pruned_states(rc, H, X_raw, Y):
    T = H.shape[0]
    parts = []
    if rc.readout.include_bias:
        parts.append(np.ones((T, 1), dtype=np.float64))
    if rc.readout.include_input:
        parts.append(np.asarray(X_raw, dtype=np.float64))
    parts.append(np.asarray(H, dtype=np.float64))
    Phi = np.concatenate(parts, axis=1)
    w = int(rc.readout.washout)
    Phi_w = Phi[w:]
    Y_w = np.asarray(Y, dtype=np.float64)[w:]
    lam = float(rc.readout.regularization)
    A = Phi_w.T @ Phi_w + lam * np.eye(Phi_w.shape[1])
    B = Phi_w.T @ Y_w
    return np.linalg.solve(A, B).T


def _predict_from_pruned(m, H, X_raw, W_out_override=None):
    if W_out_override is None:
        W_out = np.asarray(m.weights["W_out"], dtype=np.float64)
    else:
        W_out = np.asarray(W_out_override, dtype=np.float64)
    T = H.shape[0]
    parts = []
    md = m.metadata
    if bool(md.get("include_bias", True)):
        parts.append(np.ones((T, 1), dtype=np.float64))
    if bool(md.get("include_input", False)):
        parts.append(np.asarray(X_raw, dtype=np.float64))
    parts.append(H)
    Phi = np.concatenate(parts, axis=1)
    return Phi @ W_out.T


def _light_refit_rmse(rc, m, X_train, Y_train, X_eval, Y_eval):
    if (
        "W_res" not in m.weights
        or "W_in" not in m.weights
        or "W_out" not in m.weights
    ):
        return float("nan")
    H_tr = _collect_states_pruned(rc, m, X_train)
    W_new = _fit_readout_on_pruned_states(rc, H_tr, X_train, Y_train)
    # Keep the same dynamics (W_in/W_res) and evaluate with refit W_out.
    H_ev = _collect_states_pruned(rc, m, X_eval)
    y = _predict_from_pruned(m, H_ev, X_eval, W_out_override=W_new)
    return _rmse(y, Y_eval)


def _single_run(
    rc,
    exe,
    H_train,
    X_train,
    Y_train,
    X_eval,
    Y_eval,
    keep_ratio: float,
    w_readout: float,
    w_variance: float,
    w_corr: float,
    refit_readout: bool,
):
    base_passes = [StructuralSpecialize(), FuseStepReadout()]

    cases = [
        (
            "baseline",
            base_passes,
        ),
        (
            "prune_readout_norm",
            [
                PruneInactiveNodes(
                    keep_ratio=keep_ratio,
                    criterion="readout_norm",
                ),
                *base_passes,
            ],
        ),
        (
            "prune_profile",
            [
                ProfileReservoir(H_train, drop_prefix=rc.readout.washout),
                PruneInactiveNodes(
                    keep_ratio=keep_ratio,
                    criterion="low_variance_or_high_corr",
                    w_readout=w_readout,
                    w_variance=w_variance,
                    w_corr=w_corr,
                ),
                *base_passes,
            ],
        ),
    ]

    rows = [_run_case(n, rc, exe, X_eval, Y_eval, p) for n, p in cases]
    base = rows[0]

    print("baseline vs prune (host LLVM JIT)\n")
    hdr = (
        f"{'case':<20} {'N':>5} {'weights KB':>11} {'pred ms':>10} "
        f"{'speedup':>8} {'RMSE':>10} {'dRMSE':>10}"
    )
    if refit_readout:
        hdr += f" {'RMSE(refit)':>12} {'dRMSE(refit)':>13}"
    print(hdr)
    print("-" * 80)
    for r in rows:
        speedup = base["pred_ms"] / max(r["pred_ms"], 1e-30)
        drmse = r["rmse"] - base["rmse"]
        line = (
            f"{r['name']:<20} {r['N']:>5d} {r['weight_bytes'] / 1024:>11.1f} "
            f"{r['pred_ms']:>10.3f} {speedup:>7.2f}x {r['rmse']:>10.6f} "
            f"{drmse:>+10.6f}"
        )
        if refit_readout and r["name"] != "baseline":
            rr = _light_refit_rmse(
                rc, r["module"], X_train, Y_train, X_eval, Y_eval
            )
            line += f" {rr:>12.6f} {rr - base['rmse']:>+13.6f}"
        print(line)

    print("\nNotes:")
    print("- speedup is relative to baseline (higher is better).")
    print("- dRMSE is RMSE - baseline RMSE (lower is better).")


def _sweep(
    rc,
    exe,
    H_train,
    X_train,
    Y_train,
    X_eval,
    Y_eval,
    ratios,
    w_readout: float,
    w_variance: float,
    w_corr: float,
    refit_readout: bool,
):
    base_passes = [StructuralSpecialize(), FuseStepReadout()]
    base = _run_case("baseline", rc, exe, X_eval, Y_eval, base_passes)

    print("baseline vs prune sweep (host LLVM JIT)\n")
    hdr = (
        f"{'criterion':<16} {'keep':>6} {'N':>5} {'weights KB':>11} "
        f"{'pred ms':>10} {'speedup':>8} {'RMSE':>10} {'dRMSE':>10}"
    )
    if refit_readout:
        hdr += f" {'RMSE(refit)':>12} {'dRMSE(refit)':>13}"
    print(hdr)
    print("-" * 96)

    for criterion in ("readout_norm", "low_variance_or_high_corr"):
        for keep_ratio in ratios:
            passes = []
            if criterion == "low_variance_or_high_corr":
                passes.append(
                    ProfileReservoir(H_train, drop_prefix=rc.readout.washout)
                )
            passes.extend(
                [
                    PruneInactiveNodes(
                        keep_ratio=keep_ratio,
                        criterion=criterion,
                        w_readout=w_readout,
                        w_variance=w_variance,
                        w_corr=w_corr,
                    ),
                    *base_passes,
                ]
            )
            r = _run_case("prune", rc, exe, X_eval, Y_eval, passes)
            speedup = base["pred_ms"] / max(r["pred_ms"], 1e-30)
            drmse = r["rmse"] - base["rmse"]
            line = (
                f"{criterion:<16} {keep_ratio:>6.2f} {r['N']:>5d} "
                f"{r['weight_bytes'] / 1024:>11.1f} {r['pred_ms']:>10.3f} "
                f"{speedup:>7.2f}x {r['rmse']:>10.6f} {drmse:>+10.6f}"
            )
            if refit_readout:
                rr = _light_refit_rmse(
                    rc, r["module"], X_train, Y_train, X_eval, Y_eval
                )
                line += f" {rr:>12.6f} {rr - base['rmse']:>+13.6f}"
            print(line)

    print("\nbaseline")
    print(
        f"N={base['N']}, weights={base['weight_bytes'] / 1024:.1f} KB, "
        f"pred={base['pred_ms']:.3f} ms, RMSE={base['rmse']:.6f}"
    )
    print("\nNotes:")
    print("- speedup is relative to baseline (higher is better).")
    print("- dRMSE is RMSE - baseline RMSE (lower is better).")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--keep-ratio",
        type=float,
        default=0.65,
        help="single-run keep ratio (ignored with --sweep)",
    )
    ap.add_argument(
        "--sweep",
        action="store_true",
        help="run keep_ratio sweep for both criteria",
    )
    ap.add_argument(
        "--ratios",
        type=str,
        default="0.95,0.90,0.85,0.80,0.75,0.70,0.65,0.60",
        help="comma-separated keep ratios for --sweep",
    )
    ap.add_argument("--w-readout", type=float, default=1.0)
    ap.add_argument("--w-variance", type=float, default=1.0)
    ap.add_argument("--w-corr", type=float, default=1.0)
    ap.add_argument(
        "--refit-readout",
        action="store_true",
        help="also report RMSE after light readout refit on pruned states",
    )
    args = ap.parse_args()

    rc, exe, H_train, X_train, Y_train, X_eval, Y_eval = _train()
    if args.sweep:
        ratios = [
            float(x.strip()) for x in args.ratios.split(",") if x.strip()
        ]
        _sweep(
            rc,
            exe,
            H_train,
            X_train,
            Y_train,
            X_eval,
            Y_eval,
            ratios,
            args.w_readout,
            args.w_variance,
            args.w_corr,
            args.refit_readout,
        )
    else:
        _single_run(
            rc,
            exe,
            H_train,
            X_train,
            Y_train,
            X_eval,
            Y_eval,
            args.keep_ratio,
            args.w_readout,
            args.w_variance,
            args.w_corr,
            args.refit_readout,
        )


if __name__ == "__main__":
    main()
