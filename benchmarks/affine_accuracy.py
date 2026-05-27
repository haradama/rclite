"""Accuracy benchmark across the full quantization matrix.

Task: Mackey-Glass one-step-ahead prediction. Metric: NRMSE (= RMSE /
signal std), so 100% means "error as large as the signal itself" and a
few percent is production-grade.

Three tables:
  A. Storage width × method (float / symmetric Q-format / affine ± QAT)
  B. LUT strategy × storage width (affine, single-pass calibration)
  C. QAT iteration effect (affine i8 / i16)

Run:  uv run python benchmarks/affine_accuracy.py
"""
from __future__ import annotations
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec,
    I32FixedPoint, I16FixedPoint, I8Symmetric,
    quantize_model, QuantizedExecutor, search_quantization,
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    search_quantization_affine, LUTStrategy,
)
from examples.mackey_glass_esn import mackey_glass


N_TRAIN = 2000
N_EVAL = 200
RESERVOIR_N = 80


def build_and_train(seed):
    series = mackey_glass(n=N_TRAIN + N_EVAL + 600)
    X, Y = series[:-1, None], series[1:, None]
    rc = ReservoirComputer(
        input=InputNode(units=1, input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=RESERVOIR_N, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.3, seed=seed),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=300, include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:N_TRAIN], Y[:N_TRAIN])
    return rc, exe, X, Y


def _nrmse(pred_full, eval_Y, sig_std):
    sl = slice(N_TRAIN, N_TRAIN + N_EVAL)
    mse = float(np.mean((pred_full[sl] - eval_Y) ** 2))
    return float(np.sqrt(mse)) / sig_std * 100.0


# ----------------------------------------------------------------------------
# Per-method evaluators (return NRMSE %)


def eval_float(rc, exe, X, eval_Y, sig_std):
    return _nrmse(exe.predict(X[:N_TRAIN + N_EVAL]), eval_Y, sig_std)


def eval_sym_qat(rc, exe, X, Y, eval_Y, sig_std, target, sf_range, ifrac, wfrac,
                  lut_n):
    res = search_quantization(
        rc, exe, X[:N_TRAIN], Y[:N_TRAIN],
        X[N_TRAIN:N_TRAIN + N_EVAL], Y[N_TRAIN:N_TRAIN + N_EVAL],
        target=target, state_frac_range=sf_range,
        input_frac=ifrac, weight_frac=wfrac, lut=TanhLUTSpec(n=lut_n),
    )
    qexe = QuantizedExecutor(res.best_qmodel)
    return _nrmse(qexe.predict(X[:N_TRAIN + N_EVAL]), eval_Y, sig_std)


def eval_affine_1pass(rc, exe, X, eval_Y, sig_std, storage_bits, strategy=None):
    cfg = calibrate_from_data(rc, exe, X[:N_TRAIN], storage_bits=storage_bits)
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=strategy)
    qexe = AffineQuantizedExecutor(qm)
    return _nrmse(qexe.predict(X[:N_TRAIN + N_EVAL]), eval_Y, sig_std)


def eval_affine_qat(rc, exe, X, Y, eval_Y, sig_std, storage_bits, n_iter=1,
                     w_out_storage_bits=None):
    res = search_quantization_affine(
        rc, exe, X[:N_TRAIN], Y[:N_TRAIN],
        X[N_TRAIN:N_TRAIN + N_EVAL], Y[N_TRAIN:N_TRAIN + N_EVAL],
        storage_bits=storage_bits, w_out_storage_bits=w_out_storage_bits,
        n_iterations=n_iter,
    )
    qexe = AffineQuantizedExecutor(res.best_qmodel)
    return _nrmse(qexe.predict(X[:N_TRAIN + N_EVAL]), eval_Y, sig_std)


# ----------------------------------------------------------------------------


def main():
    seeds = (42, 7, 123)
    print(f"Mackey-Glass 1-step-ahead | N={RESERVOIR_N} SCR | "
          f"T_train={N_TRAIN} T_eval={N_EVAL} | {len(seeds)} seeds")
    print("NRMSE = RMSE / signal_std (%), lower is better\n")

    # Accumulate per-method NRMSE across seeds.
    table_a = {}   # method -> [nrmse per seed]
    table_b = {}   # (storage, strategy) -> [nrmse]
    table_c = {}   # (storage, n_iter) -> [nrmse]

    for seed in seeds:
        rc, exe, X, Y = build_and_train(seed)
        eval_Y = Y[N_TRAIN:N_TRAIN + N_EVAL]
        sig_std = float(np.std(eval_Y))

        # ---- Table A ----
        table_a.setdefault("float", []).append(
            eval_float(rc, exe, X, eval_Y, sig_std))
        table_a.setdefault("i32 sym (QAT)", []).append(
            eval_sym_qat(rc, exe, X, Y, eval_Y, sig_std, I32FixedPoint(),
                          (12, 22), 14, 14, 256))
        table_a.setdefault("i16 sym (QAT)", []).append(
            eval_sym_qat(rc, exe, X, Y, eval_Y, sig_std, I16FixedPoint(),
                          (8, 14), 8, 8, 128))
        table_a.setdefault("i16 affine (1-pass)", []).append(
            eval_affine_1pass(rc, exe, X, eval_Y, sig_std, 16))
        table_a.setdefault("i16 affine (QAT)", []).append(
            eval_affine_qat(rc, exe, X, Y, eval_Y, sig_std, 16))
        table_a.setdefault("i8 affine (1-pass)", []).append(
            eval_affine_1pass(rc, exe, X, eval_Y, sig_std, 8))
        table_a.setdefault("i8 affine (QAT)", []).append(
            eval_affine_qat(rc, exe, X, Y, eval_Y, sig_std, 8))
        table_a.setdefault("i8 + i16 W_out (QAT)", []).append(
            eval_affine_qat(rc, exe, X, Y, eval_Y, sig_std, 8,
                             w_out_storage_bits=16))

        # ---- Table B: LUT strategy (single-pass calibration) ----
        strategies = [
            ("direct",        LUTStrategy.direct()),
            ("interp n=256",  LUTStrategy.linear_interp(256)),
            ("interp n=64",   LUTStrategy.linear_interp(64)),
            ("interp n=16",   LUTStrategy.linear_interp(16)),
            ("poly deg=3",    LUTStrategy.polynomial(degree=3)),
            ("poly deg=5",    LUTStrategy.polynomial(degree=5)),
        ]
        for sb in (16, 8):
            for label, strat in strategies:
                table_b.setdefault((sb, label), []).append(
                    eval_affine_1pass(rc, exe, X, eval_Y, sig_std, sb, strat))

        # ---- Table C: QAT iterations ----
        for sb in (16, 8):
            for n_iter in (0, 1, 2):
                table_c.setdefault((sb, n_iter), []).append(
                    eval_affine_qat(rc, exe, X, Y, eval_Y, sig_std, sb, n_iter))

    def mean(vals):
        return float(np.mean(vals))

    # ---- Print Table A ----
    print("=" * 60)
    print("TABLE A — storage width × method (NRMSE %, seed-mean)")
    print("-" * 60)
    order_a = ["float", "i32 sym (QAT)", "i16 sym (QAT)",
               "i16 affine (1-pass)", "i16 affine (QAT)",
               "i8 affine (1-pass)", "i8 affine (QAT)",
               "i8 + i16 W_out (QAT)"]
    for m in order_a:
        vals = table_a[m]
        print(f"  {m:<24} {mean(vals):8.2f}%   "
              f"(per-seed: {', '.join(f'{v:.1f}' for v in vals)})")
    print("=" * 60)

    # ---- Print Table B ----
    print("\nTABLE B — LUT strategy × storage (affine, 1-pass, NRMSE %)")
    print("-" * 60)
    print(f"  {'strategy':<16} {'i16':>10} {'i8':>10}")
    print("  " + "-" * 38)
    for label, _ in strategies:
        v16 = mean(table_b[(16, label)])
        v8 = mean(table_b[(8, label)])
        print(f"  {label:<16} {v16:>9.2f}% {v8:>9.2f}%")
    print("=" * 60)

    # ---- Print Table C ----
    print("\nTABLE C — QAT iteration effect (affine, NRMSE %)")
    print("-" * 60)
    print(f"  {'n_iterations':<16} {'i16':>10} {'i8':>10}")
    print("  " + "-" * 38)
    for n_iter in (0, 1, 2):
        v16 = mean(table_c[(16, n_iter)])
        v8 = mean(table_c[(8, n_iter)])
        tag = "(1-pass)" if n_iter == 0 else ""
        print(f"  n_iter={n_iter} {tag:<9} {v16:>9.2f}% {v8:>9.2f}%")
    print("=" * 60)


if __name__ == "__main__":
    main()
