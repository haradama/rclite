"""Train + quantize an rclite reservoir on the *same* Mackey-Glass task and
measure host accuracy on the *same* held-out targets as the TF MLP.

Run with the rclite virtualenv:

    .venv/bin/python benchmarks/tflm_vs_rclite/eval_rclite.py

Outputs (benchmarks/tflm_vs_rclite/out/):
  rc_result.json   NRMSE per quant variant + reservoir geometry / byte sizes
  rc_pred.npz      best-variant predictions + targets (for the report)
"""
from __future__ import annotations
import json
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402

from rclite import (  # noqa: E402
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor  # noqa: E402
from rclite.quant import (  # noqa: E402
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    search_quantization_affine, LUTStrategy,
)

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)

N_UNITS = 80
TOPOLOGY = Topology.SCR


def build_rc(input_offset: float) -> ReservoirComputer:
    return ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_offset=input_offset, input_scaling=1.0,
                        name="in"),
        reservoir=ReservoirNode(units=N_UNITS, activation=Activation.TANH,
                                topology=TOPOLOGY, chain_weight=0.9,
                                leak_rate=0.3, seed=42, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                            trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=common.RC_WASHOUT, include_bias=True,
                            include_input=True, name="out"),
    )


def _nrmse_on_test(pred_full: np.ndarray, s: np.ndarray, test_t: np.ndarray):
    # pred_full[t] is the prediction of s[t+1] from inputs up to step t.
    pred = pred_full.ravel()[test_t]
    true = s[test_t + 1]
    return common.nrmse(pred, true)


def main() -> int:
    s = common.series().astype(np.float64)
    X = s[:-1, None]                 # input at step t is s[t]
    Y = s[1:, None]                  # target is s[t+1]
    train_t, test_t = common.target_indices()
    n_fit = common.TRAIN_END         # use steps [0 .. TRAIN_END-1] for fit

    input_offset = float(X[:n_fit].mean())
    rc = build_rc(input_offset)
    exe = RCExecutor(rc)
    exe.fit(X[:n_fit], Y[:n_fit])

    # float baseline
    pred_f = exe.predict(X)
    nrmse_float = _nrmse_on_test(pred_f, s, test_t)

    # persistence baseline (s[t+1] ~= s[t]) for context
    persist = common.nrmse(s[test_t], s[test_t + 1])

    variants = {}

    def eval_qm(qm):
        qexe = AffineQuantizedExecutor(qm)
        qexe.reset()
        pred = qexe.predict(X)
        return _nrmse_on_test(pred, s, test_t)

    # --- PTQ i8 (calibrate then quantize) ---
    cfg8 = calibrate_from_data(rc, exe, X[:n_fit], storage_bits=8)
    qm8_ptq = quantize_model_affine(rc, exe, cfg8,
                                    lut_strategy=LUTStrategy.linear_interp(64))
    variants["i8_affine_ptq"] = eval_qm(qm8_ptq)

    # --- QAT i8 (refit W_out on quantized states) ---
    res8 = search_quantization_affine(
        rc, exe, X[:n_fit], Y[:n_fit], X[:n_fit], Y[:n_fit],
        storage_bits=8, lut_strategy=LUTStrategy.linear_interp(64),
        n_iterations=3)
    variants["i8_affine_qat"] = eval_qm(res8.best_qmodel)

    # --- QAT i8 reservoir + i16 W_out (mixed precision) ---
    res_mix = search_quantization_affine(
        rc, exe, X[:n_fit], Y[:n_fit], X[:n_fit], Y[:n_fit],
        storage_bits=8, w_out_storage_bits=16,
        lut_strategy=LUTStrategy.linear_interp(64), n_iterations=3)
    variants["i8_i16wout_qat"] = eval_qm(res_mix.best_qmodel)

    # --- QAT i16 ---
    res16 = search_quantization_affine(
        rc, exe, X[:n_fit], Y[:n_fit], X[:n_fit], Y[:n_fit],
        storage_bits=16, lut_strategy=LUTStrategy.linear_interp(64),
        n_iterations=3)
    variants["i16_affine_qat"] = eval_qm(res16.best_qmodel)

    # geometry / parameter storage (authoritative Flash comes from the ELF)
    K, M = 1, 1
    F = (1 if rc.readout.include_bias else 0) + (K if rc.readout.include_input else 0) + N_UNITS
    n_w_in = N_UNITS * K
    n_w_res = N_UNITS            # SCR scalar chain -> N nonzeros (1 scalar stored)
    n_w_out = M * F
    n_params = n_w_in + N_UNITS + n_w_out  # chain stored as 1 scalar in kernel

    result = {
        "topology": TOPOLOGY.name,
        "reservoir_units": N_UNITS,
        "K": K, "M": M, "F": F,
        "n_W_in": n_w_in, "n_W_res_nonzero": n_w_res, "n_W_out": n_w_out,
        "n_params_stored": n_params,
        "nrmse_float_test": nrmse_float,
        "nrmse_persistence_test": persist,
        "nrmse_quant_test": variants,
        "n_test": int(len(test_t)),
    }
    (OUT / "rc_result.json").write_text(json.dumps(result, indent=2))

    # best quant variant predictions for the report
    best_name = min(variants, key=variants.get)
    best_qm = {"i8_affine_qat": res8.best_qmodel,
               "i8_i16wout_qat": res_mix.best_qmodel,
               "i16_affine_qat": res16.best_qmodel,
               "i8_affine_ptq": qm8_ptq}[best_name]
    bexe = AffineQuantizedExecutor(best_qm); bexe.reset()
    pred_best = bexe.predict(X).ravel()[test_t]
    np.savez(OUT / "rc_pred.npz", pred=pred_best, true=s[test_t + 1],
             test_t=test_t, best_variant=best_name)

    print(json.dumps(result, indent=2))
    print(f"\nbest quant variant: {best_name} = {variants[best_name]*100:.2f}% NRMSE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
