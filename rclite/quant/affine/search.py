"""QAT search for affine quantization.

Mirror of the symmetric Q-format `search_quantization`, adapted for the
affine path. The mechanism is iterative refinement:

  Round 0:
    cfg_0 = calibrate_from_data(rc, exe, X)   # from float traces
    qmodel_0 = quantize_model_affine(rc, exe, cfg_0)

  Round k ≥ 1 (QAT refit):
    H_q = qmodel_{k-1}.collect_states(train_X)     # dequantized states
    W_out_k = ridge_fit([1?, X?, H_q], train_Y, λ) # refit on quant states
    cfg_k   = recalibrate W_out_* blocks + output from W_out_k
    qmodel_k = quantize_model_affine(rc, exe, cfg_k, W_out_override=W_out_k)

The refit absorbs the quantization noise into W_out — the readout
"learns" to compensate. In our experiments this consistently improves
NRMSE versus the single-pass calibration.

`n_iterations` controls how many refit rounds run. The best round (by
held-out MSE on eval_X / eval_Y) is returned. 1-2 iterations usually
suffice; more is rarely worth the compute.
"""

from __future__ import annotations
from dataclasses import dataclass, field, replace as _replace
from typing import List, Optional, Tuple

import numpy as np

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from .._ridge import augment_phi, fit_ridge
from .types import AffineParams, AffineQuantConfig
from .calibrate import calibrate_from_data
from .quantize import AffineQuantizedModel, quantize_model_affine
from .executor import AffineQuantizedExecutor


@dataclass
class AffineSearchResult:
    best_config: AffineQuantConfig
    best_mse: float
    best_qmodel: AffineQuantizedModel
    best_iteration: int
    # (iteration, eval_mse) per round
    history: List[Tuple[int, float]] = field(default_factory=list)


def _recalibrate_for_new_W_out(
    cfg: AffineQuantConfig,
    rc: ReservoirComputer,
    W_out_new: np.ndarray,
    train_Y: np.ndarray,
    washout: int,
) -> AffineQuantConfig:
    """Update only the W_out blocks and output param to match a refit W_out.

    Reservoir-side params (input, u_pre, state, pre, W_in, W_res) come
    from float-trace calibration and don't depend on W_out, so we leave
    them untouched. Output range is re-derived from the training targets
    (the model is *trying* to match these). W_out blocks keep their
    (possibly wider) mixed-precision width.
    """
    sb = cfg.storage_bits  # activations / output width
    wob = cfg.w_out_storage_bits  # readout-weight width (mixed precision)
    K = rc.input.units
    N = rc.reservoir.units
    off = 0
    W_out_bias_p = None
    W_out_input_p = None
    if rc.readout.include_bias:
        W_out_bias_p = AffineParams.symmetric_absmax(
            W_out_new[:, 0:1], storage_bits=wob
        )
        off = 1
    if rc.readout.include_input:
        W_out_input_p = AffineParams.symmetric_absmax(
            W_out_new[:, off : off + K], storage_bits=wob
        )
        off += K
    W_out_state_p = AffineParams.symmetric_absmax(
        W_out_new[:, off : off + N], storage_bits=wob
    )
    # Use the post-washout targets — that's what the model is fit to.
    output_p = AffineParams.asymmetric_minmax(
        train_Y[washout:], storage_bits=sb
    )
    return _replace(
        cfg,
        W_out_bias=W_out_bias_p,
        W_out_input=W_out_input_p,
        W_out_state=W_out_state_p,
        output=output_p,
    )


def search_quantization_affine(
    rc: ReservoirComputer,
    exe: RCExecutor,
    train_X: np.ndarray,
    train_Y: np.ndarray,
    eval_X: np.ndarray,
    eval_Y: np.ndarray,
    *,
    storage_bits: int = 8,
    w_out_storage_bits: Optional[int] = None,
    lut_strategy=None,
    n_iterations: int = 1,
    calibration_X: Optional[np.ndarray] = None,
    ridge_lambda: Optional[float] = None,
    verbose: bool = False,
) -> AffineSearchResult:
    """Iteratively refit W_out under affine quantization.

    Returns the best (`AffineQuantizedModel`, eval MSE) over `n_iterations`
    refit rounds plus the initial (round 0) single-pass calibration.

    `calibration_X` defaults to `train_X`. `w_out_storage_bits` selects a
    wider readout-weight width (mixed precision); the QAT refit then fits
    that wider W_out to the quantized state trajectory — the combination
    that recovers single-output i8 accuracy. `lut_strategy` (a
    `LUTStrategy`) is passed through to every model built during the
    search, so the returned `best_qmodel` already uses the chosen tanh
    approximation.
    """
    if exe.W_out is None:
        raise ValueError("exe has no trained readout — call exe.fit() first")
    if calibration_X is None:
        calibration_X = train_X
    if train_X.ndim == 1:
        train_X = train_X[:, None]
    if train_Y.ndim == 1:
        train_Y = train_Y[:, None]
    if eval_X.ndim == 1:
        eval_X = eval_X[:, None]
    if eval_Y.ndim == 1:
        eval_Y = eval_Y[:, None]
    if ridge_lambda is None:
        ridge_lambda = float(rc.readout.regularization)
    washout = int(rc.readout.washout)

    # Round 0: single-pass calibration from float traces.
    cfg = calibrate_from_data(
        rc,
        exe,
        calibration_X,
        storage_bits=storage_bits,
        w_out_storage_bits=w_out_storage_bits,
    )
    W_out_current = np.asarray(exe.W_out).copy()

    history: List[Tuple[int, float]] = []
    best = (float("inf"), None, None, -1)  # (mse, cfg, qmodel, iteration)

    for it in range(n_iterations + 1):
        # Build the quantized model with the current cfg + W_out
        qm = quantize_model_affine(
            rc,
            exe,
            cfg,
            W_out_override=W_out_current,
            lut_strategy=lut_strategy,
        )

        # Held-out evaluation: warm up with train_X, then measure on eval_X.
        qexe_eval = AffineQuantizedExecutor(qm)
        qexe_eval.predict(train_X)
        Y_hat = qexe_eval.predict(eval_X)
        mse_eval = float(np.mean((Y_hat - eval_Y) ** 2))
        history.append((it, mse_eval))
        if verbose:
            print(f"  iter={it}: eval MSE = {mse_eval:.6e}")
        if mse_eval < best[0]:
            best = (mse_eval, cfg, qm, it)

        if it == n_iterations:
            break

        # QAT refit step: collect dequantized states under the current quant
        # model, ridge-fit W_out against the original targets, then recalibrate
        # W_out blocks + output for the new readout.
        qexe_refit = AffineQuantizedExecutor(qm)
        H_train_q = qexe_refit.collect_states(train_X)
        phi = augment_phi(rc, train_X, H_train_q)
        W_out_new = fit_ridge(phi, train_Y, ridge_lambda, washout)
        cfg = _recalibrate_for_new_W_out(cfg, rc, W_out_new, train_Y, washout)
        W_out_current = W_out_new

    return AffineSearchResult(
        best_config=best[1],
        best_mse=best[0],
        best_qmodel=best[2],
        best_iteration=best[3],
        history=history,
    )
