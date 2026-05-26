"""QAT search: pick the `state_frac` that minimizes eval MSE.

Implements proper QAT in mirage's style:
  1. Quantize W_in, W_res at the candidate config.
  2. Run the quantized reservoir on training inputs → dequantized states.
  3. Ridge-regression a new W_out on the quantized state trajectory.
  4. Re-quantize W_out with the same config.
  5. Replay on a held-out window and measure MSE.

W_in/W_res quantization is held fixed across the sweep; only `state_frac`
moves. The new W_out is fit to whatever the quantized reservoir actually
produces, so the readout absorbs the quantization noise.
"""
from __future__ import annotations
from dataclasses import dataclass, field, replace
from typing import List, Optional, Tuple

import numpy as np

from rclite.core.composite import ReservoirComputer
from rclite.runtime.reference import RCExecutor

from .config import QuantConfig
from .executor import QuantizedExecutor
from .model import QuantizedModel
from .quantize import quantize_model, quantize_W_out
from .target import I32FixedPoint, QuantTarget
from .tanh_lut import TanhLUTSpec


@dataclass
class SearchResult:
    best_config: QuantConfig
    best_mse: float
    best_qmodel: QuantizedModel
    history: List[Tuple[QuantConfig, float]] = field(default_factory=list)


def derive_frac_bits(
    data: np.ndarray,
    *,
    available_bits: int = 30,
    max_frac: int = 24,
) -> int:
    """Pick a fractional width that doesn't overflow the data range."""
    m = float(np.abs(data).max())
    if m <= 0:
        return max_frac
    int_bits = max(1, int(np.ceil(np.log2(m))) + 1)
    return int(max(0, min(max_frac, available_bits - int_bits)))


def _augment_phi(rc: ReservoirComputer, X: np.ndarray,
                  H: np.ndarray) -> np.ndarray:
    """Build phi = [1?] ++ [u?] ++ h for a (T, K) input and (T, N) state."""
    T = H.shape[0]
    parts = []
    if rc.readout.include_bias:
        parts.append(np.ones((T, 1)))
    if rc.readout.include_input:
        parts.append(X)
    parts.append(H)
    return np.concatenate(parts, axis=1)


def _fit_ridge(phi: np.ndarray, Y: np.ndarray,
                ridge_lambda: float, washout: int) -> np.ndarray:
    """Ridge regression on (T, F) features, (T, M) targets. Returns (M, F)."""
    phi_w = phi[washout:]
    Y_w = Y[washout:]
    F = phi_w.shape[1]
    A = phi_w.T @ phi_w + ridge_lambda * np.eye(F)
    B = phi_w.T @ Y_w
    return np.linalg.solve(A, B).T


def search_quantization(
    rc: ReservoirComputer,
    exe: RCExecutor,
    train_X: np.ndarray,
    train_Y: np.ndarray,
    eval_X: np.ndarray,
    eval_Y: np.ndarray,
    *,
    target: Optional[QuantTarget] = None,
    lut: Optional[TanhLUTSpec] = None,
    state_frac_range: Tuple[int, int] = (8, 24),
    input_frac: Optional[int] = None,
    weight_frac: Optional[int] = None,
    ridge_lambda: Optional[float] = None,
    verbose: bool = False,
) -> SearchResult:
    """Sweep `state_frac` over the range; pick the best config by eval MSE.

    For each candidate, refit W_out via ridge regression on the *quantized*
    state trajectory (QAT). Returns the best config and its QuantizedModel.
    """
    if target is None:
        target = I32FixedPoint()
    if lut is None:
        lut = TanhLUTSpec()
    if input_frac is None:
        input_frac = derive_frac_bits(np.concatenate([train_X.ravel(), eval_X.ravel()]))
    if weight_frac is None:
        weight_frac = derive_frac_bits(exe.W_res)
    if ridge_lambda is None:
        ridge_lambda = float(rc.readout.regularization)

    if train_X.ndim == 1:
        train_X = train_X[:, None]
    if train_Y.ndim == 1:
        train_Y = train_Y[:, None]
    if eval_X.ndim == 1:
        eval_X = eval_X[:, None]
    if eval_Y.ndim == 1:
        eval_Y = eval_Y[:, None]

    washout = int(rc.readout.washout)

    history: List[Tuple[QuantConfig, float]] = []
    best_mse = float("inf")
    best_cfg: Optional[QuantConfig] = None
    best_qmodel: Optional[QuantizedModel] = None

    lo, hi = state_frac_range
    for sf in range(lo, hi + 1):
        cfg = QuantConfig(state_frac=sf, input_frac=input_frac,
                            weight_frac=weight_frac)
        try:
            # Build a draft model with the old W_out (will be replaced)
            qm = quantize_model(rc, exe, cfg, target=target, lut=lut)

            # Run quantized forward on training inputs → quantized states
            qexe = QuantizedExecutor(qm)
            H_train = qexe.collect_states(train_X)

            # Refit W_out on the quantized state trajectory
            phi = _augment_phi(rc, train_X, H_train)
            W_out_new = _fit_ridge(phi, train_Y, ridge_lambda, washout)

            # Re-quantize W_out under the same config
            qm = QuantizedModel(
                rc=qm.rc, target=qm.target, config=qm.config, lut=qm.lut,
                W_in_q=qm.W_in_q, W_res_q=qm.W_res_q,
                W_out_q=quantize_W_out(W_out_new, rc, cfg, target),
                lut_table_q=qm.lut_table_q,
                state_init_q=qm.state_init_q,
            )

            # Evaluate: warmup with train_X, then measure on eval_X
            qexe = QuantizedExecutor(qm)
            qexe.predict(train_X)
            Y_hat = qexe.predict(eval_X)
            mse = float(np.mean((Y_hat - eval_Y) ** 2))
        except (ValueError, OverflowError) as e:
            if verbose:
                print(f"  state_frac={sf}: ABORT ({e})")
            history.append((cfg, float("inf")))
            continue

        history.append((cfg, mse))
        if verbose:
            print(f"  state_frac={sf}: mse={mse:.6e}")

        if mse < best_mse:
            best_mse = mse
            best_cfg = cfg
            best_qmodel = qm

    if best_cfg is None or best_qmodel is None:
        raise RuntimeError("No quantization config in the sweep produced a finite MSE")
    return SearchResult(best_config=best_cfg, best_mse=best_mse,
                         best_qmodel=best_qmodel, history=history)
