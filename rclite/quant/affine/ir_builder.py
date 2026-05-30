"""Build an rclite IR `Module` for the affine integer path from an
`AffineQuantizedModel`.

The structural ops (`ReservoirStep`, `BuildPhi`, `ReadoutLinear`,
`TimeLoop`) are reused unchanged from the symmetric path — what varies is
the arithmetic the lowering emits. `metadata["quantization"] = "affine"`
discriminates the two paths in `emit_quantized_module`, which dispatches
to `_AffineLowerer` when set.

Carries in `weights`:
  - Integer weights `W_in`, `W_res`, `W_out` (storage dtype)
  - Direct tanh LUT (`lut_table`, storage dtype, 2^storage_bits entries)
  - Precomputed `row_sum_W_in`, `row_sum_W_res`, `row_sum_Wout_*` (i32)

Carries in `metadata`:
  - Storage / topology / readout flags
  - Per-tensor `zero_point` values
  - Integer `(M0, n)` for every requantize multiplier
  - `bias_pre` and `lut_offset`

MVP restriction: only supports `input_offset == 0` and
`input_scaling == 1` so the kernel can treat the raw input as u_pre
directly. (Calibration produces identical params for input and u_pre in
this case.) Non-trivial preprocess will need integer preprocess support
in the lowering and is left for a follow-up.
"""
from __future__ import annotations

import numpy as np

from rclite.core.profile import Aggregation, Topology
from rclite.ir.module import Module
from rclite.ir.ops import (
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear, TimeLoop,
    Argmax, Softmax,
)

from .lut import LUTKind
from .quantize import AffineQuantizedModel
from ..softmax_lut import SoftmaxLUTSpec, build_params as build_softmax_params


def build_ir_from_quantized_affine(qmodel: AffineQuantizedModel,
                                   *, head=None) -> Module:
    rc = qmodel.rc
    cfg = qmodel.config
    if head not in (None, "logits", "classify", "proba"):
        raise NotImplementedError(
            f"affine integer path supports head in (None, 'logits', "
            f"'classify', 'proba'); got {head!r}"
        )
    if rc.readout.aggregation != Aggregation.NONE:
        raise NotImplementedError(
            "Quantized classification currently supports per-step readouts "
            "(aggregation=NONE) only; sequence pooling is not yet quantized."
        )

    K, N, M = qmodel.K, qmodel.N, qmodel.M
    F = qmodel.F

    # SCR/DLR/DLRB have at most 1–2 non-zero entries per W_res row, all
    # equal to (the quantised) chain_weight / chain_feedback. Skip emitting
    # the dense W_res + row_sum_W_res globals; the lowering reads the
    # chain constants from metadata and folds zp inline.
    is_structured = rc.reservoir.topology in (
        Topology.DLR, Topology.DLRB, Topology.SCR,
    )

    weights: dict[str, np.ndarray] = {
        "W_in": qmodel.W_in_q,
        "W_out": qmodel.W_out_q,
        "row_sum_W_in": qmodel.row_sum_W_in,
        "row_sum_Wout_state": qmodel.row_sum_Wout_state,
    }
    if not is_structured:
        weights["W_res"] = qmodel.W_res_q
        weights["row_sum_W_res"] = qmodel.row_sum_W_res
    # The LUT global is only emitted for the table-based strategies.
    if qmodel.lut_strategy.kind in (LUTKind.DIRECT, LUTKind.LINEAR_INTERP):
        weights["lut_table"] = qmodel.lut_q
    if rc.readout.include_input:
        if qmodel.row_sum_Wout_input is None:
            raise RuntimeError(
                "qmodel has include_input=True but row_sum_Wout_input is None"
            )
        weights["row_sum_Wout_input"] = qmodel.row_sum_Wout_input

    # When input_offset != 0 or input_scaling != 1, the kernel needs an
    # integer preprocess step that writes u_pre[k] into a scratch buffer
    # the W_in matmul then reads from. We piggy-back on the existing
    # `PreprocessInput` IR op; the affine lowerer recognises it via
    # `has_integer_preprocess` metadata and emits the affine integer form
    # (M_pre,n,pre_const) instead of the symmetric one.
    body_ops = []
    if qmodel.has_integer_preprocess:
        body_ops.append(PreprocessInput(
            offset=float(rc.input.input_offset),
            scale=float(rc.input.input_scaling),
            K=K,
        ))
    body_ops += [
        ReservoirStep(
            leak=float(rc.reservoir.leak_rate),
            bias=float(rc.reservoir.bias),
            N=N, K=K,
            topology=rc.reservoir.topology,
            chain_weight=float(rc.reservoir.chain_weight),
            chain_feedback=float(rc.reservoir.chain_feedback),
            W_res_name=None if is_structured else "W_res",
        ),
        BuildPhi(
            include_bias=rc.readout.include_bias,
            include_input=rc.readout.include_input,
            K=K, N=N,
        ),
        ReadoutLinear(M=M, F=F),
    ]
    sm = None
    if head == "classify":
        body_ops.append(Argmax(M=M))
    elif head == "proba":
        import numpy as _np
        sm = build_softmax_params(
            SoftmaxLUTSpec(), s_diff=cfg.output.scale,
            storage_bits=qmodel.storage_bits,
            storage_dtype=_np.dtype(f"int{qmodel.storage_bits}"),
        )
        body_ops.append(Softmax(M=M))
    body = tuple(body_ops)

    # Pre-quantize the chain constants once (matches what dense W_res_q
    # holds at the chain positions); the lowering uses them as scalar i32
    # operands so the symmetric chain matmul collapses to one multiply.
    weight_qmin = -(1 << (qmodel.storage_bits - 1))
    weight_qmax = (1 << (qmodel.storage_bits - 1)) - 1
    def _qweight(v: float) -> int:
        q = int(round(v / cfg.W_res.scale))
        return max(weight_qmin, min(weight_qmax, q))
    chain_weight_q   = _qweight(float(rc.reservoir.chain_weight))
    chain_feedback_q = _qweight(float(rc.reservoir.chain_feedback))

    art = qmodel.lut_artifacts
    strat = qmodel.lut_strategy
    md = {
        "quantization": "affine",
        "dtype": f"i{qmodel.storage_bits}",
        "storage_bits": qmodel.storage_bits,
        "w_out_storage_bits": qmodel.w_out_storage_bits,
        "topology": rc.reservoir.topology.name,
        "include_bias": rc.readout.include_bias,
        "include_input": rc.readout.include_input,
        "feature_dim": F,
        # Zero points
        "zp_input":  cfg.input.zero_point,
        "zp_u_pre":  cfg.u_pre.zero_point,
        "zp_state":  cfg.state.zero_point,
        "zp_pre":    cfg.pre.zero_point,
        "zp_output": cfg.output.zero_point,
        # Bias (already at pre scale; constant addend after requantize)
        "bias_pre": qmodel.bias_pre,
        # Reservoir-step multipliers (integer (M0, n))
        "M_in_M0":  qmodel.M_in_M0,  "M_in_n":  qmodel.M_in_n,
        "M_res_M0": qmodel.M_res_M0, "M_res_n": qmodel.M_res_n,
        "leak_M0":  qmodel.leak_M0,  "leak_n":  qmodel.leak_n,
        # Readout multipliers
        "M_out_bias_M0":  qmodel.M_out_bias_M0,
        "M_out_bias_n":   qmodel.M_out_bias_n,
        "M_out_input_M0": qmodel.M_out_input_M0,
        "M_out_input_n":  qmodel.M_out_input_n,
        "M_out_state_M0": qmodel.M_out_state_M0,
        "M_out_state_n":  qmodel.M_out_state_n,
        # LUT strategy
        "lut_kind": strat.kind.value,
        "lut_offset": qmodel.lut_offset,
        # Topology + chain constants (for structured-topology specialisation)
        "structured":       is_structured,
        "chain_weight_q":   chain_weight_q,
        "chain_feedback_q": chain_feedback_q,
        # Integer preprocess (only used when input_offset != 0 or
        # input_scaling != 1; otherwise the kernel reads X directly as u_pre)
        "has_integer_preprocess": qmodel.has_integer_preprocess,
        "pre_M0":    qmodel.pre_M0,
        "pre_n":     qmodel.pre_n,
        "pre_const": qmodel.pre_const,
        "head": head or "logits",
    }
    if sm is not None:
        weights["sm_lut"] = sm.lut_q
        md["sm_dmin_q"] = sm.dmin_q
        md["sm_n"] = sm.n
        md["sm_idx_frac"] = sm.idx_frac
        md["sm_prob_frac"] = sm.prob_frac
    if strat.kind == LUTKind.LINEAR_INTERP:
        md["lut_n_entries"]       = strat.n_entries
        md["lut_interp_frac_bits"] = strat.interp_frac_bits
        md["lut_idx_M0"]           = art.idx_M0
        md["lut_idx_n"]            = art.idx_n
    elif strat.kind == LUTKind.POLYNOMIAL:
        md["poly_qf_bits"]    = strat.poly_qf_bits
        md["poly_degree"]     = strat.poly_degree
        md["poly_x_M0"]       = art.x_to_qf_M0
        md["poly_x_n"]        = art.x_to_qf_n
        md["poly_back_M0"]    = art.qf_to_state_M0
        md["poly_back_n"]     = art.qf_to_state_n
        md["poly_clip_qf"]    = art.x_clip_qf
        md["poly_one_qf"]     = art.one_qf
        md["poly_a1_qf"]      = art.poly_a1_qf
        md["poly_a3_qf"]      = art.poly_a3_qf
        md["poly_a5_qf"]      = art.poly_a5_qf

    return Module(K=K, N=N, M=M, weights=weights,
                   ops=[TimeLoop(body=body)], metadata=md)
