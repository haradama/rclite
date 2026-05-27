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

from rclite.core.profile import Topology
from rclite.ir.module import Module
from rclite.ir.ops import ReservoirStep, BuildPhi, ReadoutLinear, TimeLoop

from .lut import LUTKind
from .quantize import AffineQuantizedModel


def build_ir_from_quantized_affine(qmodel: AffineQuantizedModel) -> Module:
    rc = qmodel.rc
    cfg = qmodel.config
    if rc.input.input_offset != 0.0:
        raise NotImplementedError(
            "affine IR currently requires input_offset == 0 "
            f"(got {rc.input.input_offset})"
        )
    if rc.input.input_scaling != 1.0:
        raise NotImplementedError(
            "affine IR currently requires input_scaling == 1 "
            f"(got {rc.input.input_scaling})"
        )

    K, N, M = qmodel.K, qmodel.N, qmodel.M
    F = qmodel.F

    # MVP: always emit dense W_res, even for structured topologies. The Python
    # executor already does dense matmul + zp folding; structured-specialised
    # affine codegen is a future optimisation.
    weights: dict[str, np.ndarray] = {
        "W_in": qmodel.W_in_q,
        "W_res": qmodel.W_res_q,
        "W_out": qmodel.W_out_q,
        "row_sum_W_in": qmodel.row_sum_W_in,
        "row_sum_W_res": qmodel.row_sum_W_res,
        "row_sum_Wout_state": qmodel.row_sum_Wout_state,
    }
    # The LUT global is only emitted for the table-based strategies.
    if qmodel.lut_strategy.kind in (LUTKind.DIRECT, LUTKind.LINEAR_INTERP):
        weights["lut_table"] = qmodel.lut_q
    if rc.readout.include_input:
        if qmodel.row_sum_Wout_input is None:
            raise RuntimeError(
                "qmodel has include_input=True but row_sum_Wout_input is None"
            )
        weights["row_sum_Wout_input"] = qmodel.row_sum_Wout_input

    body = (
        ReservoirStep(
            leak=float(rc.reservoir.leak_rate),
            bias=float(rc.reservoir.bias),
            N=N, K=K,
            topology=rc.reservoir.topology,
            chain_weight=float(rc.reservoir.chain_weight),
            chain_feedback=float(rc.reservoir.chain_feedback),
            W_res_name="W_res",
        ),
        BuildPhi(
            include_bias=rc.readout.include_bias,
            include_input=rc.readout.include_input,
            K=K, N=N,
        ),
        ReadoutLinear(M=M, F=F),
    )

    art = qmodel.lut_artifacts
    strat = qmodel.lut_strategy
    md = {
        "quantization": "affine",
        "dtype": f"i{qmodel.storage_bits}",
        "storage_bits": qmodel.storage_bits,
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
    }
    if strat.kind == LUTKind.LINEAR_INTERP:
        md["lut_n_entries"]       = strat.n_entries
        md["lut_interp_frac_bits"] = strat.interp_frac_bits
        md["lut_idx_M0"]           = art.idx_M0
        md["lut_idx_n"]            = art.idx_n
    elif strat.kind == LUTKind.POLYNOMIAL:
        md["poly_qf_bits"]    = strat.poly_qf_bits
        md["poly_x_M0"]       = art.x_to_qf_M0
        md["poly_x_n"]        = art.x_to_qf_n
        md["poly_back_M0"]    = art.qf_to_state_M0
        md["poly_back_n"]     = art.qf_to_state_n
        md["poly_clip_qf"]    = art.x_clip_qf
        md["poly_one_qf"]     = art.one_qf

    return Module(K=K, N=N, M=M, weights=weights,
                   ops=[TimeLoop(body=body)], metadata=md)
