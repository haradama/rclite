"""Build an rclite IR `Module` for the integer (i32) path from a QuantizedModel.

The integer path differs from the float builder in two ways:

  - **LUT-based tanh.** The reservoir step always uses a quantized LUT
    for tanh; libm tanhf is unavailable in int math.

  - **Pre-quantized weights.** `weights` carries i32 arrays and the LUT
    table; the lowering treats them as integer globals.

The IR body starts with `PreprocessInput`, mirroring the float pipeline:
the kernel consumes already-input-quantized raw samples and computes
`u_pre_q` internally via fixed-point arithmetic. This keeps the
include_input readout passthrough bit-exact with the float reference
when `input_offset != 0` or `input_scaling != 1`.

Module-level metadata records `dtype="i32"`, the Q-format fractional
widths, and the LUT geometry — the LLVM lowering reads these to pick
shift amounts and clamp constants.
"""
from __future__ import annotations

from rclite.core.profile import Aggregation, Topology
from rclite.ir.module import Module
from rclite.ir.ops import (
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear, TimeLoop,
    Argmax, Softmax,
)
from .model import QuantizedModel
from .softmax_lut import SoftmaxLUTSpec, build_params as build_softmax_params


def build_ir_from_quantized(qmodel: QuantizedModel, *, head=None) -> Module:
    """Construct an rclite IR module for the i32 integer path.

    `head="classify"` appends an Argmax so the kernel emits an int32 class
    id per step instead of M quantized scores. argmax is monotone in the
    quantized logits, so the result is identical to argmax over the float
    readout — quantization introduces no class errors except at exact ties.
    `head="proba"` appends a fixed-point Softmax (exp LUT) emitting M
    probabilities at Q.prob_frac in the storage type.
    """
    rc = qmodel.rc
    cfg = qmodel.config
    if qmodel.lut is None or qmodel.lut_table_q is None:
        raise ValueError("QuantizedModel must have a LUT for the integer path")
    if head not in (None, "logits", "classify", "proba"):
        raise NotImplementedError(
            f"quantized integer path supports head in (None, 'logits', "
            f"'classify', 'proba'); got {head!r}"
        )
    if rc.readout.aggregation != Aggregation.NONE:
        raise NotImplementedError(
            "Quantized classification currently supports per-step readouts "
            "(aggregation=NONE) only; sequence pooling is not yet quantized."
        )

    K, N, M = qmodel.K, qmodel.N, qmodel.M
    F = qmodel.F

    is_structured = rc.reservoir.topology in (
        Topology.DLR, Topology.DLRB, Topology.SCR
    )

    weights = {
        "W_in": qmodel.W_in_q,
        "W_out": qmodel.W_out_q,
        "lut_table": qmodel.lut_table_q,
    }
    # Structured topologies (DLR/DLRB/SCR) get a topology-aware kernel in
    # the lowering — no need to emit the (mostly-zero) W_res matrix.
    if not is_structured:
        weights["W_res"] = qmodel.W_res_q

    body = (
        PreprocessInput(
            offset=float(rc.input.input_offset),
            scale=float(rc.input.input_scaling),
            K=K,
        ),
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
    )
    sm = None
    if head == "classify":
        body = body + (Argmax(M=M),)
    elif head == "proba":
        sm = build_softmax_params(
            SoftmaxLUTSpec(), s_diff=1.0 / cfg.state_scale,
            storage_bits=qmodel.target.storage_bits,
            storage_dtype=qmodel.target.storage_dtype,
        )
        body = body + (Softmax(M=M),)
        weights["sm_lut"] = sm.lut_q

    # Pick IR-level dtype string from the target's storage width
    if qmodel.target.storage_bits == 32:
        dtype = "i32"
    elif qmodel.target.storage_bits == 16:
        dtype = "i16"
    elif qmodel.target.storage_bits == 8:
        dtype = "i8"
    else:
        raise NotImplementedError(
            f"storage width {qmodel.target.storage_bits} not supported by IR"
        )

    return Module(
        K=K, N=N, M=M,
        weights=weights,
        ops=[TimeLoop(body=body)],
        metadata={
            "dtype": dtype,
            "topology": rc.reservoir.topology.name,
            "include_bias": rc.readout.include_bias,
            "include_input": rc.readout.include_input,
            "feature_dim": F,
            "state_frac": cfg.state_frac,
            "input_frac": cfg.input_frac,
            "weight_frac": cfg.weight_frac,
            "lut_n": qmodel.lut.n,
            "lut_xmin_q": int(qmodel.lut.xmin * cfg.state_scale),
            "lut_xmax_q": int(qmodel.lut.xmax * cfg.state_scale),
            "leak_q": qmodel.target.quantize_state(rc.reservoir.leak_rate, cfg),
            "bias_q": qmodel.target.quantize_state(rc.reservoir.bias, cfg),
            "head": head or "logits",
            **({} if sm is None else {
                "sm_dmin_q": sm.dmin_q,
                "sm_n": sm.n,
                "sm_idx_frac": sm.idx_frac,
                "sm_prob_frac": sm.prob_frac,
            }),
        },
    )
