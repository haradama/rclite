"""Lower a trained ReservoirComputer into rclite IR."""
from __future__ import annotations

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Aggregation, Task, Topology
from rclite.runtime.reference import RCExecutor

from .module import Module
from .ops import (
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear, TimeLoop,
    Argmax, Softmax, AccumulateState, FinalizeAggregate,
)


_HEADS = (None, "logits", "proba", "classify")


def _head_op(head, M):
    """Return the post-readout op for a classification head, or None."""
    if head in (None, "logits"):
        return None
    if head == "proba":
        return Softmax(M=M)
    if head == "classify":
        return Argmax(M=M)
    raise ValueError(f"unknown head {head!r}; expected one of {_HEADS}")


def build_ir(rc: ReservoirComputer, exe: RCExecutor, *, head=None) -> Module:
    """Construct an rclite IR module from a trained ReservoirComputer.

    `head` selects the output format:
      None / "logits" — raw linear readout scores (regression / logits)
      "proba"          — softmax probabilities (classification)
      "classify"       — argmax class id (int32 output)

    Per-step lowering (aggregation == NONE):

        for t in 0..T:
            preprocess_input; reservoir_step; build_phi; readout_linear; [head]

    Sequence-to-label lowering (aggregation in {MEAN, LAST}) pools the state
    over time, then runs the readout once after the loop:

        for t in 0..T:
            preprocess_input; reservoir_step; accumulate_state
        finalize_aggregate; build_phi; readout_linear; [head]
    """
    if exe.W_out is None:
        raise ValueError("Readout has not been trained — call fit() first")
    if head not in _HEADS:
        raise ValueError(f"unknown head {head!r}; expected one of {_HEADS}")

    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()
    agg = rc.readout.aggregation

    is_structured = rc.reservoir.topology in (
        Topology.DLR, Topology.DLRB, Topology.SCR
    )

    weights = {"W_in": exe.W_in, "W_out": exe.W_out}
    if not is_structured:
        weights["W_res"] = exe.W_res

    preprocess = PreprocessInput(
        offset=float(rc.input.input_offset),
        scale=float(rc.input.input_scaling),
        K=K,
    )
    step = ReservoirStep(
        leak=float(rc.reservoir.leak_rate),
        bias=float(rc.reservoir.bias),
        N=N, K=K,
        topology=rc.reservoir.topology,
        chain_weight=float(rc.reservoir.chain_weight),
        chain_feedback=float(rc.reservoir.chain_feedback),
        W_res_name=None if is_structured else "W_res",
        activation=rc.reservoir.activation,
    )
    build_phi = BuildPhi(
        include_bias=bool(rc.readout.include_bias),
        include_input=bool(rc.readout.include_input),
        K=K, N=N,
    )
    readout = ReadoutLinear(M=M, F=F)
    head_op = _head_op(head, M)

    if agg == Aggregation.NONE:
        body = [preprocess, step, build_phi, readout]
        if head_op is not None:
            body.append(head_op)
        ops = [TimeLoop(body=tuple(body))]
    else:
        if rc.readout.include_input:
            raise NotImplementedError(
                "Sequence-aggregation codegen does not yet support "
                "include_input=True; set include_input=False on the readout."
            )
        mode = "mean" if agg == Aggregation.MEAN else "last"
        washout = int(rc.readout.washout)
        loop = TimeLoop(body=(
            preprocess, step,
            AccumulateState(N=N, mode=mode, washout=washout),
        ))
        ops = [loop, FinalizeAggregate(N=N, mode=mode, washout=washout),
               build_phi, readout]
        if head_op is not None:
            ops.append(head_op)

    return Module(
        K=K, N=N, M=M,
        weights=weights,
        ops=ops,
        metadata={
            "topology": rc.reservoir.topology.name,
            "include_bias": bool(rc.readout.include_bias),
            "include_input": bool(rc.readout.include_input),
            "feature_dim": F,
            "task": rc.readout.task.name,
            "aggregation": agg.name,
            "head": head or "logits",
            "n_classes": M if rc.readout.task == Task.CLASSIFICATION else 0,
        },
    )
