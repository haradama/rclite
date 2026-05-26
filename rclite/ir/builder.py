"""Lower a trained ReservoirComputer into rclite IR."""
from __future__ import annotations

from rclite.core.composite import ReservoirComputer
from rclite.core.profile import Topology
from rclite.runtime.reference import RCExecutor

from .module import Module
from .ops import (
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear, TimeLoop,
)


def build_ir(rc: ReservoirComputer, exe: RCExecutor) -> Module:
    """Construct an rclite IR module from a trained ReservoirComputer.

    Default lowering shape (one op per logical phase, lined up under a
    `TimeLoop`):

        for t in 0..T:
            preprocess_input
            reservoir_step
            build_phi
            readout_linear
    """
    if exe.W_out is None:
        raise ValueError("Readout has not been trained — call fit() first")

    K = rc.input.units
    N = rc.reservoir.units
    M = rc.readout.units
    F = exe._feature_dim()

    is_structured = rc.reservoir.topology in (
        Topology.DLR, Topology.DLRB, Topology.SCR
    )

    weights = {"W_in": exe.W_in, "W_out": exe.W_out}
    if not is_structured:
        weights["W_res"] = exe.W_res

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
            include_bias=bool(rc.readout.include_bias),
            include_input=bool(rc.readout.include_input),
            K=K, N=N,
        ),
        ReadoutLinear(M=M, F=F),
    )

    return Module(
        K=K, N=N, M=M,
        weights=weights,
        ops=[TimeLoop(body=body)],
        metadata={
            "topology": rc.reservoir.topology.name,
            "include_bias": bool(rc.readout.include_bias),
            "include_input": bool(rc.readout.include_input),
            "feature_dim": F,
        },
    )
