"""Textual MLIR printer for rclite IR.

Emits a representation using the `rc.*` op names and an `scf.for` time
loop, intentionally matching rc-bench's dialect shape so the output can
be inspected with `mlir-opt --verify-each` when the toolchain is around.

The printer is for inspection and future cross-tool use. It does NOT
emit dense weight initializers (they would dominate the text); a
`dense<...>` placeholder is shown instead. The full numerical values are
preserved on the in-memory `Module` and are emitted by the LLVM lowering.
"""
from __future__ import annotations
from typing import List

from .module import Module
from .ops import (
    Op,
    PreprocessInput, ReservoirStep, BuildPhi, ReadoutLinear,
    FusedStepReadout, TimeLoop,
    Argmax, Softmax, AccumulateState, FinalizeAggregate,
)


def to_mlir_text(module: Module) -> str:
    out: List[str] = []
    out.append("// rclite IR (textual MLIR; rc-bench dialect compatible)")
    md = module.metadata
    out.append(f"// topology={md.get('topology', '?')} "
               f"N={module.N} K={module.K} M={module.M} "
               f"F={md.get('feature_dim', '?')}")
    out.append("module {")
    fty = "f32"
    for name, arr in module.weights.items():
        shape = "x".join(str(d) for d in arr.shape) + f"x{fty}"
        out.append(f'  memref.global "private" constant @{name} '
                   f": memref<{shape}> = dense<...>")
    out.append("  func.func @rc_predict(%T: i64, %X: memref<?x?xf32>, "
               "%Y: memref<?x?xf32>) {")
    out.append("    %c0 = arith.constant 0 : index")
    out.append("    %c1 = arith.constant 1 : index")
    for op in module.ops:
        out.extend(_emit_op(op, 4))
    out.append("    return")
    out.append("  }")
    out.append("}")
    return "\n".join(out)


def _emit_op(op: Op, indent: int) -> List[str]:
    pad = " " * indent
    if isinstance(op, TimeLoop):
        lines: List[str] = []
        if op.unroll > 1:
            lines.append(f"{pad}// rc.time_loop unroll={op.unroll}")
        else:
            lines.append(f"{pad}// rc.time_loop")
        lines.append(f"{pad}scf.for %t = %c0 to %T step %c1 {{")
        for body_op in op.body:
            lines.extend(_emit_op(body_op, indent + 2))
        lines.append(f"{pad}}}")
        return lines
    if isinstance(op, PreprocessInput):
        return [f"{pad}rc.preprocess_input %X[%t], %u_pre "
                f'{{offset = {op.offset:.6g} : f32, '
                f'scale = {op.scale:.6g} : f32}}']
    if isinstance(op, ReservoirStep):
        topo = op.topology.name
        attrs = (f'leak = {op.leak:.6g} : f32, bias = {op.bias:.6g} : f32, '
                 f'topology = "{topo}"')
        if topo in ("DLR", "DLRB", "SCR"):
            attrs += f", chain_weight = {op.chain_weight:.6g} : f32"
            if topo == "DLRB":
                attrs += f", chain_feedback = {op.chain_feedback:.6g} : f32"
        operands = "%h, %u_pre, @" + op.W_in_name
        if op.W_res_name is not None:
            operands += ", @" + op.W_res_name
        if op.res_sparse is not None:
            attrs += (f', sparse = "{op.res_sparse.kind}", '
                      f"nnz = {op.res_sparse.nnz} : i64")
        return [f"{pad}rc.reservoir_step {operands} {{{attrs}}}"]
    if isinstance(op, BuildPhi):
        return [f"{pad}rc.build_phi %h, %X[%t], %phi "
                f'{{include_bias = {str(op.include_bias).lower()}, '
                f'include_input = {str(op.include_input).lower()}}}']
    if isinstance(op, ReadoutLinear):
        return [f"{pad}rc.readout_linear %phi, @{op.W_out_name}, %Y[%t]"]
    if isinstance(op, FusedStepReadout):
        topo = op.topology.name
        attrs = (f'leak = {op.leak:.6g} : f32, bias = {op.bias:.6g} : f32, '
                 f'topology = "{topo}"')
        if topo in ("DLR", "DLRB", "SCR"):
            attrs += f", chain_weight = {op.chain_weight:.6g} : f32"
            if topo == "DLRB":
                attrs += f", chain_feedback = {op.chain_feedback:.6g} : f32"
        attrs += (f", include_bias_phi = {str(op.include_bias_phi).lower()}"
                  f", include_input_phi = {str(op.include_input_phi).lower()}")
        operands = "%h, %u_pre, %X[%t], @" + op.W_in_name
        if op.W_res_name is not None:
            operands += ", @" + op.W_res_name
        operands += ", @" + op.W_out_name
        if op.res_sparse is not None:
            attrs += (f', sparse = "{op.res_sparse.kind}", '
                      f"nnz = {op.res_sparse.nnz} : i64")
        return [f"{pad}rc.fused_step_readout {operands}, %Y[%t] {{{attrs}}}"]
    if isinstance(op, AccumulateState):
        return [f'{pad}rc.accumulate_state %h {{mode = "{op.mode}"}}']
    if isinstance(op, FinalizeAggregate):
        return [f'{pad}rc.finalize_aggregate %h {{mode = "{op.mode}"}}']
    if isinstance(op, Argmax):
        return [f"{pad}rc.argmax %Y[%t] : i32"]
    if isinstance(op, Softmax):
        return [f"{pad}rc.softmax %Y[%t]"]
    return [f"{pad}// <unknown op {type(op).__name__}>"]
