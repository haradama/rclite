"""xDSL-built first-class `!quant.uniform` type signature of an affine model.

Stage-(1) "IR construction" migration of `mlir_quant_types.py`. xDSL has no
`quant` dialect, so this module gives it a **minimal `quant.uniform` type**: a
`ParametrizedAttribute` whose verbatim body reproduces mlir-opt's surface
syntax (`<i8:f32, scale[:zp]>` per-tensor / `<i8:f32:0, {s0,s1,...}>` per-axis).
The signature `func.func` is then assembled with the xDSL API and printed; the
emitted module verifies under mlir-opt exactly like the text emitter.

The scale/zero-point string formatting (`_f`) is shared with the text emitter,
so the `!quant.uniform<...>` types are byte-identical between the two paths.
This stays a declarative, type-level contract — the arith emitters
(`mlir_affine` / `mlir_symmetric`) remain the executable realization.
"""
from __future__ import annotations
import io

import numpy as np

from xdsl.irdl import irdl_attr_definition
from xdsl.ir import ParametrizedAttribute, TypeAttribute
from xdsl.dialects import func, memref
from xdsl.dialects.builtin import ModuleOp, TensorType, StringAttr
from xdsl.ir import Region, Block
from xdsl.builder import ImplicitBuilder
from xdsl.printer import Printer

from rclite.core.profile import Topology
from rclite.quant.affine.quantize import AffineQuantizedModel

_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)
_DYN = memref.DYNAMIC_INDEX


@irdl_attr_definition
class QuantUniform(ParametrizedAttribute, TypeAttribute):
    """`!quant.uniform<...>` — body holds the verbatim inner syntax."""
    name = "quant.uniform"
    body: StringAttr

    def print_parameters(self, printer: Printer) -> None:
        printer.print_string("<" + self.body.data + ">")


def _f(x) -> str:
    return f"{float(x):.8e}"


def uniform_type(sb: int, scale, zero_point: int = 0) -> QuantUniform:
    """`!quant.uniform` type. `scale` may be scalar (per-tensor) or 1-D array
    (per-axis along output axis 0). zero_point applies per-tensor."""
    arr = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if arr.size == 1:
        zp = f":{int(zero_point)}" if zero_point else ""
        inner = f"i{sb}:f32, {_f(arr[0])}{zp}"
    else:
        scales = ",".join(_f(s) for s in arr)
        inner = f"i{sb}:f32:0, {{{scales}}}"
    return QuantUniform(StringAttr(inner))


def emit_quant_types_xdsl(qmodel: AffineQuantizedModel) -> str:
    """Build (via xDSL) a verifiable MLIR module declaring the affine model's
    quantities as `!quant.uniform` types (per-axis where per-channel)."""
    rc, cfg = qmodel.rc, qmodel.config
    sb = qmodel.storage_bits
    wob = qmodel.w_out_storage_bits
    structured = rc.reservoir.topology in _STRUCTURED
    N, K, M, F = qmodel.N, qmodel.K, qmodel.M, qmodel.F

    t_input = uniform_type(sb, cfg.input.scale, cfg.input.zero_point)
    t_state = uniform_type(sb, cfg.state.scale, cfg.state.zero_point)
    t_out = uniform_type(sb, cfg.output.scale, cfg.output.zero_point)
    t_win = uniform_type(sb, cfg.W_in.scale)
    wout_scale = (cfg.W_out_state_scales if cfg.W_out_state_scales is not None
                  else cfg.W_out_state.scale)
    t_wout = uniform_type(wob, wout_scale)

    # operands mirror the text emitter's order: x, h, w_in, [w_res], w_out, y
    tys = [TensorType(t_input, [_DYN]),
           TensorType(t_state, [N]),
           TensorType(t_win, [N, K])]
    if not structured:
        wres_scale = (cfg.W_res_scales if cfg.W_res_scales is not None
                      else cfg.W_res.scale)
        tys.append(TensorType(uniform_type(sb, wres_scale), [N, N]))
    tys.append(TensorType(t_wout, [M, F]))
    tys.append(TensorType(t_out, [_DYN]))

    region = Region([Block(arg_types=tys)])
    with ImplicitBuilder(region.block):
        func.ReturnOp()
    sig = func.FuncOp("rc_quant_signature", (tys, []), region)

    mod = ModuleOp([sig])
    mod.verify()
    buf = io.StringIO()
    Printer(stream=buf).print_op(mod)
    return buf.getvalue() + "\n"
