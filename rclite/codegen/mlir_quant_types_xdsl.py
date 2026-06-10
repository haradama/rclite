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
import pathlib
import shutil
import subprocess
import tempfile

from xdsl.dialects import func, memref
from xdsl.dialects.builtin import ModuleOp, TensorType
from xdsl.ir import Region, Block
from xdsl.builder import ImplicitBuilder
from xdsl.printer import Printer

from rclite.core.profile import Topology
from rclite.quant.affine.quantize import AffineQuantizedModel

# The `!quant.uniform` type and its formatter now live in the shared quant
# layer; this module keeps only the declarative type-signature emitter and
# re-exports `uniform_type` for the type-string test.
from .mlir_quant_xdsl import uniform_type  # noqa: F401 (re-exported)

_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)
_DYN = memref.DYNAMIC_INDEX


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
    wout_scale = (
        cfg.W_out_state_scales
        if cfg.W_out_state_scales is not None
        else cfg.W_out_state.scale
    )
    t_wout = uniform_type(wob, wout_scale)

    # operands mirror the text emitter's order: x, h, w_in, [w_res], w_out, y
    tys = [
        TensorType(t_input, [_DYN]),
        TensorType(t_state, [N]),
        TensorType(t_win, [N, K]),
    ]
    if not structured:
        wres_scale = (
            cfg.W_res_scales
            if cfg.W_res_scales is not None
            else cfg.W_res.scale
        )
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


def verify(mlir_text: str) -> bool:
    """Return True if mlir-opt parses & verifies the module (in-process subprocess)."""
    if shutil.which("mlir-opt") is None:
        raise RuntimeError("mlir-opt not on PATH")
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "q.mlir"
        p.write_text(mlir_text)
        r = subprocess.run(
            ["mlir-opt", str(p)], capture_output=True, text=True
        )
        return r.returncode == 0
