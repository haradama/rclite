"""First-class `!quant.uniform` type representation of an affine model.

Expresses the model's quantization (scale / zero-point per quantity, and
**per-axis** scales for per-channel W_res / W_out) as MLIR `quant` dialect
types, and emits a verifiable MLIR module declaring the kernel signature in
those types. This makes the quantization *first-class in the type system*
(the phase-2 "量子化を第一級の型に" goal), and `verify()` confirms the types
are well-formed under mlir-opt.

This is intentionally a **type-level / declarative** artifact, not the
executable path: lowering `!quant.uniform` through to native code leaves
residual `quant` ops and would risk the host↔device bit-exactness that the
arith-based emitters (`mlir_affine` / `mlir_symmetric`) guarantee. The arith
kernels are the executable realization of exactly these types; this module
documents the type-level contract and validates it with the toolchain.
"""
from __future__ import annotations
import shutil
import subprocess
import tempfile
import pathlib
from typing import List, Optional

import numpy as np

from rclite.core.profile import Topology
from rclite.quant.affine.quantize import AffineQuantizedModel

_STRUCTURED = (Topology.DLR, Topology.DLRB, Topology.SCR)


def _f(x) -> str:
    return f"{float(x):.8e}"


def uniform_type(sb: int, scale, zero_point: int = 0) -> str:
    """`!quant.uniform` type string. `scale` may be a scalar (per-tensor) or a
    1-D array (per-axis along output axis 0). zero_point applies per-tensor."""
    arr = np.atleast_1d(np.asarray(scale, dtype=np.float64))
    if arr.size == 1:
        zp = f":{int(zero_point)}" if zero_point else ""
        return f"!quant.uniform<i{sb}:f32, {_f(arr[0])}{zp}>"
    scales = ",".join(_f(s) for s in arr)
    # per-axis (axis 0); symmetric (zp omitted) — per-channel weights are zp=0
    return f"!quant.uniform<i{sb}:f32:0, {{{scales}}}>"


def emit_quant_types(qmodel: AffineQuantizedModel) -> str:
    """Emit a verifiable MLIR module declaring the affine model's quantities
    as `!quant.uniform` types (per-axis where the model is per-channel)."""
    rc = qmodel.rc
    cfg = qmodel.config
    sb = qmodel.storage_bits
    wob = qmodel.w_out_storage_bits
    structured = rc.reservoir.topology in _STRUCTURED

    # activations (asymmetric: carry zero-point)
    t_input = uniform_type(sb, cfg.input.scale, cfg.input.zero_point)
    t_state = uniform_type(sb, cfg.state.scale, cfg.state.zero_point)
    t_out = uniform_type(sb, cfg.output.scale, cfg.output.zero_point)
    # W_res: per-axis if per-channel, else per-tensor (symmetric)
    if not structured:
        wres_scale = (cfg.W_res_scales if cfg.W_res_scales is not None
                      else cfg.W_res.scale)
        t_wres = uniform_type(sb, wres_scale)
    # W_out state block: per-axis if per-channel, else per-tensor
    wout_scale = (cfg.W_out_state_scales if cfg.W_out_state_scales is not None
                  else cfg.W_out_state.scale)
    t_wout = uniform_type(wob, wout_scale)
    t_win = uniform_type(sb, cfg.W_in.scale)

    N, K, M = qmodel.N, qmodel.K, qmodel.M
    pc_res = cfg.W_res_scales is not None
    pc_out = cfg.W_out_state_scales is not None

    L: List[str] = []
    a = L.append
    a("// rclite affine model — first-class quant.uniform type signature")
    a(f"// topology={rc.reservoir.topology.name} N={N} K={K} M={M} "
      f"storage=i{sb} w_out=i{wob}")
    a(f"// per-channel: W_res={pc_res} W_out={pc_out}")
    a("module {")
    # A signature func whose operands carry the quantized types. The body is a
    # no-op; this is a declarative, verifiable type contract.
    # quant.uniform is a valid *tensor* element type (not memref).
    args = [f"%x: tensor<?x{t_input}>",
            f"%h: tensor<{N}x{t_state}>",
            f"%w_in: tensor<{N}x{K}x{t_win}>"]
    if not structured:
        args.append(f"%w_res: tensor<{N}x{N}x{t_wres}>")
    args.append(f"%w_out: tensor<{M}x{qmodel.F}x{t_wout}>")
    args.append(f"%y: tensor<?x{t_out}>")
    a(f"  func.func @rc_quant_signature({', '.join(args)}) {{")
    a("    return")
    a("  }")
    a("}")
    return "\n".join(L) + "\n"


def verify(mlir_text: str) -> bool:
    """Return True if mlir-opt parses & verifies the module."""
    if shutil.which("mlir-opt") is None:
        raise RuntimeError("mlir-opt not on PATH")
    with tempfile.TemporaryDirectory() as td:
        p = pathlib.Path(td) / "q.mlir"
        p.write_text(mlir_text)
        r = subprocess.run(["mlir-opt", str(p)], capture_output=True, text=True)
        return r.returncode == 0
