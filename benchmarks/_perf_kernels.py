"""Shared model / quantization / object-emit / reference helpers for the
LLVM-path performance benches (Cortex-M0 and WASM share these).

A "dtype" is one of "float" (f32) or "i8"/"i16"/"i32" (symmetric quantized).
For each dtype + W_res kernel strategy this builds an `rc_predict` object for
a given triple and the matching embedded input X + reference Y (as numpy
arrays; each target formats the literals itself, C vs Rust). The integer
reference is the host quantized kernel (bit-exact with the cross-compiled
kernel); the float reference is the host f64 kernel cast to f32 (compared
with a small tolerance — the dense/csr/unroll f32 kernels are mutually
bit-exact, but f32-vs-f64 differs by rounding).
"""
from __future__ import annotations
import pathlib

import numpy as np
import llvmlite.binding as llvm

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I8Symmetric, I16FixedPoint, I32FixedPoint,
    quantize_model,
)
from rclite.codegen.llvm import (
    CompiledRC, CompiledQuantizedRC, cross_compile_rc,
    emit_quantized_module, _ensure_all_targets,
)
from rclite.ir import sparse_passes

FLOAT_EPS = 1e-2   # f32 device vs f64 host tolerance (contractive ESN, bounded)
INT_EPS = 0.5      # integer kernels are exact → diff must round to 0

# state_frac must keep (1<<state_frac) inside the signed storage width.
_QCFG = {
    8:  (QuantConfig(state_frac=5, input_frac=6, weight_frac=6), I8Symmetric()),
    16: (QuantConfig(state_frac=12, input_frac=12, weight_frac=12),
         I16FixedPoint()),
    32: (QuantConfig(state_frac=16, input_frac=12, weight_frac=12),
         I32FixedPoint()),
}
_BITS = {"i8": 8, "i16": 16, "i32": 32}
_NP = {8: np.int8, 16: np.int16, 32: np.int32}


def train_model(units, density, t_seq, seed=7):
    """Train an ESN; return (rc, exe, x_seq) with x_seq a raw (T,1) float input."""
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=60,
                            include_bias=True, include_input=False, name="out"),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((400, 1)) * 0.15
    exe.fit(X[:340], np.sin(np.arange(340) * 0.1)[:, None])
    return rc, exe, X[340:340 + t_seq]


def sym_qmodel(rc, exe, bits):
    cfg, target = _QCFG[bits]
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128), target=target)


def _optimize_and_emit(mod_ir, triple, cpu, out_path):
    mod_ir.triple = triple
    _ensure_all_targets()
    m = llvm.parse_assembly(str(mod_ir))
    m.verify()
    tgt = llvm.Target.from_triple(triple)
    tm = tgt.create_target_machine(cpu=cpu, opt=2, reloc="static")
    pto = llvm.create_pipeline_tuning_options()
    pto.speed_level = 2
    pto.loop_vectorization = False
    pto.slp_vectorization = False
    pb = llvm.create_pass_builder(tm, pto)
    pb.getModulePassManager().run(m, pb)
    out_path.write_bytes(tm.emit_object(m))
    return out_path


def build_object(dtype, qm_or_rcexe, sparse, *, triple, cpu, out_path):
    """Emit an optimized rc_predict object for `dtype`+`sparse` at `triple`.

    For "float", qm_or_rcexe is (rc, exe); otherwise it is the symmetric
    quantized model. `sparse` is None / "csr" / "unroll".
    """
    if dtype == "float":
        rc, exe = qm_or_rcexe
        cc = cross_compile_rc(
            rc, exe, triple=triple, cpu=cpu, dtype="f32",
            passes=sparse_passes(sparse, include_structural=True))
        cc.emit_object(str(out_path))
        return out_path
    mod = emit_quantized_module(
        qm_or_rcexe, passes=sparse_passes(sparse, include_structural=False))
    return _optimize_and_emit(mod, triple, cpu, out_path)


def reference_data(dtype, qm_or_rcexe, x_seq):
    """Return (X_arr, Y_ref_arr, eps, np_dtype, K, M, T) for the harness.

    X_arr/Y_ref_arr are numpy arrays the caller formats into C/Rust literals.
    """
    x_in = x_seq if x_seq.ndim > 1 else x_seq[:, None]
    T, K = x_in.shape
    if dtype == "float":
        rc, exe = qm_or_rcexe
        X = np.ascontiguousarray(x_in, dtype=np.float32)
        Yf = CompiledRC(rc, exe).predict(x_in)
        if Yf.ndim == 1:
            Yf = Yf[:, None]
        Y = np.ascontiguousarray(Yf, dtype=np.float32)
        return X, Y, FLOAT_EPS, np.float32, K, Y.shape[1], T

    qm = qm_or_rcexe
    bits = qm.target.storage_bits
    npd = _NP[bits]
    cfg = qm.config
    X = np.ascontiguousarray(
        qm.target.quantize_input_array(x_in, cfg).astype(npd))
    if X.ndim == 1:
        X = X[:, None]
    yf = CompiledQuantizedRC(qm).predict(x_in)
    if yf.ndim == 1:
        yf = yf[:, None]
    Y = np.rint(yf * cfg.state_scale).astype(npd)
    return X, Y, INT_EPS, npd, K, qm.M, T


def wres_bytes(dtype, qm_or_rcexe, sparse, N):
    """W_res table bytes for the size column (None for float / unroll)."""
    if dtype == "float":
        return None            # float W_res table size not the headline here
    if sparse == "unroll":
        return 0               # baked into code
    bits = qm_or_rcexe.target.storage_bits
    sb = bits // 8
    if sparse is None:
        return N * N * sb
    from rclite.ir.passes.sparsify import build_csr
    val, col, rowptr = build_csr(np.asarray(qm_or_rcexe.W_res_q))
    col_b = 2 if N <= 32767 else 4
    return len(val) * sb + len(col) * col_b + len(rowptr) * 4


KERNEL_SPARSE = {"dense": None, "csr": "csr", "unroll": "unroll"}
