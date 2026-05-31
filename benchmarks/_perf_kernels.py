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
    QuantConfig, TanhLUTSpec, I32FixedPoint, LUTStrategy, quantize_model,
)
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.codegen.llvm import (
    CompiledRC, CompiledQuantizedRC, CompiledAffineRC, cross_compile_rc,
    emit_quantized_module, emit_quantized_affine_module, _ensure_all_targets,
)
from rclite.ir import sparse_passes

# Quantization scheme per width:
#   i8 / i16 → AFFINE (data-calibrated scales) — accurate; symmetric fixed
#     Q-format saturates the ridge W_out coefficients and gives misleading,
#     non-monotonic accuracy. Matches the AVR C-kernel path (also affine).
#   i32 → SYMMETRIC fixed-point — affine i32 overflows the i64 requantize, and
#     i32 is wide enough that symmetric does not saturate (accuracy ≈ float).
# A linear-interp LUT keeps the activation table small (the DIRECT LUT is
# 2**bits entries — 128 KB at i16).
_AFFINE_BITS = {"i8": 8, "i16": 16}
_BITS = {"i8": 8, "i16": 16, "i32": 32}
_NP = {8: np.int8, 16: np.int16, 32: np.int32}

FLOAT_EPS = 1e-2   # f32 device vs f64 host tolerance (contractive ESN, bounded)
INT_EPS = 0.5      # integer kernels are exact → diff must round to 0


def train_model(units, density, t_seq, seed=7):
    """Train a one-step-ahead forecaster on a noisy sine series.

    Returns (rc, exe, x_seq, y_true, x_cal): x_seq is the raw (T,1) float eval
    input, y_true the (T,1) ground-truth next-step target (for the MSE
    column), and x_cal the training slice used to calibrate affine scales.
    """
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=60,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    series = np.sin(np.arange(1200) * 0.05) + 0.1 * rng.standard_normal(1200)
    X = series[:-1, None]
    Yt = series[1:, None]
    exe.fit(X[:900], Yt[:900])
    return rc, exe, X[900:900 + t_seq], Yt[900:900 + t_seq], X[:900]


def quant_model(dtype, rc, exe, x_cal):
    """Build the quantized model for `dtype`: affine for i8/i16, symmetric i32."""
    if dtype in _AFFINE_BITS:
        cfg = calibrate_from_data(rc, exe, x_cal,
                                  storage_bits=_AFFINE_BITS[dtype])
        return quantize_model_affine(
            rc, exe, cfg, lut_strategy=LUTStrategy.linear_interp(64))
    cfg = QuantConfig(state_frac=16, input_frac=12, weight_frac=12)
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128),
                          target=I32FixedPoint())


def _host_predict(dtype, src, x_seq):
    """Dequantized host output (real units) for the active quant scheme."""
    if dtype == "float":
        rc, exe = src
        return CompiledRC(rc, exe).predict(x_seq)
    if dtype in _AFFINE_BITS:
        return CompiledAffineRC(src).predict(x_seq)
    return CompiledQuantizedRC(src).predict(x_seq)


def accuracy_mse(dtype, src, x_seq, y_true):
    """MSE of the host (dequantized) model output vs the target, real units.

    The on-device kernel is bit-exact with this host reference (parity gate),
    so the host MSE equals the device MSE. Depends on dtype, not the kernel.
    """
    out = np.asarray(_host_predict(dtype, src, x_seq),
                     dtype=np.float64).reshape(y_true.shape)
    return float(np.mean((out - y_true) ** 2))


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


def build_object(dtype, src, sparse, *, triple, cpu, out_path):
    """Emit an optimized rc_predict object for `dtype`+`sparse` at `triple`.

    `src` is (rc, exe) for float, else the quantized model. `sparse` is
    None / "csr" / "unroll". i8/i16 lower the affine kernel, i32 the symmetric.
    """
    if dtype == "float":
        rc, exe = src
        cc = cross_compile_rc(
            rc, exe, triple=triple, cpu=cpu, dtype="f32",
            passes=sparse_passes(sparse, include_structural=True))
        cc.emit_object(str(out_path))
        return out_path
    emit = (emit_quantized_affine_module if dtype in _AFFINE_BITS
            else emit_quantized_module)
    mod = emit(src, passes=sparse_passes(sparse, include_structural=False))
    return _optimize_and_emit(mod, triple, cpu, out_path)


def reference_data(dtype, src, x_seq):
    """Return (X_arr, Y_ref_arr, eps, np_dtype, K, M, T) for the harness.

    X_arr/Y_ref_arr are numpy arrays the caller formats into C/Rust literals.
    The integer reference recovers the exact kernel storage ints from the
    host twin (CompiledAffineRC for i8/i16, CompiledQuantizedRC for i32).
    """
    x_in = x_seq if x_seq.ndim > 1 else x_seq[:, None]
    T, K = x_in.shape
    if dtype == "float":
        rc, exe = src
        X = np.ascontiguousarray(x_in, dtype=np.float32)
        Yf = np.asarray(CompiledRC(rc, exe).predict(x_in))
        if Yf.ndim == 1:
            Yf = Yf[:, None]
        return X, np.ascontiguousarray(Yf, np.float32), FLOAT_EPS, \
            np.float32, K, Yf.shape[1], T

    npd = _NP[_BITS[dtype]]
    cfg = src.config
    yf = np.asarray(_host_predict(dtype, src, x_in))
    if yf.ndim == 1:
        yf = yf[:, None]
    if dtype in _AFFINE_BITS:
        X = cfg.input.quantize_array(x_in).astype(npd)
        # quantize∘dequantize is the identity on storage ints → exact kernel Y
        Y = cfg.output.quantize_array(yf).astype(npd)
    else:
        X = src.target.quantize_input_array(x_in, cfg).astype(npd)
        Y = np.rint(yf * cfg.state_scale).astype(npd)
    if X.ndim == 1:
        X = X[:, None]
    return np.ascontiguousarray(X), Y, INT_EPS, npd, K, src.M, T


def wres_bytes(dtype, src, sparse, N):
    """W_res table bytes for the size column (None for float / unroll)."""
    if dtype == "float":
        return None            # float W_res table size not the headline here
    if sparse == "unroll":
        return 0               # baked into code
    sb = _BITS[dtype] // 8
    if sparse is None:
        return N * N * sb
    from rclite.ir.passes.sparsify import build_csr
    val, col, rowptr = build_csr(np.asarray(src.W_res_q))
    col_b = 2 if N <= 32767 else 4
    return len(val) * sb + len(col) * col_b + len(rowptr) * 4


KERNEL_SPARSE = {"dense": None, "csr": "csr", "unroll": "unroll"}
