"""per-channel (per output-row) W_out affine quantization — M>1 readouts.

Opt-in via `calibrate_from_data(..., per_channel_W_out=True)`: each readout
output channel gets its own W_out block scales, and the readout requantize
uses per-row (M0[m], n[m]). Unlike per-channel W_res, this directly targets
multi-output readouts (classification / MIMO), where output rows genuinely
differ in coefficient magnitude. Verifies:

  - host JIT == Python executor (integer, bit-exact) for i8/i16, M>1
  - emitted C (gcc) == executor (bit-exact), M>1
  - composes bit-exactly with per-channel W_res and sparse (csr/unroll)
  - per-tensor default unchanged
  - accuracy: per-channel improves (or ties) MIMO quantized MSE
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
)
from rclite.codegen.llvm import CompiledAffineRC
from rclite.targets.arduino import emit_affine_kernel_c
from rclite.ir import SparsifyReservoir


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE_GCC = shutil.which("gcc") is not None


def _model(M=4, K=2, units=44, density=0.2, seed=4):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=M, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=60,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((700, K)) * 0.3
    # heterogeneous-amplitude targets so output rows differ in scale
    Y = np.stack([(k + 1) * 0.5 * np.sin(np.arange(700) * 0.03 * (k + 1))
                  for k in range(M)], axis=1)
    exe.fit(X[:520], Y[:520])
    return rc, exe, X, Y


def _qm(rc, exe, X, sb, pc_out, pc_res=False):
    cfg = calibrate_from_data(rc, exe, X[:520], storage_bits=sb,
                              per_channel_W_out=pc_out, per_channel_W_res=pc_res)
    return quantize_model_affine(rc, exe, cfg)


def _python_qy(qm, X_float):
    qexe = AffineQuantizedExecutor(qm)
    qexe.reset()
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        x_raw_q = qexe._quantize_raw_input(X_float[t])
        u_pre_q = qexe._quantize_u_pre(X_float[t])
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(x_raw_q, qexe.state_q)
    return out


def _run_c(kernel_src, q_x, T, K, M, ctype):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(kernel_src)
        xs = ", ".join(str(int(v)) for v in q_x)
        (td / "main.c").write_text(
            '#include <stdint.h>\n#include <stdio.h>\n'
            f'extern void rc_predict(int32_t, const {ctype}*, {ctype}*);\n'
            'int main(void){\n'
            f'  {ctype} X[{T * K}] = {{ {xs} }};\n'
            f'  {ctype} Y[{T * M}];\n'
            f'  rc_predict({T}, X, Y);\n'
            f'  for (int i = 0; i < {T * M}; i++) printf("%d\\n", (int)Y[i]);\n'
            '  return 0;\n}\n')
        exe_path = td / "a.out"
        r = subprocess.run(
            ["gcc", "-O2", "-std=c99", "-o", str(exe_path),
             str(td / "main.c"), str(td / "kernel.c")],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run([str(exe_path)], capture_output=True,
                             text=True).stdout
        return np.array([int(v) for v in out.strip().split("\n")],
                        dtype=np.int64).reshape(T, M)


# ---------------------------------------------------------------------------

def test_jit_matches_executor():
    rc, exe, X, _ = _model(M=4)
    Xt = X[520:560]
    for sb in (8, 16):
        qm = _qm(rc, exe, X, sb, pc_out=True)
        assert qm.M_out_state_M0_arr is not None
        assert qm.M_out_state_M0_arr.shape == (qm.M,)
        yref = AffineQuantizedExecutor(qm).predict(Xt)
        yj = CompiledAffineRC(qm).predict(Xt)
        assert float(np.max(np.abs(yj - yref))) == 0.0, f"i{sb} JIT diff"
    print("  per-channel W_out i8/i16 (M=4): JIT == executor (bit-exact)")


def test_compose_with_wres_and_sparse():
    rc, exe, X, _ = _model(M=3)
    Xt = X[520:560]
    qm = _qm(rc, exe, X, 8, pc_out=True, pc_res=True)
    yref = AffineQuantizedExecutor(qm).predict(Xt)
    assert float(np.max(np.abs(CompiledAffineRC(qm).predict(Xt) - yref))) == 0
    for strat in ("csr", "unroll"):
        ys = CompiledAffineRC(
            qm, passes=[SparsifyReservoir(strategy=strat)]).predict(Xt)
        assert float(np.max(np.abs(ys - yref))) == 0, f"+{strat} diff"
    print("  per-channel W_out + W_res + sparse(csr/unroll): bit-exact")


def test_c_matches_executor():
    if not HAVE_GCC:
        print("  (skip: gcc not on PATH)")
        return
    rc, exe, X, _ = _model(M=4)
    Xt = X[520:555]
    T = Xt.shape[0]
    for sb in (8, 16):
        qm = _qm(rc, exe, X, sb, pc_out=True)
        ctype = {8: "int8_t", 16: "int16_t"}[sb]
        q_x = qm.config.input.quantize_array(Xt).astype(np.int64).reshape(-1)
        yref = _python_qy(qm, Xt)
        yc = _run_c(emit_affine_kernel_c(qm), q_x, T, qm.K, qm.M, ctype)
        assert int(np.max(np.abs(yref - yc))) == 0, f"i{sb} C diff"
        ycs = _run_c(emit_affine_kernel_c(qm, sparse="csr"),
                     q_x, T, qm.K, qm.M, ctype)
        assert int(np.max(np.abs(yref - ycs))) == 0, "C +csr diff"
    print("  per-channel W_out i8/i16 (M=4) C(gcc) == executor (incl. +csr)")


def test_per_tensor_unchanged():
    rc, exe, X, _ = _model(M=4)
    qm = _qm(rc, exe, X, 8, pc_out=False)
    assert qm.M_out_state_M0_arr is None
    assert qm.config.W_out_state_scales is None
    Xt = X[520:560]
    yref = AffineQuantizedExecutor(qm).predict(Xt)
    assert float(np.max(np.abs(CompiledAffineRC(qm).predict(Xt) - yref))) == 0
    print("  per-tensor W_out default unchanged (scalar M_out, JIT==executor)")


def test_accuracy_mimo():
    """On heterogeneous-amplitude MIMO targets per-channel W_out should help."""
    ratios = []
    for seed in (1, 2, 3, 4):
        rc, exe, X, Y = _model(M=4, seed=seed)
        Xt, Yt = X[520:640], Y[520:640]
        mse = {}
        for pc in (False, True):
            qm = _qm(rc, exe, X, 8, pc_out=pc)
            yq = AffineQuantizedExecutor(qm).predict(Xt)
            mse[pc] = float(np.mean((yq - Yt) ** 2))
        ratios.append(mse[True] / max(mse[False], 1e-12))
    arr = np.array(ratios)
    assert arr.mean() <= 1.0, (
        f"per-channel W_out did not help MIMO on avg (mean ratio {arr.mean():.3f})")
    print(f"  MIMO M=4 per-channel/per-tensor MSE ratios "
          f"{[round(r,3) for r in ratios]} mean={arr.mean():.3f} (<1 = better)")


TESTS = [
    test_jit_matches_executor,
    test_compose_with_wres_and_sparse,
    test_c_matches_executor,
    test_per_tensor_unchanged,
    test_accuracy_mimo,
]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
        except Exception:
            failures += 1
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
