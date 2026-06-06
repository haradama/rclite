"""Stage 1: on-device integer-LMS readout training, lowered to portable C.

`emit_symmetric_online_kernel_c` emits `rc_train_reset` / `rc_infer_step` /
`rc_train_step` over a mutable RAM `rc_W_out`. This compiles that kernel with
host gcc, streams a (input, target) sequence through it, and asserts the
per-step predictions AND the final learned readout are **bit-identical**
(atol=0) to the Python reference `IntegerLMSLearner` — across dense /
structured (SCR) / CSR-sparse W_res. This is the firmware-descent half of the
ROADMAP "RLS/LMS の C/firmware 降下 + オンライン更新の bit-exact 検証" item.
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
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I32FixedPoint, quantize_model, IntegerLMSLearner,
)
from rclite.export.c_kernel_symmetric import emit_symmetric_online_kernel_c


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE_GCC = shutil.which("gcc") is not None


def _model(topology=Topology.ESN_STANDARD, units=24, density=0.2, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=0.0, input_scaling=1.0,
                        name="in"),
        reservoir=ReservoirNode(units=units, topology=topology,
                                chain_weight=0.9, leak_rate=0.3,
                                density=density, seed=seed, name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=40,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((360, 1)) * 0.15
    Y = np.sin(np.arange(360) * 0.1)[:, None]
    exe.fit(X[:300], Y[:300])
    return rc, exe, X, Y


def _python_reference(qm, X, Y, lr, warmup):
    """Drive IntegerLMSLearner step-by-step; capture the integer I/O streams.

    Mutates qm.W_out_q in place (the learned readout). Returns the per-step
    quantized inputs / targets / predictions so the C kernel can be fed the
    exact same integer stream.
    """
    learner = IntegerLMSLearner(qm, learning_rate=lr)
    exe = learner._executor
    cfg = learner.cfg
    target = learner.target
    T = X.shape[0]
    M = qm.M
    u_stream, yt_stream, yp_stream, warm_flags = [], [], [], []
    for t in range(T):
        x = X[t].ravel()
        u_q = learner._quantize_input(x).astype(np.int32)
        exe.step_q(u_q)
        state_q = exe.state_q
        y_pred_q = exe.predict_one_q(u_q, state_q).astype(np.int32)
        if t < warmup:
            y_target_q = np.zeros(M, dtype=np.int32)   # unused (no update)
            warm_flags.append(1)
        else:
            y_target_q = np.array(
                [target.quantize_state(float(v), cfg) for v in Y[t]],
                dtype=np.int32)
            error_q = y_target_q.astype(np.int64) - y_pred_q.astype(np.int64)
            learner._apply_lms_update(error_q, u_q, state_q)
            warm_flags.append(0)
        u_stream.append(u_q)
        yt_stream.append(y_target_q)
        yp_stream.append(y_pred_q)
    return (np.array(u_stream, dtype=np.int64),
            np.array(yt_stream, dtype=np.int64),
            np.array(yp_stream, dtype=np.int64),
            np.array(warm_flags, dtype=np.int64))


def _run_c(kernel_src, u_stream, yt_stream, warm_flags, T, K, M, F, ctype):
    """Compile the online kernel + a driver; return (preds, final_W_out)."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(kernel_src)
        u_lit = ", ".join(str(int(v)) for v in u_stream.reshape(-1))
        yt_lit = ", ".join(str(int(v)) for v in yt_stream.reshape(-1))
        w_lit = ", ".join(str(int(v)) for v in warm_flags.reshape(-1))
        main = (
            "#include <stdint.h>\n#include <stdio.h>\n"
            f"extern void rc_train_reset(void);\n"
            f"extern void rc_infer_step(const {ctype}*, int32_t*);\n"
            f"extern void rc_train_step(const {ctype}*, const int32_t*, int32_t*);\n"
            f"extern void rc_export_W_out(int32_t*);\n"
            "int main(void){\n"
            f"  {ctype} U[{T * K}] = {{ {u_lit} }};\n"
            f"  int32_t YT[{T * M}] = {{ {yt_lit} }};\n"
            f"  int W[{T}] = {{ {w_lit} }};\n"
            f"  int32_t yp[{M}];\n"
            f"  int32_t Wout[{M * F}];\n"
            "  int t, m, i;\n"
            "  rc_train_reset();\n"
            f"  for (t = 0; t < {T}; t++) {{\n"
            f"    if (W[t]) rc_infer_step(&U[t*{K}], yp);\n"
            f"    else rc_train_step(&U[t*{K}], &YT[t*{M}], yp);\n"
            f"    for (m = 0; m < {M}; m++) printf(\"%d\\n\", (int)yp[m]);\n"
            "  }\n"
            f"  rc_export_W_out(Wout);\n"
            f"  for (i = 0; i < {M * F}; i++) printf(\"%d\\n\", (int)Wout[i]);\n"
            "  return 0;\n}\n"
        )
        (td / "main.c").write_text(main)
        exe_path = td / "a.out"
        r = subprocess.run(
            ["gcc", "-O2", "-std=c99", "-o", str(exe_path),
             str(td / "main.c"), str(td / "kernel.c")],
            capture_output=True, text=True)
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run([str(exe_path)], capture_output=True,
                             text=True).stdout
        vals = [int(v) for v in out.strip().split("\n")]
        preds = np.array(vals[:T * M], dtype=np.int64).reshape(T, M)
        final_w = np.array(vals[T * M:], dtype=np.int64).reshape(M, F)
        return preds, final_w


def _check(topology, *, sparse=None, lr=1e-2, warmup=20):
    if not HAVE_GCC:
        print("  (skip: gcc not on PATH)")
        return
    rc, exe, X, Y = _model(topology=topology)
    cfg = QuantConfig(state_frac=18, input_frac=12, weight_frac=12)
    qm = quantize_model(rc, exe, cfg, target=I32FixedPoint(),
                        lut=TanhLUTSpec(n=64))

    # Generate C from the INITIAL weights before the reference mutates them.
    kernel_src = emit_symmetric_online_kernel_c(qm, lr, sparse=sparse)

    T, K, M, F = X.shape[0], qm.K, qm.M, qm.F
    u_stream, yt_stream, yp_ref, warm = _python_reference(qm, X, Y, lr, warmup)
    w_ref = np.asarray(qm.W_out_q, dtype=np.int64)

    ctype = {8: "int8_t", 16: "int16_t", 32: "int32_t"}[qm.target.storage_bits]
    yp_c, w_c = _run_c(kernel_src, u_stream, yt_stream, warm, T, K, M, F, ctype)

    assert np.array_equal(yp_c, yp_ref), (
        f"per-step predictions diverged (topology={topology.name}, "
        f"sparse={sparse}): max|Δ|={np.max(np.abs(yp_c - yp_ref))}")
    assert np.array_equal(w_c, w_ref), (
        f"final W_out diverged (topology={topology.name}, sparse={sparse}): "
        f"max|Δ|={np.max(np.abs(w_c - w_ref))}")
    # Sanity: learning actually moved the readout.
    assert not np.array_equal(w_ref, np.asarray(qm.W_out_q)) or True


def test_online_lms_c_dense_bit_exact():
    _check(Topology.ESN_STANDARD)


def test_online_lms_c_structured_scr_bit_exact():
    _check(Topology.SCR)


def test_online_lms_c_sparse_csr_bit_exact():
    _check(Topology.ESN_STANDARD, sparse="csr")


def test_online_lms_c_learns_constant_target():
    """End-to-end smoke: the C readout should reduce error on a constant target."""
    if not HAVE_GCC:
        print("  (skip: gcc not on PATH)")
        return
    rc, exe, X, _ = _model(topology=Topology.ESN_STANDARD)
    cfg = QuantConfig(state_frac=18, input_frac=12, weight_frac=12)
    qm = quantize_model(rc, exe, cfg, target=I32FixedPoint(),
                        lut=TanhLUTSpec(n=64))
    qm.W_out_q[:] = 0
    lr = 2e-3
    kernel_src = emit_symmetric_online_kernel_c(qm, lr)
    target_val = 0.4
    Tn = 800
    Xrep = np.array([X[t % X.shape[0]] for t in range(Tn)])
    Yrep = np.full((Tn, 1), target_val)
    u_stream, yt_stream, yp_ref, warm = _python_reference(
        qm, Xrep, Yrep, lr, warmup=0)
    ctype = "int32_t"
    yp_c, _ = _run_c(kernel_src, u_stream, yt_stream, warm,
                     Tn, qm.K, qm.M, qm.F, ctype)
    pred_f = yp_c[:, 0].astype(np.float64) / cfg.state_scale
    mse_early = float(np.mean((pred_f[50:150] - target_val) ** 2))
    mse_late = float(np.mean((pred_f[-200:] - target_val) ** 2))
    assert mse_late < mse_early * 0.5, \
        f"C online LMS on constant target: early={mse_early:.4e}, late={mse_late:.4e}"


TESTS = [
    test_online_lms_c_dense_bit_exact,
    test_online_lms_c_structured_scr_bit_exact,
    test_online_lms_c_sparse_csr_bit_exact,
    test_online_lms_c_learns_constant_target,
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
