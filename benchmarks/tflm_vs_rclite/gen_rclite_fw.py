"""Emit rclite portable-C firmware sources for the micro:bit (Cortex-M0).

Trains the reservoir on the same Mackey-Glass task, QAT-quantizes (i8 and
i16 affine), and writes — per variant — a self-contained firmware dir under
firmware/rclite_<v>/ with:
  rc_kernel.c   portable integer kernel (rclite.export emitter)
  rc_data.h     storage typedef, T/K/M, embedded int test sequence + the
                bit-exact AffineQuantizedExecutor reference outputs

Run with the rclite venv:
    .venv/bin/python benchmarks/tflm_vs_rclite/gen_rclite_fw.py
"""
from __future__ import annotations
import pathlib
import sys

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import common  # noqa: E402
from eval_rclite import build_rc  # noqa: E402

from rclite.runtime import RCExecutor  # noqa: E402
from rclite.quant import (  # noqa: E402
    search_quantization_affine, AffineQuantizedExecutor, LUTStrategy,
)
from rclite.targets.arduino import emit_affine_kernel_c  # noqa: E402

HERE = pathlib.Path(__file__).resolve().parent
FW = HERE / "firmware"

T_FW = 200          # firmware test-sequence length
START = common.TRAIN_END   # start the test sequence in the held-out region


def _c_type(bits: int) -> str:
    return {8: "signed char", 16: "short"}[bits]


def emit_variant(name: str, qm):
    out = FW / f"rclite_{name}"
    out.mkdir(parents=True, exist_ok=True)
    (out / "rc_kernel.c").write_text(emit_affine_kernel_c(qm))

    cfg = qm.config
    sb = qm.storage_bits
    np_t = {8: np.int8, 16: np.int16}[sb]

    s = common.series().astype(np.float64)
    X = s[:-1, None]
    x_seq = X[START:START + T_FW]
    Xq = cfg.input.quantize_array(x_seq).astype(np_t)

    qexe = AffineQuantizedExecutor(qm)
    qexe.reset()
    Yref = np.zeros((T_FW, qm.M), dtype=np_t)
    for t in range(T_FW):
        x_raw_q = qexe._quantize_raw_input(x_seq[t])
        u_pre_q = qexe._quantize_u_pre(x_seq[t])
        qexe.step_q(u_pre_q)
        Yref[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(np_t)

    ct = _c_type(sb)
    h = "\n".join([
        "#ifndef RC_DATA_H_",
        "#define RC_DATA_H_",
        f"typedef {ct} rc_fw_storage_t;",
        f"#define RC_FW_T {T_FW}",
        f"#define RC_FW_K {qm.K}",
        f"#define RC_FW_M {qm.M}",
        f"static const rc_fw_storage_t g_x[{T_FW * qm.K}] = {{ "
        + ",".join(str(int(v)) for v in Xq.ravel()) + " };",
        f"static const rc_fw_storage_t g_y_ref[{T_FW * qm.M}] = {{ "
        + ",".join(str(int(v)) for v in Yref.ravel()) + " };",
        "#endif",
        "",
    ])
    (out / "rc_data.h").write_text(h)
    print(f"  wrote {out}/rc_kernel.c + rc_data.h  (storage=i{sb}, T={T_FW})")


def main() -> int:
    s = common.series().astype(np.float64)
    X = s[:-1, None]
    Y = s[1:, None]
    n_fit = common.TRAIN_END
    rc = build_rc(float(X[:n_fit].mean()))
    exe = RCExecutor(rc)
    exe.fit(X[:n_fit], Y[:n_fit])

    print("QAT i8 ...")
    r8 = search_quantization_affine(
        rc, exe, X[:n_fit], Y[:n_fit], X[:n_fit], Y[:n_fit],
        storage_bits=8, lut_strategy=LUTStrategy.linear_interp(64), n_iterations=3)
    emit_variant("i8", r8.best_qmodel)

    print("QAT i16 ...")
    r16 = search_quantization_affine(
        rc, exe, X[:n_fit], Y[:n_fit], X[:n_fit], Y[:n_fit],
        storage_bits=16, lut_strategy=LUTStrategy.linear_interp(64), n_iterations=3)
    emit_variant("i16", r16.best_qmodel)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
