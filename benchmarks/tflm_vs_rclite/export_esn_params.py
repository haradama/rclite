"""Export the trained float ESN (the exact model rclite deploys) so the TF
venv can rebuild the *same* reservoir as a TFLite single-step cell.

Run with the rclite venv:
    .venv/bin/python benchmarks/tflm_vs_rclite/export_esn_params.py

Writes out/esn_params.npz with W_in, W_res (dense), W_out (split into
bias/input/state blocks), leak, reservoir bias, input_offset, and the dims.
This is the SAME rc as eval_rclite.build_rc / gen_rclite_fw, so the rclite
firmware and the TFLM firmware run a byte-identical float reservoir.
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

OUT = pathlib.Path(__file__).resolve().parent / "out"
OUT.mkdir(exist_ok=True)


def main() -> int:
    s = common.series().astype(np.float64)
    X, Y = s[:-1, None], s[1:, None]
    n_fit = common.TRAIN_END
    rc = build_rc(float(X[:n_fit].mean()))
    exe = RCExecutor(rc)
    exe.fit(X[:n_fit], Y[:n_fit])

    K = rc.input.units
    N = rc.reservoir.units
    W_out = np.asarray(exe.W_out)  # (M, F)
    off = 0
    w_bias = None
    w_in_out = None
    if rc.readout.include_bias:
        w_bias = W_out[:, 0:1]
        off = 1
    if rc.readout.include_input:
        w_in_out = W_out[:, off : off + K]
        off += K
    w_state_out = W_out[:, off : off + N]

    np.savez(
        OUT / "esn_params.npz",
        W_in=np.asarray(exe.W_in),  # (N, K)
        W_res=np.asarray(exe.W_res),  # (N, N) dense (SCR is sparse here)
        W_out_bias=(
            w_bias if w_bias is not None else np.zeros((W_out.shape[0], 0))
        ),
        W_out_input=(
            w_in_out if w_in_out is not None else np.zeros((W_out.shape[0], 0))
        ),
        W_out_state=w_state_out,
        leak=np.float64(rc.reservoir.leak_rate),
        res_bias=np.float64(rc.reservoir.bias),
        input_offset=np.float64(rc.input.input_offset),
        input_scaling=np.float64(rc.input.input_scaling),
        K=K,
        N=N,
        M=rc.readout.units,
        include_bias=rc.readout.include_bias,
        include_input=rc.readout.include_input,
    )
    nnz = int(np.count_nonzero(exe.W_res))
    print(
        f"exported ESN: N={N} K={K} M={rc.readout.units}, "
        f"W_res nonzeros={nnz}/{N * N} (SCR), leak={float(rc.reservoir.leak_rate)}"
    )
    print(f"  -> {OUT / 'esn_params.npz'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
