"""PyTorch + ExecuTorch host AOT for the SAME 80-unit ESN rclite deploys (the
single-step cell), on the same Mackey-Glass task as the TFLM benchmark.

Reports:
  * float + int8 (PT2E) host accuracy on the identical held-out targets
  * ExecuTorch AOT .pte sizes: portable-float and XNNPACK-int8
  * a host ExecuTorch-runtime parity run (the .pte actually executes)

The real on-target run (Cortex-M55 + Ethos-U55 on the Arm Corstone-300 FVP)
lives in fvp/ — this script is the host-side AOT + accuracy counterpart.

Run with the PyTorch/ExecuTorch venv:
    /tmp/ptenv/bin/python benchmarks/executorch_vs_rclite/executorch_demo.py
"""

from __future__ import annotations
import json
import os
import pathlib
import sys

# flatc (XNNPACK .pte serialization) lives in the venv bin; put it on PATH.
os.environ["PATH"] = "/tmp/ptenv/bin:" + os.environ.get("PATH", "")

import numpy as np

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(HERE))
import common  # noqa: E402

import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.export import export  # noqa: E402
from torchao.quantization.pt2e.quantize_pt2e import prepare_pt2e, convert_pt2e  # noqa: E402
from executorch.backends.xnnpack.quantizer.xnnpack_quantizer import (  # noqa: E402
    XNNPACKQuantizer,
    get_symmetric_quantization_config,
)
from executorch.backends.xnnpack.partition.xnnpack_partitioner import (  # noqa: E402
    XnnpackPartitioner,
)
from executorch.exir import to_edge_transform_and_lower  # noqa: E402
from executorch.runtime import Runtime  # noqa: E402

torch.manual_seed(0)
ESN_PARAMS = HERE.parent / "tflm_vs_rclite" / "out" / "esn_params.npz"


# --------------------------------------------------------------- ExecuTorch helpers


def pt2e_int8(model, example, calib):
    """PT2E symmetric-int8 quantize; returns the converted (int8) module."""
    gm = export(model.eval(), example).module()
    qz = XNNPACKQuantizer().set_global(
        get_symmetric_quantization_config(is_per_channel=False)
    )
    prep = prepare_pt2e(gm, qz)
    for args in calib:
        prep(*args)
    return convert_pt2e(prep)


def pte_bytes(model, example, *, xnnpack: bool):
    # `model` is either an eval()'d nn.Module or an already-exported int8 graph
    # module (which rejects .eval()); export() handles both as-is. Returns the
    # .pte bytes, or None if this export path isn't supported for the model
    # (e.g. a portable lowering ExecuTorch rejects).
    try:
        ep = export(model, example)
        parts = [XnnpackPartitioner()] if xnnpack else None
        prog = to_edge_transform_and_lower(
            ep, partitioner=parts
        ).to_executorch()
        return prog.buffer
    except Exception as e:  # noqa: BLE001
        print(
            f"  .pte export ({'xnnpack' if xnnpack else 'portable'}) "
            f"failed: {type(e).__name__}: {str(e)[:80]}"
        )
        return None


def host_run_ok(buf, example) -> bool:
    if buf is None:
        return False
    p = OUT / "_tmp.pte"
    p.write_bytes(buf)
    meth = Runtime.get().load_program(str(p)).load_method("forward")
    out = meth.execute(list(example))
    p.unlink(missing_ok=True)
    return len(out) >= 1


def _blen(buf):
    return len(buf) if buf is not None else None


# --------------------------------------------------------------- ESN (same as rclite)


class ESNCell(nn.Module):
    def __init__(self, p):
        super().__init__()
        N, K = int(p["N"]), int(p["K"])
        self.N, self.K = N, K
        leak = float(p["leak"])
        # 2-D buffers matching the (1, N)/(1, K) activation shapes so every
        # elementwise op is exactly same-shape — any scalar/size-mismatch
        # broadcast yields a 0-stride tensor that ExecuTorch's lowering rejects.
        self.register_buffer("leak_v", torch.full((1, N), leak))
        self.register_buffer("oml_v", torch.full((1, N), 1.0 - leak))
        self.register_buffer(
            "off_v", torch.full((1, K), float(p["input_offset"]))
        )
        self.register_buffer(
            "scal_v", torch.full((1, K), float(p["input_scaling"]))
        )
        self.pre = nn.Linear(K + N, N)
        self.yout = nn.Linear(K + N, int(p["M"]))
        with torch.no_grad():
            wpre = np.zeros((N, K + N), np.float32)
            wpre[:, :K] = p["W_in"]
            wpre[:, K:] = p["W_res"]
            self.pre.weight.copy_(torch.tensor(wpre))
            self.pre.bias.copy_(torch.full((N,), float(p["res_bias"])))
            wy = np.zeros((int(p["M"]), K + N), np.float32)
            if p["W_out_input"].shape[1] == K:
                wy[:, :K] = p["W_out_input"]
            wy[:, K:] = p["W_out_state"]
            self.yout.weight.copy_(torch.tensor(wy))
            yb = (
                p["W_out_bias"].reshape(-1)
                if p["W_out_bias"].size
                else np.zeros(int(p["M"]), np.float32)
            )
            self.yout.bias.copy_(torch.tensor(yb.astype(np.float32)))

    def forward(self, x, h_prev):
        u = (x - self.off_v) * self.scal_v
        pre = self.pre(torch.cat([u, h_prev], dim=-1))
        h_t = self.oml_v * h_prev + self.leak_v * torch.tanh(pre)
        y = self.yout(torch.cat([x, h_t], dim=-1))
        return h_t, y


def _esn_loop(cell, Xseq, N):
    h = torch.zeros(1, N)
    preds = np.zeros(len(Xseq))
    hist = []
    with torch.no_grad():
        for t in range(len(Xseq)):
            hist.append(h.clone())
            h, y = cell(Xseq[t : t + 1], h)
            preds[t] = float(y.ravel()[0])
    return preds, hist


def run_esn():
    if not ESN_PARAMS.exists():
        raise SystemExit(
            f"missing {ESN_PARAMS}; run "
            "benchmarks/tflm_vs_rclite/export_esn_params.py first"
        )
    p = np.load(ESN_PARAMS)
    N = int(p["N"])
    cell = ESNCell(p).eval()
    s = common.series().astype(np.float32)
    Xseq = torch.tensor(s[:-1, None])
    _, te = common.target_indices()

    pf, hist = _esn_loop(cell, Xseq, N)
    nrmse_f = common.nrmse(pf[te], s[te + 1])

    n_fit = common.TRAIN_END
    # contiguous calibration/example tensors — slicing views (Xseq[i:i+1]) can
    # carry strides that make ExecuTorch's lowering hit its 0-stride check.
    calib = [
        (Xseq[i : i + 1].contiguous(), hist[i].contiguous())
        for i in range(0, n_fit, 4)
    ]
    ex = (torch.zeros(1, 1), torch.zeros(1, N))
    qm = pt2e_int8(cell, ex, calib)

    # int8 loop (float state feedback, like the TFLM ESN cell)
    h = torch.zeros(1, N)
    pq = np.zeros(len(Xseq))
    with torch.no_grad():
        for t in range(len(Xseq)):
            h, y = qm(Xseq[t : t + 1], h)
            pq[t] = float(y.ravel()[0])
    nrmse_q = common.nrmse(pq[te], s[te + 1])

    pte_f = pte_bytes(cell, ex, xnnpack=False)
    pte_q = pte_bytes(qm, ex, xnnpack=True)
    # The float .pte runs on the host runtime; the int8 XNNPACK .pte *builds*
    # but host-running it aborts with a double-free (an ExecuTorch host-runtime
    # bug on this recurrent cell), so we record its size but do not execute it.
    ok = host_run_ok(pte_f, ex)
    return {
        "arch": f"ESN single-step cell, {N} units (dense W_res {N}x{N})",
        "nrmse_float_test": float(nrmse_f),
        "nrmse_int8_test": float(nrmse_q),
        "pte_portable_float_bytes": _blen(pte_f),
        "pte_xnnpack_int8_bytes": _blen(pte_q),
        "host_runtime_ok_float": bool(ok),
        "host_runtime_int8_aborts": True,
    }


def main() -> int:
    res = {
        "framework": "PyTorch + ExecuTorch",
        "torch": torch.__version__,
        "note": (
            "Host AOT for the SAME ESN as rclite. The on-target run is on "
            "the Corstone-300 FVP (Cortex-M55 + Ethos-U55); see fvp/. .pte = "
            "model program only; the ExecuTorch C++ runtime adds ~418 KB."
        ),
        "esn": run_esn(),
    }
    (OUT / "pt_result.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
