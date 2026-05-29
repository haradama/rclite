"""ESN single-step cell as a model module for examples/arm/aot_arm_compiler.py
(same reservoir rclite deploys). Defines ModelUnderTest + ModelInputs.

Reads the float ESN params exported by
benchmarks/tflm_vs_rclite/export_esn_params.py.
"""
import pathlib
import sys

import numpy as np
import torch
import torch.nn as nn

HERE = pathlib.Path(__file__).resolve().parent
ESN_PARAMS = HERE.parent.parent / "tflm_vs_rclite" / "out" / "esn_params.npz"
sys.path.insert(0, str(HERE.parent))


class ESNCell(nn.Module):
    def __init__(self, p):
        super().__init__()
        N, K = int(p["N"]), int(p["K"])
        self.K = K
        leak = float(p["leak"])
        self.register_buffer("leak_v", torch.full((1, N), leak))
        self.register_buffer("oml_v", torch.full((1, N), 1.0 - leak))
        self.register_buffer("off_v", torch.full((1, K), float(p["input_offset"])))
        self.register_buffer("scal_v", torch.full((1, K), float(p["input_scaling"])))
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
            yb = (p["W_out_bias"].reshape(-1) if p["W_out_bias"].size
                  else np.zeros(int(p["M"]), np.float32))
            self.yout.bias.copy_(torch.tensor(yb.astype(np.float32)))

    def forward(self, x, h_prev):
        u = (x - self.off_v) * self.scal_v
        pre = self.pre(torch.cat([u, h_prev], dim=-1))
        h_t = self.oml_v * h_prev + self.leak_v * torch.tanh(pre)
        y = self.yout(torch.cat([x, h_t], dim=-1))
        return h_t, y


_p = np.load(ESN_PARAMS)
_N = int(_p["N"])
ModelUnderTest = ESNCell(_p).eval()
ModelInputs = (torch.zeros(1, 1), torch.zeros(1, _N))
