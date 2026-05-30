#!/usr/bin/env python3
"""Build the interactive rclite WebAssembly demo for GitHub Pages.

Trains two small Echo State Networks and cross-compiles each to a zero-WASI
WebAssembly *reactor* module (via `rclite.targets.wasm.BrowserWasm`), then
assembles a self-contained static site that runs entirely in the browser:

  * ``forecast.wasm`` -- a broadband 1-step-ahead predictor. The page feeds it
    a user-controlled (or hand-drawn) waveform and overlays the ESN's forecast.
  * ``dream.wasm``    -- a clean quasi-periodic attractor. The page closes the
    loop in JS (feed each output back as the next input) so the network
    "dreams" its learned waveform autonomously.

Usage::

    python examples/wasm_pages_demo/build.py            # -> dist/
    python examples/wasm_pages_demo/build.py --out docs/demo
    (cd dist && python -m http.server)                  # then open localhost:8000

Requires the WASI rust target (``rustup target add wasm32-wasip1``) and a wasm
linker -- `wasm-ld`, or the `rust-lld` that ships with rustc (used
automatically as a fallback). No browser/node is needed to *build*.
"""
from __future__ import annotations
import argparse
import json
import pathlib
import shutil
import sys
import tempfile
import time

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
                    Activation, Topology, Trainer)
from rclite.runtime import RCExecutor
from rclite.targets import BrowserWasm

HERE = pathlib.Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------
# Model 1: broadband 1-step-ahead forecaster (drives the "Predict" mode).
# --------------------------------------------------------------------------

def build_forecast_model():
    t = np.arange(6001)
    comps = [(0.013, 0.50, 0.0), (0.026, 0.35, 1.1),
             (0.041, 0.30, 2.2), (0.058, 0.22, 0.4)]
    s = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in comps)
    s = s / np.max(np.abs(s)) * 0.9
    X, Y = s[:-1, None], s[1:, None]
    n_tr = 4500
    esn = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=float(X[:n_tr].mean()),
                        name="in"),
        reservoir=ReservoirNode(units=140, activation=Activation.TANH,
                                 spectral_radius=0.90, leak_rate=0.30,
                                 density=0.10, topology=Topology.ESN_STANDARD,
                                 seed=2, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                            trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=200, include_bias=True,
                            include_input=True, name="out"),
    )
    exe = RCExecutor(esn)
    exe.fit(X[:n_tr], Y[:n_tr])
    nrmse = _rmse(exe.predict(X[n_tr:]), Y[n_tr:]) / float(np.std(Y[n_tr:]))
    return esn, exe, {"task": "1-step forecast", "units": 140,
                      "test_nrmse": round(nrmse, 4)}


# --------------------------------------------------------------------------
# Model 2: clean quasi-periodic attractor (drives the autonomous "Dream").
# --------------------------------------------------------------------------

def _dream_signal(t):
    return 0.6 * np.sin(2 * np.pi * 0.040 * t) + \
        0.4 * np.sin(2 * np.pi * 0.017 * t + 0.7)


def build_dream_model():
    t = np.arange(4001)
    s = _dream_signal(t)
    X, Y = s[:-1, None], s[1:, None]
    n_tr = 3000
    esn = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_scaling=1.0, input_offset=float(X[:n_tr].mean()),
                        name="in"),
        reservoir=ReservoirNode(units=120, activation=Activation.TANH,
                                 spectral_radius=0.92, leak_rate=0.25,
                                 density=0.10, topology=Topology.ESN_STANDARD,
                                 seed=1, name="res"),
        readout=ReadoutNode(units=1, activation=Activation.IDENTITY,
                            trainer=Trainer.RIDGE, regularization=1e-7,
                            washout=200, include_bias=True,
                            include_input=True, name="out"),
    )
    exe = RCExecutor(esn)
    exe.fit(X[:n_tr], Y[:n_tr])
    nrmse = _rmse(exe.predict(X[n_tr:]), Y[n_tr:]) / float(np.std(Y[n_tr:]))
    seed = s[n_tr - 300:n_tr].astype(np.float32).tolist()
    return esn, exe, {"task": "autonomous attractor", "units": 120,
                      "test_nrmse": round(nrmse, 4), "seed": seed}


# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(HERE / "dist"),
                    help="output directory for the static site")
    ap.add_argument("--no-simd", action="store_true",
                    help="disable wasm SIMD128 (smaller compatibility baseline)")
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    simd = not args.no_simd

    sample = np.linspace(-0.8, 0.8, 16, dtype=np.float32)[:, None]
    meta = {"simd": simd, "built_unix": int(time.time()), "models": {}}

    print("[1/2] training + compiling forecast model …")
    esn_f, exe_f, info_f = build_forecast_model()
    print("[2/2] training + compiling dream model …")
    esn_d, exe_d, info_d = build_dream_model()

    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        bf = BrowserWasm(simd=simd, wasm_name="forecast.wasm")
        art_f = bf.compile(esn_f, exe_f, output_dir=tdp / "f",
                           test_inputs=sample)
        bd = BrowserWasm(simd=simd, wasm_name="dream.wasm")
        art_d = bd.compile(esn_d, exe_d, output_dir=tdp / "d",
                           test_inputs=sample)

        shutil.copy(art_f.binary, out / "forecast.wasm")
        shutil.copy(art_d.binary, out / "dream.wasm")
        # both models are f32/K=1/M=1 -> the generated loader is identical;
        # ship a single shared copy.
        shutil.copy(tdp / "f" / "rclite.js", out / "rclite.js")

        info_f["wasm_bytes"] = art_f.metadata["wasm_size"]
        info_f["imports"] = art_f.metadata["imports"]
        info_d["wasm_bytes"] = art_d.metadata["wasm_size"]
        info_d["imports"] = art_d.metadata["imports"]

    meta["models"]["forecast"] = info_f
    meta["models"]["dream"] = info_d
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(FRONTEND / name, out / name)

    print(f"\n[done] static site written to {out}/")
    print(f"       forecast.wasm: {info_f['wasm_bytes']} B "
          f"(NRMSE {info_f['test_nrmse']})")
    print(f"       dream.wasm   : {info_d['wasm_bytes']} B "
          f"(NRMSE {info_d['test_nrmse']})")
    print(f"       serve it:  (cd {out} && python -m http.server)")


if __name__ == "__main__":
    main()
