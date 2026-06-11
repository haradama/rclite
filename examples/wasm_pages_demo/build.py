#!/usr/bin/env python3
"""Build the interactive rclite WebAssembly demo for GitHub Pages.

Trains four small Echo State Networks and cross-compiles each to a zero-WASI
WebAssembly *reactor* module (via `rclite.targets.wasm.BrowserWasm`), then
assembles a single self-contained static site -- one page, four tabs -- that
runs entirely in the browser. Two **regression** reservoirs and two
**classification** reservoirs share the same scope:

  * ``forecast.wasm`` -- a broadband 1-step-ahead predictor. The page feeds it
    a user-controlled (or hand-drawn) waveform and overlays the ESN's forecast.
  * ``dream.wasm``    -- a clean quasi-periodic attractor. The page closes the
    loop in JS (feed each output back as the next input) so the network
    "dreams" its learned waveform autonomously.
  * ``shape.wasm``    -- a 5-class sequence-to-label classifier (MEAN
    aggregation). The page feeds a whole drawn / generated window; the kernel
    pools its states and emits one class-logit vector for the curve's shape.
  * ``trend.wasm``    -- a 2-class per-step classifier (NONE aggregation). A
    scrolling signal is labelled rising / falling at every timestep.

Classification needs no special kernel: each module emits the ordinary linear
readout's logits, and the page recovers the label with ``argmax`` and the
probabilities with ``softmax`` -- exactly what rclite does in Python.

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

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
    Task,
    Aggregation,
)
from rclite.runtime import RCExecutor
from rclite.targets import BrowserWasm

HERE = pathlib.Path(__file__).resolve().parent
FRONTEND = HERE / "frontend"

# Window the "shape" classifier sees (one labelled curve == one sequence).
SHAPE_W = 80
SHAPE_CLASSES = ["rising", "falling", "peak", "valley", "sine"]

# Per-step "trend" classifier: rising/falling vs the value TREND_K steps ago.
TREND_K = 8
TREND_WASHOUT = 100
TREND_CLASSES = ["falling", "rising"]


def _rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


# --------------------------------------------------------------------------
# Model 1: broadband 1-step-ahead forecaster (drives the "Predict" mode).
# --------------------------------------------------------------------------


def build_forecast_model():
    t = np.arange(6001)
    comps = [
        (0.013, 0.50, 0.0),
        (0.026, 0.35, 1.1),
        (0.041, 0.30, 2.2),
        (0.058, 0.22, 0.4),
    ]
    s = sum(a * np.sin(2 * np.pi * f * t + p) for f, a, p in comps)
    s = s / np.max(np.abs(s)) * 0.9
    X, Y = s[:-1, None], s[1:, None]
    n_tr = 4500
    esn = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=1.0,
            input_offset=float(X[:n_tr].mean()),
            name="in",
        ),
        reservoir=ReservoirNode(
            units=140,
            activation=Activation.TANH,
            spectral_radius=0.90,
            leak_rate=0.30,
            density=0.10,
            topology=Topology.ESN_STANDARD,
            seed=2,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(esn)
    exe.fit(X[:n_tr], Y[:n_tr])
    nrmse = _rmse(exe.predict(X[n_tr:]), Y[n_tr:]) / float(np.std(Y[n_tr:]))
    return (
        esn,
        exe,
        {
            "task": "1-step forecast",
            "units": 140,
            "test_nrmse": round(nrmse, 4),
        },
    )


# --------------------------------------------------------------------------
# Model 2: clean quasi-periodic attractor (drives the autonomous "Dream").
# --------------------------------------------------------------------------


def _dream_signal(t):
    return 0.6 * np.sin(2 * np.pi * 0.040 * t) + 0.4 * np.sin(
        2 * np.pi * 0.017 * t + 0.7
    )


def build_dream_model():
    t = np.arange(4001)
    s = _dream_signal(t)
    X, Y = s[:-1, None], s[1:, None]
    n_tr = 3000
    esn = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_scaling=1.0,
            input_offset=float(X[:n_tr].mean()),
            name="in",
        ),
        reservoir=ReservoirNode(
            units=120,
            activation=Activation.TANH,
            spectral_radius=0.92,
            leak_rate=0.25,
            density=0.10,
            topology=Topology.ESN_STANDARD,
            seed=1,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-7,
            washout=200,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(esn)
    exe.fit(X[:n_tr], Y[:n_tr])
    nrmse = _rmse(exe.predict(X[n_tr:]), Y[n_tr:]) / float(np.std(Y[n_tr:]))
    seed = s[n_tr - 300 : n_tr].astype(np.float32).tolist()
    return (
        esn,
        exe,
        {
            "task": "autonomous attractor",
            "units": 120,
            "test_nrmse": round(nrmse, 4),
            "seed": seed,
        },
    )


# --------------------------------------------------------------------------
# Model 3: sequence-to-label shape classifier (drives the "Shape" mode).
# --------------------------------------------------------------------------


def _shape_window(kind: int, rng, jitter: float = 0.06) -> np.ndarray:
    t = np.linspace(0.0, 1.0, SHAPE_W)
    if kind == 0:  # rising ramp
        s = -1.0 + 2.0 * t
    elif kind == 1:  # falling ramp
        s = 1.0 - 2.0 * t
    elif kind == 2:  # peak (triangle: rise then fall)
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    elif kind == 3:  # valley (V: fall then rise)
        s = -1.0 + 4.0 * np.abs(t - 0.5)
    else:  # one sine cycle
        s = np.sin(2.0 * np.pi * t)
    return (s + jitter * rng.standard_normal(SHAPE_W))[:, None]


def build_shape_model():
    rng = np.random.default_rng(5)
    seqs, labels = [], []
    for kind in range(len(SHAPE_CLASSES)):
        for _ in range(80):
            seqs.append(_shape_window(kind, rng))
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in idx]
    labels = np.array(labels)[idx]
    n_tr = int(0.7 * len(seqs))

    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=140,
            activation=Activation.TANH,
            spectral_radius=0.90,
            leak_rate=0.30,
            density=0.10,
            topology=Topology.RANDOM,
            seed=9,
            name="res",
        ),
        readout=ReadoutNode(
            units=len(SHAPE_CLASSES),
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=12,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.MEAN,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(seqs[:n_tr], labels[:n_tr])
    acc = float(np.mean(exe.predict_sequences(seqs[n_tr:]) == labels[n_tr:]))
    return (
        rc,
        exe,
        {
            "task": "shape (sequence-to-label)",
            "units": 140,
            "classes": SHAPE_CLASSES,
            "window": SHAPE_W,
            "aggregation": "mean",
            "test_acc": round(acc, 4),
        },
    )


# --------------------------------------------------------------------------
# Model 4: per-step rising/falling classifier (drives the "Trend" mode).
# --------------------------------------------------------------------------


def _trend_series(n: int, seed: int):
    rng = np.random.default_rng(seed)
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.96 * u[t - 1] + 0.04 * rng.standard_normal()
    u = u / (np.std(u) + 1e-9) * 0.6
    y = np.zeros(n, dtype=int)
    y[TREND_K:] = (u[TREND_K:] > u[:-TREND_K]).astype(int)  # 1 == rising
    return u[:, None], y


def build_trend_model():
    X, y = _trend_series(6000, seed=0)
    n_tr = 4200
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=120,
            activation=Activation.TANH,
            spectral_radius=0.97,
            leak_rate=0.50,
            density=0.10,
            topology=Topology.RANDOM,
            seed=4,
            name="res",
        ),
        readout=ReadoutNode(
            units=len(TREND_CLASSES),
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=TREND_WASHOUT,
            include_input=True,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.NONE,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:n_tr], y[:n_tr])
    acc = float(np.mean(exe.predict_classes(X[n_tr:]) == y[n_tr:]))
    return (
        rc,
        exe,
        {
            "task": "trend (per-step)",
            "units": 120,
            "classes": TREND_CLASSES,
            "k": TREND_K,
            "washout": TREND_WASHOUT,
            "aggregation": "none",
            "test_acc": round(acc, 4),
        },
    )


# --------------------------------------------------------------------------


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--out",
        default=str(HERE / "dist"),
        help="output directory for the static site",
    )
    ap.add_argument(
        "--no-simd",
        action="store_true",
        help="disable wasm SIMD128 (smaller compatibility baseline)",
    )
    args = ap.parse_args()
    out = pathlib.Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    simd = not args.no_simd

    sample = np.linspace(-0.8, 0.8, 16, dtype=np.float32)[:, None]
    meta = {"simd": simd, "built_unix": int(time.time()), "models": {}}

    print("[1/4] training + compiling forecast model …")
    esn_f, exe_f, info_f = build_forecast_model()
    print("[2/4] training + compiling dream model …")
    esn_d, exe_d, info_d = build_dream_model()
    print("[3/4] training + compiling shape classifier …")
    rc_s, exe_s, info_s = build_shape_model()
    print(f"      shape test acc = {info_s['test_acc']}")
    print("[4/4] training + compiling trend classifier …")
    rc_t, exe_t, info_t = build_trend_model()
    print(f"      trend test acc = {info_t['test_acc']}")

    shape_sample = _shape_window(0, np.random.default_rng(0), jitter=0.0)
    trend_sample = _trend_series(TREND_WASHOUT + 200, seed=1)[0]

    with tempfile.TemporaryDirectory() as td:
        tdp = pathlib.Path(td)
        bf = BrowserWasm(simd=simd, wasm_name="forecast.wasm")
        art_f = bf.compile(
            esn_f, exe_f, output_dir=tdp / "f", test_inputs=sample
        )
        bd = BrowserWasm(simd=simd, wasm_name="dream.wasm")
        art_d = bd.compile(
            esn_d, exe_d, output_dir=tdp / "d", test_inputs=sample
        )
        # classifiers have different output widths (M baked into the loader),
        # so each ships its own generated loader.
        bs = BrowserWasm(
            simd=simd, wasm_name="shape.wasm", loader_name="rclite_shape.js"
        )
        art_s = bs.compile(
            rc_s, exe_s, output_dir=tdp / "s", test_inputs=shape_sample
        )
        bt = BrowserWasm(
            simd=simd, wasm_name="trend.wasm", loader_name="rclite_trend.js"
        )
        art_t = bt.compile(
            rc_t, exe_t, output_dir=tdp / "t", test_inputs=trend_sample
        )

        shutil.copy(art_f.binary, out / "forecast.wasm")
        shutil.copy(art_d.binary, out / "dream.wasm")
        shutil.copy(art_s.binary, out / "shape.wasm")
        shutil.copy(art_t.binary, out / "trend.wasm")
        # forecast + dream are both f32/K=1/M=1 -> identical loader, shared.
        shutil.copy(tdp / "f" / "rclite.js", out / "rclite.js")
        shutil.copy(tdp / "s" / "rclite_shape.js", out / "rclite_shape.js")
        shutil.copy(tdp / "t" / "rclite_trend.js", out / "rclite_trend.js")

        info_f["wasm_bytes"] = art_f.metadata["wasm_size"]
        info_f["imports"] = art_f.metadata["imports"]
        info_d["wasm_bytes"] = art_d.metadata["wasm_size"]
        info_d["imports"] = art_d.metadata["imports"]
        info_s.update(
            wasm="shape.wasm",
            loader="rclite_shape.js",
            M=art_s.metadata["M"],
            wasm_bytes=art_s.metadata["wasm_size"],
            imports=art_s.metadata["imports"],
        )
        info_t.update(
            wasm="trend.wasm",
            loader="rclite_trend.js",
            M=art_t.metadata["M"],
            wasm_bytes=art_t.metadata["wasm_size"],
            imports=art_t.metadata["imports"],
        )

    meta["models"]["forecast"] = info_f
    meta["models"]["dream"] = info_d
    meta["models"]["shape"] = info_s
    meta["models"]["trend"] = info_t
    (out / "meta.json").write_text(json.dumps(meta, indent=2))

    for name in ("index.html", "app.js", "style.css"):
        shutil.copy(FRONTEND / name, out / name)

    print(f"\n[done] static site written to {out}/")
    print(
        f"       forecast.wasm: {info_f['wasm_bytes']} B "
        f"(NRMSE {info_f['test_nrmse']})"
    )
    print(
        f"       dream.wasm   : {info_d['wasm_bytes']} B "
        f"(NRMSE {info_d['test_nrmse']})"
    )
    print(
        f"       shape.wasm   : {info_s['wasm_bytes']} B "
        f"(acc {info_s['test_acc']}, {len(SHAPE_CLASSES)} classes)"
    )
    print(
        f"       trend.wasm   : {info_t['wasm_bytes']} B "
        f"(acc {info_t['test_acc']}, {len(TREND_CLASSES)} classes)"
    )
    print(f"       serve it:  (cd {out} && python -m http.server)")


if __name__ == "__main__":
    main()
