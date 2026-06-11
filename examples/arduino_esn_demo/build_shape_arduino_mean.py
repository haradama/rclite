#!/usr/bin/env python3
"""Port the WASM "shape" classifier to Arduino Uno — the *real* MEAN version.

This is **option B**: the affine integer path now quantizes sequence pooling
(``Aggregation.MEAN``), so the browser demo's ``shape.wasm`` deploys to an
8-bit AVR *as-is* — one label per drawn window, no per-step reformulation.

The kernel runs the reservoir over the whole window, pools the post-washout
states (``rc_hsum`` / round-divide), and runs the linear readout **once** on
the pooled state — bit-exact with `AffineQuantizedExecutor.predict_pooled_q`,
the LLVM JIT, and the wasm build.

Pipeline:
  1. Same five shapes (rising / falling / peak / valley / sine), one label
     per 80-step window.
  2. Train a MEAN sequence-to-label classifier (``fit_sequences``).
  3. Affine-quantize (i8 reservoir + i16 W_out, direct tanh LUT).
  4. Emit + compile an Arduino sketch; the kernel pools internally and emits
     one 5-logit vector, which the sketch argmaxes into a shape name.

A structured SCR topology keeps the dense W_res out of Flash/SRAM.

Usage::

    python examples/arduino_esn_demo/build_shape_arduino_mean.py
"""

from __future__ import annotations
import pathlib
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Distribution,
    Topology,
    Trainer,
    Task,
    Aggregation,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
    LUTStrategy,
)
from rclite.targets import ArduinoUnoTarget

BUILD = (
    pathlib.Path(__file__).resolve().parents[2]
    / "build"
    / "arduino_shape_mean"
)

SHAPE_W = 80
SHAPE_CLASSES = ["rising", "falling", "peak", "valley", "sine"]
N_CLASSES = len(SHAPE_CLASSES)
N_UNITS = 100
WASHOUT = 12


def _shape_window(kind: int, rng, jitter: float = 0.06) -> np.ndarray:
    t = np.linspace(0.0, 1.0, SHAPE_W)
    if kind == 0:
        s = -1.0 + 2.0 * t
    elif kind == 1:
        s = 1.0 - 2.0 * t
    elif kind == 2:
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    elif kind == 3:
        s = -1.0 + 4.0 * np.abs(t - 0.5)
    else:
        s = np.sin(2.0 * np.pi * t)
    return (s + jitter * rng.standard_normal(SHAPE_W))[:, None]


def _make_windows(n_per_class: int, rng):
    seqs, labels = [], []
    for kind in range(N_CLASSES):
        for _ in range(n_per_class):
            seqs.append(_shape_window(kind, rng))
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    return [seqs[i] for i in idx], np.array(labels)[idx]


# Classification harness: kernel pools the window and emits RC_M logits; the
# sketch argmaxes them into a shape name (and checks bit-exact parity).
_CLASSIFY_INO = """\
/* rclite MEAN shape classifier on Arduino Uno (ATmega328P) -- generated.
 * The kernel pools the whole window internally (Aggregation.MEAN) and emits
 * one %(M)d-logit vector; we argmax it into a shape name. Serial @ 9600.
 */
#include <stdint.h>
#include <avr/pgmspace.h>

extern "C" void rc_predict(int32_t T, const %(STORAGE_T)s *X, %(STORAGE_T)s *Y);

#define RC_T %(T)d
#define RC_M %(M)d

static const %(STORAGE_T)s X_q[RC_T] = { %(X_VALUES)s };
static const %(STORAGE_T)s Y_ref[RC_M] = { %(Y_VALUES)s };
static %(STORAGE_T)s Y_out[RC_M];

const char s0[] PROGMEM = "%(C0)s"; const char s1[] PROGMEM = "%(C1)s";
const char s2[] PROGMEM = "%(C2)s"; const char s3[] PROGMEM = "%(C3)s";
const char s4[] PROGMEM = "%(C4)s";
const char* const SHAPES[] PROGMEM = { s0, s1, s2, s3, s4 };

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  %(STORAGE_T)s X[RC_T];
  for (int i = 0; i < RC_T; i++) X[i] = X_q[i];

  unsigned long t0 = micros();
  rc_predict((int32_t)RC_T, X, Y_out);     /* pools internally, RC_M logits */
  unsigned long dt = micros() - t0;

  int best = 0; int32_t bestv = Y_out[0];
  for (int m = 1; m < RC_M; m++)
    if (Y_out[m] > bestv) { bestv = Y_out[m]; best = m; }
  char name[12];
  strcpy_P(name, (char*)pgm_read_word(&SHAPES[best]));

  int32_t md = 0;
  for (int m = 0; m < RC_M; m++) {
    int32_t d = (int32_t)Y_out[m] - (int32_t)Y_ref[m];
    if (d < 0) d = -d;
    if (d > md) md = d;
  }

  Serial.println(F("rclite MEAN shape classifier on Arduino Uno"));
  Serial.print(F("window_steps=")); Serial.println(RC_T);
  Serial.print(F("elapsed_us="));   Serial.println(dt);
  Serial.print(F("predicted="));    Serial.println(name);
  Serial.print(F("max_abs_diff=")); Serial.println(md);
  Serial.println(md == 0 ? F("PARITY_OK") : F("PARITY_FAIL"));
}

void loop() {}
"""


def _write_sketch(qm, cfg, sketch_dir, window):
    np_storage = np.int8 if qm.storage_bits == 8 else np.int16
    storage_t = "int8_t" if qm.storage_bits == 8 else "int16_t"
    X = window if window.ndim == 2 else window[:, None]
    X_q = cfg.input.quantize_array(X).astype(np_storage)
    Y_ref_q = (
        AffineQuantizedExecutor(qm).predict_pooled_q(X).astype(np_storage)
    )
    subst = {
        "T": X.shape[0],
        "M": qm.M,
        "STORAGE_T": storage_t,
        "X_VALUES": ", ".join(str(int(v)) for v in X_q.ravel()),
        "Y_VALUES": ", ".join(str(int(v)) for v in Y_ref_q.ravel()),
    }
    for i, name in enumerate(SHAPE_CLASSES):
        subst["C%d" % i] = name
    (sketch_dir / "sketch.ino").write_text(_CLASSIFY_INO % subst)


def main() -> None:
    rng = np.random.default_rng(5)
    seqs, labels = _make_windows(80, rng)
    n_tr = int(0.7 * len(seqs))
    tr_s, tr_y = seqs[:n_tr], labels[:n_tr]
    te_s, te_y = seqs[n_tr:], labels[n_tr:]

    print(
        f"[1/4] train MEAN SCR classifier (N={N_UNITS}, {N_CLASSES} classes)"
    )
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=N_UNITS,
            activation=Activation.TANH,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.30,
            seed=9,
            name="res",
        ),
        readout=ReadoutNode(
            units=N_CLASSES,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-2,
            washout=WASHOUT,
            include_bias=True,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.MEAN,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences(tr_s, tr_y)
    acc_f = float(np.mean(exe.predict_sequences(te_s) == te_y))
    print(f"      float window acc = {acc_f:.3f}")

    print("[2/4] affine quantize (i8 reservoir + i16 W_out, direct tanh LUT)")
    cfg = calibrate_from_data(
        rc,
        exe,
        np.concatenate(tr_s[:60], axis=0),
        storage_bits=8,
        w_out_storage_bits=16,
    )
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=LUTStrategy.direct())
    qexe = AffineQuantizedExecutor(qm)
    acc_q = float(
        np.mean(
            [
                int(np.argmax(qexe.predict_pooled_q(X))) == y
                for X, y in zip(te_s, te_y)
            ]
        )
    )
    print(f"      quantized window acc = {acc_q:.3f}")

    print("[3/4] emit + compile Arduino sketch (arduino:avr:uno)")
    demo_kind = 2  # "peak"
    window = _shape_window(demo_kind, np.random.default_rng(0), jitter=0.0)
    target = ArduinoUnoTarget()
    art = target.compile_affine_quantized(
        qm, output_dir=BUILD, test_inputs=window, build=False
    )
    sketch_dir = BUILD / "sketch"
    _write_sketch(qm, cfg, sketch_dir, window)
    cp = subprocess.run(
        [
            "arduino-cli",
            "compile",
            "--fqbn",
            "arduino:avr:uno",
            "--output-dir",
            str(BUILD / "build"),
            str(sketch_dir),
        ],
        capture_output=True,
        text=True,
    )
    md = art.metadata
    print(
        f"      storage={md['dtype']}  W_out={md['w_out_dtype']}  "
        f"topology={md['topology']}  lut={md['lut_kind']}  agg=MEAN"
    )
    if cp.returncode != 0:
        print("      arduino-cli compile FAILED:\n" + cp.stderr)
        sys.exit(1)
    sizes = ArduinoUnoTarget._parse_sizes(cp.stdout)
    fb, sram = sizes.get("flash_bytes"), sizes.get("sram_bytes")
    if fb is not None:
        print(f"      Flash: {fb} / 32768 bytes ({fb / 32768 * 100:.1f}%)")
    if sram is not None:
        print(f"      SRAM : {sram} / 2048 bytes ({sram / 2048 * 100:.1f}%)")

    print("[4/4] device demo")
    pred = int(np.argmax(qexe.predict_pooled_q(window)))
    print(
        f"      embedded window is '{SHAPE_CLASSES[demo_kind]}'; "
        f"kernel predicts '{SHAPE_CLASSES[pred]}'"
    )
    print(
        f"\n[ok] flash with:  arduino-cli upload -p <PORT> "
        f"--fqbn arduino:avr:uno {sketch_dir}"
    )


if __name__ == "__main__":
    main()
