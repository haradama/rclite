#!/usr/bin/env python3
"""Port the WASM "shape" classifier to an Arduino Uno (ATmega328P, 8-bit AVR).

This is the **stepping-stone (option A)** toward fully quantized sequence
pooling. The browser demo's ``shape.wasm`` is a 5-class *sequence-to-label*
classifier that pools reservoir states with ``Aggregation.MEAN`` over an
80-step window. The affine integer path (the only path an Uno can run) does
**not** yet quantize that pooling -- see
``rclite/quant/affine/ir_builder.py`` ("sequence pooling is not yet
quantized"). So we reformulate the same 5-class task as a **per-step**
classifier (``Aggregation.NONE``), which the quantized kernel already
supports, and read the label off the *last* step of the window (where the
reservoir has integrated the whole curve -- the same information MEAN pools).

Pipeline (no rclite core changes):

  1. Build the same five shapes (rising / falling / peak / valley / sine).
  2. Train a per-step classifier from a *reset* state per window, fitting the
     readout on the late part of each window (where the curve is decided).
  3. Affine-quantize (i8 reservoir + i16 W_out) with an interp-64 tanh LUT.
  4. Emit + compile an Arduino sketch via ``ArduinoUnoTarget``; the stock
     harness verifies the kernel is **bit-exact** with this Python reference
     and reports Flash / SRAM. We then drop in a classification harness that
     prints the predicted shape name over Serial.

A structured **SCR** topology keeps the dense W_res out of Flash/SRAM, so the
model fits the Uno comfortably.

Usage::

    python examples/arduino_esn_demo/build_shape_arduino.py
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
                    Activation, Distribution, Topology, Trainer, Task)
from rclite.runtime import RCExecutor
from rclite.quant import (calibrate_from_data, quantize_model_affine,
                          AffineQuantizedExecutor, LUTStrategy)
from rclite.targets import ArduinoUnoTarget

BUILD = pathlib.Path(__file__).resolve().parents[2] / "build" / "arduino_shape"

# Same task as the WASM "shape" demo.
SHAPE_W = 80
SHAPE_CLASSES = ["rising", "falling", "peak", "valley", "sine"]
N_CLASSES = len(SHAPE_CLASSES)

# Per-step reformulation knobs.
N_UNITS = 100          # SCR reservoir size (fits the Uno with room to spare)
WASHOUT = 16           # transient steps to ignore for ranges + training
LATE = 24              # train/read the label off the last LATE steps
RIDGE = 1e-2           # readout L2 (a touch heavier helps the i8 path)


def _shape_window(kind: int, rng, jitter: float = 0.06) -> np.ndarray:
    """One labelled curve == one (SHAPE_W, 1) sequence (matches wasm demo)."""
    t = np.linspace(0.0, 1.0, SHAPE_W)
    if kind == 0:                         # rising ramp
        s = -1.0 + 2.0 * t
    elif kind == 1:                       # falling ramp
        s = 1.0 - 2.0 * t
    elif kind == 2:                       # peak (triangle: rise then fall)
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    elif kind == 3:                       # valley (V: fall then rise)
        s = -1.0 + 4.0 * np.abs(t - 0.5)
    else:                                 # one sine cycle
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


def _one_hot(y, c):
    Y = np.zeros((len(y), c))
    Y[np.arange(len(y)), y] = 1.0
    return Y


def _build_float_model():
    """Per-step (aggregation=NONE) 5-class classifier on the shape curves."""
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY,
                        input_offset=0.0, input_scaling=1.0,
                        input_distribution=Distribution.BERNOULLI, name="in"),
        reservoir=ReservoirNode(units=N_UNITS, activation=Activation.TANH,
                                 topology=Topology.SCR, chain_weight=0.9,
                                 leak_rate=0.30, seed=9, name="res"),
        readout=ReadoutNode(units=N_CLASSES, activation=Activation.IDENTITY,
                            trainer=Trainer.RIDGE, regularization=RIDGE,
                            washout=WASHOUT, include_bias=True,
                            task=Task.CLASSIFICATION, name="out"),  # NONE agg
    )
    return rc, RCExecutor(rc)


def _train_per_step(exe, seqs, labels):
    """Ridge-fit the per-step readout on the *late* steps of each window.

    Each window is run from a reset state (``collect_states`` resets per call,
    exactly like the device kernel), so training matches inference. The label
    is broadcast over the last LATE steps -- the regime where the reservoir
    has seen the whole curve and the shape is decided.
    """
    phis, ys = [], []
    for X, lab in zip(seqs, labels):
        H = exe.collect_states(X)                 # (T, N), fresh reset
        phi = exe._augment(X, H)                  # [bias, state] -> (T, 1+N)
        phis.append(phi[SHAPE_W - LATE:])
        ys.append(np.full(LATE, lab, dtype=int))
    Phi = np.concatenate(phis, axis=0)
    Y = _one_hot(np.concatenate(ys), N_CLASSES)
    A = Phi.T @ Phi + RIDGE * np.eye(Phi.shape[1])
    exe.W_out = np.linalg.solve(A, Phi.T @ Y).T   # (M, 1+N)
    exe.classes_ = np.arange(N_CLASSES)


def _window_label(logits):
    """Recover one label per window: majority vote over the last LATE steps."""
    votes = np.argmax(logits[SHAPE_W - LATE:], axis=1)
    return int(np.bincount(votes, minlength=N_CLASSES).argmax())


def _accuracy_float(exe, seqs, labels):
    ok = 0
    for X, lab in zip(seqs, labels):
        if _window_label(exe.predict(X)) == lab:   # predict() resets per call
            ok += 1
    return ok / len(seqs)


def _accuracy_quant(qm, seqs, labels):
    ok = 0
    for X, lab in zip(seqs, labels):
        qexe = AffineQuantizedExecutor(qm)          # fresh == device reset
        if _window_label(qexe.predict(X)) == lab:
            ok += 1
    return ok / len(seqs)


# --------------------------------------------------------------------------
# Classification harness: same kernel, but print the predicted shape name.
# Drop-in replacement for the stock parity sketch (kernel/ABI unchanged).
# --------------------------------------------------------------------------

_CLASSIFY_INO = """\
/* rclite shape classifier on Arduino Uno (ATmega328P) -- generated.
 * Runs the per-step quantized kernel on one embedded shape window, then
 * reports (a) the predicted shape via last-step argmax and (b) bit-exact
 * parity vs the host reference, over Serial @ 9600.
 */
#include <stdint.h>
#include <avr/pgmspace.h>

extern "C" void rc_predict(int32_t T, const %(STORAGE_T)s *X, %(STORAGE_T)s *Y);

#define RC_T %(T)d
#define RC_K %(K)d
#define RC_M %(M)d
#define X_LEN (RC_T * RC_K)
#define Y_LEN (RC_T * RC_M)

static const %(STORAGE_T)s X_q[X_LEN] = { %(X_VALUES)s };
static const %(STORAGE_T)s Y_ref[Y_LEN] = { %(Y_VALUES)s };
static %(STORAGE_T)s Y_out[Y_LEN];

const char s0[] PROGMEM = "%(C0)s"; const char s1[] PROGMEM = "%(C1)s";
const char s2[] PROGMEM = "%(C2)s"; const char s3[] PROGMEM = "%(C3)s";
const char s4[] PROGMEM = "%(C4)s";
const char* const SHAPES[] PROGMEM = { s0, s1, s2, s3, s4 };

void setup() {
  Serial.begin(9600);
  while (!Serial) {}

  %(STORAGE_T)s X[X_LEN];
  for (int i = 0; i < X_LEN; i++) X[i] = X_q[i];

  unsigned long t0 = micros();
  rc_predict((int32_t)RC_T, X, Y_out);
  unsigned long dt = micros() - t0;

  /* label = argmax of the final step's logits */
  int best = 0; int32_t bestv = Y_out[(RC_T - 1) * RC_M];
  for (int m = 1; m < RC_M; m++) {
    int32_t v = Y_out[(RC_T - 1) * RC_M + m];
    if (v > bestv) { bestv = v; best = m; }
  }
  char name[12];
  strcpy_P(name, (char*)pgm_read_word(&SHAPES[best]));

  int32_t max_abs_diff = 0;
  for (int i = 0; i < Y_LEN; i++) {
    int32_t d = (int32_t)Y_out[i] - (int32_t)Y_ref[i];
    if (d < 0) d = -d;
    if (d > max_abs_diff) max_abs_diff = d;
  }

  Serial.println(F("rclite shape classifier on Arduino Uno"));
  Serial.print(F("steps="));        Serial.println(RC_T);
  Serial.print(F("elapsed_us="));   Serial.println(dt);
  Serial.print(F("us_per_step="));  Serial.println(dt / (unsigned long)RC_T);
  Serial.print(F("predicted="));    Serial.println(name);
  Serial.print(F("max_abs_diff=")); Serial.println(max_abs_diff);
  Serial.println(max_abs_diff == 0 ? F("PARITY_OK") : F("PARITY_FAIL"));
}

void loop() {}
"""


def _write_classify_sketch(qm, cfg, sketch_dir, window):
    """Overwrite the stock parity sketch with a label-printing harness.

    Reuses the target-emitted rc_kernel.c untouched; only the demo harness
    changes. Inputs/reference are quantized exactly as ArduinoUnoTarget does.
    """
    np_storage = np.int8 if qm.storage_bits == 8 else np.int16
    storage_t = "int8_t" if qm.storage_bits == 8 else "int16_t"
    X = window if window.ndim == 2 else window[:, None]
    X_q = cfg.input.quantize_array(X).astype(np_storage)

    qexe = AffineQuantizedExecutor(qm)
    T = X.shape[0]
    Y_ref_q = np.zeros((T, qm.M), dtype=np_storage)
    for t in range(T):
        x_raw_q = qexe._quantize_raw_input(X[t])
        u_pre_q = qexe._quantize_u_pre(X[t])
        qexe.step_q(u_pre_q)
        Y_ref_q[t] = qexe.predict_one_q(x_raw_q, qexe.state_q).astype(np_storage)

    subst = {
        "T": T, "K": qm.K, "M": qm.M, "STORAGE_T": storage_t,
        "X_VALUES": ", ".join(str(int(v)) for v in X_q.ravel()),
        "Y_VALUES": ", ".join(str(int(v)) for v in Y_ref_q.ravel()),
    }
    for i, name in enumerate(SHAPE_CLASSES):
        subst["C%d" % i] = name
    (sketch_dir / "sketch.ino").write_text(_CLASSIFY_INO % subst)
    return Y_ref_q


def _arduino_compile(sketch_dir, out):
    build_dir = out / "build"
    cmd = ["arduino-cli", "compile", "--fqbn", "arduino:avr:uno",
           "--output-dir", str(build_dir), str(sketch_dir)]
    cp = subprocess.run(cmd, capture_output=True, text=True)
    return cp


def main() -> None:
    rng = np.random.default_rng(5)
    seqs, labels = _make_windows(80, rng)
    n_tr = int(0.7 * len(seqs))
    tr_s, tr_y = seqs[:n_tr], labels[:n_tr]
    te_s, te_y = seqs[n_tr:], labels[n_tr:]

    print(f"[1/4] train per-step SCR classifier (N={N_UNITS}, {N_CLASSES} classes)")
    rc, exe = _build_float_model()
    _train_per_step(exe, tr_s, tr_y)
    acc_f = _accuracy_float(exe, te_s, te_y)
    print(f"      float window acc = {acc_f:.3f}")

    print("[2/4] affine quantize (i8 reservoir + i16 W_out, direct tanh LUT)")
    calib_X = np.concatenate(tr_s[:60], axis=0)   # one stream is fine for ranges
    cfg = calibrate_from_data(rc, exe, calib_X,
                              storage_bits=8, w_out_storage_bits=16)
    # A direct 256-entry tanh LUT (256 B in Flash) is needed here: the
    # interp-64 approximation smears the 5-way late-step decision boundary
    # (drops this task to ~0.58); the full table recovers float accuracy.
    qm = quantize_model_affine(rc, exe, cfg,
                               lut_strategy=LUTStrategy.direct())
    acc_q = _accuracy_quant(qm, te_s, te_y)
    print(f"      quantized window acc = {acc_q:.3f}")

    print("[3/4] emit + compile Arduino sketch (arduino:avr:uno)")
    # Pick a clean, unambiguous test window (no jitter) for the on-device demo.
    demo_kind = 2  # "peak"
    window = _shape_window(demo_kind, np.random.default_rng(0), jitter=0.0)
    target = ArduinoUnoTarget()
    art = target.compile_affine_quantized(qm, output_dir=BUILD,
                                          test_inputs=window, build=False)
    sketch_dir = BUILD / "sketch"
    _write_classify_sketch(qm, cfg, sketch_dir, window)

    cp = _arduino_compile(sketch_dir, BUILD)
    md = art.metadata
    print(f"      storage={md['dtype']}  W_out={md['w_out_dtype']}  "
          f"topology={md['topology']}  lut={md['lut_kind']}")
    if cp.returncode != 0:
        print("      arduino-cli compile FAILED:")
        print(cp.stderr)
        sys.exit(1)
    sizes = ArduinoUnoTarget._parse_sizes(cp.stdout)
    fb, sb = sizes.get("flash_bytes"), sizes.get("sram_bytes")
    if fb is not None:
        print(f"      Flash: {fb} / 32768 bytes ({fb / 32768 * 100:.1f}%)")
    if sb is not None:
        print(f"      SRAM : {sb} / 2048 bytes ({sb / 2048 * 100:.1f}%)")

    print("[4/4] device demo")
    pred = _window_label(AffineQuantizedExecutor(qm).predict(window))
    print(f"      embedded window is '{SHAPE_CLASSES[demo_kind]}'; "
          f"kernel predicts '{SHAPE_CLASSES[pred]}'")
    print(f"\n[ok] flash with:  arduino-cli upload -p <PORT> "
          f"--fqbn arduino:avr:uno {sketch_dir}")


if __name__ == "__main__":
    main()
