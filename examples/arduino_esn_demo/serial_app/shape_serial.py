"""Core logic for the Streamlit + Arduino shape-classifier demo.

Kept free of Streamlit so it can be unit-tested and reused:

  * train + affine-quantize the MEAN shape classifier (option B)
  * generate / quantize the five shape windows
  * a tiny serial protocol (host <-> Arduino)
  * a "simulate" path (`AffineQuantizedExecutor`) that is **bit-exact** with
    what the on-device kernel computes — so the GUI works with no board
  * emit + (optionally) compile the serial-server firmware

Serial protocol
---------------
Host -> device :  b'S' + uint16-LE T + T signed int8 samples
Device -> host :  text line  "OK <best> <logit0> ... <logit{M-1}>\\n"

The samples are the *quantized* input (the kernel reads X directly as the
reservoir input), produced by `cfg.input.quantize_array` — identical to what
`ArduinoUnoTarget` embeds and what the Python reference quantizes internally.
"""

from __future__ import annotations
import pathlib
import struct
import subprocess
import sys

import numpy as np

# Allow `import rclite` when run from anywhere in the repo.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[3]))

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
from rclite.targets.arduino import emit_affine_kernel_c

SHAPE_W = 80
SHAPE_CLASSES = ["rising", "falling", "peak", "valley", "sine"]
N_CLASSES = len(SHAPE_CLASSES)
N_UNITS = 100
WASHOUT = 12
BAUD = 115200


# --------------------------------------------------------------------------
# shapes


def shape_window(
    kind: int,
    *,
    jitter: float = 0.0,
    amp: float = 1.0,
    phase: float = 0.0,
    seed: int = 0,
) -> np.ndarray:
    """One labelled curve == one (SHAPE_W, 1) sequence (matches the wasm demo)."""
    rng = np.random.default_rng(seed)
    t = np.linspace(0.0, 1.0, SHAPE_W)
    if kind == 0:  # rising ramp
        s = -1.0 + 2.0 * t
    elif kind == 1:  # falling ramp
        s = 1.0 - 2.0 * t
    elif kind == 2:  # peak (triangle)
        s = 1.0 - 4.0 * np.abs(t - 0.5)
    elif kind == 3:  # valley (V)
        s = -1.0 + 4.0 * np.abs(t - 0.5)
    else:  # one sine cycle
        s = np.sin(2.0 * np.pi * t + phase)
    s = amp * s + jitter * rng.standard_normal(SHAPE_W)
    return s[:, None]


def resample_to_window(y: np.ndarray) -> np.ndarray:
    """Resample an arbitrary 1-D trace to a (SHAPE_W, 1) window for freehand."""
    y = np.asarray(y, dtype=float).reshape(-1)
    if y.size == 0:
        return np.zeros((SHAPE_W, 1))
    xs = np.linspace(0.0, 1.0, y.size)
    xt = np.linspace(0.0, 1.0, SHAPE_W)
    return np.interp(xt, xs, y)[:, None]


# --------------------------------------------------------------------------
# model


class ShapeModel:
    """Trained + quantized shape classifier bundle."""

    def __init__(self, rc, exe, qm, cfg, classes):
        self.rc = rc
        self.exe = exe
        self.qm = qm
        self.cfg = cfg
        self.classes = classes

    @property
    def M(self) -> int:
        return self.qm.M

    @property
    def T(self) -> int:
        return SHAPE_W


def build_model(seed: int = 5) -> ShapeModel:
    """Train + affine-quantize the MEAN (sequence-to-label) shape classifier."""
    rng = np.random.default_rng(seed)
    seqs, labels = [], []
    for kind in range(N_CLASSES):
        for _ in range(80):
            seqs.append(
                shape_window(
                    kind, jitter=0.06, seed=int(rng.integers(1 << 30))
                )
            )
            labels.append(kind)
    idx = rng.permutation(len(seqs))
    seqs = [seqs[i] for i in idx]
    labels = np.array(labels)[idx]
    n_tr = int(0.7 * len(seqs))

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
    exe.fit_sequences(seqs[:n_tr], labels[:n_tr])
    cfg = calibrate_from_data(
        rc,
        exe,
        np.concatenate(seqs[:60], axis=0),
        storage_bits=8,
        w_out_storage_bits=16,
    )
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=LUTStrategy.direct())
    return ShapeModel(
        rc=rc, exe=exe, qm=qm, cfg=cfg, classes=list(SHAPE_CLASSES)
    )


# --------------------------------------------------------------------------
# quantize / predict


def quantize_window(model: ShapeModel, window: np.ndarray) -> np.ndarray:
    """Float window (T,1) -> int8 samples (T,), exactly as the kernel expects."""
    X = window if window.ndim == 2 else window[:, None]
    return model.cfg.input.quantize_array(X).astype(np.int8).reshape(-1)


def softmax(logits: np.ndarray) -> np.ndarray:
    z = np.asarray(logits, dtype=float)
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def predict_simulated(model: ShapeModel, window: np.ndarray) -> dict:
    """Bit-exact device-equivalent prediction via the Python integer kernel."""
    q_logits = AffineQuantizedExecutor(model.qm).predict_pooled_q(window)
    deq = model.cfg.output.dequantize_array(q_logits[None, :])[0]
    best = int(np.argmax(q_logits))
    return {
        "best": best,
        "label": model.classes[best],
        "logits_q": q_logits.astype(int).tolist(),
        "logits": deq.tolist(),
        "proba": softmax(deq).tolist(),
    }


# --------------------------------------------------------------------------
# serial protocol


def encode_request(samples_i8: np.ndarray) -> bytes:
    """b'S' + uint16-LE T + T signed int8 samples."""
    s = np.asarray(samples_i8, dtype=np.int8)
    return b"S" + struct.pack("<H", s.size) + s.tobytes()


def parse_response(line: str, M: int) -> dict | None:
    """Parse "OK <best> <l0> ... <l{M-1}>"; return None if malformed."""
    parts = line.strip().split()
    if len(parts) != M + 2 or parts[0] != "OK":
        return None
    try:
        best = int(parts[1])
        logits_q = [int(v) for v in parts[2 : 2 + M]]
    except ValueError:
        return None
    return {"best": best, "logits_q": logits_q}


def predict_serial(
    model: ShapeModel,
    window: np.ndarray,
    *,
    port: str,
    baud: int = BAUD,
    timeout: float = 2.0,
) -> dict:
    """Send the window to a connected Arduino and read back its prediction."""
    import serial  # pyserial; imported lazily so simulation needs no hardware

    samples = quantize_window(model, window)
    with serial.Serial(port, baud, timeout=timeout) as ser:
        # Arduino resets on open; give the bootloader a moment, then flush.
        import time

        time.sleep(2.0)
        ser.reset_input_buffer()
        ser.write(encode_request(samples))
        ser.flush()
        deadline = time.time() + timeout + 1.0
        while time.time() < deadline:
            line = ser.readline().decode("ascii", "ignore")
            r = parse_response(line, model.M)
            if r is not None:
                break
        else:
            raise TimeoutError("no valid 'OK ...' reply from device")
    deq = model.cfg.output.dequantize_array(np.array(r["logits_q"])[None, :])[
        0
    ]
    return {
        "best": r["best"],
        "label": model.classes[r["best"]],
        "logits_q": r["logits_q"],
        "logits": deq.tolist(),
        "proba": softmax(deq).tolist(),
    }


# --------------------------------------------------------------------------
# firmware

_SERVER_INO = """\
/* rclite serial shape-classifier server for Arduino Uno (ATmega328P).
 *
 * Protocol @ %(BAUD)d 8N1:
 *   host -> 'S', uint16-LE T, then T signed int8 samples (quantized input)
 *   here -> "OK <best> <logit0> ... <logit%(MM1)d>\\n"
 *
 * The kernel (rc_kernel.c) pools the whole window (Aggregation.MEAN) and
 * emits %(M)d logits; we argmax them and report logits + class.
 */
#include <stdint.h>

extern "C" void rc_predict(int32_t T, const int8_t *X, int8_t *Y);

#define RC_M %(M)d
#define MAXT %(MAXT)d
static int8_t Xbuf[MAXT];
static int8_t Y[RC_M];

static int read_byte() {            /* blocking single-byte read */
  while (Serial.available() <= 0) {}
  return Serial.read();
}

void setup() { Serial.begin(%(BAUD)d); }

void loop() {
  if (Serial.available() <= 0) return;
  if (Serial.read() != 'S') return;            /* resync on start byte */
  int lo = read_byte(), hi = read_byte();
  int32_t T = (int32_t)((lo & 0xFF) | ((hi & 0xFF) << 8));
  if (T > MAXT) T = MAXT;
  for (int32_t i = 0; i < T; i++) Xbuf[i] = (int8_t)read_byte();

  rc_predict(T, Xbuf, Y);
  int best = 0;
  for (int m = 1; m < RC_M; m++) if (Y[m] > Y[best]) best = m;

  Serial.print("OK "); Serial.print(best);
  for (int m = 0; m < RC_M; m++) { Serial.print(' '); Serial.print((int)Y[m]); }
  Serial.print('\\n');
}
"""


def emit_firmware(model: ShapeModel, out_dir, *, maxt: int = 256) -> dict:
    """Write the serial-server sketch + kernel; compile if arduino-cli exists.

    Returns a dict with sketch path and (when compiled) Flash/SRAM bytes.
    """
    import shutil

    out = pathlib.Path(out_dir)
    sketch_dir = out / "sketch"
    sketch_dir.mkdir(parents=True, exist_ok=True)
    (sketch_dir / "rc_kernel.c").write_text(emit_affine_kernel_c(model.qm))
    (sketch_dir / "sketch.ino").write_text(
        _SERVER_INO
        % {"BAUD": BAUD, "M": model.M, "MM1": model.M - 1, "MAXT": maxt}
    )

    info = {"sketch_dir": str(sketch_dir)}
    if shutil.which("arduino-cli") is None:
        info["compiled"] = False
        return info
    cp = subprocess.run(
        [
            "arduino-cli",
            "compile",
            "--fqbn",
            "arduino:avr:uno",
            "--output-dir",
            str(out / "build"),
            str(sketch_dir),
        ],
        capture_output=True,
        text=True,
    )
    info["compiled"] = cp.returncode == 0
    info["log"] = cp.stdout + cp.stderr
    if cp.returncode == 0:
        from rclite.targets.arduino import ArduinoUnoTarget

        info.update(ArduinoUnoTarget._parse_sizes(cp.stdout))
    return info


def list_serial_ports() -> list:
    """Return candidate serial port device names (best-effort)."""
    try:
        from serial.tools import list_ports

        return [p.device for p in list_ports.comports()]
    except Exception:
        return []
