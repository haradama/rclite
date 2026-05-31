# Streamlit ↔ Arduino shape classifier

A small GUI that lets you **draw a curve and classify its shape**
(`rising / falling / peak / valley / sine`) with the quantized Echo State
Network from the WASM "shape" demo — either simulated in Python or running on a
real **Arduino Uno (ATmega328P)** over USB serial.

The same i8-reservoir / i16-readout / MEAN-pooling kernel runs in the browser
tab and on the chip and produces **bit-for-bit identical** outputs, so the
*Simulate* mode is an exact stand-in when no board is attached.

![flow](https://img.shields.io/badge/draw-%E2%86%92%20quantize%20%E2%86%92%20serial%20%E2%86%92%20classify-blue)

## Install

```bash
pip install streamlit pyserial            # required
pip install streamlit-drawable-canvas     # optional: freehand mouse drawing
```

## Run

```bash
streamlit run examples/arduino_esn_demo/serial_app/app.py
```

- **Draw** a shape on the *Preset + sliders* tab (or the *Freehand* tab if the
  canvas component is installed). The window renders live.
- Pick the backend in the sidebar:
  - **Simulate** — runs rclite's integer kernel in Python (no board needed).
  - **Serial device** — sends the window to a connected Arduino and reads back
    its prediction.
- Hit **Classify** to see the predicted shape, the per-class quantized logits,
  and softmax probabilities.

## Putting it on a real Uno

1. In the sidebar, open **⚙️ Build / flash firmware → Build firmware**
   (needs `arduino-cli` with `arduino:avr` installed). It generates and
   compiles the serial-server sketch under `build/arduino_shape_serial/`.
2. Flash it:
   ```bash
   arduino-cli upload -p <PORT> --fqbn arduino:avr:uno build/arduino_shape_serial/sketch
   ```
3. Switch the sidebar to **Serial device**, pick the port, and **Classify**.

Typical footprint: **Flash ≈ 6.2 KB / 32 KB, SRAM ≈ 1.0 KB / 2 KB.**

## Serial protocol (115200 8N1)

```
host → device :  'S'  +  uint16-LE T  +  T signed int8 samples
device → host :  "OK <best> <logit0> ... <logit4>\n"
```

The samples are the *quantized* reservoir input (`cfg.input.quantize_array`) —
exactly what `ArduinoUnoTarget` embeds. The device kernel pools the whole
window (`Aggregation.MEAN`) and emits one 5-logit vector; the firmware argmaxes
it and reports both the class and the logits.

## Files

| file | what |
|------|------|
| `app.py` | Streamlit UI (draw · plot · send · show prediction) |
| `shape_serial.py` | model build + quantize, shape generators, serial protocol, simulate path, firmware emitter — all Streamlit-free and unit-testable |

`shape_serial.build_model()` trains the MEAN classifier and affine-quantizes it
(i8 reservoir, i16 `W_out`, direct tanh LUT); `emit_firmware()` writes and
compiles the server sketch; `predict_simulated()` / `predict_serial()` are the
two interchangeable inference backends.

## Related

- `../build_shape_arduino_mean.py` — the underlying option-B build (true MEAN
  pooling on AVR), bit-exact across the Python reference, LLVM JIT, and the C
  kernel (`tests/quant_pooling_test.py`).
- `../build_shape_arduino.py` — the earlier per-step stepping-stone.
