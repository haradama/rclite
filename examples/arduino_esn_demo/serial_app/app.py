"""Streamlit GUI: draw a shape, send it to an Arduino over serial, classify.

    streamlit run examples/arduino_esn_demo/serial_app/app.py

Two modes (sidebar):
  * **Simulate** — runs rclite's integer kernel in Python. Bit-exact with the
    on-device kernel, so the GUI is fully usable with no board attached.
  * **Serial device** — sends the quantized window to a connected Arduino Uno
    running the generated server firmware and reads back its prediction.

Draw a curve (preset + sliders, or freehand with the mouse), watch it render,
then classify it into one of: rising / falling / peak / valley / sine.
"""

from __future__ import annotations
import pathlib
import sys

import numpy as np
import pandas as pd
import streamlit as st

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import shape_serial as ss

st.set_page_config(
    page_title="rclite • Arduino shape classifier",
    page_icon="〰️",
    layout="wide",
)


@st.cache_resource(show_spinner="Training + quantizing the shape model …")
def get_model():
    return ss.build_model()


model = get_model()

# --------------------------------------------------------------------------
# sidebar — connection + firmware
# --------------------------------------------------------------------------
st.sidebar.title("rclite shape classifier")
st.sidebar.caption(
    "Echo State Network · i8 reservoir + i16 readout · MEAN "
    "pooling · runs on an ATmega328P"
)

mode = st.sidebar.radio(
    "Inference backend",
    ["Simulate (bit-exact device model)", "Serial device"],
    help="Simulate runs the same integer kernel in Python — identical output "
    "to the chip. Serial sends the window to a real Arduino.",
)
serial_mode = mode.startswith("Serial")

port = None
if serial_mode:
    ports = ss.list_serial_ports()
    cols = st.sidebar.columns([3, 1])
    port = cols[0].selectbox("Port", ports or ["(none found)"])
    if cols[1].button("↻", help="rescan ports"):
        st.rerun()
    st.sidebar.caption(
        f"{ss.BAUD} baud · 8N1 · flash the firmware below first"
    )

with st.sidebar.expander("⚙️ Build / flash firmware"):
    st.write(
        "Generate the serial-server sketch and compile it for "
        "`arduino:avr:uno`."
    )
    out_dir = pathlib.Path("build/arduino_shape_serial")
    if st.button("Build firmware"):
        with st.spinner("arduino-cli compile …"):
            info = ss.emit_firmware(model, out_dir)
        if info.get("compiled"):
            fb, sram = info.get("flash_bytes"), info.get("sram_bytes")
            st.success(f"Built ✓  Flash {fb}/32768 · SRAM {sram}/2048")
        elif "log" in info:
            st.error("Compile failed")
            st.code(info["log"][-1500:])
        else:
            st.warning("arduino-cli not found — sketch written, not compiled.")
        st.code(
            f"arduino-cli upload -p <PORT> --fqbn arduino:avr:uno "
            f"{info['sketch_dir']}",
            language="bash",
        )

st.sidebar.divider()
st.sidebar.metric("Reservoir units", ss.N_UNITS)
st.sidebar.metric("Window length T", model.T)
st.sidebar.write("Classes: " + ", ".join(f"`{c}`" for c in model.classes))

# --------------------------------------------------------------------------
# main — draw the shape
# --------------------------------------------------------------------------
st.title("〰️ Draw a shape → classify on Arduino")

tab_preset, tab_free = st.tabs(["🎚️ Preset + sliders", "✏️ Freehand"])
window = None

with tab_preset:
    c1, c2 = st.columns([1, 1])
    kind = ss.SHAPE_CLASSES.index(c1.selectbox("Shape", ss.SHAPE_CLASSES))
    amp = c1.slider("Amplitude", 0.2, 1.5, 1.0, 0.05)
    jitter = c2.slider("Noise", 0.0, 0.4, 0.06, 0.01)
    phase = c2.slider(
        "Phase (sine)", 0.0, 6.28, 0.0, 0.05, disabled=(kind != 4)
    )
    seed = c2.number_input("Noise seed", 0, 9999, 0, 1)
    preset_window = ss.shape_window(
        kind, jitter=jitter, amp=amp, phase=phase, seed=int(seed)
    )

with tab_free:
    try:
        from streamlit_drawable_canvas import st_canvas

        st.caption(
            f"Draw a left-to-right curve; it is resampled to {model.T} points."
        )
        W, H = 560, 220
        canvas = st_canvas(
            stroke_width=3,
            stroke_color="#1f77b4",
            background_color="#0e1117",
            height=H,
            width=W,
            drawing_mode="freedraw",
            key="canvas",
        )
        free_window = None
        if canvas.json_data is not None:
            xs, ys = [], []
            for obj in canvas.json_data.get("objects", []):
                path = obj.get("path", [])
                left, top = obj.get("left", 0), obj.get("top", 0)
                for seg in path:
                    for i in range(1, len(seg) - 1, 2):
                        xs.append(left + seg[i])
                        ys.append(top + seg[i + 1])
            if len(xs) >= 2:
                order = np.argsort(xs)
                y = np.array(ys)[order]
                y = 1.0 - 2.0 * (y - y.min()) / (
                    np.ptp(y) + 1e-9
                )  # screen→[-1,1]
                free_window = ss.resample_to_window(y)
    except ModuleNotFoundError:
        free_window = None
        st.info(
            "Freehand needs `streamlit-drawable-canvas`:\n\n"
            "`pip install streamlit-drawable-canvas`"
        )

use_free = st.radio(
    "Input",
    ["Preset", "Freehand"],
    horizontal=True,
    label_visibility="collapsed",
)
window = (
    free_window
    if use_free == "Freehand"
    and "free_window" in dir()
    and free_window is not None
    else preset_window
)

# --------------------------------------------------------------------------
# render + classify
# --------------------------------------------------------------------------
left, right = st.columns([3, 2])

with left:
    st.subheader("Window")
    st.line_chart(pd.DataFrame({"value": window.ravel()}), height=240)
    samples = ss.quantize_window(model, window)
    with st.expander("Quantized int8 samples sent over serial"):
        st.code(", ".join(str(int(v)) for v in samples), language="text")

with right:
    st.subheader("Prediction")
    go = st.button("▶️  Classify", type="primary", width="stretch")
    if go:
        try:
            if serial_mode:
                if not port or port == "(none found)":
                    st.error("No serial port selected.")
                    st.stop()
                with st.spinner(f"sending {model.T} samples to {port} …"):
                    res = ss.predict_serial(model, window, port=port)
                src = f"Arduino @ {port}"
            else:
                res = ss.predict_simulated(model, window)
                src = "Python integer kernel (= device, bit-exact)"
        except Exception as e:  # noqa: BLE001 — surface any serial/timeout error
            st.error(f"{type(e).__name__}: {e}")
            st.stop()

        st.metric("Shape", res["label"].upper())
        st.caption(f"source: {src}")
        df = pd.DataFrame(
            {
                "class": model.classes,
                "logit (q)": res["logits_q"],
                "probability": res["proba"],
            }
        ).set_index("class")
        st.bar_chart(df["logit (q)"], height=180)
        st.dataframe(
            df.style.format({"probability": "{:.3f}"}).highlight_max(
                subset=["logit (q)"], color="#2ca02c"
            ),
            width="stretch",
        )
    else:
        st.info("Pick / draw a shape, then **Classify**.")

st.caption(
    "rclite · the same quantized ESN runs in this browser tab and on "
    "the microcontroller — bit-for-bit identical outputs."
)
