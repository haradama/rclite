# rclite examples

Runnable examples, grouped by purpose. Host examples need only `numpy`
(plus `llvmlite` for the JIT/benchmarks); deploy demos additionally need a
C/toolchain as noted.

Run from the repository root, e.g.:

```bash
python examples/forecasting/mackey_glass_esn.py
python examples/classification/classification_deploy.py
```

## `forecasting/` — host training & inference

The canonical time-series demos and the shared Mackey-Glass helper.

| File | What it shows |
|------|---------------|
| `mackey_glass_esn.py` | One-step + free-run ESN forecasting. **Shared helper** (`mackey_glass`, `rmse`, `nrmse`) imported by the others. |
| `mackey_glass_sweep.py` | Hyper-parameter sweep over reservoir settings. |
| `online_esn.py` | Online readout training (RLS / LMS) vs batch ridge. |
| `structured_esn.py` | Minimum-complexity reservoirs (DLR / DLRB / SCR). |
| `yildiz_esp_demo.py` | Echo State Property verification. |

## `classification/` — classification task

| File | What it shows |
|------|---------------|
| `classification_esn.py` | Per-step and sequence-to-label classification in the reference runtime (`predict_proba` / `predict_classes`). |
| `classification_deploy.py` | Quantize → `export_bundle(head="classify"/"proba")` → compile the generated C kernel with host gcc. Needs `gcc`. |

## Deploy demos (one directory per target)

Each builds an end-to-end artifact for a specific device. Most need a
cross-toolchain (and optionally an emulator) on `PATH`.

| Directory | Target | Needs |
|-----------|--------|-------|
| `c_library_demo/` | Host shared library + C header | `gcc` |
| `arduino_esn_demo/` | Arduino Uno (AVR) | `arduino-cli` / avr-gcc |
| `microbit_esn_demo/` | BBC micro:bit (Cortex-M0) — float, quantized, i8, i16-affine | arm-none-eabi-gcc, qemu |
| `gba_esn_demo/` | Game Boy Advance | devkitARM / mGBA |
| `nes_esn_demo/` | Nintendo Entertainment System | cc65 / FCEUX |

The `microbit_esn_demo/` scripts (`build_microbit*.py`) cover the float and
the three quantized paths; `c_library_demo/build_c_library.py` is the simplest
host deploy and is a good first read.

## Performance benchmarks

Benchmarks (JIT vs NumPy, host backend / quantization comparisons, and
vs-TFLM / vs-ExecuTorch on MCUs) live in [`../benchmarks/`](../benchmarks/README.md).
