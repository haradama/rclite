# TFLM (LiteRT for Microcontrollers) vs rclite — micro-controller benchmark

Apples-to-apples: the **identical 80-unit ESN** (same float weights) deployed two
ways on the same MCU — via the **TFLM interpreter** (as a single-step cell) and
via **rclite codegen** — so the comparison isolates the deployment stack.

Task: Mackey-Glass one-step-ahead prediction. See
**[out/RESULTS.md](out/RESULTS.md)** for the generated tables.

## Headline — same 80-unit ESN, two deployment stacks (Cortex-M0)

| stack | Flash | static RAM | int8 NRMSE | instr/step | host↔device |
|---|--:|--:|--:|--:|:--|
| TFLM ESN cell | 70.2 KB | 4.5 KB | 36% PTQ | 314 K | float drift ~0.05 |
| rclite reservoir i8 | **3.4 KB** | **364 B** | 38% PTQ / **2.7% QAT** | **31 K** | **bit-exact** |
| rclite reservoir i16 | 4.1 KB | 724 B | **0.42% QAT** | 21 K | **bit-exact** |

On the *same reservoir*, rclite is **~21× smaller Flash, ~12× smaller RAM,
~10× fewer instructions/step**. TFLM has no reservoir op, so it stores a dense
80×80 `W_res` and dispatches ~11 ops per step through the interpreter; rclite
keeps the SCR chain as one scalar in a flat integer kernel.

* **LiteRT-Micro can't target the requested Arduino Uno at all** (8-bit AVR is
  unsupported — every TFLM Arduino library is 32-bit-only). So the head-to-head
  runs on the smallest 32-bit core both support: **Cortex-M0 (BBC micro:bit v1,
  nRF51822)**, under `qemu-system-arm -M microbit`.
* Naive int8 PTQ collapses for **both** on this chaotic regression (≈36% / 38%)
  — it's the quantization scheme + task, not the framework. rclite's cheap
  built-in QAT (refit the readout on quantized states, no backprop) gets i8 to
  2.7% and i16 to 0.42%; TFLM has no equivalent in this flow.
* rclite's **pure-integer kernel is bit-exact host↔device**; TFLM's float-I/O
  cell drifts (~0.05) over the chaotic recurrence (x86 vs ARM float32 ULPs).
* Latency is a qemu `-icount` instruction estimate, *not* silicon cycles.

## Files

| file | venv | what |
|---|---|---|
| `common.py` | both | shared MG task: data, splits, NRMSE |
| `eval_rclite.py` | rclite | train RC, QAT quantize, host NRMSE per variant |
| `export_esn_params.py` | rclite | dump the trained float ESN weights for TF |
| `train_tf_esn.py` | TF | rebuild the *same* ESN as an int8 TFLite cell, host NRMSE, model + fw data |
| `gen_rclite_fw.py` | rclite | emit portable-C kernels + firmware test data |
| `firmware/` | — | startup, linker, semihosting harness, ESN/rclite `main`s, build scripts |
| `measure_firmware.py` | rclite | build + qemu-run all firmwares → `out/fw_result.json` |
| `report.py` | rclite | combine the JSONs → `out/RESULTS.md` |

## Reproduce

Prerequisites: `arm-none-eabi-gcc`, `qemu-system-arm`, and the rclite venv.
TensorFlow needs Python ≤3.12, so it lives in a separate venv (rclite's is 3.14).

```bash
# 1. TF venv (isolated; TF doesn't disturb rclite's numpy/llvmlite)
uv venv /tmp/tfenv --python python3.12
VIRTUAL_ENV=/tmp/tfenv uv pip install tensorflow-cpu pillow

# 2. Build the TFLM static lib for Cortex-M0 (release).
#    The make scripts need `unzip` and a numpy-enabled `python3`; if missing,
#    put shims first on PATH (a Python zipfile-based `unzip`, and a `python3`
#    that execs the TF venv).
git clone --depth=1 https://github.com/tensorflow/tflite-micro.git /tmp/tflite-micro
cd /tmp/tflite-micro && PATH=/tmp/shimbin:$PATH make \
  -f tensorflow/lite/micro/tools/make/Makefile \
  TARGET=cortex_m_generic TARGET_ARCH=cortex-m0 BUILD_TYPE=release microlite -j6

# 3. Run everything (host accuracy + AOT + firmware build + qemu + report)
cd <repo> && bash benchmarks/tflm_vs_rclite/run_all.sh
```

## Caveats / fairness notes

* **Same model, two stacks.** The identical float ESN (SCR, 80 units) is deployed
  on both; only the deployment path differs (TFLM interpreter vs rclite codegen).
* **int8 PTQ on both.** Neither stack's int8 PTQ is competitive on this chaotic
  task; rclite additionally offers a cheap readout-refit QAT (2.7% / 0.42%).
* **Latency is a qemu `-icount` instruction estimate**, not cycle-accurate
  (qemu does not model the Cortex-M0 pipeline; no `simavr`/silicon here). Treat
  it as an order-of-magnitude, like-for-like number only.
* **TFLM** is built `BUILD_TYPE=release` (`-Os`, error strings stripped) — its
  smallest realistic footprint. rclite is `-Os`, `--gc-sections`. Same board,
  startup, linker, toolchain (TFLM-pinned arm-gcc 14.3.1), and semihosting
  harness for both, so Flash (text+data) and static RAM (data+bss) compare 1:1.
