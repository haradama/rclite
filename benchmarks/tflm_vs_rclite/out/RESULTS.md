# TFLM (LiteRT for Microcontrollers) vs rclite — Cortex-M0 benchmark

Board: **BBC micro:bit v1 (nRF51822, Cortex-M0, 256KB flash / 16KB SRAM)** (qemu `microbit`). Toolchain: arm-none-eabi-gcc (TFLM-pinned 14.3.1), -Os, --gc-sections.
Task: Mackey-Glass one-step-ahead prediction (999 held-out targets). The **same 80-unit ESN** is deployed two ways — apples-to-apples, isolating the deployment stack.

> **Why Cortex-M0 and not Arduino Uno?** LiteRT for Microcontrollers / TFLM does not support 8-bit AVR (every TFLM Arduino library targets only 32-bit cores: mbed_nano/Cortex-M, esp32, portenta). rclite *does* run on the Uno, but to compare both we use the smallest 32-bit core they share.

## Same ESN, two stacks (Cortex-M0)

The **identical 80-unit ESN** (same float weights), deployed two ways: rclite emits a flat integer `rc_predict`; TFLM runs it as a single-step cell (FullyConnected×2 + Tanh + Mul/Add, invoked per step with state feedback). TFLM has no reservoir op, so it stores a **dense 80×80 W_res** where rclite keeps the SCR chain as **one scalar**.

| stack | Flash | static RAM | int8 NRMSE | latency¹ | host↔device |
|---|--:|--:|--:|--:|:--|
| TFLM ESN cell | **70.2 KB** | **4520 B** (arena 2688 B) | 36% (PTQ) | 314,577 | float drift 0.052 |
| rclite reservoir i8 | **3.4 KB** | **364 B** | 38% PTQ / **2.7% QAT** | 31,567 | **bit-exact** |
| rclite reservoir i16 | **4.1 KB** | **724 B** | **0.42% QAT** | 21,602 | **bit-exact** |

On the **same reservoir**: rclite i8 is **21× smaller Flash**, **12× smaller RAM**, and **10× fewer instructions/step**. The TFLM ESN cell is mostly interpreter + kernels + flatbuffer framework + the dense 80×80 W_res; rclite emits a bare `rc_predict`, so its whole firmware is a fraction of the size.

¹ qemu `-icount` instruction estimate (NOT silicon cycles), per one prediction step.

## Accuracy detail (host, identical targets)

| config | NRMSE |
|---|--:|
| persistence baseline (s[t+1]≈s[t]) | 15.3% |
| ESN float (reference) | 0.30% |
| TFLM ESN cell **int8 PTQ** (deployed) | 36% |
| rclite ESN **i8 PTQ** | 38% |
| rclite ESN **i8 QAT** (deployed) | 2.73% |
| rclite ESN i8 + i16 W_out QAT | 2.68% |
| rclite ESN **i16 QAT** (deployed) | 0.42% |

Naive **int8 PTQ collapses for both** stacks on this chaotic regression (TFLM 36%, rclite 38%) — it's the quantization scheme + task, not the framework. rclite's cheap built-in QAT (refit only the readout on quantized states, no backprop) recovers it to 2.7% (i8) / 0.42% (i16); TFLM has no equivalent in this flow.

Model: SCR reservoir, 80 units (same float weights on both stacks).

