# PyTorch + ExecuTorch vs rclite — same ESN

Framework: **PyTorch + ExecuTorch** (torch 2.12.0+cu130). Task: Mackey-Glass one-step-ahead (same data/splits + the **same 80-unit ESN** as the TFLM benchmark). ExecuTorch is run in its **real MCU environment**.

> **ExecuTorch's MCU target is Cortex-M55 + Ethos-U NPU on the Arm Corstone FVP** (it does not target 8-bit AVR or Cortex-M0). We built that environment (Corstone-300 FVP + arm-gnu-toolchain + Vela + TOSA tools) and ran the ESN through the full ExecuTorch arm flow — export → EthosU int8 quantize → Vela → `.pte` → `arm_executor_runner` → FVP. It runs and verifies **bit-exact** vs the AOT reference.

## On-target: Arm Corstone-300 FVP (Cortex-M55 + Ethos-U55)

FVP: `FVP_Corstone_SSE-300_Ethos-U55 (Fast Models 11.27.42)`.

| stack (same ESN) | target | runtime code | `.pte` | arena | NPU cyc/step¹ | int8 NRMSE |
|---|---|--:|--:|--:|--:|--:|
| **ExecuTorch** (Ethos-U int8) | Cortex-M55 **+ Ethos-U55 NPU** | ~418 KB | 7,632 B | 1626 B | 4,009 | 41% (PTQ)² |
| **rclite** (affine i8) | bare **Cortex-M0** (no NPU) | 3.4 KB *(whole firmware)* | — | 364 B | 31,567 CPU instr | 38% PTQ / **2.7% QAT** |
| **rclite** (affine i16) | bare **Cortex-M0** (no NPU) | 4.1 KB *(whole firmware)* | — | 724 B | 21,602 CPU instr | **0.42% QAT** |

ExecuTorch needs a **Cortex-M55 + Ethos-U55 NPU** and a ~418 KB runtime (interpreter + kernels + NPU driver) plus the `.pte`; the ESN runs bit-exact on the NPU. rclite's **whole 3.4 KB firmware** is pure-CPU code on a bare Cortex-M0 — no NPU, no interpreter. (The example runner also reserves a 60 MB scratch pool — a demo default, excluded; the *used* arena is 1626 B.)

¹ Vela's static estimate (the Corstone FVP is explicitly *not* cycle-accurate); rclite's is a qemu `-icount` instruction count — the two are not directly comparable (NPU cycles vs CPU instructions). ² the FVP runs one ESN cell step bit-exact; the sequence NRMSE is the host int8 figure (the recurrence is software-looped, not on the NPU).

## Accuracy (host, identical held-out targets)

| stack (same ESN) | float | int8 |
|---|--:|--:|
| ExecuTorch (PT2E) | 0.30% | 41% (PTQ) |
| rclite (codegen) | 0.30% | 38% PTQ / **2.7% QAT** (i8) / **0.42%** (i16) |

The ExecuTorch ESN float matches rclite exactly (0.30%) — same reservoir. int8 PTQ is comparably lossy on both stacks (ExecuTorch 41%, rclite 38%) on this chaotic task; rclite's cheap readout-only QAT recovers it to 2.7% (i8) / 0.42% (i16), with no equivalent in the ExecuTorch flow.

## Host AOT `.pte` sizes (desktop XNNPACK/portable, no NPU)

| artifact | size |
|---|--:|
| ExecuTorch ESN `.pte` (portable float) | 30.4 KB |
| ExecuTorch ESN `.pte` (XNNPACK int8) | 12.8 KB |
| rclite ESN firmware (affine i8, *complete*) | 3.4 KB |

The `.pte` is only the model program; a deployable image also needs the ExecuTorch runtime (~418 KB above). rclite's entire Cortex-M0 firmware is smaller than ExecuTorch's ESN `.pte` by itself.

## Takeaways

* **Targeting:** ExecuTorch's MCU floor is Cortex-M55 + Ethos-U (Corstone); rclite's code-gen runs on bare Cortex-M0 (and 8-bit AVR).
* **Footprint:** ExecuTorch needs an NPU + a ~418 KB runtime; rclite's whole firmware is ~3.4 KB of pure-CPU code.
* **Accuracy:** identical float ESN; int8 PTQ ~equally lossy, but rclite's readout-refit QAT reaches 2.7% (i8) / 0.42% (i16).
* **Robustness:** rclite's pure-integer kernel is bit-exact host↔device; the ExecuTorch ESN runs bit-exact on the FVP/NPU but its host XNNPACK int8 `.pte` aborted (double-free) in the host runtime.

