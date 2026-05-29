# PyTorch + ExecuTorch vs rclite — same ESN

Framework: **PyTorch + ExecuTorch** (torch 2.12.0+cu130). Task: Mackey-Glass one-step-ahead (same data/splits + the **same 80-unit ESN** as the TFLM benchmark). ExecuTorch is run in its **real MCU environment**.

> **ExecuTorch's MCU target is Cortex-M55 + Ethos-U NPU on the Arm Corstone FVP** (it does not target 8-bit AVR or Cortex-M0). We built that environment (Corstone-300 FVP + arm-gnu-toolchain + Vela + TOSA tools) and ran the ESN through the full ExecuTorch arm flow — export → EthosU int8 quantize → Vela → `.pte` → `arm_executor_runner` → FVP. It runs and verifies **bit-exact** vs the AOT reference.

## Same target — all three on one Cortex-M55 (Corstone-300 FVP)

Identical 80-unit ESN, identical FVP/CPU. ExecuTorch is shown both with the Ethos-U55 NPU and CPU-only (portable kernels); rclite is its codegen kernel on the same M55 core. All verify **bit-exact** vs the reference.

| stack (same ESN, same M55) | engine | code (`.text`) | model + working RAM | CPU cycles/step¹ | NRMSE |
|---|---|--:|--:|--:|--:|
| ExecuTorch + Ethos-U55 | **NPU** (int8) | 418 KB | `.pte` 7.5 KB + arena 1626 B | 4,640 (+6,329 NPU) | 41% (int8 PTQ) |
| ExecuTorch CPU-only | M55 CPU (float32) | 801 KB | `.pte` 30 KB + arena 28 KB | 536,229 | 0.30% (float32, no quant) |
| **rclite** | M55 CPU (int8) | **2.3 KB** *(whole firmware)* | **364 B** | **848** | 2.7% / 0.42% (QAT) |

On the **identical M55 CPU**, rclite is **~349× smaller code** and **~632× fewer CPU cycles** than ExecuTorch CPU-only. Even against the **NPU-accelerated** ExecuTorch, rclite on the bare CPU uses **~5× fewer host CPU cycles** and **~182× less code** — a tiny inference spends thousands of CPU cycles just dispatching to the NPU through the interpreter, while rclite's flat kernel finishes in ~848.

¹ FVP cycle model (rclite: M55 DWT CYCCNT; ExecuTorch: `arm_perf_monitor` *Inference runtime*) — the **same** FVP, *not* silicon-cycle-accurate, but a like-for-like relative measure. Part of the gap is rclite exploiting the SCR structure (scalar chain vs a dense 80×80 `W_res`), part is codegen vs interpreter + int8 vs float. ExecuTorch CPU-only is **float32** because its int8 portable path lacks the per-channel quantized out-variants in this build — so its NRMSE is the un-quantized float figure (0.30%, the best of the three) but at the largest `.pte`/arena and the slowest run; the NPU and rclite rows are int8.

> rclite is not limited to the M55: the *same* ESN also runs on a bare **Cortex-M0** (no FPU/NPU) in a complete **3.4 KB** firmware (see ../tflm_vs_rclite) — below ExecuTorch's MCU floor entirely.

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

