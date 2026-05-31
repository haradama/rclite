# rclite benchmarks

Performance and accuracy benchmarks, grouped by scope. Run from the
repository root, e.g.:

```bash
python benchmarks/micro/jit_benchmark.py
python benchmarks/host/compare_host_4way.py
```

## `micro/` — rclite internals (host, fast)

Quick micro-benchmarks of the compilation pipeline; need `numpy` + `llvmlite`.

| File | What it measures |
|------|------------------|
| `jit_benchmark.py` | LLVM JIT kernel vs the NumPy reference: speed + parity. |
| `ir_pass_bench.py` | Effect of the IR optimization passes (fuse / specialize / unroll). |

## `host/` — backend & quantization comparisons (host)

Compare rclite's code paths against each other on the host. The
`compare_*` scripts emit a naive scratch-C baseline (templates in
`host/scratch_c/`) and need `gcc`; `compare_wasm.py` needs an Emscripten /
wasm runtime.

| File | What it measures |
|------|------------------|
| `affine_accuracy.py` | Affine-quantized accuracy vs the float reference. |
| `affine_speed.py` | Affine-quantized kernel throughput. |
| `compare_host_float.py` | rclite LLVM (float) vs naive scratch C. |
| `compare_host_full.py` | Float path incl. callgrind instruction counts. |
| `compare_host_4way.py` | float / i32 / i16 / i8 side by side. |
| `compare_wasm.py` | rclite WASM target vs host. |
| `scratch_c/` | Hand-written naive C templates used as the baseline. |

## Per-target perf benches — unified schema

The Cortex-M0 / AVR / wasm benches below share the **same columns**
(`_perf_schema.py`) over a common matrix: **dtype ∈ {float, i8, i16, i32} ×
kernel ∈ {dense, csr, value-spec unroll}**. A cell a target cannot measure is
rendered **`-`** ("not measured" — AVR has no float/unroll path; wasm has no
Flash/RAM). Columns:

- **speed** — a per-target *deterministic* op-count proxy (`ops/step`: SysTick
  ticks / AVR cycles / wasmtime fuel, unit per caption) and `vs float`
  (float-dense ÷ row at the same N) as the unit-free cross-dtype headline.
- **size** — Flash/RAM (MCU) or wasm bytes, plus the W_res table bytes.
- **accuracy** — `MSE` of the dequantized output vs the ground-truth target
  (real units; a function of dtype only, so it repeats across kernels).

Quant scheme: **i8/i16 = affine** (data-calibrated; symmetric fixed-point
saturates the ridge W_out and gives misleading non-monotonic accuracy), and
**i32 = symmetric** (affine i32 overflows the i64 requantize). Shared
model/quant/object/reference helpers live in `_perf_kernels.py`.

## `sparse_mcu/` — Cortex-M0 (QEMU)

On-device impact of `SparsifyReservoir` + quantization, built as real
nRF51/micro:bit firmware and measured under `qemu -icount shift=0`. Need
`arm-none-eabi-gcc` + `qemu-system-arm`.

| File | What it measures |
|------|------------------|
| `bench.py` | C kernel template (Arduino/turnkey path): dense vs CSR Flash/RAM/speed. |
| `bench_llvm.py` | LLVM codegen path, full **float + i8/i16/i32 × dense/csr/unroll** matrix; SysTick ticks/step + Flash/RAM. `--md`/`--json`; the CI `qemu-bench` job runs this. |

Speed is a deterministic op-count proxy via SysTick (ticks ∝ executed
instructions; **not** silicon cycles), so the speedup ratios are bit-stable
run to run.

## `avr_mcu/` — Arduino Uno (simavr)

ATmega328P (`emit_affine_kernel_c`, **linear-interp LUT**) under **simavr**
(cycle-accurate, deterministic). Need `avr-gcc` + `avr-libc` and host `gcc` +
`libsimavr-dev`. Measures **i8/i16 dense/csr**; the rest are blank: i32 affine
overflows the i64 requantize (i32 uses the symmetric path, on M0/WASM), and
there is no float / value-spec unroll C path. (NB: the default DIRECT LUT is
128 KB at i16 and overflows the Uno's 32 KB Flash, so the bench uses a small
interp LUT — a LUT-size issue, works on the stock avr-gcc.)

| File | What it measures |
|------|------------------|
| `bench_avr.py` | i8 dense/csr: Flash/RAM (avr-size) + AVR cycles/step (simavr). CI `avr-bench` job. |
| `sim_driver.c` | libsimavr driver: captures `avr->cycle` deltas via GPIOR markers. |
| `main_bench.c` | AVR harness: brackets `rc_predict` with cycle markers + parity. |

## `wasm_target/` — wasm32 (wasmtime fuel)

wasm32-wasip1 via the LLVM path (same as Cortex-M0): full float + i8/i16/i32
× dense/csr/unroll matrix. Speed = **wasmtime fuel** (deterministic op-count
proxy, two-point measurement). Need `rustc` (+ `rustup target add
wasm32-wasip1`) and the `wasmtime` Python package.

| File | What it measures |
|------|------------------|
| `bench_wasm.py` | float + i8/i16/i32 × dense/csr/unroll: module bytes + fuel/step. CI `wasm-bench` job. |
| `bench_fuel.rs` | harness: runs `rc_predict` N times (N from WASI arg) + tolerance/exact parity. |

## `executorch_vs_rclite/` — vs ExecuTorch on Cortex-M55 (FVP)

End-to-end comparison against ExecuTorch on an Arm FVP. See its `README.md`.

## `tflm_vs_rclite/` — vs TensorFlow Lite Micro on nRF51

Firmware-level size/speed comparison against TFLM. See its `README.md`.

---

Smaller, dependency-light **usage examples** (not benchmarks) live in
[`../examples/`](../examples/README.md).
