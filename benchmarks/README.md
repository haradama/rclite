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

## `executorch_vs_rclite/` — vs ExecuTorch on Cortex-M55 (FVP)

End-to-end comparison against ExecuTorch on an Arm FVP. See its `README.md`.

## `tflm_vs_rclite/` — vs TensorFlow Lite Micro on nRF51

Firmware-level size/speed comparison against TFLM. See its `README.md`.

---

Smaller, dependency-light **usage examples** (not benchmarks) live in
[`../examples/`](../examples/README.md).
