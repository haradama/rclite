# PyTorch + ExecuTorch vs rclite — same ESN

The PyTorch/ExecuTorch counterpart of [`../tflm_vs_rclite`](../tflm_vs_rclite):
the **same 80-unit ESN** on the **same** Mackey-Glass task, deployed via
ExecuTorch and via rclite, so the numbers line up across all three stacks
(TF/TFLM, PyTorch/ExecuTorch, rclite).

See **[out/RESULTS.md](out/RESULTS.md)** for the generated tables.

## Headline — all three on the SAME Cortex-M55 (Corstone-300 FVP)

The identical 80-unit ESN on the identical M55 FVP, three ways (all bit-exact):

| stack | engine | code (`.text`) | model + RAM | CPU cyc/step¹ | NRMSE |
|---|---|--:|--:|--:|--:|
| ExecuTorch + Ethos-U55 | NPU (int8) | 418 KB | .pte 7.6 KB + arena 1.6 KB | 4,640 (+6,329 NPU) | 41% PTQ |
| ExecuTorch CPU-only | M55 CPU (float32) | 801 KB | .pte 30 KB + arena 28 KB | 536,229 | 0.30% (float32, **not** quantized) |
| **rclite** | M55 CPU (int8) | **2.3 KB** *(whole fw)* | **364 B** | **848** | 2.7%/0.42% QAT |

* On the **identical M55 CPU**, rclite is **~349× smaller code** and **~632× fewer
  CPU cycles** than ExecuTorch CPU-only — and even vs the **NPU-accelerated**
  ExecuTorch, rclite on the bare CPU uses **~5× fewer host CPU cycles** and
  **~182× less code** (a tiny inference spends thousands of CPU cycles just
  dispatching to the NPU through the interpreter).
* The ESN runs **bit-exact** through ExecuTorch's full arm flow on the Corstone-300
  FVP (export → EthosU int8 quantize → Vela → `.pte` → `arm_executor_runner`).
* int8 PTQ is comparably lossy on both stacks (ExecuTorch 41%, rclite 38%);
  rclite's cheap readout-only QAT recovers it to 2.7% (i8) / 0.42% (i16).
* rclite is not limited to M55: the same ESN also runs on a bare **Cortex-M0**
  (no FPU/NPU) in a complete **3.4 KB** firmware — below ExecuTorch's MCU floor.

¹ FVP cycle model (rclite: M55 DWT CYCCNT; ExecuTorch: `arm_perf_monitor`) — same
FVP, not silicon-cycle-accurate. Part of the gap is rclite exploiting the SCR
structure (scalar chain vs dense 80×80), part is codegen vs interpreter + int8 vs
float (ExecuTorch CPU-only is float32: its int8 portable path lacks per-channel
quantized out-variants in this build).

## Files

| file | venv | what |
|---|---|---|
| `common.py` | both | shared MG task (identical to the TFLM benchmark) |
| `executorch_demo.py` | PyTorch/ET | build the same ESN; float/int8 (PT2E) host accuracy; export portable-float + XNNPACK-int8 `.pte`; host-runtime run |
| `report_pt.py` | any | combine host + on-target FVP + rclite numbers → `out/RESULTS.md` |
| `fvp/mg_esn.py` | ET | ESN cell model module for `aot_arm_compiler` |
| `fvp/run_fvp.sh` | — | deploy + run the ESN on the Corstone-300 FVP (Cortex-M55 + Ethos-U55 NPU; also `--no_delegate` for CPU-only) |
| `fvp/rclite_m55/` | — | bare-metal rclite ESN firmware for the **same** Cortex-M55 FVP (`build_run_m55.sh`, SSE-300 linker, DWT-cycle main) |
| `run_all.sh` | — | chains the host steps |

## Reproduce

```bash
# PyTorch + ExecuTorch venv (Python <= 3.12; rclite's venv is 3.14)
uv venv /tmp/ptenv --python python3.12
VIRTUAL_ENV=/tmp/ptenv uv pip install executorch        # pulls a matching torch

# rclite-side reference (ESN weights + rclite accuracy/firmware numbers):
# run the sibling benchmark first (it also writes out/esn_params.npz):
#   benchmarks/tflm_vs_rclite/run_all.sh
.venv/bin/python benchmarks/tflm_vs_rclite/export_esn_params.py   # ESN weights

/tmp/ptenv/bin/python benchmarks/executorch_vs_rclite/executorch_demo.py
.venv/bin/python      benchmarks/executorch_vs_rclite/report_pt.py
```

`flatc` (needed for XNNPACK `.pte` serialization) ships in the venv at
`/tmp/ptenv/bin/flatc`; `executorch_demo.py` puts it on `PATH` automatically.

### On-target (Corstone-300 FVP) — ExecuTorch's real environment

```bash
# one-time: get the ExecuTorch source (dir MUST be named `executorch`) + submodules,
# then fetch the Corstone FVP + arm toolchain + Vela + TOSA tools:
git clone --branch v1.2.0 https://github.com/pytorch/executorch.git /tmp/executorch
( cd /tmp/executorch && git submodule update --init --recursive --depth 1 )
( cd /tmp/executorch && examples/arm/setup.sh --i-agree-to-the-contained-eula )
VIRTUAL_ENV=/tmp/ptenv uv pip install ethos-u-vela cmake ninja scikit-build-core
# build tosa_serializer + tosa_reference_model from arm-scratch/tosa-tools
# the FVP links libpython3.9 — provide one, e.g.  uv python install 3.9

# ExecuTorch on the FVP — NPU (Ethos-U) and CPU-only (float32, --no_delegate):
bash benchmarks/executorch_vs_rclite/fvp/run_fvp.sh
( cd /tmp/executorch && bash examples/arm/run.sh --model_name=.../fvp/mg_esn.py \
    --target=ethos-u55-128 --no_delegate --no_quantize --bundleio )   # CPU-only

# rclite on the SAME Cortex-M55 FVP (CPU, int8), bit-exact, DWT cycle count:
bash benchmarks/executorch_vs_rclite/fvp/rclite_m55/build_run_m55.sh

.venv/bin/python benchmarks/executorch_vs_rclite/report_pt.py   # reads fvp/out/*.json
```

`run_fvp.sh` documents every env quirk (PATH/LD_LIBRARY_PATH, the
`CMAKE_POLICY_VERSION_MINIMUM` shim for cmake 4, and a one-line PRI-macro patch
the fetched Ethos-U core driver needs to compile with this newlib).

## Caveats / fairness notes

* **Same model, two stacks** — the identical float ESN (SCR, 80 units) deployed
  via ExecuTorch (Cortex-M55 + Ethos-U NPU) and via rclite (bare Cortex-M0).
* **On-target footprint** counts the ExecuTorch+Ethos-U runtime `.text` (~418 KB)
  + `.pte` + used arena. The example runner's 60 MB scratch pool is a demo
  default, excluded. rclite's 3.4 KB is a *complete* Cortex-M0 firmware.
* **The FVP is not cycle-accurate** (it says so itself); NPU cycle figures are
  Vela's static estimate. The FVP run is used for correctness (bit-exact vs
  reference) + footprint, not timing. ExecuTorch NPU cycles and rclite CPU
  instructions are not directly comparable.
* **The ESN on the FVP runs one cell step** bit-exact; the sequence NRMSE is the
  host int8 figure (the recurrence is software-looped, not on the NPU).
* The host-only `.pte`/XNNPACK numbers (separate desktop path) and rclite's
  deployed Cortex-M0 numbers (reused from `../tflm_vs_rclite`, same ESN) round
  out the comparison.
