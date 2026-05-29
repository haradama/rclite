# PyTorch + ExecuTorch vs rclite — same ESN

The PyTorch/ExecuTorch counterpart of [`../tflm_vs_rclite`](../tflm_vs_rclite):
the **same 80-unit ESN** on the **same** Mackey-Glass task, deployed via
ExecuTorch and via rclite, so the numbers line up across all three stacks
(TF/TFLM, PyTorch/ExecuTorch, rclite).

See **[out/RESULTS.md](out/RESULTS.md)** for the generated tables.

## Headline

* **Run in ExecuTorch's real environment.** ExecuTorch's MCU target is
  **Cortex-M55 + Ethos-U NPU on the Arm Corstone FVP** (no 8-bit-AVR / Cortex-M0
  path). We built that environment (Corstone-300 FVP + arm-gnu-toolchain + Vela
  + TOSA tools) and ran the **same ESN through the full ExecuTorch arm flow onto
  the FVP** — export → EthosU int8 quantize → Vela → `.pte` → `arm_executor_runner`
  → FVP. It runs and verifies **bit-exact** vs the AOT reference.
* **On-target footprint** (Cortex-M55 + Ethos-U55): ExecuTorch+Ethos-U **runtime
  code ~418 KB**, `.pte` 7.6 KB, tensor arena 1.6 KB, NPU ~4.0 K cycles/step
  (Vela est.).
* **Contrast with rclite** (same ESN, bare **Cortex-M0, no NPU**): a complete
  **3.4 KB** firmware / **364 B** RAM, bit-exact, 2.7% (i8) / 0.42% (i16) NRMSE.
  ExecuTorch needs a Cortex-M55 **+ Ethos-U55 NPU** and a ~418 KB runtime;
  rclite's *whole* firmware is ~3 KB of pure-CPU code.
* int8 PTQ of the ESN is comparably lossy across stacks (ExecuTorch 41%, rclite
  38%); rclite's cheap readout-only QAT recovers it to 2.7% (i8) / 0.42% (i16),
  with no equivalent in the ExecuTorch flow.

## Files

| file | venv | what |
|---|---|---|
| `common.py` | both | shared MG task (identical to the TFLM benchmark) |
| `executorch_demo.py` | PyTorch/ET | build the same ESN; float/int8 (PT2E) host accuracy; export portable-float + XNNPACK-int8 `.pte`; host-runtime run |
| `report_pt.py` | any | combine host + on-target FVP + rclite numbers → `out/RESULTS.md` |
| `fvp/mg_esn.py` | ET | ESN cell model module for `aot_arm_compiler` |
| `fvp/run_fvp.sh` | — | deploy + run the ESN on the Corstone-300 FVP (Cortex-M55 + Ethos-U55), write `out/fvp_result.json` |
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

# deploy + run the ESN on the FVP (Cortex-M55 + Ethos-U55):
bash benchmarks/executorch_vs_rclite/fvp/run_fvp.sh
.venv/bin/python benchmarks/executorch_vs_rclite/report_pt.py   # picks up fvp/out/fvp_result.json
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
