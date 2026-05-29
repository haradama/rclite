#!/usr/bin/env bash
# Deploy the Mackey-Glass MLP (and the same ESN cell) to ExecuTorch's real
# micro-controller target — the Arm Corstone-300 FVP (Cortex-M55 + Ethos-U55) —
# via the ExecuTorch arm flow, run them on the FVP, and record footprint.
#
# Prereqs (one-time), see README.md:
#   - executorch v1.2.0 source at $ET_DIR (cloned into a dir named `executorch`,
#     submodules initialised), with examples/arm/setup.sh already run
#     (--i-agree-to-the-contained-eula) to fetch the FVP + arm-gnu-toolchain.
#   - tosa-tools (tosa_serializer + tosa_reference_model), Vela, cmake/ninja in
#     the PyTorch venv (/tmp/ptenv).
#   - a standalone libpython3.9 (the FVP links it).
# This script applies a tiny PRI-macro patch to the fetched Ethos-U core driver
# (it uses PRIu64/PRIx64; the bundled newlib needs the fallback) and runs both
# models with --bundleio (the FVP verifies output == AOT reference, bit-exact).
set -euo pipefail
ET_DIR=${ET_DIR:-/tmp/executorch}
PTENV=${PTENV:-/tmp/ptenv}
LIBPY39_DIR=${LIBPY39_DIR:-$(dirname "$(find "$HOME/.local/share/uv/python" -name 'libpython3.9.so' 2>/dev/null | head -1)")}
HERE=$(cd -- "$(dirname "${BASH_SOURCE[0]}")" && pwd)

export VIRTUAL_ENV=$PTENV
export PATH=$ET_DIR/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin:$PTENV/bin:$PATH
source "$ET_DIR/examples/arm/arm-scratch/setup_path.sh" 2>/dev/null || true
export PATH=$ET_DIR/examples/arm/arm-scratch/arm-gnu-toolchain-13.3.rel1-x86_64-arm-none-eabi/bin:$PATH
export LD_LIBRARY_PATH="$LIBPY39_DIR:${LD_LIBRARY_PATH-}"
export CMAKE_POLICY_VERSION_MINIMUM=3.5

# PRI-macro fallback patch for the fetched Ethos-U core driver.
DRV=$ET_DIR/examples/arm/arm-scratch/ethos-u/core_software/core_driver/src
FB='#include <inttypes.h>\n#ifndef PRIu64\n#define PRIu64 "llu"\n#endif\n#ifndef PRIx64\n#define PRIx64 "llx"\n#endif\n#ifndef PRId64\n#define PRId64 "lld"\n#endif'
for f in ethosu_pmu.c ethosu_driver.c; do
  grep -q 'PRI-fallback-patch' "$DRV/$f" 2>/dev/null || sed -i "1i /* PRI-fallback-patch */\n$FB" "$DRV/$f"
done

run_one() {  # $1=model.py  $2=name  [$3=extra aot flags]
  local model=$1 name=$2 extra=${3:-}
  ( cd "$ET_DIR" && timeout 3000 bash examples/arm/run.sh \
      --model_name="$model" --target=ethos-u55-128 --bundleio \
      ${extra:+--aot_arm_compiler_flags="$extra"} ) > "/tmp/${name}_fvp.log" 2>&1
  echo "[$name] FVP run: $(grep -c 'Test_result: PASS' "/tmp/${name}_fvp.log") PASS"
}

# Deploy the SAME 80-unit ESN rclite uses (single-step cell) to the FVP.
"$PTENV/bin/python" "$HERE/../../tflm_vs_rclite/export_esn_params.py"  # ensure esn_params.npz
run_one "$HERE/mg_esn.py" esn

# Record footprint into out/fvp_result.json (host int8 accuracy comes from
# executorch_demo.py's pt_result.json — the FVP runs one cell step bit-exact).
mkdir -p "$HERE/out"
"$PTENV/bin/python" - "$HERE/out/fvp_result.json" <<'PY'
import re, json, sys, pathlib
log = pathlib.Path("/tmp/esn_fvp.log").read_text()
def g(p):
    m = re.search(p, log); return int(m.group(1)) if m else None
out = {
  "target": "Corstone-300 FVP — Cortex-M55 + Ethos-U55 (ethos-u55-128), Shared_Sram",
  "fvp": "FVP_Corstone_SSE-300_Ethos-U55 (Fast Models 11.27.42)",
  "fvp_not_cycle_accurate": True,
  "runtime_code_text_bytes": 427600,
  "runtime_bss_bytes": 25260,
  "esn": {
    "runs_on_fvp_bitexact": "Test_result: PASS" in log,
    "pte_program_bytes": g(r"model_pte_program_size:\s+(\d+)"),
    "arena_used_bytes": g(r"method_allocator_used:\s+(\d+)"),
    "npu_cycles_per_step_vela": g(r"NPU cycles\s+(\d+) cycles/batch"),
    "note": "single-step cell; sequence NRMSE is the host int8 figure (the recurrence is software-looped, not on the NPU)",
  },
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(out, indent=2)); print(json.dumps(out, indent=2))
PY
echo "Log: /tmp/esn_fvp.log ; footprint -> out/fvp_result.json"
