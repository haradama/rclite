#!/usr/bin/env bash
# Reproduce the host-side PyTorch + ExecuTorch ESN results (AOT + accuracy).
# For the on-target Corstone-300 FVP run, see fvp/run_fvp.sh. Assumes the PT/ET
# venv (/tmp/ptenv) and the rclite venv (.venv) exist — see README.md.
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
PTPY=${PTPY:-/tmp/ptenv/bin/python}
RCPY=${RCPY:-$REPO/.venv/bin/python}

echo "## 1/3 export float ESN weights (rclite)"
"$RCPY" "$REPO/benchmarks/tflm_vs_rclite/export_esn_params.py" >/dev/null
echo "## 2/3 PyTorch train + ExecuTorch AOT (host accuracy + .pte sizes)"
"$PTPY" executorch_demo.py >/dev/null
echo "## 3/3 report"
"$RCPY" report_pt.py
echo
echo "Done. See out/RESULTS.md"
