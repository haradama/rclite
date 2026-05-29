#!/usr/bin/env bash
# Reproduce the host + firmware stages of the TFLM-vs-rclite benchmark.
# Assumes the TF venv (/tmp/tfenv), the rclite venv (.venv), and the TFLM
# Cortex-M0 microlite lib are already built — see README.md "Reproduce".
set -euo pipefail
cd "$(dirname "$0")"
REPO=$(cd ../.. && pwd)
TFPY=${TFPY:-/tmp/tfenv/bin/python}
RCPY=${RCPY:-$REPO/.venv/bin/python}

echo "## 1/6 eval rclite accuracy";          "$RCPY" eval_rclite.py >/dev/null
echo "## 2/6 export float ESN params";       "$RCPY" export_esn_params.py >/dev/null
echo "## 3/6 build same ESN as TFLM cell";   "$TFPY" train_tf_esn.py >/dev/null
echo "## 4/6 gen rclite firmware kernels";   "$RCPY" gen_rclite_fw.py >/dev/null
echo "## 5/6 build + run firmwares on qemu"; "$RCPY" measure_firmware.py >/dev/null
echo "## 6/6 report";                        "$RCPY" report.py
echo
echo "Done. See out/RESULTS.md"
