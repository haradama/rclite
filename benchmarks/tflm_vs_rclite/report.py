"""Consolidate the rclite + firmware JSONs into a Markdown report
(out/RESULTS.md + stdout). Run after the measurement stages.

    .venv/bin/python benchmarks/tflm_vs_rclite/report.py
"""
from __future__ import annotations
import json
import pathlib

OUT = pathlib.Path(__file__).resolve().parent / "out"


def _kb(b):
    return f"{b/1024:.1f} KB"


def main() -> int:
    rc = json.loads((OUT / "rc_result.json").read_text())
    fw = json.loads((OUT / "fw_result.json").read_text())
    esn = json.loads((OUT / "esn_tf_result.json").read_text())
    q = rc["nrmse_quant_test"]

    L = []
    a = L.append
    a("# TFLM (LiteRT for Microcontrollers) vs rclite — Cortex-M0 benchmark")
    a("")
    a(f"Board: **{fw['board']}** (qemu `{fw['qemu_machine']}`). "
      f"Toolchain: {fw['toolchain']}.")
    a(f"Task: Mackey-Glass one-step-ahead prediction "
      f"({rc['n_test']} held-out targets). The **same 80-unit ESN** is deployed "
      f"two ways — apples-to-apples, isolating the deployment stack.")
    a("")
    a("> **Why Cortex-M0 and not Arduino Uno?** LiteRT for Microcontrollers / "
      "TFLM does not support 8-bit AVR (every TFLM Arduino library targets only "
      "32-bit cores: mbed_nano/Cortex-M, esp32, portenta). rclite *does* run on "
      "the Uno, but to compare both we use the smallest 32-bit core they share.")
    a("")

    te = fw["tflm_esn"]
    r8, r16 = fw["rc_i8"], fw["rc_i16"]
    a("## Same ESN, two stacks (Cortex-M0)")
    a("")
    a("The **identical 80-unit ESN** (same float weights), deployed two ways: "
      "rclite emits a flat integer `rc_predict`; TFLM runs it as a single-step "
      "cell (FullyConnected×2 + Tanh + Mul/Add, invoked per step with state "
      "feedback). TFLM has no reservoir op, so it stores a **dense 80×80 W_res** "
      "where rclite keeps the SCR chain as **one scalar**.")
    a("")
    a("| stack | Flash | static RAM | int8 NRMSE | latency¹ | host↔device |")
    a("|---|--:|--:|--:|--:|:--|")
    a(f"| TFLM ESN cell | **{_kb(te['flash'])}** | **{te['ram']} B** "
      f"(arena {te['arena_used']} B) | {esn['nrmse_int8_test']*100:.0f}% (PTQ) | "
      f"{te['instr_per_inference']:,} | float drift {te['max_abs_diff_scaled']/10000.0:.3f} |")
    a(f"| rclite reservoir i8 | **{_kb(r8['flash'])}** | **{r8['ram']} B** | "
      f"{q['i8_affine_ptq']*100:.0f}% PTQ / **{q['i8_affine_qat']*100:.1f}% QAT** | "
      f"{r8['instr_per_inference']:,} | **bit-exact** |")
    a(f"| rclite reservoir i16 | **{_kb(r16['flash'])}** | **{r16['ram']} B** | "
      f"**{q['i16_affine_qat']*100:.2f}% QAT** | "
      f"{r16['instr_per_inference']:,} | **bit-exact** |")
    a("")
    a(f"On the **same reservoir**: rclite i8 is **{te['flash']/r8['flash']:.0f}× "
      f"smaller Flash**, **{te['ram']/r8['ram']:.0f}× smaller RAM**, and "
      f"**{te['instr_per_inference']/r8['instr_per_inference']:.0f}× fewer "
      f"instructions/step**. The TFLM ESN cell is mostly interpreter + kernels + "
      f"flatbuffer framework + the dense 80×80 W_res; rclite emits a bare "
      f"`rc_predict`, so its whole firmware is a fraction of the size.")
    a("")
    a("¹ qemu `-icount` instruction estimate (NOT silicon cycles), per one "
      "prediction step.")
    a("")

    a("## Accuracy detail (host, identical targets)")
    a("")
    a("| config | NRMSE |")
    a("|---|--:|")
    a(f"| persistence baseline (s[t+1]≈s[t]) | {rc['nrmse_persistence_test']*100:.1f}% |")
    a(f"| ESN float (reference) | {rc['nrmse_float_test']*100:.2f}% |")
    a(f"| TFLM ESN cell **int8 PTQ** (deployed) | {esn['nrmse_int8_test']*100:.0f}% |")
    a(f"| rclite ESN **i8 PTQ** | {q['i8_affine_ptq']*100:.0f}% |")
    a(f"| rclite ESN **i8 QAT** (deployed) | {q['i8_affine_qat']*100:.2f}% |")
    a(f"| rclite ESN i8 + i16 W_out QAT | {q['i8_i16wout_qat']*100:.2f}% |")
    a(f"| rclite ESN **i16 QAT** (deployed) | {q['i16_affine_qat']*100:.2f}% |")
    a("")
    a("Naive **int8 PTQ collapses for both** stacks on this chaotic regression "
      f"(TFLM {esn['nrmse_int8_test']*100:.0f}%, rclite {q['i8_affine_ptq']*100:.0f}%) "
      "— it's the quantization scheme + task, not the framework. rclite's cheap "
      "built-in QAT (refit only the readout on quantized states, no backprop) "
      f"recovers it to {q['i8_affine_qat']*100:.1f}% (i8) / "
      f"{q['i16_affine_qat']*100:.2f}% (i16); TFLM has no equivalent in this flow.")
    a("")
    a(f"Model: {rc['topology']} reservoir, {rc['reservoir_units']} units "
      f"(same float weights on both stacks).")
    a("")

    md = "\n".join(L)
    (OUT / "RESULTS.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
