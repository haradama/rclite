"""Consolidate the PyTorch/ExecuTorch ESN results (host AOT + on-target FVP)
against the rclite numbers (reused from ../tflm_vs_rclite/out, same ESN) into
out/RESULTS.md.

    .venv/bin/python benchmarks/executorch_vs_rclite/report_pt.py
"""
from __future__ import annotations
import json
import pathlib

HERE = pathlib.Path(__file__).resolve().parent
OUT = HERE / "out"
RC = HERE.parent / "tflm_vs_rclite" / "out"


def _kb(b):
    return f"{b/1024:.1f} KB" if b is not None else "—"


def main() -> int:
    pt = json.loads((OUT / "pt_result.json").read_text())
    esn = pt["esn"]
    rc = json.loads((RC / "rc_result.json").read_text()) if (RC / "rc_result.json").exists() else None
    fw = json.loads((RC / "fw_result.json").read_text()) if (RC / "fw_result.json").exists() else None
    fvp = (json.loads((HERE / "fvp" / "out" / "fvp_result.json").read_text())
           if (HERE / "fvp" / "out" / "fvp_result.json").exists() else None)
    q = rc["nrmse_quant_test"] if rc else {}

    L = []
    a = L.append
    a("# PyTorch + ExecuTorch vs rclite — same ESN")
    a("")
    a(f"Framework: **{pt['framework']}** (torch {pt['torch']}). Task: Mackey-Glass "
      "one-step-ahead (same data/splits + the **same 80-unit ESN** as the TFLM "
      "benchmark). ExecuTorch is run in its **real MCU environment**.")
    a("")
    a("> **ExecuTorch's MCU target is Cortex-M55 + Ethos-U NPU on the Arm Corstone "
      "FVP** (it does not target 8-bit AVR or Cortex-M0). We built that environment "
      "(Corstone-300 FVP + arm-gnu-toolchain + Vela + TOSA tools) and ran the ESN "
      "through the full ExecuTorch arm flow — export → EthosU int8 quantize → Vela "
      "→ `.pte` → `arm_executor_runner` → FVP. It runs and verifies **bit-exact** "
      "vs the AOT reference.")
    a("")

    # ---- ON-TARGET: Corstone-300 FVP ----
    if fvp:
        e = fvp["esn"]
        a("## On-target: Arm Corstone-300 FVP (Cortex-M55 + Ethos-U55)")
        a("")
        a(f"FVP: `{fvp['fvp']}`.")
        a("")
        a("| stack (same ESN) | target | runtime code | `.pte` | arena | NPU cyc/step¹ | int8 NRMSE |")
        a("|---|---|--:|--:|--:|--:|--:|")
        a(f"| **ExecuTorch** (Ethos-U int8) | Cortex-M55 **+ Ethos-U55 NPU** | "
          f"~{fvp['runtime_code_text_bytes']/1024:.0f} KB | {e['pte_program_bytes']:,} B | "
          f"{e['arena_used_bytes']} B | {e['npu_cycles_per_step_vela']:,} | "
          f"{esn['nrmse_int8_test']*100:.0f}% (PTQ)² |")
        if fw:
            r8 = fw["rc_i8"]
            a(f"| **rclite** (affine i8) | bare **Cortex-M0** (no NPU) | "
              f"{_kb(r8['flash'])} *(whole firmware)* | — | {r8['ram']} B | "
              f"{r8['instr_per_inference']:,} CPU instr | "
              f"{q['i8_affine_ptq']*100:.0f}% PTQ / **{q['i8_affine_qat']*100:.1f}% QAT** |")
            r16 = fw["rc_i16"]
            a(f"| **rclite** (affine i16) | bare **Cortex-M0** (no NPU) | "
              f"{_kb(r16['flash'])} *(whole firmware)* | — | {r16['ram']} B | "
              f"{r16['instr_per_inference']:,} CPU instr | **{q['i16_affine_qat']*100:.2f}% QAT** |")
        a("")
        a(f"ExecuTorch needs a **Cortex-M55 + Ethos-U55 NPU** and a "
          f"~{fvp['runtime_code_text_bytes']/1024:.0f} KB runtime (interpreter + kernels "
          f"+ NPU driver) plus the `.pte`; the ESN runs bit-exact on the NPU. rclite's "
          f"**whole {_kb(fw['rc_i8']['flash']) if fw else '~3.4 KB'} firmware** is "
          f"pure-CPU code on a bare Cortex-M0 — no NPU, no interpreter. (The example "
          f"runner also reserves a 60 MB scratch pool — a demo default, excluded; the "
          f"*used* arena is {e['arena_used_bytes']} B.)")
        a("")
        a("¹ Vela's static estimate (the Corstone FVP is explicitly *not* "
          "cycle-accurate); rclite's is a qemu `-icount` instruction count — the two "
          "are not directly comparable (NPU cycles vs CPU instructions). ² the FVP "
          "runs one ESN cell step bit-exact; the sequence NRMSE is the host int8 "
          "figure (the recurrence is software-looped, not on the NPU).")
        a("")

    # ---- host accuracy ----
    a("## Accuracy (host, identical held-out targets)")
    a("")
    a("| stack (same ESN) | float | int8 |")
    a("|---|--:|--:|")
    a(f"| ExecuTorch (PT2E) | {esn['nrmse_float_test']*100:.2f}% | "
      f"{esn['nrmse_int8_test']*100:.0f}% (PTQ) |")
    if rc:
        a(f"| rclite (codegen) | {rc['nrmse_float_test']*100:.2f}% | "
          f"{q['i8_affine_ptq']*100:.0f}% PTQ / **{q['i8_affine_qat']*100:.1f}% QAT** "
          f"(i8) / **{q['i16_affine_qat']*100:.2f}%** (i16) |")
    a("")
    a(f"The ExecuTorch ESN float matches rclite exactly "
      f"({esn['nrmse_float_test']*100:.2f}%) — same reservoir. int8 PTQ is comparably "
      f"lossy on both stacks (ExecuTorch {esn['nrmse_int8_test']*100:.0f}%, rclite "
      f"{q['i8_affine_ptq']*100:.0f}%) on this chaotic task; rclite's cheap "
      f"readout-only QAT recovers it to {q['i8_affine_qat']*100:.1f}% (i8) / "
      f"{q['i16_affine_qat']*100:.2f}% (i16), with no equivalent in the ExecuTorch flow.")
    a("")

    # ---- host AOT .pte sizes (secondary) ----
    a("## Host AOT `.pte` sizes (desktop XNNPACK/portable, no NPU)")
    a("")
    a("| artifact | size |")
    a("|---|--:|")
    a(f"| ExecuTorch ESN `.pte` (portable float) | {_kb(esn['pte_portable_float_bytes'])} |")
    a(f"| ExecuTorch ESN `.pte` (XNNPACK int8) | {_kb(esn['pte_xnnpack_int8_bytes'])} |")
    if fw:
        a(f"| rclite ESN firmware (affine i8, *complete*) | {_kb(fw['rc_i8']['flash'])} |")
    a("")
    a("The `.pte` is only the model program; a deployable image also needs the "
      "ExecuTorch runtime (~418 KB above). rclite's entire Cortex-M0 firmware is "
      "smaller than ExecuTorch's ESN `.pte` by itself.")
    a("")

    a("## Takeaways")
    a("")
    a("* **Targeting:** ExecuTorch's MCU floor is Cortex-M55 + Ethos-U (Corstone); "
      "rclite's code-gen runs on bare Cortex-M0 (and 8-bit AVR).")
    a("* **Footprint:** ExecuTorch needs an NPU + a ~418 KB runtime; rclite's whole "
      "firmware is ~3.4 KB of pure-CPU code.")
    a("* **Accuracy:** identical float ESN; int8 PTQ ~equally lossy, but rclite's "
      "readout-refit QAT reaches 2.7% (i8) / 0.42% (i16).")
    a("* **Robustness:** rclite's pure-integer kernel is bit-exact host↔device; the "
      "ExecuTorch ESN runs bit-exact on the FVP/NPU but its host XNNPACK int8 `.pte` "
      "aborted (double-free) in the host runtime.")
    a("")
    md = "\n".join(L)
    (OUT / "RESULTS.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
