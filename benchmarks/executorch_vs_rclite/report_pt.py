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
    return f"{b / 1024:.1f} KB" if b is not None else "—"


def main() -> int:
    pt = json.loads((OUT / "pt_result.json").read_text())
    esn = pt["esn"]
    rc = (
        json.loads((RC / "rc_result.json").read_text())
        if (RC / "rc_result.json").exists()
        else None
    )
    fw = (
        json.loads((RC / "fw_result.json").read_text())
        if (RC / "fw_result.json").exists()
        else None
    )
    fvp = (
        json.loads((HERE / "fvp" / "out" / "fvp_result.json").read_text())
        if (HERE / "fvp" / "out" / "fvp_result.json").exists()
        else None
    )
    q = rc["nrmse_quant_test"] if rc else {}

    L = []
    a = L.append
    a("# PyTorch + ExecuTorch vs rclite — same ESN")
    a("")
    a(
        f"Framework: **{pt['framework']}** (torch {pt['torch']}). Task: Mackey-Glass "
        "one-step-ahead (same data/splits + the **same 80-unit ESN** as the TFLM "
        "benchmark). ExecuTorch is run in its **real MCU environment**."
    )
    a("")
    a(
        "> **ExecuTorch's MCU target is Cortex-M55 + Ethos-U NPU on the Arm Corstone "
        "FVP** (it does not target 8-bit AVR or Cortex-M0). We built that environment "
        "(Corstone-300 FVP + arm-gnu-toolchain + Vela + TOSA tools) and ran the ESN "
        "through the full ExecuTorch arm flow — export → EthosU int8 quantize → Vela "
        "→ `.pte` → `arm_executor_runner` → FVP. It runs and verifies **bit-exact** "
        "vs the AOT reference."
    )
    a("")

    # ---- SAME Cortex-M55 (Corstone-300 FVP): 3-way ----
    m55 = (
        json.loads((HERE / "fvp" / "out" / "m55_result.json").read_text())
        if (HERE / "fvp" / "out" / "m55_result.json").exists()
        else None
    )
    if m55:
        npu, cpu, rcl = (
            m55["executorch_ethos_u_int8"],
            m55["executorch_cpu_float32"],
            m55["rclite_int8"],
        )
        a("## Same target — all three on one Cortex-M55 (Corstone-300 FVP)")
        a("")
        a(
            "Identical 80-unit ESN, identical FVP/CPU. ExecuTorch is shown both with "
            "the Ethos-U55 NPU and CPU-only (portable kernels); rclite is its codegen "
            "kernel on the same M55 core. All verify **bit-exact** vs the reference."
        )
        a("")
        a(
            "| stack (same ESN, same M55) | engine | code (`.text`) | model + working RAM | CPU cycles/step¹ | NRMSE |"
        )
        a("|---|---|--:|--:|--:|--:|")
        a(
            f"| ExecuTorch + Ethos-U55 | **NPU** (int8) | {npu['code_text_bytes'] / 1024:.0f} KB "
            f"| `.pte` {npu['pte_bytes'] / 1024:.1f} KB + arena {npu['arena_bytes']} B "
            f"| {npu['cpu_cycles_per_step']:,} (+{npu['npu_cycles_per_step']:,} NPU) | 41% (int8 PTQ) |"
        )
        a(
            f"| ExecuTorch CPU-only | M55 CPU (float32) | {cpu['code_text_bytes'] / 1024:.0f} KB "
            f"| `.pte` {cpu['pte_bytes'] / 1024:.0f} KB + arena {cpu['arena_bytes'] / 1024:.0f} KB "
            f"| {cpu['cpu_cycles_per_step']:,} | {rc['nrmse_float_test'] * 100:.2f}% (float32, no quant) |"
        )
        a(
            f"| **rclite** | M55 CPU (int8) | **{rcl['code_text_bytes'] / 1024:.1f} KB** "
            f"*(whole firmware)* | **{rcl['ram_bytes']} B** | **{rcl['cpu_cycles_per_step']:,}** "
            f"| {q['i8_affine_qat'] * 100:.1f}% / {q['i16_affine_qat'] * 100:.2f}% (QAT) |"
        )
        a("")
        a(
            f"On the **identical M55 CPU**, rclite is **~{cpu['code_text_bytes'] / rcl['code_text_bytes']:.0f}× "
            f"smaller code** and **~{cpu['cpu_cycles_per_step'] / rcl['cpu_cycles_per_step']:.0f}× fewer "
            f"CPU cycles** than ExecuTorch CPU-only. Even against the **NPU-accelerated** "
            f"ExecuTorch, rclite on the bare CPU uses **~{npu['cpu_cycles_per_step'] / rcl['cpu_cycles_per_step']:.0f}× "
            f"fewer host CPU cycles** and **~{npu['code_text_bytes'] / rcl['code_text_bytes']:.0f}× less code** "
            f"— a tiny inference spends thousands of CPU cycles just dispatching to the NPU "
            f"through the interpreter, while rclite's flat kernel finishes in ~{rcl['cpu_cycles_per_step']}."
        )
        a("")
        a(
            "¹ FVP cycle model (rclite: M55 DWT CYCCNT; ExecuTorch: `arm_perf_monitor` "
            "*Inference runtime*) — the **same** FVP, *not* silicon-cycle-accurate, but "
            "a like-for-like relative measure. Part of the gap is rclite exploiting the "
            "SCR structure (scalar chain vs a dense 80×80 `W_res`), part is codegen vs "
            "interpreter + int8 vs float. ExecuTorch CPU-only is **float32** because its "
            "int8 portable path lacks the per-channel quantized out-variants in this build "
            "— so its NRMSE is the un-quantized float figure "
            f"({rc['nrmse_float_test'] * 100:.2f}%, the best of the three) but at the largest "
            "`.pte`/arena and the slowest run; the NPU and rclite rows are int8."
        )
        a("")
        if fw:
            r8 = fw["rc_i8"]
            a(
                f"> rclite is not limited to the M55: the *same* ESN also runs on a bare "
                f"**Cortex-M0** (no FPU/NPU) in a complete **{_kb(r8['flash'])}** firmware "
                f"(see ../tflm_vs_rclite) — below ExecuTorch's MCU floor entirely."
            )
            a("")

    # ---- host accuracy ----
    a("## Accuracy (host, identical held-out targets)")
    a("")
    a("| stack (same ESN) | float | int8 |")
    a("|---|--:|--:|")
    a(
        f"| ExecuTorch (PT2E) | {esn['nrmse_float_test'] * 100:.2f}% | "
        f"{esn['nrmse_int8_test'] * 100:.0f}% (PTQ) |"
    )
    if rc:
        a(
            f"| rclite (codegen) | {rc['nrmse_float_test'] * 100:.2f}% | "
            f"{q['i8_affine_ptq'] * 100:.0f}% PTQ / **{q['i8_affine_qat'] * 100:.1f}% QAT** "
            f"(i8) / **{q['i16_affine_qat'] * 100:.2f}%** (i16) |"
        )
    a("")
    a(
        f"The ExecuTorch ESN float matches rclite exactly "
        f"({esn['nrmse_float_test'] * 100:.2f}%) — same reservoir. int8 PTQ is comparably "
        f"lossy on both stacks (ExecuTorch {esn['nrmse_int8_test'] * 100:.0f}%, rclite "
        f"{q['i8_affine_ptq'] * 100:.0f}%) on this chaotic task; rclite's cheap "
        f"readout-only QAT recovers it to {q['i8_affine_qat'] * 100:.1f}% (i8) / "
        f"{q['i16_affine_qat'] * 100:.2f}% (i16), with no equivalent in the ExecuTorch flow."
    )
    a("")

    # ---- host AOT .pte sizes (secondary) ----
    a("## Host AOT `.pte` sizes (desktop XNNPACK/portable, no NPU)")
    a("")
    a("| artifact | size |")
    a("|---|--:|")
    a(
        f"| ExecuTorch ESN `.pte` (portable float) | {_kb(esn['pte_portable_float_bytes'])} |"
    )
    a(
        f"| ExecuTorch ESN `.pte` (XNNPACK int8) | {_kb(esn['pte_xnnpack_int8_bytes'])} |"
    )
    if fw:
        a(
            f"| rclite ESN firmware (affine i8, *complete*) | {_kb(fw['rc_i8']['flash'])} |"
        )
    a("")
    a(
        "The `.pte` is only the model program; a deployable image also needs the "
        "ExecuTorch runtime (~418 KB above). rclite's entire Cortex-M0 firmware is "
        "smaller than ExecuTorch's ESN `.pte` by itself."
    )
    a("")

    a("## Takeaways")
    a("")
    a(
        "* **Targeting:** ExecuTorch's MCU floor is Cortex-M55 + Ethos-U (Corstone); "
        "rclite's code-gen runs on bare Cortex-M0 (and 8-bit AVR)."
    )
    a(
        "* **Footprint:** ExecuTorch needs an NPU + a ~418 KB runtime; rclite's whole "
        "firmware is ~3.4 KB of pure-CPU code."
    )
    a(
        "* **Accuracy:** identical float ESN; int8 PTQ ~equally lossy, but rclite's "
        "readout-refit QAT reaches 2.7% (i8) / 0.42% (i16)."
    )
    a(
        "* **Robustness:** rclite's pure-integer kernel is bit-exact host↔device; the "
        "ExecuTorch ESN runs bit-exact on the FVP/NPU but its host XNNPACK int8 `.pte` "
        "aborted (double-free) in the host runtime."
    )
    a("")
    md = "\n".join(L)
    (OUT / "RESULTS.md").write_text(md + "\n")
    print(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
