"""One `rc` vector dialect, many SIMD ISAs (cross-compile demonstration).

Stage-3 showed the `rc`-dialect `vector` float kernel beats the scalar baseline
on the host because LLVM won't auto-vectorise a float reduction. This script
shows the *unification* payoff: the SAME `lower_fused_float(vlen=...)` output,
cross-compiled with `mlir_jit.cross_compile_object` (+`--convert-vector-to-llvm`),
emits real target SIMD on every backend — no per-ISA kernel.

For each target it cross-compiles both the vector kernel and the scalar baseline
to assembly and counts the ISA's SIMD instructions, demonstrating that (a) the
vector kernel vectorises and (b) the scalar baseline stays scalar (LLVM keeps
the float reduction ordered) — the win is the explicit `vector` lowering, cross
-ISA.

  x86_64  +avx2,+fma    vfmadd...pd  (ymm, 4xf64)
  wasm32  +simd128      f64x2.mul/add
  aarch64 +neon         fmla v.2d    (2xf64)
  armv7   +neon         f64 NEON unsupported on armv7 -> scalar VFP (honest)

Needs llc with the targets registered (the nix devShell's LLVM-20, or a system
llvm-20). Usage:
    python benchmarks/host/rc_dialect_simd_targets.py
"""

from __future__ import annotations
import pathlib
import re
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.ir import build_ir
from rclite.codegen import mlir_jit
from rclite.codegen.rc_dialect_xdsl import (
    build_rc_module,
    fuse_step_readout,
    lower_fused_float,
)

# (label, triple, features, simd-mnemonic regex, vlen)
TARGETS = [
    (
        "x86_64 AVX2",
        "x86_64-unknown-linux-gnu",
        "+avx2,+fma",
        r"vfmadd\w*pd|\bmulpd|\baddpd",
        4,
    ),
    (
        "wasm32 SIMD128",
        "wasm32-unknown-unknown",
        "+simd128",
        r"f64x2\.(mul|add)",
        2,
    ),
    (
        "aarch64 NEON",
        "aarch64-unknown-linux-gnu",
        "+neon",
        r"\bfmla\b.*\.2d|\bfmul\b.*\.2d|\bfadd\b.*\.2d",
        2,
    ),
    (
        "armv7 NEON",
        "armv7-unknown-linux-gnueabihf",
        "+neon",
        r"\bvfma\.f64\b.*q|f64.*q\d",
        2,
    ),
]


def _kernel_mlir(vlen):
    rc = ReservoirComputer(
        input=InputNode(units=4, name="in"),
        reservoir=ReservoirNode(
            units=64,
            topology=Topology.ESN_STANDARD,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=1.0,
            seed=5,
            activation=Activation.IDENTITY,
            name="res",
        ),
        readout=ReadoutNode(
            units=8,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=50,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(
        np.random.default_rng(5).standard_normal((400, 4)) * 0.1,
        np.stack(
            [np.sin(np.arange(400) * 0.03 * (i + 1)) for i in range(8)], 1
        ),
    )
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    return lower_fused_float(mod, m.weights, vlen=vlen)


def _simd_count(triple, features, regex, vlen):
    asm = mlir_jit.cross_compile_object(
        _kernel_mlir(vlen),
        triple=triple,
        features=features,
        extra_passes=["--convert-vector-to-llvm"],
        filetype="asm",
    ).decode()
    asm_s = mlir_jit.cross_compile_object(
        _kernel_mlir(1), triple=triple, features=features, filetype="asm"
    ).decode()
    pat = re.compile(regex)
    n_vec = sum(1 for ln in asm.splitlines() if pat.search(ln))
    n_scalar = sum(1 for ln in asm_s.splitlines() if pat.search(ln))
    return n_vec, n_scalar


def main():
    if (
        "llc" not in [p for p in ("llc",) if __import__("shutil").which(p)]
        or not mlir_jit.tools_available()
    ):
        print("skip: LLVM-20 mlir-opt/llc not on PATH (use the nix devShell)")
        return
    print(
        "One rc vector dialect -> target SIMD (cross-compile, vlen per ISA)\n"
    )
    header = (
        f"{'target':16} {'features':14} {'vector SIMD':>11} "
        f"{'scalar SIMD':>11}  verdict"
    )
    print(header)
    print("-" * len(header))
    for label, triple, feat, regex, vlen in TARGETS:
        try:
            nv, ns = _simd_count(triple, feat, regex, vlen)
            if nv > ns:
                v = "VECTORISED" if ns == 0 else f"vectorised (+{nv - ns})"
            else:
                v = "scalar (ISA has no f64 SIMD)"
            print(f"{label:16} {feat:14} {nv:>11} {ns:>11}  {v}")
        except Exception as e:
            print(f"{label:16} {feat:14}  FAIL: {type(e).__name__}: {e}")
    print(
        "\nvector/scalar SIMD = count of the ISA's f64-SIMD mnemonics in the "
        "vector(vlen) vs scalar(vlen=1) kernel.\nSame rc dialect, one "
        "--convert-vector-to-llvm; the scalar baseline stays scalar because "
        "LLVM won't reassociate the float reduction.\narmv7 has no f64 NEON, so "
        "it correctly falls back to scalar VFP (use f32 for armv7 SIMD)."
    )


if __name__ == "__main__":
    main()
