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

It sweeps both dtypes (128-bit SIMD = 2xf64 or 4xf32):

  x86_64  f64 vfmadd...pd / f32 vfmadd...ps   (AVX)
  wasm32  f64 f64x2.*     / f32 f32x4.*       (SIMD128)
  aarch64 f64 *.2d        / f32 *.4s          (NEON)
  armv7   f64 (none — no f64 NEON) / f32 vmla.f32 q  (NEON; f32 unlocks armv7)

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

# (label, dtype, triple, features, simd-mnemonic regex, vlen). 128-bit SIMD = 2
# f64 lanes or 4 f32 lanes; armv7 NEON has no f64 SIMD, so only its f32 row
# vectorises — the same rc dialect, just dtype/vlen.
TARGETS = [
    (
        "x86_64 AVX",
        "f64",
        "x86_64-unknown-linux-gnu",
        "+avx2,+fma",
        r"vfmadd\w*pd|\bmulpd|\baddpd",
        4,
    ),
    (
        "x86_64 AVX",
        "f32",
        "x86_64-unknown-linux-gnu",
        "+avx2,+fma",
        r"vfmadd\w*ps|\bmulps|\baddps",
        8,
    ),
    (
        "wasm32 SIMD128",
        "f64",
        "wasm32-unknown-unknown",
        "+simd128",
        r"f64x2\.(mul|add)",
        2,
    ),
    (
        "wasm32 SIMD128",
        "f32",
        "wasm32-unknown-unknown",
        "+simd128",
        r"f32x4\.(mul|add)",
        4,
    ),
    (
        "aarch64 NEON",
        "f64",
        "aarch64-unknown-linux-gnu",
        "+neon",
        r"\.2d\b",
        2,
    ),
    (
        "aarch64 NEON",
        "f32",
        "aarch64-unknown-linux-gnu",
        "+neon",
        r"\.4s\b",
        4,
    ),
    (
        "armv7 NEON",
        "f64",
        "armv7-unknown-linux-gnueabihf",
        "+neon",
        r"\.f64\b.*\bq\d",
        2,
    ),
    (
        "armv7 NEON",
        "f32",
        "armv7-unknown-linux-gnueabihf",
        "+neon",
        r"\b(vmla|vmul|vfma|vadd)\.f32\b.*\bq\d",
        4,
    ),
]


def _kernel_mlir(vlen, dtype):
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
    return lower_fused_float(mod, m.weights, vlen=vlen, dtype=dtype)


def _simd_count(triple, features, regex, vlen, dtype):
    asm = mlir_jit.cross_compile_object(
        _kernel_mlir(vlen, dtype),
        triple=triple,
        features=features,
        extra_passes=["--convert-vector-to-llvm"],
        filetype="asm",
    ).decode()
    asm_s = mlir_jit.cross_compile_object(
        _kernel_mlir(1, dtype),
        triple=triple,
        features=features,
        filetype="asm",
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
        "One rc vector dialect -> target SIMD (cross-compile; 128b = 2xf64 / "
        "4xf32)\n"
    )
    header = (
        f"{'target':16} {'dtype':5} {'vlen':>4} {'vec SIMD':>9} "
        f"{'scalar':>7}  verdict"
    )
    print(header)
    print("-" * len(header))
    for label, dtype, triple, feat, regex, vlen in TARGETS:
        try:
            nv, ns = _simd_count(triple, feat, regex, vlen, dtype)
            v = "VECTORISED" if nv > ns else "scalar (no SIMD for this dtype)"
            print(f"{label:16} {dtype:5} {vlen:>4} {nv:>9} {ns:>7}  {v}")
        except Exception as e:
            print(f"{label:16} {dtype:5}  FAIL: {type(e).__name__}: {e}")
    print(
        "\nvec/scalar = ISA SIMD-mnemonic count in the vector(vlen) vs "
        "scalar(vlen=1) kernel. Same rc dialect, one --convert-vector-to-llvm;\n"
        "the scalar baseline stays scalar because LLVM won't reassociate the "
        "float reduction. armv7 NEON has no f64 SIMD -> only its f32 row "
        "vectorises.\nThe wasm32 kernel also *runs* under wasmtime "
        "(tests/rc_dialect_test.py::test_wasm_simd_execution)."
    )


if __name__ == "__main__":
    main()
