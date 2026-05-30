"""Stage 2: W_res sparse specialization reaches the LLVM-IR MCU targets.

The Cortex-M0 / GBA / WASM targets cross-compile from the same rclite IR
Module, so threading `sparse=` into their compile methods feeds the
`SparsifyReservoir` pass through `emit_quantized_module` /
`emit_quantized_affine_module` / `cross_compile_rc`.

Two levels of check:
  1. Object-emit smoke (always): the sparse quantized IR parses, verifies,
     and lowers to a non-empty object for the thumbv6m / thumbv4t / wasm32
     backends via llvmlite — no external toolchain needed. This catches
     backend-specific issues (i32 CSR index globals, unrolled blocks).
  2. Full Cortex-M0 pipeline on QEMU (gated on arm-none-eabi-gcc + qemu):
     compile_quantized(sparse="csr") builds and the on-device integer
     verification (host-reference embedded) passes.
"""
from __future__ import annotations
import pathlib
import shutil
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np
import llvmlite.binding as llvm

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import QuantConfig, TanhLUTSpec, quantize_model
from rclite.codegen.llvm import emit_quantized_module, _ensure_all_targets
from rclite.codegen import cross_compile_rc
from rclite.ir import SparsifyReservoir, sparse_passes


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def _model(units=40, density=0.15, seed=7):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(units=units, topology=Topology.ESN_STANDARD,
                                leak_rate=0.3, density=density, seed=seed,
                                name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=60,
                            include_bias=True, include_input=False, name="out"),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((400, 1)) * 0.15
    exe.fit(X[:340], np.sin(np.arange(340) * 0.1)[:, None])
    return rc, exe, X[340:370]


def _emit_object(ll_mod, triple, cpu=""):
    ll_mod.triple = triple
    _ensure_all_targets()
    m = llvm.parse_assembly(str(ll_mod))
    m.verify()
    tgt = llvm.Target.from_triple(triple)
    tm = tgt.create_target_machine(cpu=cpu, opt=2, reloc="static")
    return tm.emit_object(m)


MCU_TRIPLES = [
    ("thumbv6m-none-eabi", "cortex-m0"),  # Cortex-M0
    ("thumbv4t-none-eabi", "arm7tdmi"),   # GBA
    ("wasm32-unknown-unknown", ""),       # WASM
]


def test_quantized_sparse_object_emit():
    """Sparse i32 quantized IR lowers to an object on every MCU backend."""
    rc, exe, _ = _model()
    qm = quantize_model(rc, exe,
                        QuantConfig(state_frac=16, input_frac=12, weight_frac=12),
                        lut=TanhLUTSpec(n=128))
    for strat in ("unroll", "csr"):
        for triple, cpu in MCU_TRIPLES:
            mod = emit_quantized_module(
                qm, passes=[SparsifyReservoir(strategy=strat)])
            obj = _emit_object(mod, triple, cpu)
            assert obj and len(obj) > 0, f"{triple} {strat}: empty object"
    print("  quantized sparse (unroll/csr) → object on thumbv6m/thumbv4t/wasm32")


def test_float_sparse_object_emit():
    """Sparse f32 IR cross-compiles to an object on every MCU backend."""
    rc, exe, _ = _model()
    for triple, cpu in MCU_TRIPLES:
        cc = cross_compile_rc(
            rc, exe, triple=triple, cpu=cpu, dtype="f32",
            passes=sparse_passes("csr", include_structural=True))
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "rc.o"
            cc.emit_object(str(p))
            assert p.exists() and p.stat().st_size > 0
    print("  float sparse → object on thumbv6m/thumbv4t/wasm32")


def test_cortex_m0_quantized_sparse_qemu():
    """Full Cortex-M0 sparse quantized pipeline on QEMU (gated)."""
    if shutil.which("arm-none-eabi-gcc") is None:
        print("  (skip: arm-none-eabi-gcc not on PATH)")
        return
    if shutil.which("qemu-system-arm") is None:
        print("  (skip: qemu-system-arm not on PATH)")
        return
    from rclite.targets.cortex_m0 import Microbit
    rc, exe, sample = _model(units=32)
    qm = quantize_model(rc, exe,
                        QuantConfig(state_frac=16, input_frac=12, weight_frac=12),
                        lut=TanhLUTSpec(n=128))
    target = Microbit()
    with tempfile.TemporaryDirectory() as td:
        art = target.compile_quantized(
            qm, output_dir=td, test_inputs=sample, sparse="csr")
        assert art.binary.exists()
        res = target.run(art)
        assert res.success, f"QEMU output:\n{res.output}"
        assert "EMULATOR_EXIT" in res.output
    print("  Cortex-M0 sparse(csr) quantized: on-device bit-exact via QEMU")


TESTS = [
    test_quantized_sparse_object_emit,
    test_float_sparse_object_emit,
    test_cortex_m0_quantized_sparse_qemu,
]


def main():
    failures = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
        except Exception:
            failures += 1
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
    print(f"\n{len(TESTS) - failures}/{len(TESTS)} passed")
    sys.exit(1 if failures else 0)


if __name__ == "__main__":
    main()
