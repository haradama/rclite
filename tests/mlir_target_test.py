"""MLIR path -> embedded target cross-compilation.

`cross_compile_object` lowers the MLIR affine/symmetric kernel and emits a
relocatable object for the embedded triples the llvmlite path serves
(thumbv6m=Cortex-M0, thumbv4t=GBA, wasm32=WASM), connecting the MLIR path to
the targets. The integer kernel stays scalar (SIMD vectorization would break
bit-exactness — same policy as the existing WASM quantized target); `features`
only selects the ISA. Skipped when the MLIR toolchain is absent.
"""

from __future__ import annotations
import pathlib
import shutil
import sys
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.quant import QuantConfig, TanhLUTSpec, I8Symmetric, quantize_model
from rclite.codegen import mlir_jit
from rclite.codegen.mlir_jit import cross_compile_object


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
try:
    import xdsl  # noqa: F401
    from rclite.codegen.mlir_affine_xdsl import emit_affine_mlir_xdsl
    from rclite.codegen.mlir_symmetric_xdsl import (
        emit_symmetric_mlir_xdsl,
    )

    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

HAVE = (
    mlir_jit.tools_available()
    and shutil.which("llc") is not None
    and _HAVE_XDSL
)

# (triple, cpu, features)
TARGETS = [
    ("thumbv6m-none-eabi", "cortex-m0", ""),  # Cortex-M0
    ("thumbv4t-none-eabi", "arm7tdmi", ""),  # GBA
    ("wasm32-unknown-unknown", "", ""),  # WASM (scalar)
    (
        "wasm32-unknown-unknown",
        "",
        "+simd128",
    ),  # WASM SIMD ISA (kernel scalar)
]


def _affine_qm():
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=16,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.2,
            seed=4,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=40,
            include_bias=True,
            include_input=False,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.sin(np.arange(400) * 0.05)[:, None]
    exe.fit(X[:300], np.sin(np.arange(1, 301) * 0.05)[:, None])
    cfg = calibrate_from_data(rc, exe, X[:300], storage_bits=8)
    return quantize_model_affine(rc, exe, cfg)


def _symmetric_qm():
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=16,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.2,
            seed=4,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=40,
            include_bias=True,
            include_input=False,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.sin(np.arange(400) * 0.05)[:, None]
    exe.fit(X[:300], np.sin(np.arange(1, 301) * 0.05)[:, None])
    cfg = QuantConfig(state_frac=5, input_frac=6, weight_frac=6)
    return quantize_model(
        rc, exe, cfg, lut=TanhLUTSpec(n=128), target=I8Symmetric()
    )


def test_affine_cross_compile():
    if not HAVE:
        print("  (skip: MLIR toolchain not on PATH)")
        return
    mlir = emit_affine_mlir_xdsl(_affine_qm())
    for triple, cpu, feat in TARGETS:
        obj = cross_compile_object(mlir, triple=triple, cpu=cpu, features=feat)
        assert obj and len(obj) > 0, f"{triple} {feat}: empty object"
    print("  affine MLIR -> object on thumbv6m/thumbv4t/wasm32 (+simd128 ISA)")


def test_symmetric_cross_compile():
    if not HAVE:
        print("  (skip)")
        return
    mlir = emit_symmetric_mlir_xdsl(_symmetric_qm())
    for triple, cpu, feat in TARGETS:
        obj = cross_compile_object(mlir, triple=triple, cpu=cpu, features=feat)
        assert obj and len(obj) > 0, f"{triple} {feat}: empty object"
    print("  symmetric MLIR -> object on thumbv6m/thumbv4t/wasm32")


TESTS = [test_affine_cross_compile, test_symmetric_cross_compile]


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
