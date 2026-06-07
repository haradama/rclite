"""First-class quant.uniform type representation of an affine model.

`emit_quant_types` expresses the model's quantization (scale/zero-point per
quantity, per-axis scales for per-channel W_res/W_out) as MLIR `quant` dialect
types. Verifies the emitted module is well-formed under mlir-opt, and that
per-channel models produce per-axis (`:f32:0, {...}`) types. Type-level /
declarative (the arith emitters are the executable realization). Skipped when
mlir-opt is absent.
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
from rclite.codegen.mlir_quant_types import (
    emit_quant_types,
    uniform_type,
    verify,
)


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE = shutil.which("mlir-opt") is not None


def _qm(pcr=False, pco=False, topology=Topology.ESN_STANDARD):
    rc = ReservoirComputer(
        input=InputNode(units=2, name="in"),
        reservoir=ReservoirNode(
            units=12,
            topology=topology,
            leak_rate=0.3,
            density=0.3,
            seed=4,
            name="res",
        ),
        readout=ReadoutNode(
            units=3,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=30,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(0).standard_normal((300, 2)) * 0.3
    exe.fit(
        X[:240],
        np.stack(
            [np.sin(np.arange(240) * 0.03 * (k + 1)) for k in range(3)], axis=1
        ),
    )
    cfg = calibrate_from_data(
        rc,
        exe,
        X[:240],
        storage_bits=8,
        per_channel_W_res=pcr,
        per_channel_W_out=pco,
    )
    return quantize_model_affine(rc, exe, cfg)


def test_uniform_type_strings():
    assert uniform_type(8, 0.031) == "!quant.uniform<i8:f32, 3.10000000e-02>"
    assert uniform_type(8, 0.031, 15).endswith(":15>")
    pa = uniform_type(8, np.array([0.1, 0.2, 0.3]))
    assert ":f32:0, {" in pa and pa.count(",") >= 2
    print("  uniform_type: per-tensor / asymmetric / per-axis strings correct")


def test_per_tensor_verifies():
    if not HAVE:
        print("  (skip: mlir-opt not on PATH)")
        return
    for topo in (Topology.ESN_STANDARD, Topology.SCR):
        txt = emit_quant_types(_qm(topology=topo))
        assert verify(txt), f"per-tensor {topo.name} did not verify"
        assert ":f32:0" not in txt  # no per-axis
    print("  per-tensor quant.uniform signature verifies (dense + structured)")


def test_per_channel_per_axis_verifies():
    if not HAVE:
        print("  (skip)")
        return
    for pcr, pco in [(True, False), (False, True), (True, True)]:
        txt = emit_quant_types(_qm(pcr=pcr, pco=pco))
        assert verify(txt), f"per-channel ({pcr},{pco}) did not verify"
        assert ":f32:0, {" in txt, "expected per-axis quant type"
    print("  per-channel models emit verifiable per-axis quant.uniform types")


TESTS = [
    test_uniform_type_strings,
    test_per_tensor_verifies,
    test_per_channel_per_axis_verifies,
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
