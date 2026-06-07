"""xDSL-built quant.uniform signature == text-emitted, and verifies under mlir-opt.

Stage-(1) migration of `mlir_quant_types`. Checks the xDSL path (a) verifies
under mlir-opt, (b) carries the per-axis `:f32:0, {...}` types for per-channel
models, and (c) emits exactly the same multiset of `!quant.uniform<...>` types
as the text emitter (the scale/zp formatting is shared).

Skipped unless the `mlir` extra (xdsl) is installed AND mlir-opt is on PATH.
"""
from __future__ import annotations
import io
import re
import shutil

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.codegen import mlir_quant_types

try:
    import xdsl  # noqa: F401
    from rclite.codegen.mlir_quant_types_xdsl import emit_quant_types_xdsl, uniform_type
    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

PASS = "\033[32m[PASS]\033[0m"
SKIP = "\033[33m[SKIP]\033[0m"
_QTYPE = re.compile(r"!quant\.uniform<[^>]*>")


def _qm(pcr=False, pco=False, topology=Topology.ESN_STANDARD):
    rc = ReservoirComputer(
        input=InputNode(units=2, name="in"),
        reservoir=ReservoirNode(units=12, topology=topology, leak_rate=0.3,
                                density=0.3, seed=4, name="res"),
        readout=ReadoutNode(units=3, trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=30, include_bias=True, include_input=True,
                            name="out"),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(0).standard_normal((300, 2)) * 0.3
    exe.fit(X[:240], np.stack([np.sin(np.arange(240) * 0.03 * (k + 1))
                               for k in range(3)], axis=1))
    cfg = calibrate_from_data(rc, exe, X[:240], storage_bits=8,
                              per_channel_W_res=pcr, per_channel_W_out=pco)
    return quantize_model_affine(rc, exe, cfg)


def _show(attr):
    from xdsl.printer import Printer
    buf = io.StringIO()
    Printer(stream=buf).print_attribute(attr)
    return buf.getvalue()


def _guard():
    if not _HAVE_XDSL:
        print(f"{SKIP} xdsl not installed (pip install 'rclite[mlir]')")
        return False
    if shutil.which("mlir-opt") is None:
        print(f"{SKIP} mlir-opt not on PATH")
        return False
    return True


def _check(label, qm):
    txt = emit_quant_types_xdsl(qm)
    assert mlir_quant_types.verify(txt), f"{label}: xDSL output did not verify"
    # same multiset of quant.uniform types as the text emitter
    xdsl_types = sorted(_QTYPE.findall(txt))
    text_types = sorted(_QTYPE.findall(mlir_quant_types.emit_quant_types(qm)))
    assert xdsl_types == text_types, (
        f"{label}: type mismatch\n xDSL: {xdsl_types}\n text: {text_types}")
    print(f"{PASS} {label}: verifies + {len(xdsl_types)} quant types == text emitter")
    return txt


def test_xdsl_quant_types_strings():
    if not _HAVE_XDSL:
        print(f"{SKIP} xdsl not installed")
        return
    # the xDSL uniform_type returns the type attribute; its printed form matches
    assert _show(uniform_type(8, 0.031)) == "!quant.uniform<i8:f32, 3.10000000e-02>"
    assert _show(uniform_type(8, 0.031, 15)).endswith(":15>")
    pa = _show(uniform_type(8, np.array([0.1, 0.2, 0.3])))
    assert ":f32:0, {" in pa and pa.count(",") >= 2
    print(f"{PASS} uniform_type: per-tensor / asymmetric / per-axis attrs correct")


def test_xdsl_quant_types_per_tensor():
    if not _guard():
        return
    for topo in (Topology.ESN_STANDARD, Topology.SCR):
        txt = _check(f"per-tensor {topo.name}", _qm(topology=topo))
        assert ":f32:0" not in txt


def test_xdsl_quant_types_per_channel():
    if not _guard():
        return
    for pcr, pco in [(True, False), (False, True), (True, True)]:
        txt = _check(f"per-channel ({pcr},{pco})", _qm(pcr=pcr, pco=pco))
        assert ":f32:0, {" in txt


if __name__ == "__main__":
    test_xdsl_quant_types_strings()
    test_xdsl_quant_types_per_tensor()
    test_xdsl_quant_types_per_channel()
