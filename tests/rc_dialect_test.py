"""Stage-1 `rc` dialect spike: dialect build, FuseStepReadout rewrite, lowering.

Covers the three legs of the spike:
  1. rclite IR -> `rc` dialect ModuleOp builds and xDSL-verifies.
  2. The FuseStepReadout structural optimization (xDSL RewritePattern) collapses
     reservoir_step;build_phi;readout_linear -> fused_step_readout.
  3. The fused op lowers to an arith/memref/scf `rc_predict` kernel that, JITed
     via the MLIR pipeline, matches the float runtime reference numerically.
Leg 3 is skipped when the LLVM-20 MLIR toolchain is absent (needs the nix shell).
"""

from __future__ import annotations
import ctypes
import pathlib
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
from rclite.core.profile import Activation
from rclite.runtime import RCExecutor
from rclite.ir import build_ir
from rclite.codegen import mlir_jit

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
try:
    import xdsl  # noqa: F401
    from rclite.codegen.rc_dialect_xdsl import (
        build_rc_module,
        fuse_step_readout,
        lower_fused_float,
        count_ops,
    )

    _HAVE_XDSL = True
except ImportError:
    _HAVE_XDSL = False

HAVE = mlir_jit.tools_available() and _HAVE_XDSL


def _model(units=12, K=2, M=3, activation=Activation.TANH, seed=3):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.3,
            seed=seed,
            activation=activation,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=20,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((140, K)) * 0.2
    exe.fit(
        X[:100],
        np.stack(
            [np.sin(np.arange(100) * 0.05 * (m + 1)) for m in range(M)], axis=1
        ),
    )
    return rc, exe, X


def test_dialect_build_and_verify():
    if not _HAVE_XDSL:
        print("  (skip: xdsl not installed)")
        return
    rc, exe, _ = _model()
    mod = build_rc_module(build_ir(rc, exe))
    mod.verify()
    h = count_ops(mod)
    assert h.get("rc.reservoir_step") == 1
    assert h.get("rc.build_phi") == 1
    assert h.get("rc.readout_linear") == 1
    assert "rc.fused_step_readout" not in h
    print(
        "  rc dialect builds + xDSL-verifies (unfused: step;build_phi;readout)"
    )


def test_fuse_pattern():
    if not _HAVE_XDSL:
        print("  (skip)")
        return
    rc, exe, _ = _model()
    mod = build_rc_module(build_ir(rc, exe))
    fuse_step_readout(mod)
    mod.verify()
    h = count_ops(mod)
    assert h.get("rc.fused_step_readout") == 1
    assert "rc.build_phi" not in h and "rc.readout_linear" not in h
    print(
        "  FuseStepReadout rewrite: 3 body ops -> 1 fused op (xDSL-verified)"
    )


def _jit_float(mlir_text, X, K, M):
    """JIT a float `rc_predict` kernel and run it (f64 c-interface ABI)."""
    mlir_jit._ensure_llvm()
    import llvmlite.binding as llvm

    mod = llvm.parse_assembly(mlir_jit.mlir_to_llvm_ir(mlir_text))
    mod.verify()
    tm = llvm.Target.from_triple(
        llvm.get_default_triple()
    ).create_target_machine(opt=3)
    eng = llvm.create_mcjit_compiler(mod, tm)
    eng.finalize_object()
    eng.run_static_constructors()
    fn = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int64,
        ctypes.POINTER(mlir_jit._MemRef1D),
        ctypes.POINTER(mlir_jit._MemRef1D),
    )(eng.get_function_address("_mlir_ciface_rc_predict"))
    Xt = np.ascontiguousarray(X, dtype=np.float64).reshape(-1)
    T = X.shape[0]
    Y = np.zeros(T * M, dtype=np.float64)
    dx, dy = mlir_jit._desc(Xt), mlir_jit._desc(Y)
    fn(ctypes.c_int64(T), ctypes.byref(dx), ctypes.byref(dy))
    return Y.reshape(T, M)


def test_lower_fused_float_numeric():
    if not HAVE:
        print("  (skip: MLIR toolchain not on PATH)")
        return
    rc, exe, X = _model(activation=Activation.IDENTITY)
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    txt = lower_fused_float(mod, m.weights)
    got = _jit_float(txt, X, rc.input.units, rc.readout.units)
    ref = exe.predict(X)
    d = float(np.max(np.abs(got - ref)))
    assert d < 1e-9, f"rc-dialect kernel vs runtime max|diff|={d:.2e}"
    print(f"  fused rc -> arith kernel matches runtime (max|diff|={d:.1e})")


TESTS = [
    test_dialect_build_and_verify,
    test_fuse_pattern,
    test_lower_fused_float_numeric,
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
