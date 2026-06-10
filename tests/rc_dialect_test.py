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

import shutil  # noqa: E402

try:
    import wasmtime  # noqa: F401

    _HAVE_WASMTIME = True
except ImportError:
    _HAVE_WASMTIME = False
_WASM_LD = shutil.which("wasm-ld")
HAVE_WASM = HAVE and _HAVE_WASMTIME and _WASM_LD is not None


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


def _jit_float(mlir_text, X, M, func_name, extra_passes=()):
    """JIT a float kernel and run it (f64 c-interface ABI). `func_name` is made
    unique per kernel so two engines can coexist in one process without their
    `_mlir_ciface_*` symbols colliding."""
    mlir_jit._ensure_llvm()
    import llvmlite.binding as llvm

    mod = llvm.parse_assembly(
        mlir_jit.mlir_to_llvm_ir(mlir_text, extra_passes=extra_passes)
    )
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
    )(eng.get_function_address("_mlir_ciface_" + func_name))
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
    txt = lower_fused_float(mod, m.weights, func_name="rc_predict_scalar")
    got = _jit_float(txt, X, rc.readout.units, "rc_predict_scalar")
    ref = exe.predict(X)
    d = float(np.max(np.abs(got - ref)))
    assert d < 1e-9, f"rc-dialect kernel vs runtime max|diff|={d:.2e}"
    print(f"  fused rc -> arith kernel matches runtime (max|diff|={d:.1e})")


def test_lower_fused_float_vector():
    if not HAVE:
        print("  (skip)")
        return
    rc, exe, X = _model(units=40, activation=Activation.IDENTITY)
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    # vlen=4 vectorises the N-wide reductions; N=40 also exercises N%vlen==0,
    # and a distinct func name avoids the _mlir_ciface_* collision with the
    # scalar kernel JITed in the same process.
    txt = lower_fused_float(mod, m.weights, vlen=4, func_name="rc_predict_vec")
    got = _jit_float(
        txt,
        X,
        rc.readout.units,
        "rc_predict_vec",
        extra_passes=["--convert-vector-to-llvm"],
    )
    ref = exe.predict(X)
    d = float(np.max(np.abs(got - ref)))
    # float reassociation from the SIMD partial sums, not a logic error.
    assert d < 1e-9, f"vectorized rc kernel vs runtime max|diff|={d:.2e}"
    print(
        f"  vectorized rc -> arith kernel matches runtime (max|diff|={d:.1e})"
    )


def _simd_lines(triple, features, regex, vlen, extra):
    import re

    rc, exe, _ = _model(units=64, activation=Activation.IDENTITY)
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    asm = mlir_jit.cross_compile_object(
        lower_fused_float(mod, m.weights, vlen=vlen),
        triple=triple,
        features=features,
        extra_passes=extra,
        filetype="asm",
    ).decode()
    pat = re.compile(regex)
    return sum(1 for ln in asm.splitlines() if pat.search(ln))


def test_cross_isa_simd():
    """One rc vector dialect -> real SIMD on each ISA; scalar baseline stays
    scalar (LLVM won't reassociate the float reduction). Skips targets llc
    can't register on this host."""
    if not HAVE:
        print("  (skip)")
        return
    targets = [
        (
            "x86_64",
            "x86_64-unknown-linux-gnu",
            "+avx2,+fma",
            r"vfmadd\w*pd|\bmulpd|\baddpd",
            4,
        ),
        (
            "wasm32",
            "wasm32-unknown-unknown",
            "+simd128",
            r"f64x2\.(mul|add)",
            2,
        ),
        (
            "aarch64",
            "aarch64-unknown-linux-gnu",
            "+neon",
            r"\bfmla\b.*\.2d|\bfmul\b.*\.2d|\bfadd\b.*\.2d",
            2,
        ),
    ]
    vectorised = []
    for name, tri, feat, rx, vlen in targets:
        try:
            n_vec = _simd_lines(
                tri, feat, rx, vlen, ["--convert-vector-to-llvm"]
            )
            n_scalar = _simd_lines(tri, feat, rx, 1, [])
        except RuntimeError:
            continue  # target not registered in this llc build
        assert n_scalar == 0, (
            f"{name}: scalar baseline unexpectedly vectorised"
        )
        assert n_vec > 0, f"{name}: vector kernel produced no {name} SIMD"
        vectorised.append(name)
    assert "x86_64" in vectorised, "host x86_64 SIMD must be reachable"
    print(
        f"  one rc dialect -> SIMD on {', '.join(vectorised)} (scalar=0 each)"
    )


def test_f32_armv7_neon_simd():
    """The f32 path packs 4 lanes per 128-bit register, so armv7 NEON (which has
    no f64 SIMD) vectorises — the same rc dialect, dtype='f32', vlen=4."""
    if not HAVE:
        print("  (skip)")
        return
    import re

    rc, exe, _ = _model(units=64, activation=Activation.IDENTITY)
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)

    def simd(vlen, extra):
        asm = mlir_jit.cross_compile_object(
            lower_fused_float(mod, m.weights, vlen=vlen, dtype="f32"),
            triple="armv7-unknown-linux-gnueabihf",
            features="+neon",
            extra_passes=extra,
            filetype="asm",
        ).decode()
        return sum(
            1
            for ln in asm.splitlines()
            if re.search(r"\b(vmla|vmul|vfma|vadd)\.f32\b.*\bq\d", ln)
        )

    try:
        n_vec = simd(4, ["--convert-vector-to-llvm"])
        n_scalar = simd(1, [])
    except RuntimeError:
        print("  (skip: armv7 target not registered)")
        return
    assert n_scalar == 0 and n_vec > 0, (
        f"armv7 f32 vec={n_vec} scalar={n_scalar}"
    )
    print(
        f"  f32 path -> armv7 NEON SIMD (q-regs): {n_vec} vec, {n_scalar} scalar"
    )


def _wasm_run(mlir_text, X, T, K, M):
    """Link a wasm32+simd128 kernel and run it under wasmtime; returns Y (T,M)."""
    import pathlib
    import subprocess
    import tempfile

    import wasmtime

    obj = mlir_jit.cross_compile_object(
        mlir_text,
        triple="wasm32-unknown-unknown",
        features="+simd128",
        extra_passes=["--convert-vector-to-llvm"],
        filetype="obj",
    )
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "rc.o").write_bytes(obj)
        subprocess.run(
            [
                _WASM_LD,
                "--no-entry",
                "--allow-undefined",
                "--export=rc_predict",
                "--export=memory",
                "--export=__heap_base",
                "--strip-debug",
                str(td / "rc.o"),
                "-o",
                str(td / "rc.wasm"),
            ],
            check=True,
            capture_output=True,
        )
        store = wasmtime.Store()
        mw = wasmtime.Module.from_file(store.engine, str(td / "rc.wasm"))
        inst = wasmtime.Instance(store, mw, [])
        ex = inst.exports(store)
        mem, rcf = ex["memory"], ex["rc_predict"]
        hb = ex["__heap_base"].value(store)
        xb = np.ascontiguousarray(X, dtype=np.float32).tobytes()
        xa = (hb + 15) // 16 * 16
        ya = (xa + len(xb) + 15) // 16 * 16
        need = (ya + T * M * 4 + 0xFFFF) // 0x10000
        if need > mem.size(store):
            mem.grow(store, need - mem.size(store))
        mem.write(store, xb, xa)
        rcf(store, T, xa, xa, 0, T * K, 1, ya, ya, 0, T * M, 1)
        yb = mem.read(store, ya, ya + T * M * 4)
    return np.frombuffer(yb, dtype=np.float32).reshape(T, M).astype(np.float64)


def test_wasm_simd_execution():
    """End-to-end: the rc f32 vector kernel cross-compiles to wasm32+simd128,
    links with wasm-ld, and runs under wasmtime with output matching the f64
    runtime (to f32 precision). One rc dialect -> running WASM SIMD."""
    if not HAVE_WASM:
        print("  (skip: wasmtime / wasm-ld not available)")
        return
    rc, exe, X = _model(units=37, activation=Activation.IDENTITY)
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    txt = lower_fused_float(mod, m.weights, vlen=4, dtype="f32")
    got = _wasm_run(txt, X, X.shape[0], rc.input.units, rc.readout.units)
    ref = exe.predict(X)
    d = float(np.max(np.abs(got - ref)))
    assert d < 1e-3, f"wasm kernel vs runtime max|diff|={d:.2e}"
    print(f"  wasm32+simd128 kernel runs under wasmtime (max|diff|={d:.1e})")


TESTS = [
    test_dialect_build_and_verify,
    test_fuse_pattern,
    test_lower_fused_float_numeric,
    test_lower_fused_float_vector,
    test_cross_isa_simd,
    test_f32_armv7_neon_simd,
    test_wasm_simd_execution,
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
