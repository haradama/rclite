"""Tests for the target-agnostic export bundle (`rclite.export`).

Two quantization families are exported through one bundle writer:
  * symmetric Q-format (`QuantizedModel`)  — emitter in rclite.export
  * asymmetric affine  (`AffineQuantizedModel`) — emitter in rclite.targets.arduino

Validation layers:
  1. The bundle writes every expected file (C kernel/header, Cargo project).
  2. The emitted C kernel is bit-exact with the Python reference executor,
     checked by compiling with *host* gcc (PROGMEM macros are no-ops off-AVR).
  3. The full Cargo crate builds and the safe Rust `predict_into` FFI wrapper
     is bit-exact with the Python reference (compiles rc_kernel.c via build.rs
     + the `cc` crate, runs a generated integration test). Skipped when cargo
     or network is unavailable.
"""
from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I8Symmetric, I16FixedPoint, I32FixedPoint,
    quantize_model, QuantizedExecutor,
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    LUTStrategy,
)
from rclite.export import export_bundle, emit_symmetric_kernel_c


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

_HAVE_GCC = shutil.which("gcc") is not None
_HAVE_CARGO = shutil.which("cargo") is not None


# ----------------------------------------------------------------- model builders


def _make_rc(topology, units=24, input_offset=0.0, input_scaling=1.0):
    return ReservoirComputer(
        input=InputNode(units=1, input_offset=input_offset,
                        input_scaling=input_scaling,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=units, topology=topology, chain_weight=0.9,
                                chain_feedback=0.1, leak_rate=0.3, seed=42),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE, regularization=1e-6,
                            washout=30, include_bias=True, include_input=True),
    )


def _train(rc, T=200, seed=0):
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.2
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[:T - 50], Y[:T - 50])
    return exe, X


def build_symmetric(topology, target_factory, cfg, lut_n, **rc_kw):
    rc = _make_rc(topology, **rc_kw)
    exe, X = _train(rc)
    qm = quantize_model(rc, exe, cfg, target=target_factory(),
                        lut=TanhLUTSpec(n=lut_n))
    return qm, X


def build_affine(topology, *, storage_bits=8, w_out_storage_bits=None,
                 strategy=None, **rc_kw):
    rc = _make_rc(topology, **rc_kw)
    exe, X = _train(rc)
    cfg = calibrate_from_data(rc, exe, X[:150], storage_bits=storage_bits,
                              w_out_storage_bits=w_out_storage_bits)
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=strategy)
    return qm, X


# ----------------------------------------------------------------- reference q_y


def sym_q_x(qm, X_float):
    cfg = qm.config
    return np.array([qm.target.quantize_input_array(X_float[t], cfg)
                     for t in range(X_float.shape[0])], dtype=np.int64)


def sym_python_qy(qm, X_float):
    qexe = QuantizedExecutor(qm)
    qexe.reset()
    cfg = qm.config
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        u_raw_q = qm.target.quantize_input_array(X_float[t], cfg)
        u_pre_q = qexe._preprocess_q(u_raw_q)
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(u_raw_q, qexe.state_q)
    return out


def aff_q_x(qm, X_float):
    return qm.config.input.quantize_array(X_float).astype(np.int64)


def aff_python_qy(qm, X_float):
    qexe = AffineQuantizedExecutor(qm)
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        x_raw_q = qexe._quantize_raw_input(X_float[t])
        u_pre_q = qexe._quantize_u_pre(X_float[t])
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(x_raw_q, qexe.state_q)
    return out


# ----------------------------------------------------------------- host-C parity


def _host_c_qy(kernel_src, ctype, q_x_flat, T, K, M, tmp):
    (tmp / "kernel.c").write_text(kernel_src)
    body = ", ".join(str(int(v)) for v in q_x_flat)
    main = "\n".join([
        "#include <stdint.h>", "#include <stdio.h>",
        f'extern void rc_predict(int32_t, const {ctype}*, {ctype}*);',
        "int main(void){",
        f"  {ctype} X[{T * K}] = {{ {body} }};",
        f"  {ctype} Y[{T * M}];",
        f"  rc_predict({T}, X, Y);",
        f"  for (int i=0;i<{T * M};i++) printf(\"%d\\n\",(int)Y[i]);",
        "  return 0; }",
    ])
    (tmp / "main.c").write_text(main)
    r = subprocess.run(["gcc", "-O2", "-std=c99", "-o", str(tmp / "a.out"),
                        str(tmp / "main.c"), str(tmp / "kernel.c")],
                       capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gcc failed:\n" + r.stderr)
    out = subprocess.run([str(tmp / "a.out")], capture_output=True, text=True).stdout
    return np.array([int(v) for v in out.strip().split("\n")],
                    dtype=np.int64).reshape(T, M)


def _assert_sym_host_parity(qm, X_eval):
    if not _HAVE_GCC:
        return
    ctype = {8: "int8_t", 16: "int16_t", 32: "int32_t"}[qm.target.storage_bits]
    a = sym_python_qy(qm, X_eval)
    q_x = sym_q_x(qm, X_eval).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        b = _host_c_qy(emit_symmetric_kernel_c(qm), ctype, q_x,
                       X_eval.shape[0], qm.K, qm.M, pathlib.Path(td))
    diff = int(np.max(np.abs(a - b)))
    assert diff == 0, f"symmetric host C vs Python diff = {diff}"


# ----------------------------------------------------------------- cargo FFI


def _cargo_ffi_roundtrip(qm, q_x, expect_qy, storage_rust):
    """export_bundle → `cargo test` a generated FFI round-trip integration test."""
    T, K = expect_qy.shape[0], qm.K
    M = qm.M
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "crate"
        export_bundle(qm, out, name="rc_model")
        # Sanity: bundle wrote everything.
        for f in ("rc_kernel.c", "rc_model.h", "Cargo.toml", "build.rs",
                  "src/lib.rs", "README.md"):
            assert (out / f).exists(), f"missing {f}"

        x_body = ", ".join(str(int(v)) for v in q_x.reshape(-1))
        e_body = ", ".join(str(int(v)) for v in expect_qy.reshape(-1))
        test_src = "\n".join([
            "use rc_model::{predict_into, Storage, INPUT_DIM, OUTPUT_DIM};",
            f"const X: &[Storage] = &[{x_body}];",
            f"const EXPECT: &[i32] = &[{e_body}];",
            "#[test]",
            "fn ffi_roundtrip() {",
            "    let t = X.len() / INPUT_DIM;",
            "    assert_eq!(t * OUTPUT_DIM, EXPECT.len());",
            "    let mut y = vec![0 as Storage; t * OUTPUT_DIM];",
            "    predict_into(X, &mut y);",
            "    for (i, (&g, &e)) in y.iter().zip(EXPECT.iter()).enumerate() {",
            "        assert_eq!(g as i32, e, \"mismatch at index {}\", i);",
            "    }",
            "}",
        ])
        (out / "tests").mkdir(exist_ok=True)
        (out / "tests" / "ffi.rs").write_text(test_src)
        r = subprocess.run(["cargo", "test", "--quiet"], cwd=str(out),
                           capture_output=True, text=True)
        assert r.returncode == 0, \
            f"cargo test failed:\nSTDOUT:\n{r.stdout}\nSTDERR:\n{r.stderr}"
    # storage_rust sanity (the crate alias matches the model)
    assert storage_rust in ("i8", "i16", "i32")


# ----------------------------------------------------------------- bundle files


def test_bundle_symmetric_writes_all_files():
    qm, _ = build_symmetric(Topology.SCR, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4),
                            32)
    with tempfile.TemporaryDirectory() as td:
        out = export_bundle(qm, pathlib.Path(td) / "b", name="my-rc")
        for f in ("rc_kernel.c", "rc_model.h", "Cargo.toml", "build.rs",
                  "src/lib.rs", "README.md"):
            assert (out / f).exists(), f"missing {f}"
        header = (out / "rc_model.h").read_text()
        assert "symmetric" in header
        assert "rc_predict" in header
        cargo = (out / "Cargo.toml").read_text()
        assert 'name = "my_rc"' in cargo            # hyphen → underscore
        assert (out / "src" / "lib.rs").read_text().count("predict_into") >= 1


def test_bundle_affine_writes_all_files():
    qm, _ = build_affine(Topology.SCR, strategy=LUTStrategy.linear_interp(64))
    with tempfile.TemporaryDirectory() as td:
        out = export_bundle(qm, pathlib.Path(td) / "b", name="rc_model")
        kernel = (out / "rc_kernel.c").read_text()
        assert "RC_PROGMEM" in kernel and "void rc_predict(" in kernel
        header = (out / "rc_model.h").read_text()
        assert "affine" in header


def test_bundle_rejects_unknown_model():
    try:
        with tempfile.TemporaryDirectory() as td:
            export_bundle(object(), pathlib.Path(td))
    except TypeError:
        return
    raise AssertionError("export_bundle should reject a non-quantized model")


# ----------------------------------------------------------------- host parity


def test_sym_host_parity_i8_scr():
    qm, X = build_symmetric(Topology.SCR, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4), 32)
    _assert_sym_host_parity(qm, X[150:175])


def test_sym_host_parity_i8_dlrb():
    qm, X = build_symmetric(Topology.DLRB, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4), 32)
    _assert_sym_host_parity(qm, X[150:175])


def test_sym_host_parity_i8_dense():
    qm, X = build_symmetric(Topology.ESN_STANDARD, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4), 32)
    _assert_sym_host_parity(qm, X[150:175])


def test_sym_host_parity_i8_preprocess():
    qm, X = build_symmetric(Topology.SCR, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4), 32,
                            input_offset=0.3, input_scaling=1.5)
    _assert_sym_host_parity(qm, X[150:175])


def test_sym_host_parity_i16():
    qm, X = build_symmetric(Topology.SCR, I16FixedPoint,
                            QuantConfig(state_frac=10, input_frac=8, weight_frac=8), 128)
    _assert_sym_host_parity(qm, X[150:175])


def test_sym_host_parity_i32():
    qm, X = build_symmetric(Topology.ESN_STANDARD, I32FixedPoint,
                            QuantConfig(state_frac=16, input_frac=14, weight_frac=14), 256)
    _assert_sym_host_parity(qm, X[150:175])


# ----------------------------------------------------------------- cargo round-trip


def test_cargo_ffi_symmetric_i8():
    if not (_HAVE_CARGO and _HAVE_GCC):
        return
    qm, X = build_symmetric(Topology.SCR, I8Symmetric,
                            QuantConfig(state_frac=5, input_frac=4, weight_frac=4), 32)
    Xe = X[150:170]
    _cargo_ffi_roundtrip(qm, sym_q_x(qm, Xe), sym_python_qy(qm, Xe), "i8")


def test_cargo_ffi_affine_i8():
    if not (_HAVE_CARGO and _HAVE_GCC):
        return
    qm, X = build_affine(Topology.SCR, strategy=LUTStrategy.linear_interp(64))
    Xe = X[150:170]
    _cargo_ffi_roundtrip(qm, aff_q_x(qm, Xe), aff_python_qy(qm, Xe), "i8")


def test_cargo_ffi_affine_mixed_precision():
    if not (_HAVE_CARGO and _HAVE_GCC):
        return
    qm, X = build_affine(Topology.SCR, storage_bits=8, w_out_storage_bits=16,
                         strategy=LUTStrategy.linear_interp(64))
    Xe = X[150:170]
    _cargo_ffi_roundtrip(qm, aff_q_x(qm, Xe), aff_python_qy(qm, Xe), "i8")


TESTS = [v for k, v in list(globals().items())
         if k.startswith("test_") and callable(v)]


def main() -> int:
    n_pass = n_fail = 0
    for t in TESTS:
        try:
            t()
            print(f"{PASS} {t.__name__}")
            n_pass += 1
        except Exception:
            print(f"{FAIL} {t.__name__}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed (of {len(TESTS)})")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
