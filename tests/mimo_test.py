"""Multi-input / multi-output (MIMO) end-to-end parity tests.

The core pipeline is generic over the input dimension K (`input.units`) and
the output dimension M (`readout.units`); these tests lock that in across the
whole stack for a genuine MIMO model (K > 1 *and* M > 1):

  1. reference runtime  — fit/predict produce (T, M) outputs
  2. LLVM JIT codegen   — bit-exact with the runtime (float)
  3. quantized executor — symmetric i16 fixed-point on-host
  4. C kernel export    — the generated rc_kernel.c is bit-exact with the
                          quantized executor (compiled with host gcc)

The C-export checks are skipped when gcc is unavailable.
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
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Distribution,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.codegen import compile_rc
from rclite.quant import (
    QuantConfig,
    TanhLUTSpec,
    I16FixedPoint,
    quantize_model,
    QuantizedExecutor,
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
)
from rclite.export import export_bundle, emit_symmetric_kernel_c


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

_HAVE_GCC = shutil.which("gcc") is not None

K, N, M = 3, 48, 2  # 3 inputs, 2 outputs


def _build_and_train(*, topology=Topology.SCR, include_input=True):
    rc = ReservoirComputer(
        input=InputNode(
            units=K,
            input_scaling=0.5,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=N,
            topology=topology,
            chain_weight=0.9,
            chain_feedback=0.1,
            leak_rate=0.3,
            spectral_radius=0.9,
            density=0.3,
            seed=42,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=40,
            include_bias=True,
            include_input=include_input,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    T = 300
    X = rng.standard_normal((T, K)) * 0.3
    # Two outputs mixing delayed, distinct input channels.
    Y = np.zeros((T, M))
    for t in range(2, T):
        Y[t, 0] = 0.5 * X[t - 1, 0] + 0.3 * X[t - 2, 1]
        Y[t, 1] = -0.4 * X[t - 1, 2] + 0.2 * X[t, 0]
    exe.fit(X, Y)
    return rc, exe, X


def _assert_close(a, b, atol=1e-9):
    diff = float(np.max(np.abs(a - b)))
    if diff >= atol:
        raise AssertionError(f"arrays differ: max|diff|={diff:.3e} >= {atol}")


# ----------------------------------------------------------------- quant helpers


def _quantize_i16(rc, exe, X):
    cfg = QuantConfig(state_frac=10, input_frac=8, weight_frac=8)
    return quantize_model(
        rc, exe, cfg, target=I16FixedPoint(), lut=TanhLUTSpec(n=256)
    ), cfg


def _sym_python_qy(qm, cfg, X):
    qexe = QuantizedExecutor(qm)
    qexe.reset()
    out = np.zeros((X.shape[0], qm.M), dtype=np.int64)
    for t in range(X.shape[0]):
        u_raw_q = qm.target.quantize_input_array(X[t], cfg)
        u_pre_q = qexe._preprocess_q(u_raw_q)
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(u_raw_q, qexe.state_q)
    return out


def _sym_q_x(qm, cfg, X):
    return np.array(
        [qm.target.quantize_input_array(X[t], cfg) for t in range(X.shape[0])],
        dtype=np.int64,
    )


def _host_c_qy(kernel_src, ctype, q_x_flat, T, K, M, tmp):
    (tmp / "kernel.c").write_text(kernel_src)
    body = ", ".join(str(int(v)) for v in q_x_flat)
    main = "\n".join(
        [
            "#include <stdint.h>",
            "#include <stdio.h>",
            f"extern void rc_predict(int32_t, const {ctype}*, {ctype}*);",
            "int main(void){",
            f"  {ctype} X[{T * K}] = {{ {body} }};",
            f"  {ctype} Y[{T * M}];",
            f"  rc_predict({T}, X, Y);",
            f'  for (int i=0;i<{T * M};i++) printf("%d\\n",(int)Y[i]);',
            "  return 0; }",
        ]
    )
    (tmp / "main.c").write_text(main)
    r = subprocess.run(
        [
            "gcc",
            "-O2",
            "-std=c99",
            "-o",
            str(tmp / "a.out"),
            str(tmp / "main.c"),
            str(tmp / "kernel.c"),
        ],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError("gcc failed:\n" + r.stderr)
    out = subprocess.run(
        [str(tmp / "a.out")], capture_output=True, text=True
    ).stdout
    return np.array(
        [int(v) for v in out.strip().split("\n")], dtype=np.int64
    ).reshape(T, M)


# ----------------------------------------------------------------- tests


def test_runtime_shapes():
    """Reference runtime produces (T, M) regression outputs for K>1 inputs."""
    rc, exe, X = _build_and_train()
    Y = exe.predict(X)
    assert Y.shape == (X.shape[0], M), f"got {Y.shape}, want {(X.shape[0], M)}"
    assert exe.W_in.shape == (N, K), f"W_in {exe.W_in.shape} != {(N, K)}"
    assert exe.W_out.shape[0] == M, f"W_out rows {exe.W_out.shape[0]} != {M}"


def test_jit_parity_mimo():
    """LLVM JIT codegen is bit-exact (float) with the runtime for MIMO."""
    rc, exe, X = _build_and_train()
    Y_np = exe.predict(X)
    Y_jit = compile_rc(rc, exe).predict(X)
    assert Y_jit.shape == (X.shape[0], M)
    _assert_close(Y_jit, Y_np)


def test_jit_parity_mimo_no_input_passthrough():
    """MIMO parity without include_input (phi = bias + state only)."""
    rc, exe, X = _build_and_train(include_input=False)
    _assert_close(compile_rc(rc, exe).predict(X), exe.predict(X))


def test_quant_executor_shapes():
    """Symmetric i16 quantized executor produces (T, M) for K>1."""
    rc, exe, X = _build_and_train()
    qm, cfg = _quantize_i16(rc, exe, X)
    assert (qm.K, qm.M) == (K, M)
    qy = _sym_python_qy(qm, cfg, X)
    assert qy.shape == (X.shape[0], M)


def test_quant_c_export_bit_exact():
    """Generated symmetric C kernel is bit-exact with the quantized executor."""
    if not _HAVE_GCC:
        print("  (skipped: gcc not on PATH)")
        return
    rc, exe, X = _build_and_train()
    qm, cfg = _quantize_i16(rc, exe, X)
    a = _sym_python_qy(qm, cfg, X)
    q_x = _sym_q_x(qm, cfg, X).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        b = _host_c_qy(
            emit_symmetric_kernel_c(qm),
            "int16_t",
            q_x,
            X.shape[0],
            qm.K,
            qm.M,
            pathlib.Path(td),
        )
    diff = int(np.max(np.abs(a - b)))
    assert diff == 0, f"C kernel vs quantized executor diff = {diff}"


def test_bundle_writes_all_files_mimo():
    """export_bundle emits the full C + Rust crate for a MIMO model."""
    rc, exe, X = _build_and_train()
    qm, _ = _quantize_i16(rc, exe, X)
    with tempfile.TemporaryDirectory() as td:
        out = pathlib.Path(td) / "crate"
        export_bundle(qm, out, name="rc_model")
        for f in (
            "rc_kernel.c",
            "rc_model.h",
            "Cargo.toml",
            "build.rs",
            "src/lib.rs",
            "README.md",
        ):
            assert (out / f).exists(), f"missing {f}"
        hdr = (out / "rc_model.h").read_text()
        dims = {
            ln.split()[1]: ln.split()[2]
            for ln in hdr.splitlines()
            if ln.startswith("#define RC_") and len(ln.split()) >= 3
        }
        assert dims.get("RC_INPUT_DIM") == str(K), (
            f"header RC_INPUT_DIM = {dims.get('RC_INPUT_DIM')}, want {K}"
        )
        assert dims.get("RC_OUTPUT_DIM") == str(M), (
            f"header RC_OUTPUT_DIM = {dims.get('RC_OUTPUT_DIM')}, want {M}"
        )


def test_affine_quant_executor_mimo():
    """Affine (asymmetric) quantization also handles K>1, M>1."""
    rc, exe, X = _build_and_train()
    cfg = calibrate_from_data(rc, exe, X[:200], storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)
    assert (qm.K, qm.M) == (K, M)
    qexe = AffineQuantizedExecutor(qm)
    out = np.zeros((X.shape[0], qm.M), dtype=np.int64)
    for t in range(X.shape[0]):
        x_raw_q = qexe._quantize_raw_input(X[t])
        u_pre_q = qexe._quantize_u_pre(X[t])
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(x_raw_q, qexe.state_q)
    assert out.shape == (X.shape[0], M)


def test_affine_c_kernel_bit_exact():
    """The affine C kernel (used by the Arduino target) is bit-exact for MIMO."""
    if not _HAVE_GCC:
        print("  (skipped: gcc not on PATH)")
        return
    from rclite.targets.arduino import emit_affine_kernel_c

    rc, exe, X = _build_and_train()
    Xe = X[:80]
    cfg = calibrate_from_data(rc, exe, X[:200], storage_bits=16)
    qm = quantize_model_affine(rc, exe, cfg)

    qexe = AffineQuantizedExecutor(qm)
    a = np.zeros((Xe.shape[0], qm.M), dtype=np.int64)
    for t in range(Xe.shape[0]):
        x_raw_q = qexe._quantize_raw_input(Xe[t])
        u_pre_q = qexe._quantize_u_pre(Xe[t])
        qexe.step_q(u_pre_q)
        a[t] = qexe.predict_one_q(x_raw_q, qexe.state_q)

    q_x = cfg.input.quantize_array(Xe).astype(np.int64).reshape(-1)
    with tempfile.TemporaryDirectory() as td:
        b = _host_c_qy(
            emit_affine_kernel_c(qm),
            "int16_t",
            q_x,
            Xe.shape[0],
            qm.K,
            qm.M,
            pathlib.Path(td),
        )
    diff = int(np.max(np.abs(a - b)))
    assert diff == 0, f"affine C kernel vs executor diff = {diff}"


def test_gba_symmetric_multi_input_guarded():
    """GBA symmetric i8/i16 must reject K>1 (thumbv4t miscompile) clearly."""
    from rclite.targets.gba import GbaTarget

    rc, exe, X = _build_and_train()
    qm, _ = _quantize_i16(rc, exe, X)
    assert qm.K > 1
    try:
        GbaTarget().compile_quantized(
            qm, output_dir=tempfile.mkdtemp(), test_inputs=X[:5]
        )
    except NotImplementedError:
        return
    except Exception as e:  # missing toolchain etc. — guard runs before that
        raise AssertionError(
            f"expected NotImplementedError, got {type(e).__name__}: {e}"
        )
    raise AssertionError(
        "expected NotImplementedError for GBA symmetric i16 K>1"
    )


TESTS = [
    v
    for k, v in list(globals().items())
    if k.startswith("test_") and callable(v)
]


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
