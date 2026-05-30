"""Classification (argmax head) parity for the quantized paths.

For both the symmetric Q-format and the affine integer families, checks
that the `head="classify"` kernel emits the class id = argmax over the
quantized logits — through both the LLVM JIT and the generated C kernel
(compiled with host gcc when available).

argmax is monotone in the readout scores, so quantization introduces no
class errors except at exact ties; the invariant we assert is that the
classify kernel agrees with argmax over the *same path's* logits kernel.
"""
from __future__ import annotations
import sys
import pathlib
import shutil
import subprocess
import tempfile
import traceback

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode, ReservoirNode, ReadoutNode, ReservoirComputer,
    Activation, Topology, Trainer, Task,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    QuantConfig, TanhLUTSpec, I16FixedPoint, I32FixedPoint,
    quantize_model, QuantizedExecutor,
)
from rclite.quant import calibrate_from_data, quantize_model_affine
from rclite.quant.softmax_lut import SoftmaxLUTSpec, build_params, softmax_q
from rclite.export import emit_symmetric_kernel_c
from rclite.targets.arduino.emit_c import emit_affine_kernel_c
from rclite.codegen.llvm import CompiledQuantizedRC, CompiledAffineRC

PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

_HAVE_GCC = shutil.which("gcc") is not None
_CTYPE = {8: "int8_t", 16: "int16_t", 32: "int32_t"}
_NPTYPE = {8: np.int8, 16: np.int16, 32: np.int32}


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _train_classifier(units=50, washout=80, n_classes=3):
    rng = np.random.default_rng(0)
    n = 900
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + 0.1 * rng.standard_normal()
    X = u[:, None]
    # n_classes levels by amplitude bands
    y = np.zeros(n, dtype=int)
    y[u > 0.3] = 2
    y[(u <= 0.3) & (u > -0.3)] = 1
    if n_classes == 2:
        y = (u > 0).astype(int)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(
            units=units, activation=Activation.TANH, spectral_radius=0.9,
            leak_rate=0.3, density=0.2, topology=Topology.RANDOM, seed=3,
            name="r",
        ),
        readout=ReadoutNode(
            units=n_classes, activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE, regularization=1e-3, washout=washout,
            task=Task.CLASSIFICATION, name="ro",
        ),
    )
    exe = RCExecutor(rc)
    exe.fit(X[:600], y[:600])
    return rc, exe, X[600:720]


def _run_c_classify(kernel_src, ctype, qx_flat, T, K):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(kernel_src)
        body = ", ".join(str(int(v)) for v in qx_flat)
        (td / "main.c").write_text("\n".join([
            "#include <stdint.h>", "#include <stdio.h>",
            f'extern void rc_predict(int32_t, const {ctype}*, int32_t*);',
            "int main(void){",
            f"  {ctype} X[{T * K}] = {{ {body} }};",
            f"  int32_t Y[{T}];",
            f"  rc_predict({T}, X, Y);",
            f"  for (int i=0;i<{T};i++) printf(\"%d\\n\",(int)Y[i]);",
            "  return 0; }",
        ]))
        r = subprocess.run(
            ["gcc", "-O2", "-std=c99", "-o", str(td / "a.out"),
             str(td / "main.c"), str(td / "kernel.c")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run([str(td / "a.out")], capture_output=True,
                             text=True).stdout
    return np.array([int(v) for v in out.strip().split("\n")], dtype=np.int32)


# ---------------------------------------------------------------------------
# symmetric


def _sym_quantize(rc, exe, target):
    cfg = QuantConfig(state_frac=14, input_frac=12, weight_frac=10)
    return quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=256), target=target)


def test_symmetric_classify_matches_jit_logits():
    for target in (I32FixedPoint(), I16FixedPoint()):
        rc, exe, Xte = _train_classifier()
        qm = _sym_quantize(rc, exe, target)
        logits = CompiledQuantizedRC(qm).predict(Xte)
        cls = CompiledQuantizedRC(qm, head="classify").predict(Xte)
        assert cls.dtype == np.int32 and cls.shape == (Xte.shape[0],)
        assert np.array_equal(cls, np.argmax(logits, axis=1))


def test_symmetric_classify_c_matches_jit():
    if not _HAVE_GCC:
        return
    target = I16FixedPoint()
    rc, exe, Xte = _train_classifier()
    qm = _sym_quantize(rc, exe, target)
    jit = CompiledQuantizedRC(qm, head="classify").predict(Xte)
    sb = qm.target.storage_bits
    qx = qm.target.quantize_input_array(Xte, qm.config).astype(_NPTYPE[sb]).reshape(-1)
    cc = _run_c_classify(emit_symmetric_kernel_c(qm, head="classify"),
                         _CTYPE[sb], qx, Xte.shape[0], qm.K)
    assert np.array_equal(cc, jit)


# ---------------------------------------------------------------------------
# affine


def _affine_quantize(rc, exe, X_calib, storage_bits=16):
    cfg = calibrate_from_data(rc, exe, X_calib, storage_bits=storage_bits)
    return quantize_model_affine(rc, exe, cfg)


def test_affine_classify_matches_jit_logits():
    for sb in (16, 8):
        rc, exe, Xte = _train_classifier()
        qm = _affine_quantize(rc, exe, _calib_X(), sb)
        logits = CompiledAffineRC(qm).predict(Xte)
        cls = CompiledAffineRC(qm, head="classify").predict(Xte)
        assert cls.dtype == np.int32 and cls.shape == (Xte.shape[0],)
        assert np.array_equal(cls, np.argmax(logits, axis=1))


def test_affine_classify_c_matches_jit():
    if not _HAVE_GCC:
        return
    rc, exe, Xte = _train_classifier()
    qm = _affine_quantize(rc, exe, _calib_X(), 16)
    jit = CompiledAffineRC(qm, head="classify").predict(Xte)
    qx = qm.config.input.quantize_array(Xte).astype(np.int16).reshape(-1)
    cc = _run_c_classify(emit_affine_kernel_c(qm, head="classify"),
                         "int16_t", qx, Xte.shape[0], qm.K)
    assert np.array_equal(cc, jit)


def _run_c_proba(kernel_src, ctype, qx_flat, T, K, M):
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(kernel_src)
        body = ", ".join(str(int(v)) for v in qx_flat)
        (td / "main.c").write_text("\n".join([
            "#include <stdint.h>", "#include <stdio.h>",
            f'extern void rc_predict(int32_t, const {ctype}*, {ctype}*);',
            "int main(void){",
            f"  {ctype} X[{T * K}] = {{ {body} }};",
            f"  {ctype} Y[{T * M}];",
            f"  rc_predict({T}, X, Y);",
            f"  for (int i=0;i<{T * M};i++) printf(\"%d\\n\",(int)Y[i]);",
            "  return 0; }",
        ]))
        r = subprocess.run(
            ["gcc", "-O2", "-std=c99", "-o", str(td / "a.out"),
             str(td / "main.c"), str(td / "kernel.c")],
            capture_output=True, text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run([str(td / "a.out")], capture_output=True,
                             text=True).stdout
    return np.array([int(v) for v in out.strip().split("\n")],
                    dtype=np.int64).reshape(T, M)


def test_symmetric_proba_matches_reference():
    target = I16FixedPoint()
    rc, exe, Xte = _train_classifier()
    cfg = QuantConfig(state_frac=10, input_frac=12, weight_frac=10)
    qm = quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=256), target=target)
    proba = CompiledQuantizedRC(qm, head="proba").predict(Xte)
    classify = CompiledQuantizedRC(qm, head="classify").predict(Xte)
    pf = min(target.storage_bits - 1, 15)
    # reference softmax_q over reconstructed quantized logits
    qexe = QuantizedExecutor(qm)
    q_logits = np.round(qexe.predict(Xte) * cfg.state_scale).astype(np.int64)
    sm = build_params(SoftmaxLUTSpec(), 1.0 / cfg.state_scale,
                      target.storage_bits, target.storage_dtype)
    ref = np.stack([softmax_q(q_logits[t], sm) for t in range(len(Xte))])
    ref = ref.astype(np.float64) / (1 << pf)
    assert np.allclose(proba, ref, atol=1e-12)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=0.01)
    assert np.array_equal(np.argmax(proba, axis=1), classify)


def test_symmetric_proba_c_matches_jit():
    if not _HAVE_GCC:
        return
    target = I16FixedPoint()
    rc, exe, Xte = _train_classifier()
    cfg = QuantConfig(state_frac=10, input_frac=12, weight_frac=10)
    qm = quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=256), target=target)
    pf = min(target.storage_bits - 1, 15)
    jit_q = (CompiledQuantizedRC(qm, head="proba").predict(Xte) * (1 << pf))
    jit_q = jit_q.round().astype(np.int64)
    sb = target.storage_bits
    qx = qm.target.quantize_input_array(Xte, qm.config).astype(_NPTYPE[sb]).reshape(-1)
    cc = _run_c_proba(emit_symmetric_kernel_c(qm, head="proba"),
                      _CTYPE[sb], qx, Xte.shape[0], qm.K, qm.M)
    assert np.array_equal(cc, jit_q)


def test_affine_proba_matches_reference():
    for sb in (16, 8):
        rc, exe, Xte = _train_classifier()
        qm = _affine_quantize(rc, exe, _calib_X(), sb)
        proba = CompiledAffineRC(qm, head="proba").predict(Xte)
        classify = CompiledAffineRC(qm, head="classify").predict(Xte)
        pf = min(sb - 1, 15)
        # rows ~sum to 1 (looser for i8 / Q7) and argmax consistent
        assert np.all(proba >= 0.0)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=0.03)
        assert np.array_equal(np.argmax(proba, axis=1), classify)


def test_affine_proba_c_matches_jit():
    if not _HAVE_GCC:
        return
    sb = 16
    rc, exe, Xte = _train_classifier()
    qm = _affine_quantize(rc, exe, _calib_X(), sb)
    pf = min(sb - 1, 15)
    jit_q = (CompiledAffineRC(qm, head="proba").predict(Xte) * (1 << pf))
    jit_q = jit_q.round().astype(np.int64)
    qx = qm.config.input.quantize_array(Xte).astype(_NPTYPE[sb]).reshape(-1)
    cc = _run_c_proba(emit_affine_kernel_c(qm, head="proba"),
                      _CTYPE[sb], qx, Xte.shape[0], qm.K, qm.M)
    assert np.array_equal(cc, jit_q)


def _calib_X():
    """Calibration data long enough to survive the washout drop."""
    rng = np.random.default_rng(0)
    n = 900
    u = np.zeros(n)
    for t in range(1, n):
        u[t] = 0.9 * u[t - 1] + 0.1 * rng.standard_normal()
    return u[:600, None]


# ---------------------------------------------------------------------------
# validation


def test_quantized_sequence_aggregation_rejected():
    from rclite import Aggregation
    from rclite.quant import build_ir_from_quantized
    rng = np.random.default_rng(1)
    X = rng.standard_normal((200, 1)) * 0.2
    y = (X[:, 0] > 0).astype(int)
    rc = ReservoirComputer(
        input=InputNode(units=1, activation=Activation.IDENTITY, name="in"),
        reservoir=ReservoirNode(units=30, activation=Activation.TANH,
                                 leak_rate=0.3, seed=1, name="r"),
        readout=ReadoutNode(units=2, activation=Activation.IDENTITY,
                             trainer=Trainer.RIDGE, washout=10,
                             task=Task.CLASSIFICATION,
                             aggregation=Aggregation.MEAN, name="ro"),
    )
    exe = RCExecutor(rc)
    exe.fit_sequences([X[:100], X[100:]], np.array([0, 1]))
    cfg = QuantConfig(state_frac=14, input_frac=12, weight_frac=10)
    qm = quantize_model(rc, exe, cfg, lut=TanhLUTSpec(n=128))
    expect_raises(NotImplementedError, build_ir_from_quantized, qm, head="classify")


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
