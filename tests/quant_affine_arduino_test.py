"""Tests for the Arduino Uno (8-bit AVR) affine target.

The portable C kernel is checked bit-exactly against the Python reference
by compiling it with *host* gcc (the source is platform-independent; on a
host the PROGMEM macros are no-ops). The arduino-cli build (real AVR) is a
size/smoke check, skipped when the toolchain is absent.
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
    Activation, Distribution, Topology, Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    calibrate_from_data, quantize_model_affine, AffineQuantizedExecutor,
    LUTStrategy,
)
from rclite.targets.arduino import emit_affine_kernel_c, ArduinoUnoTarget


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"

_HAVE_GCC = shutil.which("gcc") is not None
_HAVE_ARDUINO = shutil.which("arduino-cli") is not None


def _build(topology=Topology.SCR, storage_bits=8, w_out_storage_bits=None,
            strategy=None, units=24, T=200, seed=0,
            input_offset=0.0, input_scaling=1.0):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_offset=input_offset,
                        input_scaling=input_scaling,
                        input_distribution=Distribution.BERNOULLI),
        reservoir=ReservoirNode(units=units, topology=topology, chain_weight=0.9,
                                 chain_feedback=0.1, leak_rate=0.3, seed=42),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE, regularization=1e-6,
                             washout=30, include_bias=True, include_input=True),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((T, 1)) * 0.2
    Y = np.sin(np.arange(T) * 0.1)[:, None]
    exe.fit(X[:T - 50], Y[:T - 50])
    cfg = calibrate_from_data(rc, exe, X[:T - 50], storage_bits=storage_bits,
                                w_out_storage_bits=w_out_storage_bits)
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=strategy)
    return rc, exe, qm, X


def _python_qy(qm, X_float):
    qexe = AffineQuantizedExecutor(qm)
    T = X_float.shape[0]
    out = np.zeros((T, qm.M), dtype=np.int64)
    for t in range(T):
        x_raw_q = qexe._quantize_raw_input(X_float[t])
        u_pre_q = qexe._quantize_u_pre(X_float[t])
        qexe.step_q(u_pre_q)
        out[t] = qexe.predict_one_q(x_raw_q, qexe.state_q)
    return out


def _host_c_qy(qm, X_float, tmp: pathlib.Path, allow_i32_accum=False):
    """Compile the emitted C kernel with host gcc and return its q_y."""
    cfg = qm.config
    ctype = "int8_t" if qm.storage_bits == 8 else "int16_t"
    q_x = cfg.input.quantize_array(X_float).astype(np.int64).reshape(-1)
    T = X_float.shape[0]
    (tmp / "kernel.c").write_text(
        emit_affine_kernel_c(qm, allow_i32_accum=allow_i32_accum))
    main = "\n".join([
        "#include <stdint.h>", "#include <stdio.h>",
        f'extern void rc_predict(int32_t, const {ctype}*, {ctype}*);',
        "int main(void){",
        f"  {ctype} X[{T * qm.K}] = {{ {', '.join(str(int(v)) for v in q_x)} }};",
        f"  {ctype} Y[{T * qm.M}];",
        f"  rc_predict({T}, X, Y);",
        f"  for (int i = 0; i < {T * qm.M}; i++) printf(\"%d\\n\", (int)Y[i]);",
        "  return 0; }",
    ])
    (tmp / "main.c").write_text(main)
    exe_path = tmp / "a.out"
    r = subprocess.run(
        ["gcc", "-O2", "-std=c99", "-o", str(exe_path),
         str(tmp / "main.c"), str(tmp / "kernel.c")],
        capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError("gcc failed:\n" + r.stderr)
    out = subprocess.run([str(exe_path)], capture_output=True, text=True).stdout
    vals = [int(v) for v in out.strip().split("\n")]
    return np.array(vals, dtype=np.int64).reshape(T, qm.M)


def _assert_host_parity(qm, X_eval):
    if not _HAVE_GCC:
        return  # no host compiler; skip
    a = _python_qy(qm, X_eval)
    # Both accumulator modes must be bit-exact with the Python reference:
    # the default i64 path (used by the Arduino target) and the i32-accum
    # path (faster, used where the compiler handles it).
    for allow_i32 in (False, True):
        with tempfile.TemporaryDirectory() as td:
            b = _host_c_qy(qm, X_eval, pathlib.Path(td), allow_i32_accum=allow_i32)
            diff = int(np.max(np.abs(a - b)))
            assert diff == 0, \
                f"host C vs Python q_y diff = {diff} (allow_i32_accum={allow_i32})"


# ---------------------------------------------------------------- C emitter shape


def test_emitted_c_has_progmem_and_rc_predict():
    _, _, qm, _ = _build()
    src = emit_affine_kernel_c(qm)
    assert "RC_PROGMEM" in src
    assert "void rc_predict(" in src
    assert "pgm_read_byte" in src  # AVR read path present


def test_emitted_c_structured_omits_w_res():
    _, _, qm, _ = _build(topology=Topology.SCR)
    src = emit_affine_kernel_c(qm)
    assert "rc_W_res" not in src   # SCR uses scalar chain, no dense W_res


def test_emitted_c_dense_includes_w_res():
    _, _, qm, _ = _build(topology=Topology.ESN_STANDARD)
    src = emit_affine_kernel_c(qm)
    assert "rc_W_res" in src


# ---------------------------------------------------------------- host parity


def test_host_parity_scr_direct():
    _, _, qm, X = _build(topology=Topology.SCR, strategy=LUTStrategy.direct())
    _assert_host_parity(qm, X[150:175])


def test_host_parity_scr_interp():
    _, _, qm, X = _build(topology=Topology.SCR,
                          strategy=LUTStrategy.linear_interp(64))
    _assert_host_parity(qm, X[150:175])


def test_host_parity_scr_poly():
    _, _, qm, X = _build(topology=Topology.SCR,
                          strategy=LUTStrategy.polynomial(degree=5))
    _assert_host_parity(qm, X[150:175])


def test_host_parity_dlr():
    _, _, qm, X = _build(topology=Topology.DLR,
                          strategy=LUTStrategy.linear_interp(64))
    _assert_host_parity(qm, X[150:175])


def test_host_parity_dlrb():
    _, _, qm, X = _build(topology=Topology.DLRB,
                          strategy=LUTStrategy.linear_interp(64))
    _assert_host_parity(qm, X[150:175])


def test_host_parity_dense():
    _, _, qm, X = _build(topology=Topology.ESN_STANDARD,
                          strategy=LUTStrategy.direct())
    _assert_host_parity(qm, X[150:175])


def test_host_parity_mixed_precision():
    _, _, qm, X = _build(topology=Topology.SCR, storage_bits=8,
                          w_out_storage_bits=16,
                          strategy=LUTStrategy.linear_interp(64))
    assert qm.W_out_q.dtype == np.int16
    _assert_host_parity(qm, X[150:175])


def test_host_parity_i16():
    _, _, qm, X = _build(topology=Topology.SCR, storage_bits=16,
                          strategy=LUTStrategy.direct())
    _assert_host_parity(qm, X[150:175])


def test_host_parity_i16_dense():
    """i16 + dense W_res (ESN_STANDARD): host gcc must match the Python ref.

    Pins the i16 dense kernel's algorithm (the i16xi16 matvec uses i64
    accumulators). On a 32 KB-Flash Uno a 65536-entry DIRECT LUT does not fit,
    so the AVR bench uses a small interp LUT; the algorithm itself is correct
    at i16, as this exercises with a DIRECT LUT on the host.
    """
    _, _, qm, X = _build(topology=Topology.ESN_STANDARD, storage_bits=16,
                          strategy=LUTStrategy.direct())
    _assert_host_parity(qm, X[150:175])


def test_direct_lut_rejects_oversized_storage():
    """A DIRECT activation LUT for i32 would need 2**32 entries — quantizing
    must raise a clear error fast, not hang on a ~34 GB allocation."""
    from rclite.quant import calibrate_from_data, quantize_model_affine
    rc, exe, _, X = _build(topology=Topology.SCR, storage_bits=8)
    cfg = calibrate_from_data(rc, exe, X[:150], storage_bits=32)
    try:
        quantize_model_affine(rc, exe, cfg, lut_strategy=LUTStrategy.direct())
    except ValueError as e:
        assert "storage_bits" in str(e)
        return
    raise AssertionError("expected ValueError for DIRECT LUT at i32")


def test_host_parity_with_preprocess():
    _, _, qm, X = _build(topology=Topology.SCR, input_offset=0.3,
                          input_scaling=1.5,
                          strategy=LUTStrategy.linear_interp(64))
    assert qm.has_integer_preprocess
    _assert_host_parity(qm, X[150:175])


# ---------------------------------------------------------------- arduino build


def test_arduino_emit_without_build():
    """Emitting the sketch must not require arduino-cli."""
    _, _, qm, X = _build(topology=Topology.SCR,
                          strategy=LUTStrategy.linear_interp(64))
    with tempfile.TemporaryDirectory() as td:
        art = ArduinoUnoTarget().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=X[150:160], build=False,
        )
        assert art.sources[0].exists()   # sketch.ino
        assert art.sources[1].exists()   # rc_kernel.c
        assert art.binary is None
        assert art.metadata["fqbn"] == "arduino:avr:uno"


def test_arduino_cli_build_fits_uno():
    """Real AVR compile: must fit the Uno (Flash < 32K, SRAM < 2K)."""
    if not _HAVE_ARDUINO:
        return
    _, _, qm, X = _build(topology=Topology.SCR, storage_bits=8,
                          w_out_storage_bits=16,
                          strategy=LUTStrategy.linear_interp(64))
    with tempfile.TemporaryDirectory() as td:
        art = ArduinoUnoTarget().compile_affine_quantized(
            qm, output_dir=pathlib.Path(td), test_inputs=X[150:160], build=True,
        )
        md = art.metadata
        if "flash_bytes" in md:
            assert md["flash_bytes"] < 32768, f"flash {md['flash_bytes']} > 32K"
        if "sram_bytes" in md:
            assert md["sram_bytes"] < 2048, f"SRAM {md['sram_bytes']} > 2K"
        assert art.binary is not None and art.binary.exists()


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
