"""AOT shared-library tests: emit .so, dlopen it, verify parity with Python."""
from __future__ import annotations
import ctypes
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
from rclite.codegen import compile_rc


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"


def expect_raises(exc_type, fn, *args, **kwargs):
    try:
        fn(*args, **kwargs)
    except exc_type:
        return True
    raise AssertionError(f"Expected {exc_type.__name__}, none raised")


def _build_and_train(topology: Topology = Topology.ESN_STANDARD, units: int = 80):
    rc = ReservoirComputer(
        input=InputNode(units=1, input_scaling=1.0, input_offset=0.5, name="in"),
        reservoir=ReservoirNode(units=units, activation=Activation.TANH,
                                topology=topology, spectral_radius=0.9,
                                chain_weight=0.85, leak_rate=0.3,
                                density=0.2, seed=7, name="res"),
        readout=ReadoutNode(units=1, trainer=Trainer.RIDGE,
                            regularization=1e-6, washout=100,
                            include_bias=True, include_input=True, name="out"),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    X = rng.standard_normal((400, 1)) * 0.3 + 0.5
    Y = np.sin(np.arange(400) * 0.1)[:, None]
    exe.fit(X, Y)
    return rc, exe, X[300:350]


def _dlopen_predict(lib_path: str):
    lib = ctypes.CDLL(lib_path)
    lib.rc_predict.argtypes = [
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.POINTER(ctypes.c_double),
    ]
    lib.rc_predict.restype = None
    return lib


def test_emit_object_creates_file():
    rc, exe, _ = _build_and_train()
    jit = compile_rc(rc, exe)
    with tempfile.NamedTemporaryFile(suffix=".o", delete=False) as f:
        path = f.name
    try:
        jit.emit_object(path)
        size = pathlib.Path(path).stat().st_size
        assert size > 0, "object file is empty"
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def test_emit_header_includes_expected_symbols():
    rc, exe, _ = _build_and_train(topology=Topology.SCR, units=50)
    jit = compile_rc(rc, exe)
    with tempfile.NamedTemporaryFile(suffix=".h", delete=False, mode="w") as f:
        path = f.name
    try:
        jit.emit_header(path)
        text = pathlib.Path(path).read_text()
        assert "void rc_predict(int64_t T, double *X, double *Y);" in text
        assert "RC_INPUT_DIM  1" in text
        assert "RC_RES_UNITS  50" in text
        assert "topology         = SCR" in text
    finally:
        pathlib.Path(path).unlink(missing_ok=True)


def _parity_via_so(topology: Topology, units: int):
    if shutil.which("gcc") is None:
        return  # silently skip if no gcc
    rc, exe, sample = _build_and_train(topology=topology, units=units)
    jit = compile_rc(rc, exe)
    Y_jit = jit.predict(sample)

    with tempfile.TemporaryDirectory() as td:
        lib = pathlib.Path(td) / "librc_test.so"
        jit.emit_shared_library(str(lib))
        dyn = _dlopen_predict(str(lib))

        X = np.ascontiguousarray(sample, dtype=np.float64)
        Y_so = np.zeros_like(Y_jit)
        dyn.rc_predict(
            X.shape[0],
            X.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
            Y_so.ctypes.data_as(ctypes.POINTER(ctypes.c_double)),
        )
        # Free the handle before deleting tempdir (linux)
        del dyn
    diff = float(np.max(np.abs(Y_so - Y_jit)))
    assert diff < 1e-12, f"AOT .so diverges from JIT: max |diff| = {diff}"


def test_aot_parity_random():
    _parity_via_so(Topology.ESN_STANDARD, 80)


def test_aot_parity_scr():
    _parity_via_so(Topology.SCR, 80)


def test_aot_parity_dlr():
    _parity_via_so(Topology.DLR, 80)


def test_aot_parity_dlrb():
    _parity_via_so(Topology.DLRB, 80)


def test_aot_link_failure_surfaces_error():
    rc, exe, _ = _build_and_train()
    jit = compile_rc(rc, exe)
    expect_raises(RuntimeError, jit.emit_shared_library,
                  "/nonexistent/dir/that/does/not/exist/librc.so")


def test_built_c_demo_runs_and_matches_python():
    if shutil.which("gcc") is None:
        return
    rc, exe, sample = _build_and_train(topology=Topology.SCR, units=40)
    jit = compile_rc(rc, exe)
    Y_ref = jit.predict(sample)

    with tempfile.TemporaryDirectory() as td:
        td_path = pathlib.Path(td)
        lib = td_path / "librc.so"
        hdr = td_path / "rc_predict.h"
        src = td_path / "demo.c"
        bin = td_path / "demo"
        jit.emit_shared_library(str(lib))
        jit.emit_header(str(hdr))

        flat = np.ascontiguousarray(sample, dtype=np.float64).ravel()
        x_literals = ", ".join(f"{v:.17g}" for v in flat)
        T = sample.shape[0]
        src.write_text(
            "#include <stdio.h>\n"
            "#include <stdint.h>\n"
            "#include \"rc_predict.h\"\n"
            f"int main(void) {{\n"
            f"  double X[{T}] = {{ {x_literals} }};\n"
            f"  double Y[{T}] = {{0}};\n"
            f"  rc_predict((int64_t){T}, X, Y);\n"
            f"  for (int t = 0; t < {T}; t++) printf(\"%.17g\\n\", Y[t]);\n"
            f"  return 0;\n"
            f"}}\n"
        )

        cmd = ["gcc", "-O2", "-o", str(bin), str(src),
               "-L", str(td_path), "-lrc",
               f"-Wl,-rpath,{td_path}", "-lm",
               f"-I{td_path}"]
        cp = subprocess.run(cmd, capture_output=True, text=True)
        if cp.returncode != 0:
            raise AssertionError(f"gcc failed: {cp.stderr}")
        cp = subprocess.run([str(bin)], capture_output=True, text=True, check=True)
        Y_c = np.array([float(line) for line in cp.stdout.strip().split("\n")])
        diff = float(np.max(np.abs(Y_c - Y_ref.ravel())))
        assert diff < 1e-12, f"C demo diverges: max |diff| = {diff}"


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
