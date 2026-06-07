"""Stage 3: W_res sparse specialization in the bespoke C kernel templates.

The Arduino/NES affine kernel (`emit_affine_kernel_c`) and the C/Rust bundle
symmetric kernel (`emit_symmetric_kernel_c`) emit hand-written C. With
`sparse=` they emit a CSR W_res kernel (bounded code size — the right choice
for these Flash/SRAM-constrained targets; "unroll"/"auto" resolve to CSR
here). This compiles the dense and sparse C with host gcc and asserts the
runtime outputs are bit-identical (atol=0), for affine i8/i16 and symmetric.
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
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor
from rclite.quant import QuantConfig, TanhLUTSpec, quantize_model
from rclite.quant.affine import calibrate_from_data, quantize_model_affine
from rclite.targets.arduino import emit_affine_kernel_c
from rclite.export.c_kernel_symmetric import emit_symmetric_kernel_c


PASS = "\033[32m[PASS]\033[0m"
FAIL = "\033[31m[FAIL]\033[0m"
HAVE_GCC = shutil.which("gcc") is not None


def _model(units=40, density=0.15, seed=9):
    rc = ReservoirComputer(
        input=InputNode(units=1, name="in"),
        reservoir=ReservoirNode(
            units=units,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=density,
            seed=seed,
            name="res",
        ),
        readout=ReadoutNode(
            units=1,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=40,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    X = np.random.default_rng(seed).standard_normal((320, 1)) * 0.15
    exe.fit(X[:270], np.sin(np.arange(270) * 0.1)[:, None])
    return rc, exe, X


def _run_c(kernel_src, q_x, T, K, M, ctype):
    """Compile kernel + a tiny driver with host gcc, return the i64 outputs."""
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "kernel.c").write_text(kernel_src)
        xs = ", ".join(str(int(v)) for v in q_x)
        main = (
            "#include <stdint.h>\n#include <stdio.h>\n"
            f"extern void rc_predict(int32_t, const {ctype}*, {ctype}*);\n"
            "int main(void){\n"
            f"  {ctype} X[{T * K}] = {{ {xs} }};\n"
            f"  {ctype} Y[{T * M}];\n"
            f"  rc_predict({T}, X, Y);\n"
            f'  for (int i = 0; i < {T * M}; i++) printf("%d\\n", (int)Y[i]);\n'
            "  return 0;\n}\n"
        )
        (td / "main.c").write_text(main)
        exe_path = td / "a.out"
        r = subprocess.run(
            [
                "gcc",
                "-O2",
                "-std=c99",
                "-o",
                str(exe_path),
                str(td / "main.c"),
                str(td / "kernel.c"),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run(
            [str(exe_path)], capture_output=True, text=True
        ).stdout
        vals = [int(v) for v in out.strip().split("\n")]
        return np.array(vals, dtype=np.int64).reshape(T, M)


def test_affine_c_sparse_bit_exact():
    if not HAVE_GCC:
        print("  (skip: gcc not on PATH)")
        return
    for sb in (8, 16):
        rc, exe, X = _model()
        cfg = calibrate_from_data(rc, exe, X[:270], storage_bits=sb)
        qm = quantize_model_affine(rc, exe, cfg)
        Xe = X[270:295]
        T = Xe.shape[0]
        ctype = {8: "int8_t", 16: "int16_t"}[sb]
        q_x = cfg.input.quantize_array(Xe).astype(np.int64).reshape(-1)
        dense = _run_c(emit_affine_kernel_c(qm), q_x, T, qm.K, qm.M, ctype)
        for strat in ("csr", "auto", "unroll"):
            sp = _run_c(
                emit_affine_kernel_c(qm, sparse=strat),
                q_x,
                T,
                qm.K,
                qm.M,
                ctype,
            )
            d = int(np.max(np.abs(dense - sp)))
            assert d == 0, f"affine i{sb} C [{strat}] diff={d}"
    print("  affine C i8/i16 sparse(csr/auto/unroll) bit-exact vs dense")


def test_symmetric_c_sparse_bit_exact():
    if not HAVE_GCC:
        print("  (skip: gcc not on PATH)")
        return
    rc, exe, X = _model(units=40)
    qm = quantize_model(
        rc,
        exe,
        QuantConfig(state_frac=16, input_frac=12, weight_frac=12),
        lut=TanhLUTSpec(n=128),
    )
    Xe = X[270:295]
    T = Xe.shape[0]
    # symmetric i32 storage: input quantized at input_scale
    q_x = np.round(Xe * (1 << 12)).astype(np.int64).reshape(-1)
    dense = _run_c(emit_symmetric_kernel_c(qm), q_x, T, qm.K, qm.M, "int32_t")
    for strat in ("csr", "auto", "unroll"):
        sp = _run_c(
            emit_symmetric_kernel_c(qm, sparse=strat),
            q_x,
            T,
            qm.K,
            qm.M,
            "int32_t",
        )
        d = int(np.max(np.abs(dense - sp)))
        assert d == 0, f"symmetric C [{strat}] diff={d}"
    print("  symmetric C sparse(csr/auto/unroll) bit-exact vs dense")


def test_sparse_c_omits_dense_w_res():
    rc, exe, X = _model()
    cfg = calibrate_from_data(rc, exe, X[:270], storage_bits=8)
    qm = quantize_model_affine(rc, exe, cfg)
    src = emit_affine_kernel_c(qm, sparse="csr")
    assert "rc_W_res_val" in src and "rc_W_res_rowptr" in src
    assert "rc_W_res[" not in src  # dense matrix not emitted
    print("  sparse C emits CSR arrays, drops dense rc_W_res[]")


TESTS = [
    test_affine_c_sparse_bit_exact,
    test_symmetric_c_sparse_bit_exact,
    test_sparse_c_omits_dense_w_res,
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
