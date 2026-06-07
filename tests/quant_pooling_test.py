"""Quantized sequence-to-label pooling (aggregation MEAN / LAST) for the
affine integer path.

Asserts the three backends agree bit-for-bit on the pooled readout:

    AffineQuantizedExecutor   (Python integer reference, the ground truth)
    CompiledAffineRC          (LLVM JIT — host / wasm / cortex)
    emit_affine_kernel_c      (portable C — host gcc stands in for AVR)

for both heads (logits + argmax classify) over random-length sequences, plus
the float-vs-quant accuracy sanity check and the include_input guard.
"""

from __future__ import annotations
import pathlib
import shutil
import subprocess
import sys
import tempfile

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))

import numpy as np

from rclite import (
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Activation,
    Distribution,
    Topology,
    Trainer,
    Task,
    Aggregation,
)
from rclite.runtime import RCExecutor
from rclite.quant import (
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
    LUTStrategy,
)
from rclite.quant.affine.executor import _saturate
from rclite.codegen.llvm import CompiledAffineRC
from rclite.targets.arduino import emit_affine_kernel_c


_HAVE_GCC = shutil.which("gcc") is not None


def _build(
    agg, *, topo=Topology.SCR, units=32, sb=8, wob=16, M=4, washout=8, seed=0
):
    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_distribution=Distribution.BERNOULLI,
            name="in",
        ),
        reservoir=ReservoirNode(
            units=units,
            activation=Activation.TANH,
            topology=topo,
            chain_weight=0.9,
            chain_feedback=0.1,
            leak_rate=0.3,
            seed=7,
        ),
        readout=ReadoutNode(
            units=M,
            activation=Activation.IDENTITY,
            trainer=Trainer.RIDGE,
            regularization=1e-2,
            washout=washout,
            include_bias=True,
            task=Task.CLASSIFICATION,
            aggregation=agg,
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(seed)
    seqs = [rng.standard_normal((25, 1)) * 0.5 for _ in range(40)]
    labels = rng.integers(0, M, size=40)
    exe.fit_sequences(seqs, labels)
    cfg = calibrate_from_data(
        rc, exe, np.concatenate(seqs), storage_bits=sb, w_out_storage_bits=wob
    )
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=LUTStrategy.direct())
    return rc, exe, qm


def _host_c_out(qm, X, head, allow_i32):
    """Compile the emitted pooled C kernel with host gcc; return its raw out."""
    ctype = "int8_t" if qm.storage_bits == 8 else "int16_t"
    q_x = qm.config.input.quantize_array(X).astype(np.int64).reshape(-1)
    T = X.shape[0]
    M = qm.M
    yt = "int32_t" if head == "classify" else ctype
    ylen = 1 if head == "classify" else M  # pooled → one row
    with tempfile.TemporaryDirectory() as td:
        td = pathlib.Path(td)
        (td / "k.c").write_text(
            emit_affine_kernel_c(qm, head=head, allow_i32_accum=allow_i32)
        )
        (td / "m.c").write_text(
            "\n".join(
                [
                    "#include <stdint.h>",
                    "#include <stdio.h>",
                    f"extern void rc_predict(int32_t, const {ctype}*, {yt}*);",
                    "int main(void){",
                    f"  {ctype} X[{T}] = {{ {', '.join(str(int(v)) for v in q_x)} }};",
                    f"  {yt} Y[{ylen}];  rc_predict({T}, X, Y);",
                    f'  for(int i=0;i<{ylen};i++) printf("%d\\n",(int)Y[i]); return 0; }}',
                ]
            )
        )
        r = subprocess.run(
            [
                "gcc",
                "-O2",
                "-std=c99",
                "-o",
                str(td / "a.out"),
                str(td / "m.c"),
                str(td / "k.c"),
            ],
            capture_output=True,
            text=True,
        )
        if r.returncode != 0:
            raise RuntimeError("gcc failed:\n" + r.stderr)
        out = subprocess.run(
            [str(td / "a.out")], capture_output=True, text=True
        ).stdout
        return np.array(
            [int(v) for v in out.strip().split("\n")], dtype=np.int64
        )


def _check_parity(agg):
    rc, exe, qm = _build(agg)
    qexe = AffineQuantizedExecutor(qm)
    jit_l = CompiledAffineRC(qm, head=None)
    jit_c = CompiledAffineRC(qm, head="classify")
    rng = np.random.default_rng(11)
    for _ in range(12):
        T = int(rng.integers(12, 40))
        X = rng.standard_normal((T, 1)) * 0.5

        py_q = qexe.predict_pooled_q(X)  # (M,) int logits
        py_deq = qexe.predict_pooled(X)  # (M,) dequantized
        py_cls = int(np.argmax(py_q))

        # LLVM JIT: dequantized logits (1, M) and argmax (1,)
        jl = jit_l.predict(X)
        assert jl.shape == (1, qm.M)
        assert np.max(np.abs(jl[0] - py_deq)) < 1e-9
        assert int(jit_c.predict(X)[0]) == py_cls

        if _HAVE_GCC:
            py_qsat = _saturate(py_q, qm.storage_bits)
            for allow_i32 in (False, True):
                cl = _host_c_out(qm, X, None, allow_i32)
                assert np.array_equal(cl, py_qsat), (
                    f"{agg.name} C logits mismatch (i32={allow_i32})"
                )
            assert int(_host_c_out(qm, X, "classify", False)[0]) == py_cls


def test_pooling_mean_parity():
    _check_parity(Aggregation.MEAN)


def test_pooling_last_parity():
    _check_parity(Aggregation.LAST)


def test_pooling_quant_accuracy_reasonable():
    """Quantized pooled classification should track the float classifier."""
    rc, exe, qm = _build(Aggregation.MEAN, units=40)
    qexe = AffineQuantizedExecutor(qm)
    rng = np.random.default_rng(5)
    seqs = [rng.standard_normal((25, 1)) * 0.5 for _ in range(40)]
    agree = 0
    for X in seqs:
        f = int(
            np.argmax(exe.predict_sequences([X])[:1])
            if False
            else exe.predict_sequences([X])[0]
        )
        q = int(np.argmax(qexe.predict_pooled_q(X)))
        agree += f == q
    # i8 reservoir + i16 W_out + direct LUT should match the float labels
    # on almost every sequence.
    assert agree >= int(0.9 * len(seqs)), f"only {agree}/{len(seqs)} agree"


def test_pooling_rejects_include_input():
    import pytest

    rc = ReservoirComputer(
        input=InputNode(
            units=1,
            activation=Activation.IDENTITY,
            input_distribution=Distribution.BERNOULLI,
        ),
        reservoir=ReservoirNode(
            units=16,
            topology=Topology.SCR,
            chain_weight=0.9,
            leak_rate=0.3,
            seed=1,
        ),
        readout=ReadoutNode(
            units=3,
            trainer=Trainer.RIDGE,
            regularization=1e-2,
            washout=5,
            include_bias=True,
            include_input=True,
            task=Task.CLASSIFICATION,
            aggregation=Aggregation.MEAN,
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(0)
    seqs = [rng.standard_normal((20, 1)) * 0.5 for _ in range(12)]
    exe.fit_sequences(seqs, rng.integers(0, 3, size=12))
    cfg = calibrate_from_data(
        rc, exe, np.concatenate(seqs), storage_bits=8, w_out_storage_bits=16
    )
    qm = quantize_model_affine(rc, exe, cfg, lut_strategy=LUTStrategy.direct())
    with pytest.raises(NotImplementedError):
        CompiledAffineRC(qm, head=None)
    with pytest.raises(NotImplementedError):
        emit_affine_kernel_c(qm)


if __name__ == "__main__":
    test_pooling_mean_parity()
    test_pooling_last_parity()
    test_pooling_quant_accuracy_reasonable()
    test_pooling_rejects_include_input()
    print("ok")
