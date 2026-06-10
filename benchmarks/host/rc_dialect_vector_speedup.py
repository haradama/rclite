"""Benchmark: Stage-3 `vector`-dialect float kernel vs scalar baseline (host JIT).

The `rc` dialect lowers the fused float reservoir to an arith/memref/scf kernel
(`rclite.codegen.rc_dialect_xdsl.lower_fused_float`). With `vlen>1` the two
N-wide reductions (`W_res@h` and the readout `W_out_state@h`) are emitted in the
`vector` dialect: a `vector<vlen x f64>` FMA accumulator collapsed by
`vector.reduction <add>` (plus a scalar tail).

Why this beats the scalar baseline: vectorising a float reduction *reassociates*
the sum, which LLVM's auto-vectoriser refuses to do without fast-math — so the
scalar kernel's reductions stay scalar even at -O3. The explicit `vector` lowering
forces the SIMD partial sums. Both kernels go through the *same* mlir_jit ->
LLVM -O3 pipeline; the only difference is the explicit vectorisation, so the
speedup is attributable to it.

Each kernel is measured in a *fresh subprocess* (one MCJIT engine at a time):
two engines in one process would both export `_mlir_ciface_rc_predict` and
collide. Outputs differ from the scalar kernel only by float reassociation
(~1e-13 rel), reported as a correctness check.

Needs an LLVM-20 mlir-opt on PATH (the nix devShell, or a system llvm-20).
Usage:
    python benchmarks/host/rc_dialect_vector_speedup.py
"""

from __future__ import annotations
import ctypes
import json
import pathlib
import statistics
import subprocess
import sys
import tempfile
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

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
from rclite.codegen.rc_dialect_xdsl import (
    build_rc_module,
    fuse_step_readout,
    lower_fused_float,
)

K, M, T = 4, 8, 1500
SIZES = (64, 128, 256, 512, 1024)
VLEN = 4


def _kernel(N, vlen):
    rc = ReservoirComputer(
        input=InputNode(units=K, name="in"),
        reservoir=ReservoirNode(
            units=N,
            topology=Topology.ESN_STANDARD,
            spectral_radius=0.9,
            leak_rate=0.3,
            density=1.0,
            seed=5,
            activation=Activation.IDENTITY,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-6,
            washout=50,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)
    rng = np.random.default_rng(5)
    exe.fit(
        rng.standard_normal((400, K)) * 0.1,
        np.stack(
            [np.sin(np.arange(400) * 0.03 * (i + 1)) for i in range(M)], 1
        ),
    )
    m = build_ir(rc, exe)
    mod = build_rc_module(m)
    fuse_step_readout(mod)
    import llvmlite.binding as llvm

    mlir_jit._ensure_llvm()
    extra = ["--convert-vector-to-llvm"] if vlen > 1 else []
    text = lower_fused_float(mod, m.weights, vlen=vlen)
    md = llvm.parse_assembly(
        mlir_jit.mlir_to_llvm_ir(text, extra_passes=extra)
    )
    md.verify()
    tm = llvm.Target.from_triple(
        llvm.get_default_triple()
    ).create_target_machine(opt=3)
    eng = llvm.create_mcjit_compiler(md, tm)
    eng.finalize_object()
    eng.run_static_constructors()
    fn = ctypes.CFUNCTYPE(
        None,
        ctypes.c_int64,
        ctypes.POINTER(mlir_jit._MemRef1D),
        ctypes.POINTER(mlir_jit._MemRef1D),
    )(eng.get_function_address("_mlir_ciface_rc_predict"))
    return eng, fn


def _run(fn, X):
    Xt = np.ascontiguousarray(X, dtype=np.float64).reshape(-1)
    Tn = X.shape[0]
    Y = np.zeros(Tn * M, dtype=np.float64)
    dx, dy = mlir_jit._desc(Xt), mlir_jit._desc(Y)
    fn(ctypes.c_int64(Tn), ctypes.byref(dx), ctypes.byref(dy))
    return Y.reshape(Tn, M)


def _worker(N, vlen, dump_path):
    """One MCJIT engine, isolated: median predict time + dumped output."""
    _eng, fn = _kernel(N, vlen)
    X = np.random.default_rng(0).standard_normal((T, K)) * 0.1
    y = _run(fn, X)  # warm + capture for correctness
    ts = []
    for _ in range(9):
        t0 = time.perf_counter()
        _run(fn, X)
        ts.append(time.perf_counter() - t0)
    np.save(dump_path, y)
    print(json.dumps({"ms": statistics.median(ts) * 1e3}))


def _spawn(N, vlen):
    with tempfile.NamedTemporaryFile(suffix=".npy", delete=False) as f:
        dump = f.name
    r = subprocess.run(
        [sys.executable, __file__, "_worker", str(N), str(vlen), dump],
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(
            f"worker N={N} vlen={vlen} failed:\n{r.stderr[-800:]}"
        )
    ms = json.loads(r.stdout.strip().splitlines()[-1])["ms"]
    y = np.load(dump)
    pathlib.Path(dump).unlink(missing_ok=True)
    return ms, y


def main():
    if not mlir_jit.tools_available():
        print("skip: LLVM-20 mlir-opt not on PATH (use the nix devShell)")
        return
    print(
        f"Stage-3 vector vs scalar float kernel — host LLVM JIT "
        f"(dense, K={K}, M={M}, T={T}, vlen={VLEN})\n"
    )
    header = (
        f"{'N':>5} {'scalar ms':>10} {'vector ms':>10} "
        f"{'speedup':>8} {'max|diff|':>11}"
    )
    print(header)
    print("-" * len(header))
    for N in SIZES:
        t_s, y_s = _spawn(N, 1)
        t_v, y_v = _spawn(N, VLEN)
        diff = float(np.max(np.abs(y_s - y_v)))
        print(
            f"{N:>5} {t_s:>10.3f} {t_v:>10.3f} "
            f"{t_s / t_v:>7.2f}x {diff:>11.2e}"
        )
    print(
        "\nspeedup = scalar / vector host wall-clock (same -O3 pipeline). "
        "max|diff| = float reassociation only (vector reorders the reduction;\n"
        "LLVM won't without fast-math, which is exactly why the scalar baseline "
        "stays un-vectorised and the vector kernel wins)."
    )


if __name__ == "__main__":
    if len(sys.argv) >= 2 and sys.argv[1] == "_worker":
        _worker(int(sys.argv[2]), int(sys.argv[3]), sys.argv[4])
    else:
        main()
