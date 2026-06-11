#!/usr/bin/env python3
"""Benchmark: hand-written C ESN vs rclite's generated library.

Same trained ESN, three host (x86-64) contenders, identical T-step workload:

  1. manual-float   a from-scratch C ESN (the natural float formulation a
                    developer writes by hand) — gcc autovectorizes the float
                    matvec.  Built at -O0 / -O2 / -O3 -march=native -ffast-math.
  2. rclite-ctmpl   rclite's PORTABLE scalar-int kernel (`export_bundle` ->
                    rc_kernel.c) — pure C99, optimized only by the C compiler
                    (gcc -O3 -march=native).  Bit-exact with #3.
  3. rclite-objlib  rclite's OPTIMIZED library (`export_optimized_object`) — the
                    int8 kernel compiled HERE through MLIR/LLVM with AVX2 SIMD,
                    shipped as rc_kernel.o + rc_kernel.h.  Linked from a -O0 C
                    driver (the kernel's speed lives in the .o, not the caller).

#2 and #3 compute byte-identical output, so #2-vs-#3 is a pure *codegen* head-to-
head (portable C + gcc -O3  vs  rclite MLIR/LLVM SIMD).  #1 is the realistic
"should I just hand-roll it?" baseline.  All three reset state per call and run
the same T-length sequence; we report min-over-repeats ns/step.

Run:  python benchmarks/host/esn_manual_vs_rclite.py
Needs gcc + the MLIR toolchain (mlir-opt/mlir-translate/llc) + xdsl on PATH.
"""

from __future__ import annotations

import pathlib
import shutil
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))

from rclite import (  # noqa: E402
    InputNode,
    ReservoirNode,
    ReadoutNode,
    ReservoirComputer,
    Topology,
    Trainer,
)
from rclite.runtime import RCExecutor  # noqa: E402
from rclite.quant.affine import (  # noqa: E402
    calibrate_from_data,
    quantize_model_affine,
    AffineQuantizedExecutor,
)
from rclite.export import (  # noqa: E402
    export_optimized_object,
    export_bundle,
)

# ---- benchmark size -------------------------------------------------------
N = 256  # reservoir units  (the N x N matvec is the hot loop)
K = 3  # input dim
M = 4  # output dim
T = 2000  # sequence length processed per kernel call
REPS = 25  # timing repeats (we take the min)
WARM = 3
CC = "gcc"


def _carray(name, values, ctype):
    body = ",".join(
        repr(float(v))
        if "float" in ctype or "double" in ctype
        else str(int(v))
        for v in values
    )
    return f"static const {ctype} {name}[] = {{{body}}};\n"


def build_model():
    rc = ReservoirComputer(
        input=InputNode(
            units=K, input_offset=0.0, input_scaling=1.0, name="in"
        ),
        reservoir=ReservoirNode(
            units=N,
            topology=Topology.ESN_STANDARD,
            leak_rate=0.3,
            density=0.3,
            seed=7,
            spectral_radius=0.9,
            name="res",
        ),
        readout=ReadoutNode(
            units=M,
            trainer=Trainer.RIDGE,
            regularization=1e-3,
            washout=50,
            include_bias=True,
            include_input=True,
            name="out",
        ),
    )
    exe = RCExecutor(rc)

    def signal(t):
        # smooth, bounded, deterministic multi-sine input (per channel)
        return np.stack(
            [
                0.6 * np.sin(0.05 * (k + 1) * t)
                + 0.3 * np.sin(0.017 * (k + 2) * t + k)
                for k in range(K)
            ],
            axis=1,
        )

    # learnable delay task: output m = input channel (m%K) delayed by (m+1).
    # The reservoir's echo memory fits this and GENERALIZES, so test outputs
    # stay in range -> i8 quantization is accurate (not extrapolating wildly).
    n_tr = 1500
    t_tr = np.arange(n_tr)
    Xtr = signal(t_tr)
    Ytr = np.zeros((n_tr, M))
    for m in range(M):
        d = m + 1
        Ytr[d:, m] = Xtr[:-d, m % K]
    exe.fit(Xtr, Ytr)
    Xte = np.ascontiguousarray(signal(np.arange(n_tr, n_tr + T)))
    return rc, exe, Xtr, Xte


# ---- 1. hand-written float ESN (emitted C) --------------------------------
def emit_manual_float_c(rc, exe, *, fn="run"):
    leak = float(rc.reservoir.leak_rate)
    bias = float(rc.reservoir.bias)
    off = float(rc.input.input_offset)
    scl = float(rc.input.input_scaling)
    inc_b = int(rc.readout.include_bias)
    inc_i = int(rc.readout.include_input)
    W_in = np.asarray(exe.W_in, float)  # (N,K)
    W_res = np.asarray(exe.W_res, float)  # (N,N)
    W_out = np.asarray(exe.W_out, float)  # (M,F)
    F = W_out.shape[1]
    src = []
    src.append("#include <math.h>\n#include <stdint.h>\n")
    src.append(f"#define RC_N {N}\n#define RC_K {K}\n#define RC_M {M}\n")
    src.append(f"#define RC_F {F}\n")
    src.append(f"#define RC_LEAK {leak!r}f\n#define RC_BIAS {bias!r}f\n")
    src.append(f"#define RC_OFF {off!r}f\n#define RC_SCL {scl!r}f\n")
    src.append(_carray("W_in", W_in.reshape(-1), "float"))
    src.append(_carray("W_res", W_res.reshape(-1), "float"))
    src.append(_carray("W_out", W_out.reshape(-1), "float"))
    src.append(f"""
void {fn}(int64_t T, const float *X, float *Y) {{
  static float h[RC_N], hn[RC_N], xp[RC_K], phi[RC_F];
  for (int i = 0; i < RC_N; i++) h[i] = 0.0f;
  for (int64_t t = 0; t < T; t++) {{
    const float *u = X + t * RC_K;
    for (int k = 0; k < RC_K; k++) xp[k] = (u[k] - RC_OFF) * RC_SCL;
    for (int i = 0; i < RC_N; i++) {{
      float pre = RC_BIAS;
      const float *wr = W_res + (long)i * RC_N;
      for (int j = 0; j < RC_N; j++) pre += wr[j] * h[j];
      const float *wi = W_in + (long)i * RC_K;
      for (int k = 0; k < RC_K; k++) pre += wi[k] * xp[k];
      hn[i] = (1.0f - RC_LEAK) * h[i] + RC_LEAK * tanhf(pre);
    }}
    for (int i = 0; i < RC_N; i++) h[i] = hn[i];
    int f = 0;
    if ({inc_b}) phi[f++] = 1.0f;
    if ({inc_i}) for (int k = 0; k < RC_K; k++) phi[f++] = u[k];
    for (int i = 0; i < RC_N; i++) phi[f++] = h[i];
    float *y = Y + t * RC_M;
    for (int m = 0; m < RC_M; m++) {{
      float acc = 0.0f;
      const float *wo = W_out + (long)m * RC_F;
      for (int g = 0; g < RC_F; g++) acc += wo[g] * phi[g];
      y[m] = acc;
    }}
  }}
}}
""")
    return "".join(src)


# ---- timing harness + driver ----------------------------------------------
def main():
    need = ["mlir-opt", "mlir-translate", "llc", CC]
    missing = [t for t in need if shutil.which(t) is None]
    if missing:
        print(f"  (skip: need {missing} on PATH)")
        return
    rc, exe, Xtr, Xt = build_model()

    # f64 reference outputs (ground truth for accuracy)
    Yref = exe.predict(Xt)  # (T, M) float64

    # quantize (i8) — calibrate on the training data — + the int8 kernel input
    qm = quantize_model_affine(
        rc, exe, calibrate_from_data(rc, exe, Xtr, storage_bits=8)
    )
    Xq = np.ascontiguousarray(
        qm.config.input.quantize_array(Xt), dtype=np.int8
    )

    # rclite int8 reference (dequantized) for accuracy vs Yref
    qe = AffineQuantizedExecutor(qm)
    qe.reset()
    yq = np.zeros((T, M), dtype=np.int64)
    for t in range(T):
        xr = qe._quantize_raw_input(Xt[t])
        qe.step_q(qe._quantize_u_pre(Xt[t]))
        yq[t] = qe.predict_one_q(xr, qe.state_q)
    y_deq = (yq - qm.config.output.zero_point) * qm.config.output.scale

    work = pathlib.Path(tempfile.mkdtemp(prefix="esn_bench_"))
    print(f"  model: ESN N={N} K={K} M={M}, T={T} steps, i8 quant")
    print(
        f"  accuracy vs f64 reference:  rclite-int8 max|err|="
        f"{np.max(np.abs(y_deq - Yref)):.4g}"
    )

    # ---- data headers ----
    xq_csv = ",".join(str(int(v)) for v in Xq.reshape(-1))
    data_int = (
        f"#define RC_K {K}\n#define RC_M {M}\n"
        f"static const int8_t X[{T * K}] = {{{xq_csv}}};\n"
    )
    xf_csv = ",".join(repr(float(v)) for v in Xt.reshape(-1))
    data_float = (
        f"#define RC_K {K}\n#define RC_M {M}\n"
        f"static const float X[{T * K}] = {{{xf_csv}}};\n"
    )

    results = []

    # ---- 3. rclite optimized object library ----
    bundle = export_optimized_object(
        qm, target="x86_64-avx2", name="rc_kernel"
    )
    (work / "rc_kernel.o").write_bytes(bundle.object_code)
    (work / "rc_kernel.h").write_text(bundle.header)
    # adapter so harness `run(T,X,Y)` -> rc_run
    objlib_c = (
        '#include <stdint.h>\n#include "rc_kernel.h"\n'
        "void run(int64_t T, const int8_t *X, int8_t *Y){"
        " rc_run(T, X, Y); }\n"
    )
    ns, chk_obj = _run_int_variant(
        work,
        "rclite-objlib",
        {"k.c": objlib_c, "rc_kernel.h": bundle.header},
        data_int,
        ["-O0", "-no-pie"],
        extra_objs=[work / "rc_kernel.o"],
        inc=[work],
    )
    results.append(
        ("rclite-objlib (MLIR/LLVM AVX2 .o, -O0 driver)", ns, chk_obj)
    )

    # ---- 2. rclite portable scalar-int C kernel (export_bundle) ----
    bdir = work / "bundle"
    export_bundle(qm, bdir)
    ctmpl = (bdir / "rc_kernel.c").read_text()
    # the template entry is rc_predict(int32_t T, const int8_t*, int8_t*)
    tmpl_adapter = (
        "#include <stdint.h>\n"
        "void rc_predict(int32_t, const int8_t*, int8_t*);\n"
        "void run(int64_t T, const int8_t *X, int8_t *Y){"
        " rc_predict((int32_t)T, X, Y); }\n"
    )
    for label, flags in [
        ("-O3 -march=native", ["-O3", "-march=native"]),
        ("-O2", ["-O2"]),
    ]:
        ns, chk = _run_int_variant(
            work,
            f"rclite-ctmpl {label}",
            {"rc_kernel.c": ctmpl, "adapter.c": tmpl_adapter},
            data_int,
            flags,
        )
        results.append((f"rclite-ctmpl (portable C, gcc {label})", ns, chk))
        assert chk == chk_obj, "int kernels must be bit-exact"

    # ---- 1. hand-written float ESN ----
    manual_c = emit_manual_float_c(rc, exe)
    chk_f_ref = None
    for label, flags in [
        (
            "-O3 -march=native -ffast-math",
            ["-O3", "-march=native", "-ffast-math", "-funroll-loops"],
        ),
        ("-O2", ["-O2"]),
        ("-O0", ["-O0"]),
    ]:
        ns, chk = _run_float_variant(
            work,
            f"manual-float {label}",
            {"esn.c": manual_c},
            data_float,
            flags,
        )
        results.append(
            (f"manual-float (hand-written C, gcc {label})", ns, chk)
        )
        if chk_f_ref is None:
            chk_f_ref = chk

    # accuracy of the hand-written float kernel: re-run -O0 build capturing Y
    acc = _manual_accuracy(work, manual_c, data_float, Yref)
    print(f"  accuracy vs f64 reference:  manual-float max|err|={acc:.4g}\n")

    # ---- report ----
    base = next(
        ns
        for nm, ns, _ in results
        if nm.startswith("manual-float (hand-written C, gcc -O3")
    )
    objns = next(ns for nm, ns, _ in results if nm.startswith("rclite-objlib"))
    ctmpl3 = next(
        ns
        for nm, ns, _ in results
        if nm.startswith("rclite-ctmpl (portable C, gcc -O3")
    )
    w = max(len(nm) for nm, _, _ in results)
    print(f"  {'variant':<{w}}  {'ns/step':>9}  {'steps/s':>10}  speedup")
    print(f"  {'-' * w}  {'-' * 9}  {'-' * 10}  -------")
    for nm, ns, _ in results:
        sp = base / ns
        print(f"  {nm:<{w}}  {ns:9.1f}  {1e9 / ns:10.3g}  {sp:6.2f}x")
    print()
    print("  head-to-head (same int8 kernel, bit-exact):")
    print(
        f"    rclite-objlib (MLIR/LLVM SIMD)  vs  rclite-ctmpl (gcc -O3): "
        f"{ctmpl3 / objns:.2f}x faster"
    )
    print(
        f"    rclite-objlib                   vs  manual-float (gcc -O3): "
        f"{base / objns:.2f}x"
    )
    print(
        f"\n  (speedup column is relative to hand-written float @ -O3; "
        f"work dir {work})"
    )


def _run_int_variant(
    work, name, sources, data_int, flags, *, extra_objs=(), inc=()
):
    d = work / name.replace(" ", "_")
    d.mkdir(parents=True, exist_ok=True)
    for fn, txt in sources.items():
        (d / fn).write_text(txt)
    (d / "data.h").write_text(data_int)
    harness = _int_harness("int8_t")
    (d / "harness.c").write_text(harness)
    csrc = [str(d / fn) for fn in sources if fn.endswith(".c")]
    incs = []
    for p in (d, *inc):
        incs += ["-I", str(p)]
    cmd = [
        CC,
        *flags,
        str(d / "harness.c"),
        *csrc,
        *map(str, extra_objs),
        *incs,
        "-o",
        str(d / "app"),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name} compile:\n{r.stderr[:2000]}")
    out = subprocess.run([str(d / "app")], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{name} run:\n{out.stderr[:1000]}")
    chk, ns = out.stdout.split()
    return float(ns) / T, float(chk)


def _run_float_variant(work, name, sources, data_float, flags):
    d = work / name.replace(" ", "_").replace("=", "")
    d.mkdir(parents=True, exist_ok=True)
    for fn, txt in sources.items():
        (d / fn).write_text(txt)
    (d / "data.h").write_text(data_float)
    (d / "harness.c").write_text(_float_harness())
    csrc = [str(d / fn) for fn in sources if fn.endswith(".c")]
    cmd = [
        CC,
        *flags,
        str(d / "harness.c"),
        *csrc,
        "-I",
        str(d),
        "-o",
        str(d / "app"),
        "-lm",
    ]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode != 0:
        raise RuntimeError(f"{name} compile:\n{r.stderr[:2000]}")
    out = subprocess.run([str(d / "app")], capture_output=True, text=True)
    if out.returncode != 0:
        raise RuntimeError(f"{name} run:\n{out.stderr[:1000]}")
    chk, ns = out.stdout.split()
    return float(ns) / T, float(chk)


def _int_harness(out_t):
    return f"""
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include "data.h"
extern void run(int64_t, const int8_t*, {out_t}*);
int main(void) {{
  static {out_t} Y[{T} * RC_M];
  for (int w = 0; w < {WARM}; w++) run({T}, X, Y);
  long best = LONG_MAX; struct timespec a, b;
  for (int r = 0; r < {REPS}; r++) {{
    clock_gettime(CLOCK_MONOTONIC, &a);
    run({T}, X, Y);
    clock_gettime(CLOCK_MONOTONIC, &b);
    long ns = (b.tv_sec-a.tv_sec)*1000000000L + (b.tv_nsec-a.tv_nsec);
    if (ns < best) best = ns;
  }}
  long long chk = 0;
  for (int i = 0; i < {T} * RC_M; i++) chk += (long long)Y[i];
  printf("%lld %ld\\n", chk, best);
  return 0;
}}
"""


def _float_harness():
    return f"""
#include <stdint.h>
#include <stdio.h>
#include <time.h>
#include <limits.h>
#include "data.h"
extern void run(int64_t, const float*, float*);
int main(void) {{
  static float Y[{T} * RC_M];
  for (int w = 0; w < {WARM}; w++) run({T}, X, Y);
  long best = LONG_MAX; struct timespec a, b;
  for (int r = 0; r < {REPS}; r++) {{
    clock_gettime(CLOCK_MONOTONIC, &a);
    run({T}, X, Y);
    clock_gettime(CLOCK_MONOTONIC, &b);
    long ns = (b.tv_sec-a.tv_sec)*1000000000L + (b.tv_nsec-a.tv_nsec);
    if (ns < best) best = ns;
  }}
  double chk = 0;
  for (int i = 0; i < {T} * RC_M; i++) chk += (double)Y[i];
  printf("%.6g %ld\\n", chk, best);
  return 0;
}}
"""


def _manual_accuracy(work, manual_c, data_float, Yref):
    """Compile the manual kernel to dump Y, compare to the f64 reference."""
    d = work / "manual_acc"
    d.mkdir(parents=True, exist_ok=True)
    (d / "esn.c").write_text(manual_c)
    (d / "data.h").write_text(data_float)
    (d / "dump.c").write_text(f"""
#include <stdint.h>
#include <stdio.h>
#include "data.h"
extern void run(int64_t, const float*, float*);
int main(void) {{
  static float Y[{T} * RC_M];
  run({T}, X, Y);
  for (int i = 0; i < {T} * RC_M; i++) printf("%.9g\\n", Y[i]);
  return 0;
}}
""")
    cmd = [
        CC,
        "-O2",
        str(d / "dump.c"),
        str(d / "esn.c"),
        "-I",
        str(d),
        "-o",
        str(d / "app"),
        "-lm",
    ]
    subprocess.run(cmd, capture_output=True, text=True, check=True)
    out = subprocess.run([str(d / "app")], capture_output=True, text=True)
    got = np.array([float(x) for x in out.stdout.split()]).reshape(T, M)
    return float(np.max(np.abs(got - Yref)))


if __name__ == "__main__":
    main()
