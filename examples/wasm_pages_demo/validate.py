#!/usr/bin/env python3
"""End-to-end check of the built browser demo, without a browser.

Instantiates the generated reactor modules exactly as the JS loader does
(``env.tanhf`` -> ``math.tanh``, inputs written at ``__heap_base``, output read
back), runs ``rc_predict``, and compares against the in-process host model.
Also exercises the autoregressive "dream" feedback loop for stability.

Requires the wasmtime Python bindings (separate from rclite's deps)::

    pip install wasmtime
    python examples/wasm_pages_demo/build.py            # produce dist/
    python examples/wasm_pages_demo/validate.py
"""
from __future__ import annotations
import ctypes
import math
import pathlib
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))

from wasmtime import Store, Module, Instance, Func, FuncType, ValType  # noqa: E402
import build as demo  # noqa: E402

DIST = pathlib.Path(__file__).resolve().parent / "dist"


def _load(path):
    store = Store()
    mod = Module.from_file(store.engine, str(path))
    # f32 kernel imports env.tanhf; mirror the JS loader (f32-rounded tanh).
    ty = FuncType([ValType.f32()], [ValType.f32()])
    tanhf = Func(store, ty, lambda x: ctypes.c_float(math.tanh(x)).value)
    inst = Instance(store, mod, [tanhf])
    return store, inst.exports(store)


def _predict(store, ex, inp, M=1):
    """Replicates rclite.js loadRclite(...).predict for the f32 build.

    Returns the (T, M) storage-domain output the kernel wrote (squeezed to
    (T,) when M == 1, to match the original regression callers).
    """
    mem, rc = ex["memory"], ex["rc_predict"]
    heap = ex["__heap_base"].value(store)
    T = len(inp)
    align = lambda n: (n + 15) & ~15
    xptr, yptr = heap, heap + align(T * 4)
    need = yptr + align(T * M * 4)
    if need > mem.data_len(store):
        mem.grow(store, (need - mem.data_len(store)) // 65536 + 1)
    base = ctypes.addressof(mem.data_ptr(store).contents)
    xs = (ctypes.c_float * T).from_address(base + xptr)
    for i, v in enumerate(inp):
        xs[i] = float(v)
    ys = (ctypes.c_float * (T * M)).from_address(base + yptr)
    for i in range(T * M):
        ys[i] = 0.0
    rc(store, T, xptr, yptr)
    out = np.array([ys[i] for i in range(T * M)], dtype=np.float32)
    return out if M == 1 else out.reshape(T, M)


def _softmax(z):
    z = z - z.max()
    e = np.exp(z)
    return e / e.sum()


def main() -> int:
    if not (DIST / "forecast.wasm").exists() or not (DIST / "shape.wasm").exists():
        print("dist/ not found (or stale) -- run build.py first", file=sys.stderr)
        return 2

    ok = True

    _, exe_f, _ = demo.build_forecast_model()
    store, ex = _load(DIST / "forecast.wasm")
    inp = (0.8 * np.sin(2 * np.pi * 0.03 * np.arange(220))).astype(np.float32)
    yw = _predict(store, ex, inp)
    yh = exe_f.predict(inp[:, None]).ravel().astype(np.float32)
    d = float(np.max(np.abs(yw - yh)))
    print(f"forecast.wasm vs host:  max|diff|={d:.2e}  corr={np.corrcoef(yw, yh)[0,1]:.6f}")
    ok &= d < 1e-3

    _, exe_d, info_d = demo.build_dream_model()
    store2, ex2 = _load(DIST / "dream.wasm")
    seed = np.array(info_d["seed"][-256:], dtype=np.float32)
    yw2 = _predict(store2, ex2, seed)
    yh2 = exe_d.predict(seed[:, None]).ravel().astype(np.float32)
    d2 = float(np.max(np.abs(yw2 - yh2)))
    print(f"dream.wasm    vs host:  max|diff|={d2:.2e}  corr={np.corrcoef(yw2, yh2)[0,1]:.6f}")
    ok &= d2 < 1e-3

    buf = seed.copy(); gen = []
    for _ in range(800):
        y = _predict(store2, ex2, buf)
        gen.append(float(y[-1]))
        buf = np.concatenate([buf[1:], [y[-1]]]).astype(np.float32)
    gen = np.array(gen)
    stable = bool(np.all(np.isfinite(gen))) and gen.std() > 0.05 and np.abs(gen).max() < 2
    print(f"dream autoregression:   range=[{gen.min():.3f},{gen.max():.3f}] "
          f"std={gen.std():.3f} stable={stable}")
    ok &= stable

    # --- shape classifier (MEAN aggregation, M classes): first row == logits -
    _, exe_s, _ = demo.build_shape_model()
    M_s = len(demo.SHAPE_CLASSES)
    store3, ex3 = _load(DIST / "shape.wasm")
    rng = np.random.default_rng(123)
    agree = 0; max_pdiff = 0.0; n = 0
    for kind in range(M_s):
        w = demo._shape_window(kind, rng)
        out = _predict(store3, ex3, w.ravel().astype(np.float32), M_s)
        p_js = _softmax(out[0]); c_js = int(np.argmax(out[0]))
        p_host = exe_s.predict_proba_sequences([w])[0]
        c_host = int(exe_s.predict_sequences([w])[0])
        max_pdiff = max(max_pdiff, float(np.max(np.abs(p_js - p_host))))
        agree += (c_js == c_host); n += 1
    print(f"shape.wasm    vs host:  class agreement={agree}/{n}  "
          f"max|softmax diff|={max_pdiff:.2e}")
    ok &= (agree == n) and (max_pdiff < 1e-3)

    # --- trend classifier (NONE aggregation, per-step): argmax matches host ---
    _, exe_t, _ = demo.build_trend_model()
    M_t = len(demo.TREND_CLASSES)
    store4, ex4 = _load(DIST / "trend.wasm")
    Xt, yt = demo._trend_series(demo.TREND_WASHOUT + 400, seed=7)
    out = _predict(store4, ex4, Xt.ravel().astype(np.float32), M_t)
    cls_js = np.argmax(out, axis=1)
    cls_host = exe_t.predict_classes(Xt)
    w0 = demo.TREND_WASHOUT
    match = float(np.mean(cls_js[w0:] == cls_host[w0:]))
    truth = float(np.mean(cls_js[w0:] == yt[w0:]))
    print(f"trend.wasm    vs host:  per-step argmax match={match:.4f} "
          f"(post-washout)  vs-truth acc={truth:.3f}")
    ok &= match > 0.999

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
