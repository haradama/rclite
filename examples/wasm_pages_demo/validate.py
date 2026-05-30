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


def _predict(store, ex, inp):
    """Replicates rclite.js loadRclite(...).predict for the f32 build."""
    mem, rc = ex["memory"], ex["rc_predict"]
    heap = ex["__heap_base"].value(store)
    T = len(inp)
    align = lambda n: (n + 15) & ~15
    xptr, yptr = heap, heap + align(T * 4)
    need = yptr + align(T * 4)
    if need > mem.data_len(store):
        mem.grow(store, (need - mem.data_len(store)) // 65536 + 1)
    base = ctypes.addressof(mem.data_ptr(store).contents)
    xs = (ctypes.c_float * T).from_address(base + xptr)
    for i, v in enumerate(inp):
        xs[i] = float(v)
    ys = (ctypes.c_float * T).from_address(base + yptr)
    rc(store, T, xptr, yptr)
    return np.array([ys[i] for i in range(T)], dtype=np.float32)


def main() -> int:
    if not (DIST / "forecast.wasm").exists():
        print("dist/ not found -- run build.py first", file=sys.stderr)
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

    print("\n" + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
