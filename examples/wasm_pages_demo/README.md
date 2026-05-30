# Interactive WebAssembly demo (GitHub Pages)

An Echo State Network trained with **rclite**, cross-compiled to a zero-WASI
WebAssembly *reactor* module and running **entirely in the browser** — no
server-side inference. Built for publishing on GitHub Pages.

Two reservoirs drive one scrolling-oscilloscope UI:

| module          | task                              | how the page uses it                                   |
| --------------- | --------------------------------- | ------------------------------------------------------ |
| `forecast.wasm` | broadband 1-step-ahead prediction | **Predict** mode — you drive (sliders / draw), it forecasts |
| `dream.wasm`    | clean quasi-periodic attractor    | **Dream** mode — output is fed back as input (in JS), it self-generates |

Each `.wasm` is a reactor that **exports** `rc_predict` + the linear `memory`
(`__heap_base`) and **imports** only `env.tanhf` (the loader wires it to
`Math.tanh`). The model is plain `f32`, cross-compiled to `wasm32` with
SIMD128, via [`BrowserWasm`](../../rclite/targets/wasm/browser.py).

## Build

Needs the WASI rust target and a wasm linker (`wasm-ld`, or the `rust-lld`
bundled with rustc — used automatically as a fallback). No browser/node needed
to build.

```bash
rustup target add wasm32-wasip1
python examples/wasm_pages_demo/build.py --out dist
(cd dist && python -m http.server)        # open http://localhost:8000
```

`build.py` trains both ESNs, compiles the wasm, and copies the static
front-end (`frontend/`) + a `meta.json` (model sizes / NRMSE / dream seed) into
the output directory.

## Verify (optional, no browser)

`validate.py` instantiates the built modules exactly as the JS loader does and
compares against the in-process host model:

```bash
pip install wasmtime          # not an rclite dependency
python examples/wasm_pages_demo/validate.py
# forecast.wasm vs host:  max|diff|=2.44e-06  corr=1.000000
# dream.wasm    vs host:  max|diff|=4.77e-07  corr=1.000000
# dream autoregression:   range=[-0.998,0.997] std=0.515 stable=True
# PASS
```

## Publish on GitHub Pages

The repo's `docs` workflow builds this demo into `docs/demo/` alongside the API
docs, so it is served at `…/rclite/demo/`. To do it by hand:

```bash
python examples/wasm_pages_demo/build.py --out docs/demo
```

## Files

```
build.py            train + compile + assemble the site
validate.py         headless end-to-end check (needs `wasmtime`)
frontend/
  index.html        page shell
  app.js            scope rendering, controls, predict + dream loops
  style.css         styling
dist/               generated output (git-ignored)
```
