# Interactive WebAssembly demo (GitHub Pages)

Echo State Networks trained with **rclite**, cross-compiled to zero-WASI
WebAssembly *reactor* modules and running **entirely in the browser** — no
server-side inference. Built for publishing on GitHub Pages.

**One page, four tabs.** Two reservoirs do **regression**, two do
**classification**, all sharing the same scope:

| module          | task                                  | how the page uses it                                                  |
| --------------- | ------------------------------------- | --------------------------------------------------------------------- |
| `forecast.wasm` | broadband 1-step-ahead prediction     | **Predict** — you drive (sliders / draw), it forecasts                |
| `dream.wasm`    | clean quasi-periodic attractor        | **Dream** — output is fed back as input (in JS), it self-generates    |
| `shape.wasm`    | 5-class sequence-to-label (MEAN agg.) | **Shape** — draw / pick a curve, it names the whole shape             |
| `trend.wasm`    | 2-class per-step (NONE agg.)          | **Trend** — a scrolling signal is labelled rising/falling each step   |

Each `.wasm` is a reactor that **exports** `rc_predict` + the linear `memory`
(`__heap_base`) and **imports** only `env.tanhf` (the loader wires it to
`Math.tanh`). The models are plain `f32`, cross-compiled to `wasm32` with
SIMD128, via [`BrowserWasm`](../../rclite/targets/wasm/browser.py).

**Classification needs no special kernel:** each module emits the ordinary
linear readout's **logits**, and the page recovers the label with `argmax` and
the probabilities with `softmax` — exactly what rclite does in Python. The
shape model pools states over the window (`Aggregation.MEAN`) so its kernel
emits **one** logit vector per window; the trend model emits **one per step**
(`Aggregation.NONE`), which the page colours into a per-step class strip. The
forecast/dream loaders are shared (`M=1`); the classifiers ship their own
loaders (`M=5` / `M=2`).

## Build

Needs the WASI rust target and a wasm linker (`wasm-ld`, or the `rust-lld`
bundled with rustc — used automatically as a fallback). No browser/node needed
to build.

```bash
rustup target add wasm32-wasip1
python examples/wasm_pages_demo/build.py --out dist
(cd dist && python -m http.server)        # open http://localhost:8000
```

`build.py` trains all four ESNs, compiles the wasm, and copies the static
front-end (`frontend/`) + a `meta.json` (model sizes / NRMSE / accuracy / class
names / dream seed) into the output directory.

## Verify (optional, no browser)

`validate.py` instantiates the built modules exactly as the JS loader does,
reproduces the page's argmax / softmax, and compares against the in-process
host model:

```bash
pip install wasmtime          # not an rclite dependency
python examples/wasm_pages_demo/validate.py
# forecast.wasm vs host:  max|diff|=2.44e-06  corr=1.000000
# dream.wasm    vs host:  max|diff|=4.77e-07  corr=1.000000
# dream autoregression:   range=[-0.998,0.997] std=0.515 stable=True
# shape.wasm    vs host:  class agreement=5/5  max|softmax diff|=3.24e-07
# trend.wasm    vs host:  per-step argmax match=1.0000 (post-washout)  vs-truth acc=0.912
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
build.py            train + compile (4 models) + assemble the site
validate.py         headless end-to-end check (needs `wasmtime`)
frontend/
  index.html        page shell (scope + classification panel, 4 tabs)
  app.js            scope rendering, controls, predict/dream/shape/trend loops
  style.css         styling
dist/               generated output (git-ignored)
```
