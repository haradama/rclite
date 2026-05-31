// Interactive front-end for the rclite WebAssembly demo -- one page, four
// reservoirs, two of them regression and two classification:
//
//   * "predict" -- forecast.wasm overlays a one-step-ahead forecast of a
//      user/keyboard/drawn waveform.
//   * "dream"   -- dream.wasm is run autoregressively (output -> next input)
//      so the network regenerates its learned waveform on its own.
//   * "shape"   -- shape.wasm (MEAN aggregation) classifies a whole drawn /
//      generated window into one of five curve shapes; the page softmaxes the
//      emitted logits into live probability bars.
//   * "trend"   -- trend.wasm (NONE aggregation) labels every timestep of a
//      scrolling signal rising/falling; the page draws a per-step class strip.
//
// Classification is just argmax (class id) + softmax (probabilities) over the
// linear readout's logits -- exactly what rclite computes in Python.

import { loadRclite as loadReg } from "./rclite.js";          // M=1 (forecast+dream)
import { loadRclite as loadShape } from "./rclite_shape.js";  // M=5
import { loadRclite as loadTrend } from "./rclite_trend.js";  // M=2

const W = 220;                       // samples shown across the scope (regression)
const DREAM_BUF = 256;               // autoregression context (> washout)
const DISP = 200;                    // trend: visible samples (after warm-up)

const $ = (id) => document.getElementById(id);
const canvas = $("scope");
const ctx2d = canvas.getContext("2d");
const CW = canvas.width, CH = canvas.height;

const state = {
  mode: "predict",
  playing: true,
  phase: 0,
  // regression
  input: new Float32Array(W),
  drawn: null,
  dreamBuf: null,
  dreamDisp: new Float32Array(W),
  dreamSeed: null,
  // classification
  shapeWin: null,
  shapeDrawn: null,
  ctx: null,
  trendDrawn: false,
  tphase: 0,
  // shared
  drawing: false,
  drawMode: false,
  emaTime: 0,
};

let forecast = null, dream = null, shape = null, trend = null, meta = null;
let SHAPE_W = 80, TREND_WASHOUT = 100, CTX = 300;
let SHAPE_CLASSES = [], TREND_CLASSES = [];

// ----------------------------------------------------------------- utilities

const cssCache = {};
function getCss(name) {
  if (cssCache[name] === undefined) {
    cssCache[name] = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }
  return cssCache[name];
}
const classColor = (i) => getCss(`--c${i}`);

function softmax(z) {
  let m = -Infinity;
  for (const v of z) if (v > m) m = v;
  let s = 0; const e = new Float32Array(z.length);
  for (let i = 0; i < z.length; i++) { e[i] = Math.exp(z[i] - m); s += e[i]; }
  for (let i = 0; i < e.length; i++) e[i] /= s;
  return e;
}
function argmax(z) {
  let bi = 0, bv = -Infinity;
  for (let i = 0; i < z.length; i++) if (z[i] > bv) { bv = z[i]; bi = i; }
  return bi;
}

// ----------------------------------------------------------------- waveforms

function waveSample(kind, phase, freqN, amp, noise) {
  const x = 2 * Math.PI * freqN * phase;
  let v;
  switch (kind) {
    case "twotone": v = 0.6 * Math.sin(x) + 0.4 * Math.sin(1.7 * x + 0.7); break;
    case "square":   v = Math.sign(Math.sin(x)); break;
    case "triangle": v = (2 / Math.PI) * Math.asin(Math.sin(x)); break;
    default:         v = Math.sin(x);
  }
  v *= amp;
  if (noise > 0) v += noise * (Math.random() * 2 - 1);
  return Math.max(-1.2, Math.min(1.2, v));
}

function shapeSample(kind, i, n) {
  const t = i / (n - 1);
  switch (kind) {
    case 0: return -1 + 2 * t;                 // rising
    case 1: return 1 - 2 * t;                  // falling
    case 2: return 1 - 4 * Math.abs(t - 0.5);  // peak
    case 3: return -1 + 4 * Math.abs(t - 0.5); // valley
    default: return Math.sin(2 * Math.PI * t); // sine
  }
}

// ----------------------------------------------------------------- rendering

function clearScope() {
  ctx2d.fillStyle = "#0a0d14";
  ctx2d.fillRect(0, 0, CW, CH);
  ctx2d.strokeStyle = "#161d2c";
  ctx2d.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    const y = (CH / 8) * i;
    ctx2d.beginPath(); ctx2d.moveTo(0, y); ctx2d.lineTo(CW, y); ctx2d.stroke();
  }
  ctx2d.strokeStyle = "#243049";
  ctx2d.beginPath(); ctx2d.moveTo(0, CH / 2); ctx2d.lineTo(CW, CH / 2); ctx2d.stroke();
}

function plot(arr, color, glow, xShift = 0, sy = CH * 0.40, span = W) {
  const mid = CH / 2;
  ctx2d.lineWidth = 3;
  ctx2d.strokeStyle = color;
  ctx2d.shadowColor = color;
  ctx2d.shadowBlur = glow;
  ctx2d.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = ((i + xShift) / (span - 1)) * CW;
    const y = mid - arr[i] * sy;
    if (i === 0) ctx2d.moveTo(x, y); else ctx2d.lineTo(x, y);
  }
  ctx2d.stroke();
  ctx2d.shadowBlur = 0;
}

// colour strip along the bottom: classes[i] colours the i-th column
function plotStrip(classes) {
  const n = classes.length, h = 26, y0 = CH - h;
  ctx2d.globalAlpha = 0.85;
  for (let i = 0; i < n; i++) {
    ctx2d.fillStyle = classColor(classes[i]);
    ctx2d.fillRect((i / n) * CW, y0, CW / n + 1, h);
  }
  ctx2d.globalAlpha = 1;
}

// --------------------------------------------------------------- predict panel

function buildPanel(classes) {
  const probs = $("probs");
  probs.innerHTML = "";
  classes.forEach((name, i) => {
    const row = document.createElement("div");
    row.className = "prob";
    row.innerHTML = `<span class="lbl">${name}</span>` +
      `<span class="track"><span class="fill" style="background:${classColor(i)}"></span></span>` +
      `<span class="pct">0%</span>`;
    probs.appendChild(row);
  });
}

function updatePanel(classes, proba, cls) {
  $("pred-class").textContent = classes[cls];
  $("pred-class").style.color = classColor(cls);
  $("pred-conf").textContent = `confidence ${(proba[cls] * 100).toFixed(1)}%`;
  const rows = $("probs").children;
  for (let i = 0; i < rows.length; i++) {
    rows[i].querySelector(".fill").style.width = (proba[i] * 100).toFixed(1) + "%";
    rows[i].querySelector(".pct").textContent = (proba[i] * 100).toFixed(0) + "%";
    rows[i].classList.toggle("lead", i === cls);
  }
}

// ------------------------------------------------------------------- predict

function stepPredict() {
  const kind = $("wave").value;
  const freqN = +$("freq").value / 2000;
  const amp = +$("amp").value / 100;
  const noise = +$("noise").value / 100;

  if (state.drawMode && state.drawn) {
    state.input.set(state.drawn);
  } else if (state.playing) {
    state.input.copyWithin(0, 1);
    state.input[W - 1] = waveSample(kind, state.phase, freqN, amp, noise);
    state.phase++;
  }

  const t0 = performance.now();
  const pred = forecast.predict(state.input);     // Float32Array length W
  const dt = performance.now() - t0;
  state.emaTime = state.emaTime ? 0.9 * state.emaTime + 0.1 * dt : dt;

  let se = 0, n = 0, mean = 0;
  for (let i = 0; i < W - 1; i++) mean += state.input[i + 1];
  mean /= (W - 1);
  let varSum = 0;
  for (let i = 0; i < W - 1; i++) {
    const e = pred[i] - state.input[i + 1];
    se += e * e; n++;
    varSum += (state.input[i + 1] - mean) ** 2;
  }
  const nrmse = Math.sqrt(se / n) / (Math.sqrt(varSum / n) + 1e-9);

  clearScope();
  plot(state.input, getCss("--accent"), 6);
  plot(pred, getCss("--accent2"), 12, 1);

  $("r-metric").textContent = nrmse.toFixed(3);
  $("r-metric").className = "v" + (nrmse < 0.15 ? " good" : "");
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// --------------------------------------------------------------------- dream

function reseed() {
  state.dreamBuf = Float32Array.from(state.dreamSeed.slice(-DREAM_BUF));
  state.dreamDisp.fill(0);
}

function stepDream() {
  const steps = +$("speed").value;
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    const y = dream.predict(state.dreamBuf);
    const next = y[DREAM_BUF - 1];
    state.dreamBuf.copyWithin(0, 1);
    state.dreamBuf[DREAM_BUF - 1] = next;
    state.dreamDisp.copyWithin(0, 1);
    state.dreamDisp[W - 1] = next;
  }
  const dt = (performance.now() - t0) / steps;
  state.emaTime = state.emaTime ? 0.9 * state.emaTime + 0.1 * dt : dt;

  clearScope();
  plot(state.dreamDisp, getCss("--accent2"), 12);
  $("r-metric").textContent = "—";
  $("r-metric").className = "v";
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// ------------------------------------------------------------------- shape

function regenShape() {
  const kind = +$("shapeSel").value;
  const jit = +$("jitter").value / 100;
  for (let i = 0; i < SHAPE_W; i++) {
    state.shapeWin[i] = shapeSample(kind, i, SHAPE_W) +
      jit * (Math.random() * 2 - 1);
  }
}

function stepShape() {
  if (state.drawMode && state.shapeDrawn) {
    state.shapeWin.set(state.shapeDrawn);
  } else if (state.playing) {
    regenShape();
  }

  const t0 = performance.now();
  const out = shape.predict(state.shapeWin);          // length SHAPE_W * M
  const dt = performance.now() - t0;
  state.emaTime = state.emaTime ? 0.9 * state.emaTime + 0.1 * dt : dt;

  const M = SHAPE_CLASSES.length;
  const logits = out.subarray(0, M);                  // MEAN agg -> first row
  const proba = softmax(logits);
  const cls = argmax(logits);

  clearScope();
  plot(state.shapeWin, classColor(cls), 10, 0, CH * 0.40, SHAPE_W);
  updatePanel(SHAPE_CLASSES, proba, cls);
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// ------------------------------------------------------------------- trend

function prefillTrend() {
  const kind = $("twave").value;
  const freqN = +$("tfreq").value / 2000;
  const amp = +$("tamp").value / 100;
  const noise = +$("tnoise").value / 100;
  for (let i = 0; i < CTX; i++) {
    state.ctx[i] = waveSample(kind, state.tphase + i, freqN, amp, noise);
  }
  state.tphase += CTX;
}

function stepTrend() {
  if (!state.trendDrawn && state.playing) {
    const kind = $("twave").value;
    const freqN = +$("tfreq").value / 2000;
    const amp = +$("tamp").value / 100;
    const noise = +$("tnoise").value / 100;
    state.ctx.copyWithin(0, 1);
    state.ctx[CTX - 1] = waveSample(kind, state.tphase, freqN, amp, noise);
    state.tphase++;
  }

  const t0 = performance.now();
  const out = trend.predict(state.ctx);               // length CTX * M
  const dt = performance.now() - t0;
  state.emaTime = state.emaTime ? 0.9 * state.emaTime + 0.1 * dt : dt;

  const M = TREND_CLASSES.length;
  const disp = new Float32Array(DISP);
  const dispCls = new Int32Array(DISP);
  for (let i = 0; i < DISP; i++) {
    const t = TREND_WASHOUT + i;
    disp[i] = state.ctx[t];
    let bi = 0, bv = -Infinity;
    for (let c = 0; c < M; c++) {
      const v = out[t * M + c];
      if (v > bv) { bv = v; bi = c; }
    }
    dispCls[i] = bi;
  }
  const last = (CTX - 1) * M;
  const lastLogits = out.subarray(last, last + M);
  const proba = softmax(lastLogits);
  const cls = argmax(lastLogits);

  clearScope();
  plotStrip(dispCls);
  plot(disp, classColor(cls), 8, 0, CH * 0.34, DISP);
  updatePanel(TREND_CLASSES, proba, cls);
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// ----------------------------------------------------------------- main loop

function frame() {
  switch (state.mode) {
    case "predict":
      if (state.playing || (state.drawMode && state.drawn)) stepPredict();
      break;
    case "dream":
      if (state.playing) stepDream();
      break;
    case "shape":
      if (state.playing || (state.drawMode && state.shapeDrawn)) stepShape();
      break;
    case "trend":
      stepTrend();
      break;
  }
  requestAnimationFrame(frame);
}

// ------------------------------------------------------------------- drawing

function attachDrawing() {
  const onDown = (ev) => {
    if (!state.drawMode) return;
    state.drawing = true;
    if (state.mode === "predict" && !state.drawn) {
      state.drawn = Float32Array.from(state.input);
    } else if (state.mode === "shape" && !state.shapeDrawn) {
      state.shapeDrawn = Float32Array.from(state.shapeWin);
    } else if (state.mode === "trend") {
      state.trendDrawn = true;
    }
    paint(ev); ev.preventDefault();
  };
  const paint = (ev) => {
    if (!state.drawing) return;
    const r = canvas.getBoundingClientRect();
    const px = (ev.clientX - r.left) / r.width;
    const py = (ev.clientY - r.top) / r.height;
    if (state.mode === "predict") {
      const idx = clampIdx(px, W);
      const val = clampVal(py, 0.40);
      state.drawn[idx] = val;
      if (idx > 0) state.drawn[idx - 1] = (state.drawn[idx - 1] + val) / 2;
      if (idx < W - 1) state.drawn[idx + 1] = (state.drawn[idx + 1] + val) / 2;
    } else if (state.mode === "shape") {
      const idx = clampIdx(px, SHAPE_W);
      const val = clampVal(py, 0.40);
      state.shapeDrawn[idx] = val;
      if (idx > 0) state.shapeDrawn[idx - 1] = (state.shapeDrawn[idx - 1] + val) / 2;
      if (idx < SHAPE_W - 1) state.shapeDrawn[idx + 1] = (state.shapeDrawn[idx + 1] + val) / 2;
    } else if (state.mode === "trend") {
      const idx = clampIdx(px, DISP);
      const val = clampVal(py, 0.34);
      state.ctx[TREND_WASHOUT + idx] = val;
      if (idx === 0) for (let k = 0; k < TREND_WASHOUT; k++) state.ctx[k] = val;
    }
  };
  const onUp = () => { state.drawing = false; };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", paint);
  window.addEventListener("pointerup", onUp);
}
const clampIdx = (px, n) => Math.max(0, Math.min(n - 1, Math.round(px * (n - 1))));
const clampVal = (py, sy) => Math.max(-1.2, Math.min(1.2, (0.5 - py) / sy));

// --------------------------------------------------------------------- modes

const REGRESSION = (m) => m === "predict" || m === "dream";

function showFor(mode) {
  document.querySelectorAll("[data-modes]").forEach((el) => {
    const match = el.dataset.modes.split(" ").includes(mode);
    // explicit inline display (wins over the `[data-modes]{display:none}`
    // default); control groups are display:contents so their controls flow
    // straight into the row.
    const shown = el.classList.contains("cgroup") ? "contents"
      : el.classList.contains("predict-card") ? "flex" : "block";
    el.style.display = match ? shown : "none";
  });
}

function setMode(mode) {
  state.mode = mode;
  state.drawMode = false;
  state.drawn = null;
  state.shapeDrawn = null;
  state.trendDrawn = false;
  document.querySelectorAll(".btn.toggle").forEach((b) => b.classList.remove("on"));
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.mode === mode));
  showFor(mode);

  const m = meta ? meta.models[mode] : null;
  $("r-units").textContent = m ? m.units : "—";
  $("r-size").textContent = m ? (m.wasm_bytes / 1024).toFixed(1) + " KB" : "—";

  if (REGRESSION(mode)) {
    $("r-metric-k").textContent = "forecast error (NRMSE)";
    $("legend").innerHTML = mode === "predict" ? `
      <span><i class="swatch" style="background:var(--accent)"></i>input</span>
      <span><i class="swatch" style="background:var(--accent2)"></i>ESN&nbsp;1-step&nbsp;forecast</span>` : "";
    $("legend").style.display = mode === "predict" ? "" : "none";
    if (mode === "dream" && !state.dreamBuf) reseed();
  } else {
    $("r-metric-k").textContent = "held-out accuracy";
    $("r-metric").textContent = m ? (m.test_acc * 100).toFixed(1) + "%" : "—";
    $("r-metric").className = "v good";
    if (mode === "shape") {
      buildPanel(SHAPE_CLASSES);
      $("legend").style.display = "none";
      regenShape();
    } else {
      buildPanel(TREND_CLASSES);
      $("legend").innerHTML = TREND_CLASSES.map((n, i) =>
        `<span><i class="swatch" style="background:${classColor(i)}"></i>${n}</span>`).join("");
      $("legend").style.display = "";
      prefillTrend();
    }
  }
}

// ---------------------------------------------------------------------- init

async function init() {
  meta = await fetch("./meta.json").then((r) => r.json()).catch(() => null);
  [forecast, dream, shape, trend] = await Promise.all([
    loadReg("./forecast.wasm"),
    loadReg("./dream.wasm"),
    loadShape("./shape.wasm"),
    loadTrend("./trend.wasm"),
  ]);

  SHAPE_CLASSES = (meta && meta.models.shape.classes) ||
    ["rising", "falling", "peak", "valley", "sine"];
  TREND_CLASSES = (meta && meta.models.trend.classes) || ["falling", "rising"];
  SHAPE_W = (meta && meta.models.shape.window) || 80;
  TREND_WASHOUT = (meta && meta.models.trend.washout) || 100;
  CTX = TREND_WASHOUT + DISP;
  state.shapeWin = new Float32Array(SHAPE_W);
  state.ctx = new Float32Array(CTX);

  state.dreamSeed = (meta && meta.models.dream.seed) ||
    Array.from({ length: DREAM_BUF }, (_, i) =>
      0.6 * Math.sin(2 * Math.PI * 0.04 * i) + 0.4 * Math.sin(2 * Math.PI * 0.017 * i + 0.7));
  reseed();

  // badges
  const kb = (b) => (b / 1024).toFixed(1) + " KB";
  const get = (k) => (meta ? meta.models[k] : { units: "?", wasm_bytes: 0, imports: [] });
  const f = get("forecast"), sh = get("shape"), tr = get("trend");
  $("badges").innerHTML = `
    <span class="badge"><b>4</b> reservoirs &middot; trained with rclite</span>
    <span class="badge">regression <b>${f.units}u</b> &middot; ${kb(f.wasm_bytes)}</span>
    <span class="badge">shape <b>${SHAPE_CLASSES.length}-class</b> &middot; ${kb(sh.wasm_bytes)}</span>
    <span class="badge">trend <b>${TREND_CLASSES.length}-class</b> &middot; ${kb(tr.wasm_bytes)}</span>
    <span class="badge">kernel <b>f32 wasm32${meta && meta.simd ? " · SIMD128" : ""}</b></span>
    <span class="badge">imports <b>${(f.imports || ["env.tanhf"]).join(", ") || "none"}</b></span>`;

  // controls
  $("playPause").addEventListener("click", () => {
    state.playing = !state.playing;
    $("playPause").textContent = state.playing ? "Pause" : "Play";
  });
  document.querySelectorAll(".tab").forEach((t) =>
    t.addEventListener("click", () => setMode(t.dataset.mode)));
  const sync = () => {
    $("freqV").textContent = (+$("freq").value / 2000).toFixed(3);
    $("ampV").textContent = (+$("amp").value / 100).toFixed(2);
    $("noiseV").textContent = (+$("noise").value / 100).toFixed(2);
    $("speedV").textContent = $("speed").value;
    $("jitterV").textContent = (+$("jitter").value / 100).toFixed(2);
    $("tfreqV").textContent = (+$("tfreq").value / 2000).toFixed(3);
    $("tampV").textContent = (+$("tamp").value / 100).toFixed(2);
    $("tnoiseV").textContent = (+$("tnoise").value / 100).toFixed(2);
  };
  ["freq", "amp", "noise", "speed", "jitter", "tfreq", "tamp", "tnoise"].forEach((id) =>
    $(id).addEventListener("input", sync));
  $("shapeSel").addEventListener("change", () => { state.shapeDrawn = null; regenShape(); });
  sync();

  // draw toggles (one per mode that supports drawing)
  const wireDraw = (btnId, mode) => {
    $(btnId).addEventListener("click", () => {
      const on = !state.drawMode;
      state.drawMode = on;
      state.drawn = null;
      state.shapeDrawn = null;
      if (mode === "trend") state.trendDrawn = on;
      $(btnId).classList.toggle("on", on);
    });
  };
  wireDraw("drawToggle", "predict");
  wireDraw("sdrawToggle", "shape");
  wireDraw("tdrawToggle", "trend");

  $("perturb").addEventListener("click", () => {
    if (state.dreamBuf) state.dreamBuf[DREAM_BUF - 1] += 0.9;
  });
  $("reseed").addEventListener("click", reseed);
  attachDrawing();

  setMode("predict");
  requestAnimationFrame(frame);
}

init();
