// Interactive front-end for the rclite WebAssembly ESN demo.
//
// Loads two reactor modules through the rclite loader and drives them from a
// single scrolling-oscilloscope canvas:
//   * "predict" mode -- a user/keyboard/drawn waveform is fed to forecast.wasm,
//      whose one-step-ahead output is overlaid.
//   * "dream" mode -- dream.wasm is run autoregressively (output -> next input)
//      so the network regenerates its learned waveform on its own.

import { loadRclite } from "./rclite.js";

const W = 220;                       // samples shown across the scope
const DREAM_BUF = 256;               // autoregression context (> washout)

const $ = (id) => document.getElementById(id);
const canvas = $("scope");
const ctx = canvas.getContext("2d");
const CW = canvas.width, CH = canvas.height;

const state = {
  mode: "predict",
  playing: true,
  phase: 0,
  input: new Float32Array(W),        // current scope window (predict mode)
  drawing: false,
  drawMode: false,
  drawn: null,                       // user-drawn window or null
  dreamBuf: null,                    // Float32Array(DREAM_BUF)
  dreamDisp: new Float32Array(W),
  dreamSeed: null,
  emaTime: 0,
};

let forecast = null, dream = null, meta = null;

// ----------------------------------------------------------------- waveforms

function waveSample(kind, phase, freqN, amp, noise) {
  // freqN: cycles-per-sample; phase: integer sample index
  const x = 2 * Math.PI * freqN * phase;
  let v;
  switch (kind) {
    case "twotone": v = 0.6 * Math.sin(x) + 0.4 * Math.sin(1.7 * x + 0.7); break;
    case "square":   v = Math.sign(Math.sin(x)); break;
    case "triangle": v = (2 / Math.PI) * Math.asin(Math.sin(x)); break;
    case "noisy":    v = Math.sin(x); break;
    default:         v = Math.sin(x);
  }
  v *= amp;
  if (noise > 0) v += noise * (Math.random() * 2 - 1);
  return Math.max(-1.2, Math.min(1.2, v));
}

// ----------------------------------------------------------------- rendering

function clearScope() {
  ctx.fillStyle = "#0a0d14";
  ctx.fillRect(0, 0, CW, CH);
  ctx.strokeStyle = "#161d2c";
  ctx.lineWidth = 1;
  for (let i = 1; i < 8; i++) {
    const y = (CH / 8) * i;
    ctx.beginPath(); ctx.moveTo(0, y); ctx.lineTo(CW, y); ctx.stroke();
  }
  // zero line
  ctx.strokeStyle = "#243049";
  ctx.beginPath(); ctx.moveTo(0, CH / 2); ctx.lineTo(CW, CH / 2); ctx.stroke();
}

function plot(arr, color, glow, xShift = 0) {
  const mid = CH / 2, sy = CH * 0.40;
  ctx.lineWidth = 3;
  ctx.strokeStyle = color;
  ctx.shadowColor = color;
  ctx.shadowBlur = glow;
  ctx.beginPath();
  for (let i = 0; i < arr.length; i++) {
    const x = ((i + xShift) / (W - 1)) * CW;
    const y = mid - arr[i] * sy;
    if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
  }
  ctx.stroke();
  ctx.shadowBlur = 0;
}

// ------------------------------------------------------------------- predict

function stepPredict() {
  const kind = $("wave").value;
  const freqN = +$("freq").value / 2000;          // 0.004 .. 0.040 cycles/sample
  const amp = +$("amp").value / 100;
  const noise = +$("noise").value / 100;

  if (state.drawMode && state.drawn) {
    state.input.set(state.drawn);
  } else if (state.playing) {
    // scroll in one fresh sample
    state.input.copyWithin(0, 1);
    state.input[W - 1] = waveSample(kind, state.phase, freqN, amp, noise);
    state.phase++;
  }

  const t0 = performance.now();
  const pred = forecast.predict(state.input);     // Float32Array length W
  const dt = performance.now() - t0;
  state.emaTime = state.emaTime ? 0.9 * state.emaTime + 0.1 * dt : dt;

  // NRMSE: pred[i] forecasts input[i+1]
  let se = 0, n = 0, mean = 0;
  for (let i = 0; i < W - 1; i++) mean += state.input[i + 1];
  mean /= (W - 1);
  let varSum = 0;
  for (let i = 0; i < W - 1; i++) {
    const e = pred[i] - state.input[i + 1];
    se += e * e; n++;
    varSum += (state.input[i + 1] - mean) ** 2;
  }
  const rmse = Math.sqrt(se / n);
  const nrmse = rmse / (Math.sqrt(varSum / n) + 1e-9);

  clearScope();
  plot(state.input, getCss("--accent"), 6);
  plot(pred, getCss("--accent2"), 12, 1);          // shift +1: forecast leads

  $("r-nrmse").textContent = nrmse.toFixed(3);
  $("r-nrmse").className = "v" + (nrmse < 0.15 ? " good" : "");
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// --------------------------------------------------------------------- dream

function reseed() {
  const seed = state.dreamSeed;
  state.dreamBuf = Float32Array.from(seed.slice(-DREAM_BUF));
  state.dreamDisp.fill(0);
}

function stepDream() {
  const steps = +$("speed").value;
  const t0 = performance.now();
  for (let s = 0; s < steps; s++) {
    const y = dream.predict(state.dreamBuf);       // length DREAM_BUF
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
  $("r-nrmse").textContent = "—";
  $("r-nrmse").className = "v";
  $("r-time").textContent = state.emaTime.toFixed(3) + " ms";
}

// ----------------------------------------------------------------- main loop

function frame() {
  if (state.mode === "predict") {
    if (state.playing || (state.drawMode && state.drawn)) stepPredict();
  } else {
    if (state.playing) stepDream();
  }
  requestAnimationFrame(frame);
}

// --------------------------------------------------------------------- utils

const cssCache = {};
function getCss(name) {
  if (!cssCache[name]) {
    cssCache[name] = getComputedStyle(document.documentElement)
      .getPropertyValue(name).trim();
  }
  return cssCache[name];
}

function setMode(mode) {
  state.mode = mode;
  document.querySelectorAll(".tab").forEach((t) =>
    t.classList.toggle("active", t.dataset.mode === mode));
  document.querySelectorAll(".mode-only").forEach((el) => {
    el.style.display = el.classList.contains(mode) ? "" : "none";
  });
  document.querySelector(".legend").style.opacity = mode === "predict" ? 1 : 0.0;
  if (mode === "dream" && !state.dreamBuf) reseed();
}

// ------------------------------------------------------------------- drawing

function canvasToSample(ev) {
  const r = canvas.getBoundingClientRect();
  const px = (ev.clientX - r.left) / r.width;            // 0..1 across W
  const py = (ev.clientY - r.top) / r.height;            // 0..1 vertical
  const idx = Math.max(0, Math.min(W - 1, Math.round(px * (W - 1))));
  const val = Math.max(-1.2, Math.min(1.2, (0.5 - py) / 0.40));
  return { idx, val };
}

function attachDrawing() {
  const onDown = (ev) => {
    if (!state.drawMode) return;
    state.drawing = true;
    if (!state.drawn) state.drawn = Float32Array.from(state.input);
    paint(ev); ev.preventDefault();
  };
  const paint = (ev) => {
    if (!state.drawing) return;
    const { idx, val } = canvasToSample(ev);
    state.drawn[idx] = val;
    // light smoothing into neighbours so the curve is continuous
    if (idx > 0) state.drawn[idx - 1] = (state.drawn[idx - 1] + val) / 2;
    if (idx < W - 1) state.drawn[idx + 1] = (state.drawn[idx + 1] + val) / 2;
  };
  const onUp = () => { state.drawing = false; };
  canvas.addEventListener("pointerdown", onDown);
  canvas.addEventListener("pointermove", paint);
  window.addEventListener("pointerup", onUp);
}

// ---------------------------------------------------------------------- init

async function init() {
  meta = await fetch("./meta.json").then((r) => r.json()).catch(() => null);
  [forecast, dream] = await Promise.all([
    loadRclite("./forecast.wasm"),
    loadRclite("./dream.wasm"),
  ]);
  state.dreamSeed = (meta && meta.models.dream.seed) ||
    Array.from({ length: DREAM_BUF }, (_, i) =>
      0.6 * Math.sin(2 * Math.PI * 0.04 * i) + 0.4 * Math.sin(2 * Math.PI * 0.017 * i + 0.7));
  reseed();

  // badges + readouts
  const f = meta ? meta.models.forecast : { units: "?", wasm_bytes: 0 };
  const d = meta ? meta.models.dream : { units: "?", wasm_bytes: 0 };
  const kb = (b) => (b / 1024).toFixed(1) + " KB";
  $("badges").innerHTML = `
    <span class="badge"><b>2</b> reservoirs &middot; trained with rclite</span>
    <span class="badge">forecast <b>${f.units}u</b> &middot; ${kb(f.wasm_bytes)}</span>
    <span class="badge">dream <b>${d.units}u</b> &middot; ${kb(d.wasm_bytes)}</span>
    <span class="badge">kernel <b>f32 wasm32${meta && meta.simd ? " · SIMD128" : ""}</b></span>
    <span class="badge">imports <b>${(f.imports || ["env.tanhf"]).join(", ") || "none"}</b></span>`;
  $("r-units").textContent = f.units;
  $("r-size").textContent = kb(f.wasm_bytes);

  // wire controls
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
  };
  ["freq", "amp", "noise", "speed"].forEach((id) =>
    $(id).addEventListener("input", sync));
  sync();
  $("drawToggle").addEventListener("click", () => {
    state.drawMode = !state.drawMode;
    $("drawToggle").classList.toggle("on", state.drawMode);
    if (!state.drawMode) state.drawn = null;
  });
  $("perturb").addEventListener("click", () => {
    if (state.dreamBuf) state.dreamBuf[DREAM_BUF - 1] += 0.9;
  });
  $("reseed").addEventListener("click", reseed);
  attachDrawing();

  setMode("predict");
  requestAnimationFrame(frame);
}

init();
