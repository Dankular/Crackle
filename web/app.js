import * as ort from 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/ort.min.mjs';
import { StreamEnhancer, siSdr, rms } from './dsp.js';

ort.env.wasm.wasmPaths = 'https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/';
ort.env.wasm.numThreads = 1; // multi-thread WASM needs cross-origin isolation headers

const $ = (id) => document.getElementById(id);
let configs = {}, enhancer = null, loadedKey = null, abort = null;
const files = { A: null, B: null };

// ---------- model loading ----------
async function init() {
  configs = await (await fetch('models.json')).json();
  for (const [k, c] of Object.entries(configs)) $('model').add(new Option(c.label, k));
  for (const [id, unit] of [['intfDb', ' dB'], ['snr', ' dB'], ['rt60', ' s']]) {
    const upd = () => { $(id + 'V').textContent = $(id).value + unit; };
    $(id).addEventListener('input', upd); upd();
  }
  // close-mic talker vs a quieter, reverberant background talker + low-level noise
  $('preset').onclick = () => {
    $('intfDb').value = -15; $('farfield').checked = true; $('rt60').value = 0.5;
    $('noiseType').value = 'pink'; $('snr').value = 15;
    for (const id of ['intfDb', 'snr', 'rt60']) $(id).dispatchEvent(new Event('input'));
  };
}

async function fetchModel(cfg) {
  for (const url of ['models/' + cfg.file, cfg.url]) {
    try {
      const r = await fetch(url);
      if (!r.ok) continue;
      return { buf: new Uint8Array(await r.arrayBuffer()), from: url };
    } catch { /* try next */ }
  }
  throw new Error('could not fetch ' + cfg.file + ' (run scripts/fetch-models.sh or check network)');
}

$('load').onclick = async () => {
  const key = $('model').value, cfg = configs[key];
  $('load').disabled = true; $('mstat').textContent = 'loading…';
  try {
    const t0 = performance.now();
    const { buf, from } = await fetchModel(cfg);
    const session = await ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    enhancer = new StreamEnhancer(ort, session, cfg);
    loadedKey = key;
    const ms = Math.round(performance.now() - t0);
    $('mstat').textContent = `ready · ${(buf.length / 1e6).toFixed(1)} MB from ${from.startsWith('http') ? 'Hugging Face' : 'local'} · ${ms} ms · ` +
      `${cfg.sr / 1000} kHz, window ${cfg.win}, hop ${cfg.hop} (${(cfg.hop / cfg.sr * 1000).toFixed(0)} ms)`;
    updateRun();
  } catch (e) { $('mstat').textContent = 'failed: ' + e.message; console.error(e); }
  $('load').disabled = false;
};
$('model').onchange = () => { enhancer = null; loadedKey = null; $('mstat').textContent = ''; updateRun(); };

// ---------- audio io ----------
async function decode(arrayBuffer, sr) {
  const ctx = new AudioContext({ sampleRate: sr }); // decodeAudioData resamples to ctx rate
  try {
    const ab = await ctx.decodeAudioData(arrayBuffer.slice(0));
    const mono = new Float32Array(ab.length);
    for (let c = 0; c < ab.numberOfChannels; c++) {
      const d = ab.getChannelData(c);
      for (let i = 0; i < d.length; i++) mono[i] += d[i] / ab.numberOfChannels;
    }
    return mono;
  } finally { ctx.close(); }
}

function wavBlob(x, sr) {
  const b = new ArrayBuffer(44 + x.length * 2), v = new DataView(b);
  const s = (o, t) => { for (let i = 0; i < t.length; i++) v.setUint8(o + i, t.charCodeAt(i)); };
  s(0, 'RIFF'); v.setUint32(4, 36 + x.length * 2, true); s(8, 'WAVEfmt ');
  v.setUint32(16, 16, true); v.setUint16(20, 1, true); v.setUint16(22, 1, true);
  v.setUint32(24, sr, true); v.setUint32(28, sr * 2, true); v.setUint16(32, 2, true); v.setUint16(34, 16, true);
  s(36, 'data'); v.setUint32(40, x.length * 2, true);
  for (let i = 0; i < x.length; i++) v.setInt16(44 + i * 2, Math.max(-1, Math.min(1, x[i])) * 32767, true);
  return new Blob([b], { type: 'audio/wav' });
}

// ---------- inputs ----------
$('fileA').onchange = (e) => { files.A = e.target.files[0] || null; updateRun(); };
$('fileB').onchange = (e) => { files.B = e.target.files[0] || null; };

let recorder = null;
$('rec').onclick = async () => {
  if (recorder) { recorder.stop(); return; }
  try {
    // turn browser DSP off so we test the model, not Chrome's built-in suppression
    const stream = await navigator.mediaDevices.getUserMedia({ audio: { echoCancellation: false, noiseSuppression: false, autoGainControl: false } });
    const chunks = [];
    recorder = new MediaRecorder(stream);
    recorder.ondataavailable = (e) => chunks.push(e.data);
    recorder.onstop = () => {
      stream.getTracks().forEach((t) => t.stop());
      files.A = new File(chunks, 'recording.webm', { type: recorder.mimeType });
      recorder = null; $('rec').textContent = '● Record from mic';
      $('recstat').textContent = `recorded ${(files.A.size / 1024).toFixed(0)} KB — ready`; updateRun();
    };
    recorder.start(); $('rec').textContent = '■ Stop'; $('recstat').textContent = 'recording…';
  } catch (e) { $('recstat').textContent = 'mic unavailable: ' + e.message; }
};

function updateRun() { $('run').disabled = !(enhancer && files.A); }

// ---------- noise + mixing ----------
function makeNoise(type, n) {
  const x = new Float32Array(n);
  if (type === 'white') { for (let i = 0; i < n; i++) x[i] = Math.random() * 2 - 1; }
  else if (type === 'brown') { let a = 0; for (let i = 0; i < n; i++) { a = (a + (Math.random() * 2 - 1) * 0.02) / 1.0002; x[i] = a; } }
  else if (type === 'pink') { // Paul Kellet filter
    let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0;
    for (let i = 0; i < n; i++) {
      const w = Math.random() * 2 - 1;
      b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
      b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
      x[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362; b6 = w * 0.115926;
    }
  }
  return x;
}
const scaleTo = (x, targetRms) => { const g = targetRms / (rms(x) || 1); const y = new Float32Array(x.length); for (let i = 0; i < x.length; i++) y[i] = x[i] * g; return y; };

// ---------- far-field simulation (synthetic, not a measured room) ----------
function fftConvolve(x, h) {
  let n = 1; while (n < x.length + h.length - 1) n <<= 1;
  const ar = new Float64Array(n), ai = new Float64Array(n), br = new Float64Array(n), bi = new Float64Array(n);
  ar.set(x); br.set(h);
  fft(ar, ai); fft(br, bi);
  for (let i = 0; i < n; i++) { const r = ar[i] * br[i] - ai[i] * bi[i], m = ar[i] * bi[i] + ai[i] * br[i]; ar[i] = r; ai[i] = -m; } // conj for inverse
  fft(ar, ai);
  const y = new Float32Array(x.length);
  for (let i = 0; i < y.length; i++) y[i] = ar[i] / n;
  return y;
}
// Direct path + exponentially decaying noise tail (RT60), then 1-pole low-pass (~3.5 kHz) for air/wall absorption.
function farField(x, sr, rt60) {
  const len = Math.floor(rt60 * sr), h = new Float32Array(len);
  const k = 6.9078 / (rt60 * sr);
  for (let i = 0; i < len; i++) h[i] = (Math.random() * 2 - 1) * Math.exp(-k * i);
  h[Math.floor(0.003 * sr)] += 0.35 * Math.sqrt(h.reduce((s, v) => s + v * v, 0)); // weak direct path => low direct-to-reverb ratio
  const y = fftConvolve(x, h);
  const a = Math.exp(-2 * Math.PI * 3500 / sr);
  let s = 0; for (let i = 0; i < y.length; i++) { s = (1 - a) * y[i] + a * s; y[i] = s; }
  return y;
}

// ---------- spectrogram ----------
function fft(re, im) { // radix-2 in place
  const n = re.length;
  for (let i = 1, j = 0; i < n; i++) { let bit = n >> 1; for (; j & bit; bit >>= 1) j ^= bit; j ^= bit; if (i < j) { [re[i], re[j]] = [re[j], re[i]]; [im[i], im[j]] = [im[j], im[i]]; } }
  for (let len = 2; len <= n; len <<= 1) {
    const ang = -2 * Math.PI / len, wr = Math.cos(ang), wi = Math.sin(ang);
    for (let i = 0; i < n; i += len) {
      let cr = 1, ci = 0;
      for (let k = 0; k < len / 2; k++) {
        const a = i + k, b = a + len / 2, tr = re[b] * cr - im[b] * ci, ti = re[b] * ci + im[b] * cr;
        re[b] = re[a] - tr; im[b] = im[a] - ti; re[a] += tr; im[a] += ti;
        const t = cr * wr - ci * wi; ci = cr * wi + ci * wr; cr = t;
      }
    }
  }
}
function drawSpec(canvas, x, sr) {
  const N = 512, hop = 256, W = Math.min(900, Math.floor((x.length - N) / hop)), H = N / 2;
  canvas.width = W; canvas.height = H;
  const g = canvas.getContext('2d'), img = g.createImageData(W, H);
  const re = new Float32Array(N), im = new Float32Array(N), win = Float32Array.from({ length: N }, (_, i) => 0.5 - 0.5 * Math.cos(2 * Math.PI * i / N));
  const stride = Math.max(1, Math.floor((x.length - N) / hop / W));
  for (let c = 0; c < W; c++) {
    const o = c * stride * hop;
    for (let i = 0; i < N; i++) { re[i] = (x[o + i] || 0) * win[i]; im[i] = 0; }
    fft(re, im);
    for (let k = 0; k < H; k++) {
      const db = 10 * Math.log10(re[k] * re[k] + im[k] * im[k] + 1e-10);
      const t = Math.max(0, Math.min(1, (db + 70) / 90)); // -70..+20 dB (unnormalised FFT)
      const p = ((H - 1 - k) * W + c) * 4;
      img.data[p] = 255 * Math.min(1, t * 2); img.data[p + 1] = 255 * Math.max(0, t * 2 - 1) ** 1.2; img.data[p + 2] = 90 * t; img.data[p + 3] = 255;
    }
  }
  g.putImageData(img, 0, 0);
}

// ---------- run ----------
$('stop').onclick = () => abort?.abort();
$('run').onclick = async () => {
  const cfg = configs[loadedKey], sr = cfg.sr;
  abort = new AbortController();
  $('run').disabled = true; $('stop').disabled = false; $('prog').hidden = false; $('prog').value = 0;
  try {
    $('status').textContent = 'decoding…';
    const target = await decode(await files.A.arrayBuffer(), sr);
    const mix = Float32Array.from(target);
    const ra = rms(target);
    const added = [];
    if (files.B) {
      const b = await decode(await files.B.arrayBuffer(), sr);
      const looped = new Float32Array(target.length);
      for (let i = 0; i < looped.length; i++) looped[i] = b[i % b.length];
      const far = $('farfield').checked;
      const src = far ? farField(looped, sr, +$('rt60').value) : looped; // level set AFTER reverb
      const sc = scaleTo(src, ra * 10 ** ($('intfDb').value / 20));
      for (let i = 0; i < mix.length; i++) mix[i] += sc[i];
      added.push(`interferer ${$('intfDb').value} dB${far ? ' far-field RT60 ' + $('rt60').value + ' s' : ''}`);
    }
    if ($('noiseType').value !== 'none') {
      const nz = scaleTo(makeNoise($('noiseType').value, target.length), ra * 10 ** (-$('snr').value / 20));
      for (let i = 0; i < mix.length; i++) mix[i] += nz[i];
      added.push(`${$('noiseType').value} noise ${$('snr').value} dB SNR`);
    }
    let peak = 0; for (const v of mix) peak = Math.max(peak, Math.abs(v));
    if (peak > 0.99) { const g = 0.99 / peak; for (let i = 0; i < mix.length; i++) mix[i] *= g; } // SI-SDR is scale-invariant

    $('status').textContent = 'running ' + loadedKey + '…';
    const t0 = performance.now();
    const out = await enhancer.processAll(mix, (p) => { $('prog').value = p; }, abort.signal);
    const sec = (performance.now() - t0) / 1000, dur = mix.length / sr;

    $('aRef').src = URL.createObjectURL(wavBlob(target, sr));
    $('aMix').src = URL.createObjectURL(wavBlob(mix, sr));
    $('aOut').src = URL.createObjectURL(wavBlob(out, sr));
    drawSpec($('cRef'), target, sr); drawSpec($('cMix'), mix, sr); drawSpec($('cOut'), out, sr);

    const hasRef = added.length > 0;
    let rows = `<tr><th>Model</th><td colspan=2>${loadedKey}</td></tr>
      <tr><th>Processing</th><td colspan=2>${sec.toFixed(2)} s for ${dur.toFixed(1)} s audio · real-time factor ${(sec / dur).toFixed(2)} (WASM, 1 thread, this machine)</td></tr>
      <tr><th>Algorithmic latency</th><td colspan=2>${((cfg.win + cfg.delayHops * cfg.hop) / sr * 1000).toFixed(0)} ms (window ${(cfg.win / sr * 1000).toFixed(0)} ms + ${cfg.delayHops} hop delay, compensated in this page)</td></tr>`;
    if (hasRef) {
      const a = siSdr(target, mix).db, b = siSdr(target, out).db;
      rows += `<tr><th>SI-SDR vs reference</th><td>input ${a.toFixed(1)} dB</td><td>enhanced ${b.toFixed(1)} dB · <span class="${b - a > 1 ? 'good' : 'badc'}">Δ ${(b - a >= 0 ? '+' : '') + (b - a).toFixed(1)} dB</span></td></tr>
        <tr><th>Added</th><td colspan=2>${added.join(' + ')}</td></tr>`;
    } else {
      rows += `<tr><th>SI-SDR</th><td colspan=2>n/a — no interference added, so there is no clean reference. Add noise or an interferer to get objective numbers.</td></tr>`;
    }
    $('metrics').innerHTML = rows;
    $('mnote').textContent = 'SI-SDR is computed on this single clip against the target you supplied; treat it as a smoke test, not a benchmark. Listen to the Enhanced clip: models trained to denoise do not necessarily remove a competing voice.';
    $('results').hidden = false; $('status').textContent = 'done';
  } catch (e) {
    $('status').textContent = e.name === 'AbortError' ? 'cancelled' : 'error: ' + e.message; console.error(e);
  }
  $('run').disabled = false; $('stop').disabled = true; $('prog').hidden = true; updateRun();
};

init();
