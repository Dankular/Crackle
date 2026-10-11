// Streaming STFT/iSTFT engine + metrics. No DOM; works in browsers and Node.
// `ort` (onnxruntime-web or onnxruntime-node) is injected so both can use it.

function makeWindow(kind, n) {
  const w = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    if (kind === 'vorbis') {
      // same formula as the dpdfnet reference package
      const s = Math.sin(0.5 * Math.PI * (i + 0.5) / (n / 2));
      w[i] = Math.sin(0.5 * Math.PI * s * s);
    } else if (kind === 'sqrthann') {
      // sqrt of *periodic* hann (torch.hann_window default), COLA at 50% overlap
      w[i] = Math.sqrt(0.5 - 0.5 * Math.cos((2 * Math.PI * i) / n));
    } else throw new Error('unknown window ' + kind);
  }
  return w;
}

// Direct real DFT with table lookups. Needed because n_fft is 320/960 (not powers of 2).
class RealDFT {
  constructor(n) {
    this.n = n;
    this.bins = n / 2 + 1;
    this.cos = new Float64Array(n);
    this.sin = new Float64Array(n);
    for (let i = 0; i < n; i++) {
      this.cos[i] = Math.cos((2 * Math.PI * i) / n);
      this.sin[i] = Math.sin((2 * Math.PI * i) / n);
    }
  }
  // x: Float32Array(n) already windowed -> out interleaved [re,im] * bins
  forward(x, out) {
    const { n, bins, cos, sin } = this;
    for (let k = 0; k < bins; k++) {
      let re = 0, im = 0, m = 0;
      for (let t = 0; t < n; t++) {
        re += x[t] * cos[m];
        im -= x[t] * sin[m];
        m += k; if (m >= n) m -= n;
      }
      out[2 * k] = re; out[2 * k + 1] = im;
    }
  }
  // spec interleaved -> y(n)  (irfft, includes 1/n)
  inverse(spec, y) {
    const { n, bins, cos, sin } = this;
    const half = bins - 1;
    for (let t = 0; t < n; t++) {
      let acc = spec[0]; // DC real
      let m = 0;
      for (let k = 1; k < half; k++) {
        m += t; if (m >= n) m -= n;
        acc += 2 * (spec[2 * k] * cos[m] - spec[2 * k + 1] * sin[m]);
      }
      acc += spec[2 * half] * ((t & 1) ? -1 : 1); // Nyquist real
      y[t] = acc / n;
    }
  }
}

export class StreamEnhancer {
  constructor(ort, session, cfg) {
    this.ort = ort; this.session = session; this.cfg = cfg;
    this.win = cfg.win; this.hop = cfg.hop; this.bins = cfg.freqBins;
    this.window = makeWindow(cfg.window, cfg.win);
    this.dft = new RealDFT(cfg.win);
    this.frame = new Float32Array(cfg.win);
    this.spec = new Float32Array(this.bins * 2);
    this.tFrame = new Float32Array(cfg.win);
    this.reset();
  }

  reset() {
    const c = this.cfg;
    this.inBuf = new Float32Array(0);
    this.outBuf = new Float32Array(this.win);
    if (c.kind === 'dpdfnet') {
      this.state = new Float32Array(c.stateSize);
      this.state.set(c.erbNormInit, 0);
      this.state.set(c.specNormInit, c.erbNormInit.length);
    } else {
      this.caches = {};
      for (const [k, shape] of Object.entries(c.caches))
        this.caches[k] = new Float32Array(shape.reduce((a, b) => a * b, 1));
    }
  }

  async _infer(spec) {
    const { Tensor } = this.ort;
    const c = this.cfg;
    if (c.kind === 'dpdfnet') {
      const r = await this.session.run({
        spec: new Tensor('float32', spec, [1, 1, this.bins, 2]),
        state_in: new Tensor('float32', this.state, [c.stateSize]),
      });
      this.state = r.state_out.data;
      return r.spec_e.data;
    }
    const feeds = { mix: new Tensor('float32', spec, [1, this.bins, 1, 2]) };
    for (const [k, shape] of Object.entries(c.caches))
      feeds[k] = new Tensor('float32', this.caches[k], shape);
    const r = await this.session.run(feeds);
    for (const k of Object.keys(c.caches)) this.caches[k] = r[k + '_out'].data;
    return r.enh.data;
  }

  // Feed any amount of audio (model sample rate). Returns committed enhanced samples.
  async process(chunk) {
    const merged = new Float32Array(this.inBuf.length + chunk.length);
    merged.set(this.inBuf); merged.set(chunk, this.inBuf.length);
    const out = [];
    let pos = 0;
    while (merged.length - pos >= this.win) {
      for (let i = 0; i < this.win; i++) this.frame[i] = merged[pos + i] * this.window[i];
      this.dft.forward(this.frame, this.spec);
      const enh = await this._infer(this.spec);
      this.dft.inverse(enh, this.tFrame);
      for (let i = 0; i < this.win; i++) this.outBuf[i] += this.tFrame[i] * this.window[i];
      out.push(this.outBuf.slice(0, this.hop));
      this.outBuf.copyWithin(0, this.hop);
      this.outBuf.fill(0, this.win - this.hop);
      pos += this.hop;
    }
    this.inBuf = merged.slice(pos);
    const res = new Float32Array(out.length * this.hop);
    out.forEach((o, i) => res.set(o, i * this.hop));
    return res;
  }

  // Convenience: whole clip -> same-length enhanced clip.
  async processAll(x, onProgress, signal) {
    this.reset();
    // Measured: dpdfnet outputs are delayed by `delayHops` hops (checked against the
    // reference package); pad the tail so we can drop that delay and keep alignment.
    const delay = (this.cfg.delayHops || 0) * this.hop;
    const padded = new Float32Array(x.length + this.win + delay);
    padded.set(x);
    const step = this.hop * 40;
    const parts = [];
    let total = 0;
    for (let i = 0; i < padded.length; i += step) {
      if (signal?.aborted) throw new DOMException('aborted', 'AbortError');
      const o = await this.process(padded.subarray(i, Math.min(i + step, padded.length)));
      parts.push(o); total += o.length;
      if (onProgress) { onProgress(Math.min(1, i / padded.length)); await new Promise(r => setTimeout(r, 0)); }
    }
    const y = new Float32Array(total);
    let o = 0; for (const p of parts) { y.set(p, o); o += p.length; }
    return y.slice(delay, delay + x.length);
  }
}

// SI-SDR (dB) of est vs ref, searching est lag in [-maxLag, maxLag] on a segment.
export function siSdr(ref, est, maxLag = 0) {
  const n = Math.min(ref.length, est.length);
  const score = (lag) => {
    const a0 = Math.max(0, -lag), a1 = Math.min(n, n - lag);
    let dot = 0, rr = 0;
    for (let i = a0; i < a1; i++) { dot += est[i + lag] * ref[i]; rr += ref[i] * ref[i]; }
    const alpha = dot / (rr + 1e-12);
    let sig = 0, err = 0;
    for (let i = a0; i < a1; i++) {
      const s = alpha * ref[i], e = est[i + lag] - s;
      sig += s * s; err += e * e;
    }
    return 10 * Math.log10((sig + 1e-12) / (err + 1e-12));
  };
  if (!maxLag) return { db: score(0), lag: 0 };
  let best = { db: -Infinity, lag: 0 };
  for (let lag = -maxLag; lag <= maxLag; lag++) {
    const db = score(lag);
    if (db > best.db) best = { db, lag };
  }
  return best;
}

export function rms(x) { let s = 0; for (const v of x) s += v * v; return Math.sqrt(s / Math.max(1, x.length)); }
