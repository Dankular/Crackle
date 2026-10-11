# Voice isolation test bench

Static page that runs open-source speech-enhancement ONNX models in the browser and
lets you mix a target voice with an interfering voice and/or noise to see what each
model removes.

```
./scripts/fetch-models.sh      # downloads ONNX files into web/models/ (gitignored)
cd web && python3 -m http.server 8765
# open http://localhost:8765
```

Models (all streaming, frame-by-frame, state carried in model I/O): DPDFNet-2/-8 (16 kHz),
DPDFNet-2 HR (48 kHz), GTCRN (16 kHz). Per-model STFT settings and initial state are in
`models.json`; they were read from the ONNX metadata and the DPDFNet reference package.

Notes
- Runtime is `onnxruntime-web` (the engine Transformers.js uses), called directly, because
  these models need custom STFT/state handling that no Transformers.js pipeline provides.
- DPDFNet output is delayed by 4 hops (measured against the reference package); `dsp.js`
  compensates so enhanced audio is sample-aligned with the input.
- `dsp.js` output matched the `dpdfnet` Python `StreamEnhancer` to 2e-7 max abs difference
  on a 10 s clip (dpdfnet2).
