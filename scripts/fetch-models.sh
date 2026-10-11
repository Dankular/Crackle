#!/usr/bin/env bash
# Download the ONNX models the test page uses into web/models/.
set -euo pipefail
cd "$(dirname "$0")/../web/models"
HF=https://huggingface.co
for f in dpdfnet2.onnx dpdfnet8.onnx dpdfnet2_48khz_hr.onnx; do
  [ -s "$f" ] || curl -fL -o "$f" "$HF/Ceva-IP/DPDFNet/resolve/main/onnx/$f"
done
[ -s gtcrn_simple.onnx ] || curl -fL -o gtcrn_simple.onnx "$HF/bitsydarel/gtcrn-onnx/resolve/main/gtcrn_simple.onnx"
ls -l
