#!/usr/bin/env bash
# Baixa os pesos pra models/upscale/. Idempotente: pula o que já existe.
set -euo pipefail
cd "$(dirname "$0")/upscale"

models=(
  "4x-UltraSharp.pth|https://huggingface.co/lokCX/4x-Ultrasharp/resolve/main/4x-UltraSharp.pth"
  "RealESRGAN_x4plus.pth|https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth"
  "4x_NMKD-Siax_200k.pth|https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_NMKD-Siax_200k.pth"
  "4x_foolhardy_Remacri.pth|https://huggingface.co/uwg/upscaler/resolve/main/ESRGAN/4x_foolhardy_Remacri.pth"
)

for entry in "${models[@]}"; do
  name="${entry%%|*}"
  url="${entry##*|}"
  if [ -f "$name" ]; then
    echo "skip $name (exists)"
    continue
  fi
  echo "--- downloading $name ---"
  curl -fL -o "$name" "$url"
done

ls -lh *.pth
