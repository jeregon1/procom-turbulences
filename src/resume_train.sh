#!/usr/bin/env bash
TARGET_EPOCH=1200
B=0.2

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

while true; do
  # detect last saved epoch or zero
  if [ -f turbulence/pretrained/spectralLoss_b${B}_latest.ckpt ]; then
    CUR=$(python3 - <<EOF
import torch
ckpt=torch.load("turbulence/pretrained/spectralLoss_b${B}_latest.ckpt", map_location="cpu")
print(int(ckpt['epoch']))
EOF
)
  else
    CUR=0
  fi

  REMAIN=$(( TARGET_EPOCH - CUR ))
  if (( REMAIN <= 0 )); then
    echo "✅ Reached $TARGET_EPOCH epochs. Exiting."
    break
  fi

  echo "Starting at epoch $CUR, running $REMAIN more epochs..."
  python3 notebook.py \
    --epochs $REMAIN \
    --b $B \
    --resume \
    && { echo "Finished cleanly at $TARGET_EPOCH epochs."; break;} \
    || echo "❌ Crash at epoch $CUR, restarting..."
done
