#!/usr/bin/env bash
# Start an Isaac Sim 5.1 WebRTC reference replay for the macOS streaming client.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if ss -lnt | grep -q ":49100 "; then
  echo "Port 49100 is already in use. Connect to the existing viewer session first." >&2
  exit 1
fi
exec ./badminton.sh replay \
  --motion data/motions/g1_walk_10s.npz \
  --device cuda:0 --livestream 2 --steps 100000 \
  --kit_args="--/app/livestream/publicEndpointAddress=128.32.164.89 --/app/livestream/port=49100 --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720" \
  "$@"
