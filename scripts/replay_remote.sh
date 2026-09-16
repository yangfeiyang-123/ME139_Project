#!/usr/bin/env bash
# Start an Isaac Sim 5.1 WebRTC reference replay.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
: "${ISAAC_STREAM_HOST:?Set ISAAC_STREAM_HOST to a reachable server IP address}"
if [[ $# -eq 0 ]]; then
  echo "Usage: ISAAC_STREAM_HOST=<IP> ./project.sh replay-remote --motion <reference.npz>" >&2
  exit 2
fi
if ss -lnt | grep ":49100 " >/dev/null; then
  echo "Port 49100 is already in use. Connect to the existing viewer session first." >&2
  exit 1
fi
exec ./project.sh replay \
  --device cuda:0 --livestream 2 --steps 100000 \
  --kit_args="--/app/livestream/publicEndpointAddress=$ISAAC_STREAM_HOST --/app/livestream/port=49100 --/app/renderer/resolution/width=1280 --/app/renderer/resolution/height=720" \
  "$@"
