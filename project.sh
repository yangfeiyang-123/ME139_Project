#!/usr/bin/env bash
# Run all project commands relative to the repository root.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")"

usage() {
    cat <<'HELP'
Usage: ./project.sh <command> [arguments]

Commands:
  setup          Create the local overlay on an existing Isaac Sim environment
  doctor         Check package versions, CUDA, and robot assets
  smoke          Verify G1 physics and generate a synthetic standing reference
  convert        Convert a local motion CSV to a reference NPZ
  train          Train a G1 motion-tracking policy
  play           Play a saved policy checkpoint
  replay         Visualize a reference motion
  replay-remote  Stream a reference motion (requires ISAAC_STREAM_HOST)
  wham           Check/run WHAM or export world-space SMPL (separate environment)
  retarget       Convert world-space SMPL to joint-limited G1 motion CSV
  view-smpl      Export a synchronized video / SMPL / robot viewer
  python         Run Python in the project environment
  help           Show this help without loading Isaac Sim

Set ISAAC_BASE_PYTHON to a Python 3.11 interpreter with Isaac Sim 5.1.0.0
and PyTorch 2.7.0+cu128 before setup. Paths are relative to this repository.
HELP
}

command_name="${1:-help}"
if [[ $# -gt 0 ]]; then shift; fi
case "$command_name" in
    help|-h|--help) usage ;;
    setup) exec bash scripts/setup.sh "$@" ;;
    doctor) exec bash scripts/python.sh scripts/doctor.py "$@" ;;
    smoke) exec bash scripts/python.sh scripts/smoke_g1.py "$@" ;;
    convert) exec bash scripts/python.sh scripts/convert_motion.py "$@" ;;
    train) exec bash scripts/python.sh scripts/train.py "$@" ;;
    play) exec bash scripts/python.sh scripts/train.py --play "$@" ;;
    replay) exec bash scripts/python.sh scripts/replay_motion.py "$@" ;;
    replay-remote) exec bash scripts/replay_remote.sh "$@" ;;
    wham) exec python3 scripts/wham.py "$@" ;;
    retarget) exec bash scripts/python.sh scripts/retarget_smpl.py "$@" ;;
    view-smpl) exec bash scripts/python.sh scripts/visualize_smpl.py "$@" ;;
    python) exec bash scripts/python.sh "$@" ;;
    *) printf 'Unknown command: %s\n' "$command_name" >&2; usage >&2; exit 2 ;;
esac
