#!/usr/bin/env bash
# Source this file from any directory.
export HUMANOID_BADMINTON_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
source "$HUMANOID_BADMINTON_ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export ISAACLAB_PATH="$HUMANOID_BADMINTON_ROOT/third_party/IsaacLab"
# Editable finders otherwise lose precedence to the pre-existing pip Isaac Lab bundle.
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_assets:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAACLAB_PATH/source/isaaclab_rl:$ISAACLAB_PATH/source/isaaclab_mimic:$HUMANOID_BADMINTON_ROOT/third_party/whole_body_tracking/source/whole_body_tracking"
export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE="${WANDB_MODE:-offline}"
export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED=1

export TMPDIR="$HUMANOID_BADMINTON_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
