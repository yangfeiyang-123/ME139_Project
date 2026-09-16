#!/usr/bin/env bash
# Source this file from any directory.
export ME139_PROJECT_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
if [[ ! -x "$ME139_PROJECT_ROOT/.venv/bin/python" || ! -r "$ME139_PROJECT_ROOT/.venv/bin/activate" ]]; then
    echo "Project environment missing. Run ISAAC_BASE_PYTHON=/path/to/python ./project.sh setup first." >&2
    return 1
fi
source "$ME139_PROJECT_ROOT/.venv/bin/activate"
export PYTHONNOUSERSITE=1
export ISAACLAB_PATH="$ME139_PROJECT_ROOT/third_party/IsaacLab"
# Editable finders otherwise lose precedence to the pre-existing pip Isaac Lab bundle.
export PYTHONPATH="$ISAACLAB_PATH/source/isaaclab:$ISAACLAB_PATH/source/isaaclab_assets:$ISAACLAB_PATH/source/isaaclab_tasks:$ISAACLAB_PATH/source/isaaclab_rl:$ISAACLAB_PATH/source/isaaclab_mimic:$ME139_PROJECT_ROOT/third_party/whole_body_tracking/source/whole_body_tracking"
export OMNI_KIT_ACCEPT_EULA=YES
export WANDB_MODE="${WANDB_MODE:-offline}"
export TERM="${TERM:-xterm}"
export PYTHONUNBUFFERED=1

export TMPDIR="$ME139_PROJECT_ROOT/.cache/tmp"
mkdir -p "$TMPDIR"
