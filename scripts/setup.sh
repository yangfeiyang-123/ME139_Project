#!/usr/bin/env bash
# Rebuild the project overlay using the existing Isaac Sim 5.1 installation.
set -euo pipefail
cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.."
if [[ $# -gt 0 ]]; then
    echo "Usage: ISAAC_BASE_PYTHON=/path/to/python ./project.sh setup" >&2
    exit 2
fi
base_python="${ISAAC_BASE_PYTHON:-python3}"
"$base_python" - <<'CHECK'
import sys, importlib.metadata as md
try:
    assert sys.version_info[:2] == (3, 11), "Python 3.11 is required"
    assert md.version("isaacsim") == "5.1.0.0", "Isaac Sim 5.1.0.0 is required"
    assert md.version("torch") == "2.7.0+cu128", "PyTorch 2.7.0+cu128 is required"
except (AssertionError, md.PackageNotFoundError) as error:
    raise SystemExit(f"{error}. Set ISAAC_BASE_PYTHON to the matching existing environment.")
CHECK
mkdir -p third_party logs data/motions .cache/tmp
export TMPDIR="$PWD/.cache/tmp"
export PYTHONNOUSERSITE=1
ensure_pinned() {
    local repo_dir="$1" repo_revision="$2"
    if [[ ! -e "$repo_dir/.git" ]]; then
        git submodule update --init -- "$repo_dir"
    fi
    if [[ "$(git -C "$repo_dir" rev-parse HEAD)" != "$repo_revision" ]]; then
        echo "Unexpected checkout in $repo_dir; keeping it unchanged. Expected $repo_revision" >&2
        exit 1
    fi
}
ensure_pinned third_party/IsaacLab 37ddf626871758333d6ed89cf64ad702aef127d0
ensure_pinned third_party/whole_body_tracking cd65172032893724b445448818c34165846d847d
if [[ ! -x .venv/bin/python ]]; then
    "$base_python" -m venv --system-site-packages .venv
fi
.venv/bin/python - <<'CHECK'
from pathlib import Path
import sys
assert Path(sys.prefix).resolve() == (Path.cwd()/".venv").resolve(), "Refusing to install outside project .venv"
CHECK
.venv/bin/python -m pip install -c configs/constraints.txt -r configs/requirements-overlay.lock.txt
for package in isaaclab isaaclab_assets isaaclab_tasks isaaclab_rl isaaclab_mimic; do
    .venv/bin/python -m pip install --no-deps --no-build-isolation -e "third_party/IsaacLab/source/$package"
done
.venv/bin/python -m pip install --no-deps --no-build-isolation -e third_party/whole_body_tracking/source/whole_body_tracking
asset_dir=third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets
if [[ ! -f "$asset_dir/unitree_description/urdf/g1/main.urdf" ]]; then
    archive="$TMPDIR/unitree_description.tar.gz"
    curl -fL --retry 3 -o "$archive" https://storage.googleapis.com/qiayuanl_robot_descriptions/unitree_description.tar.gz
    python3 - "$archive" "$asset_dir" <<'EXTRACT'
import hashlib,sys,tarfile
from pathlib import Path
p=Path(sys.argv[1])
assert hashlib.sha256(p.read_bytes()).hexdigest() == "b514bc9ddd1039c29a0e6feea9f57f1503f6657d07d97a4ef8a7b11fbebe6674", "Robot archive checksum changed"
with tarfile.open(p) as archive:
    archive.extractall(sys.argv[2], filter="data")
EXTRACT
fi
./project.sh doctor
