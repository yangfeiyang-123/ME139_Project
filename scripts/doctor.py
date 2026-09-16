"""Read-only environment checks; does not launch Isaac Sim."""
import importlib.metadata as md
import os
from pathlib import Path
import shutil
import sys
import torch

root = Path(__file__).resolve().parents[1]
expected = {"isaacsim": "5.1.0.0", "isaaclab": "0.54.2", "isaaclab_assets": "0.2.4",
            "isaaclab_tasks": "0.11.12", "isaaclab_rl": "0.4.7", "isaaclab_mimic": "1.0.16",
            "whole_body_tracking": "0.1.0", "rsl-rl-lib": "3.1.2", "torch": "2.7.0+cu128",
            "numpy": "1.26.0", "tensordict": "0.8.3"}
errors = []
for name, version in expected.items():
    try:
        actual = md.version(name)
        print(f"{name:24s} {actual}")
        if actual != version:
            errors.append(f"{name}: expected {version}, got {actual}")
    except md.PackageNotFoundError:
        errors.append(f"Missing package: {name}")
if not torch.cuda.is_available():
    errors.append("CUDA is not available")
else:
    for i in range(torch.cuda.device_count()):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    x = torch.ones(8, device="cuda:0")
    assert float((x*x).sum()) == 8
asset = root / "third_party/whole_body_tracking/source/whole_body_tracking/whole_body_tracking/assets/unitree_description/urdf/g1/main.urdf"
if not asset.is_file():
    errors.append(f"Missing G1 asset: {asset}")
print(f"Python: {sys.executable}")
print(f"Temporary files: {os.environ.get('TMPDIR')}")
print(f"Disk available: {shutil.disk_usage(root).free / 2**30:.1f} GiB")
print("Note: source Isaac Lab v2.3.2 uses per-extension package versions shown above.")
if errors:
    print("\n".join(errors), file=sys.stderr)
    raise SystemExit(1)
print("ENVIRONMENT_CHECK_PASSED (GPU simulator validation: ./project.sh smoke --headless)")
