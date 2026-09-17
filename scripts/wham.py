"""WHAM configuration checks, inference, and world-space SMPL export."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
WHAM = ROOT / "third_party/WHAM"
ASSETS = ROOT / "configs/wham/assets.json"


def missing_assets(root, local_only=False):
    groups = {"base"} if local_only else {"base", "global"}
    return [
        item for item in json.loads(ASSETS.read_text())["assets"]
        if item["group"] in groups
        and (not (root / item["path"]).is_file() or (root / item["path"]).stat().st_size == 0)
    ]


def clean_environment():
    env = os.environ.copy()
    # Do not let Isaac Lab's editable import paths leak into WHAM.
    env.pop("PYTHONPATH", None)
    env.pop("PYTHONHOME", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def infer_command(python, video, output, local_only=False, visualize=False, calib=None):
    command = [str(python), "demo.py", "--video", str(video.resolve()),
               "--output_pth", str(output.resolve()), "--save_pkl"]
    if local_only:
        command.append("--estimate_local_only")
    if visualize:
        command.append("--visualize")
    if calib:
        command += ["--calib", str(calib.resolve())]
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    check = sub.add_parser("check", help="List missing model files without downloading anything")
    check.add_argument("--local-only", action="store_true")
    run = sub.add_parser("run", help="Run WHAM in its separate Python environment")
    run.add_argument("--video", required=True, type=Path)
    run.add_argument("--output", type=Path, default=ROOT / "outputs/wham")
    run.add_argument("--local-only", action="store_true")
    run.add_argument("--visualize", action="store_true")
    run.add_argument("--calib", type=Path)
    export = sub.add_parser("export", help="Reconstruct world-space SMPL joints for retargeting")
    export.add_argument("--input", type=Path, required=True)
    export.add_argument("--video", type=Path, required=True)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--person-id")
    export.add_argument("--vertices", action="store_true", help="Also export the full SMPL mesh")
    for p in (run, export):
        p.add_argument("--python", default=os.environ.get("WHAM_PYTHON"),
                       help="WHAM Python 3.9 executable; or set WHAM_PYTHON")
    args = parser.parse_args()
    if not (WHAM / "demo.py").is_file():
        parser.error("Initialize third_party/WHAM: git submodule update --init --recursive")
    if args.command == "check":
        missing = missing_assets(WHAM, args.local_only)
        for item in missing:
            print(f"MISSING {item['path']}\n  {item['url']}")
        source_paths = ["third-party/ViTPose/setup.py"]
        if not args.local_only:
            source_paths.append("third-party/DPVO/setup.py")
        missing_sources = False
        for path in source_paths:
            if not (WHAM / path).is_file():
                missing_sources = True
                print(f"SOURCE NOT INITIALIZED: {path}")
        print("Environment and download instructions: docs/video-smpl-retarget.md")
        if not missing and not missing_sources:
            print("Model files present; use 'wham run' for runtime checks.")
        return int(bool(missing) or missing_sources)
    if not args.python:
        parser.error("Set WHAM_PYTHON to the separate WHAM environment's Python executable.")
    interpreter = Path(args.python).expanduser().absolute()
    if not interpreter.is_file():
        parser.error(f"WHAM Python does not exist: {interpreter}")
    if not args.video.is_file():
        parser.error(f"Video not found: {args.video}")
    env = clean_environment()
    if args.command == "export":
        if not args.input.is_file():
            parser.error(f"WHAM output not found: {args.input}")
        command = [str(interpreter), str(ROOT / "scripts/export_wham.py"),
                   "--input", str(args.input.resolve()), "--video", str(args.video.resolve()),
                   "--output", str(args.output.resolve()),
                   "--model", str(WHAM / "dataset/body_models/smpl/SMPL_NEUTRAL.pkl")]
        if args.person_id is not None:
            command += ["--person-id", args.person_id]
        if args.vertices:
            command.append("--vertices")
        return subprocess.call(command, cwd=WHAM, env=env)
    if args.calib and not args.calib.is_file():
        parser.error(f"Camera calibration not found: {args.calib}")
    missing = missing_assets(WHAM, args.local_only)
    if missing:
        parser.error("Missing model files. Run './project.sh wham check' for paths and download links.")
    sequence = args.output.resolve() / args.video.stem
    if sequence.exists() and any(sequence.iterdir()):
        parser.error(f"Output already contains a run: {sequence}. Choose a new --output to avoid stale caches.")
    preflight = [
        "import sys, torch",
        "assert sys.version_info[:2] == (3, 9), 'Use the WHAM Python 3.9 environment'",
        "assert torch.__version__.split('+')[0] == '1.11.0', 'Expected WHAM PyTorch 1.11.0'",
        "assert torch.cuda.is_available(), 'WHAM inference requires CUDA'",
        "from lib.models.preproc.detector import DetectionModel",
        "from lib.models.preproc.extractor import FeatureExtractor",
    ]
    if not args.local_only:
        # Upstream otherwise silently falls back to local-only estimation.
        preflight.append("from lib.models.preproc.slam import SLAMModel")
    if args.visualize:
        preflight.append("from lib.vis.run_vis import run_vis_on_demo")
    subprocess.run([str(interpreter), "-c", "\n".join(preflight)],
                   cwd=WHAM, env=env, check=True)
    command = infer_command(interpreter, args.video, args.output, args.local_only,
                            args.visualize, args.calib)
    subprocess.run(command, cwd=WHAM, env=env, check=True)
    if not (sequence / "wham_output.pkl").is_file():
        raise RuntimeError("WHAM finished without wham_output.pkl")
    digest = hashlib.sha256()
    with args.video.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    metadata = {
        "video": str(args.video.resolve()),
        "video_sha256": digest.hexdigest(),
        "upstream_commit": json.loads(ASSETS.read_text())["upstream_commit"],
        "coordinate_mode": "camera-local" if args.local_only else "world",
        "command": command,
    }
    (sequence / "run.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"WHAM output: {sequence / 'wham_output.pkl'}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, ValueError, RuntimeError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        raise SystemExit(1)
