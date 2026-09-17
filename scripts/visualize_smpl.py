"""Export a generic video / SMPL / retargeted-robot viewer without external web dependencies."""
import argparse
import json
import os
from pathlib import Path
from urllib.parse import quote

import numpy as np
from smpl_data import SMPL_EDGES, file_sha256, load_motion, validate_frames


def export(input_path, video, output, robot=None):
    import cv2
    motion = load_motion(input_path)
    video = Path(video).resolve()
    if "source_video_sha256" in motion and str(motion["source_video_sha256"]) != file_sha256(video):
        raise ValueError("Video does not match the SMPL export")
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read video: {video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        validate_frames(motion["frame_ids"], fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        if not np.isclose(fps, motion["fps"], rtol=1e-4):
            raise ValueError("Video fps does not match the SMPL export")
    finally:
        cap.release()
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f"Output is not empty: {output}")
    robot_points, robot_edges = None, None
    if robot:
        with np.load(robot, allow_pickle=False) as r:
            if not np.array_equal(r["frame_ids"], motion["frame_ids"]):
                raise ValueError("Robot preview frame_ids do not match the SMPL export")
            digest = file_sha256(input_path)
            if str(r["source_sha256"]) != digest:
                raise ValueError("Robot preview was generated from another SMPL file")
            p, e = r["points"].copy(), r["edges"]
            if p.ndim != 3 or p.shape[0] != len(motion["frame_ids"]) or p.shape[2] != 3:
                raise ValueError("Invalid robot preview dimensions")
            if not np.isfinite(p).all() or e.ndim != 2 or e.shape[1] != 2:
                raise ValueError("Invalid robot preview points or edges")
            if e.dtype.kind not in "iu" or np.any(e < 0) or np.any(e >= p.shape[1]):
                raise ValueError("Invalid robot edge indices")
            p[:, :, :2] -= p[0, 0, :2].copy()
            robot_points, robot_edges = np.round(p, 4).tolist(), e.tolist()
    points = motion["joints_world"].copy()
    points[:, :, :2] -= points[0, 0, :2].copy()
    vertices = motion.get("vertices_world")
    if vertices is not None:
        vertices = vertices[:, ::8].copy()
        vertices[:, :, :2] -= motion["joints_world"][0, 0, :2]
    data = dict(name=Path(input_path).stem, fps=fps, ids=motion["frame_ids"].tolist(),
                video=quote(os.path.relpath(video, output.resolve()), safe="/"),
                joints=np.round(points, 4).tolist(), edges=SMPL_EDGES,
                mesh=None if vertices is None else np.round(vertices, 4).tolist(),
                robot=robot_points, robot_edges=robot_edges)
    output.mkdir(parents=True, exist_ok=True)
    serialized = json.dumps(data, ensure_ascii=True, allow_nan=False, separators=(",", ":"))
    (output / "motion.js").write_text("window.MOTION_DATA = " + serialized + ";\n")
    template = Path(__file__).with_name("smpl_viewer.html")
    (output / "index.html").write_text(template.read_text())
    return output / "index.html"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robot", type=Path)
    args = parser.parse_args()
    print(export(args.input, args.video, args.output, args.robot))
