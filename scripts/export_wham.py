"""Convert WHAM world-space SMPL parameters into a shared numeric motion NPZ."""
import argparse
import json
from pathlib import Path

import numpy as np

from smpl_data import Y_UP_TO_Z_UP, file_sha256, load_numeric, select_person, validate_frames


def world_parameters(record):
    required = ("pose_world", "trans_world", "betas", "frame_ids")
    if any(key not in record for key in required):
        raise ValueError(f"WHAM export requires {required}")
    ids = np.asarray(record["frame_ids"])
    n = len(ids)
    pose = np.asarray(record["pose_world"], dtype=np.float32)
    trans = np.asarray(record["trans_world"], dtype=np.float32)
    betas = np.asarray(record["betas"], dtype=np.float32)
    if betas.shape in ((10,), (1, 10)):
        betas = np.broadcast_to(betas.reshape(1, 10), (n, 10)).copy()
    if pose.shape != (n, 72) or trans.shape != (n, 3) or betas.shape != (n, 10):
        raise ValueError("Expected pose_world (F,72), trans_world (F,3), betas (F,10)")
    if not all(np.isfinite(a).all() for a in (pose, trans, betas)):
        raise ValueError("Non-finite SMPL parameters")
    return ids, pose, trans, betas


def align_wham_root(joints, vertices, trans, regressor):
    """Match WHAM SMPL.get_output's regressed-hip origin, retaining SMPL-24 joints."""
    if regressor.ndim != 2 or regressor.shape[0] < 13 or regressor.shape[1] != vertices.shape[1]:
        raise ValueError("J_regressor_wham.npy does not match the SMPL model")
    if not np.isfinite(regressor).all():
        raise ValueError("Non-finite WHAM joint regressor")
    hips = np.einsum("jv,fvc->fjc", regressor[[11, 12]], vertices).mean(axis=1)
    offset = hips - trans
    return joints - offset[:, None], vertices - offset[:, None]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("input", "video", "output", "model"):
        parser.add_argument("--" + name, required=True, type=Path)
    parser.add_argument("--person-id")
    parser.add_argument("--vertices", action="store_true")
    args = parser.parse_args()
    if not args.model.is_file():
        parser.error("SMPL_NEUTRAL.pkl is missing; see docs/video-smpl-retarget.md")
    run_path = args.input.parent / "run.json"
    run = json.loads(run_path.read_text()) if run_path.is_file() else {}
    if run.get("coordinate_mode") == "camera-local":
        parser.error("This is a camera-local WHAM run. Re-run WHAM with world tracking for retargeting.")
    if run.get("video_sha256"):
        if file_sha256(args.video) != run["video_sha256"]:
            parser.error("The selected video does not match this WHAM run.")
    regressor_path = args.model.parent.parent / "J_regressor_wham.npy"
    if not regressor_path.is_file():
        parser.error(f"WHAM joint regressor missing: {regressor_path}")
    regressor = np.load(regressor_path, allow_pickle=False)
    import cv2
    import smplx
    import torch
    person, record = select_person(load_numeric(args.input), args.person_id)
    ids, pose, trans, betas = world_parameters(record)
    cap = cv2.VideoCapture(str(args.video))
    try:
        if not cap.isOpened():
            raise ValueError(f"Cannot read video: {args.video}")
        fps = cap.get(cv2.CAP_PROP_FPS)
        validate_frames(ids, fps, int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
    finally:
        cap.release()
    # Reconstruct from world pose/translation: upstream 'verts' are camera-space.
    model = smplx.SMPL(str(args.model), gender="neutral", batch_size=64).eval()
    joints, vertices = [], []
    with torch.no_grad():
        for start in range(0, len(ids), 64):
            sl = slice(start, start + 64)
            body = model(
                global_orient=torch.from_numpy(pose[sl, :3]),
                body_pose=torch.from_numpy(pose[sl, 3:]),
                transl=torch.from_numpy(trans[sl]), betas=torch.from_numpy(betas[sl]),
            )
            j, v = align_wham_root(body.joints[:, :24].cpu().numpy(),
                                   body.vertices.cpu().numpy(), trans[sl], regressor)
            joints.append(j @ Y_UP_TO_Z_UP.T)
            if args.vertices:
                vertices.append(v @ Y_UP_TO_Z_UP.T)
    data = dict(
        fps=np.array([fps]), frame_ids=ids, joints_world=np.concatenate(joints),
        coordinate_system=np.array("world_z_up"), person_id=np.array(person),
        source_video=np.array(str(args.video.resolve())),
        source_video_sha256=np.array(file_sha256(args.video)),
    )
    if vertices:
        data["vertices_world"] = np.concatenate(vertices)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **data)
    print(f"Exported {len(ids)} frames for person {person}: {args.output}")


if __name__ == "__main__":
    main()
