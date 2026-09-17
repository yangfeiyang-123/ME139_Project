"""Position-based, joint-limited SMPL-to-G1 retargeting; outputs motion CSV and a preview."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from smpl_data import file_sha256, load_motion

ROOT = Path(__file__).resolve().parents[1]


def body_frames(joints):
    """Construct a right-handed pelvis frame: X forward, Y left, Z towards neck."""
    y = joints[:, 1] - joints[:, 2]
    up = joints[:, 12] - joints[:, 0]
    ynorm = np.linalg.norm(y, axis=1, keepdims=True)
    if np.any(ynorm < 1e-5):
        raise ValueError("Degenerate hip positions")
    y /= ynorm
    z = up - (up * y).sum(axis=1, keepdims=True) * y
    znorm = np.linalg.norm(z, axis=1, keepdims=True)
    if np.any(znorm < 1e-5):
        raise ValueError("Degenerate spine/hip geometry")
    z /= znorm
    x = np.cross(y, z)
    return np.stack([x, y, z], axis=-1)


class PositionIK:
    def __init__(self, urdf, config):
        import pinocchio as pin
        self.pin = pin
        self.config = config
        self.model = pin.buildModelFromUrdf(str(urdf))
        self.data = self.model.createData()
        if self.model.nq != self.model.nv:
            raise ValueError("Expected a fixed-base model with scalar joint coordinates")
        self.neutral = pin.neutral(self.model)
        self.lower = self.model.lowerPositionLimit
        self.upper = self.model.upperPositionLimit
        self.names = [item["frame"] for item in config["targets"]]
        self.target_ids = [self.frame_id(name) for name in self.names]
        self.weights = np.sqrt([item["weight"] for item in config["targets"]])
        if not np.isfinite(self.weights).all() or np.any(self.weights <= 0):
            raise ValueError("Target weights must be finite and positive")
        self.posture = float(config["posture_weight"]) ** 0.5
        self.temporal = float(config["temporal_weight"]) ** 0.5

    def frame_id(self, name):
        frame = self.model.getFrameId(name)
        if frame >= len(self.model.frames):
            raise ValueError(f"Robot frame not found: {name}")
        return frame

    def positions(self, q, names):
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        return np.array([self.data.oMf[self.frame_id(name)].translation.copy() for name in names])

    def solve(self, target, previous):
        def residual(q):
            points = self.positions(q, self.names)
            return np.concatenate([
                ((points - target) * self.weights[:, None]).ravel(),
                self.posture * (q - self.neutral),
                self.temporal * (q - previous),
            ])

        def jacobian(q):
            self.pin.computeJointJacobians(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            j = np.vstack([
                self.weights[i] * self.pin.getFrameJacobian(
                    self.model, self.data, frame, self.pin.LOCAL_WORLD_ALIGNED)[:3]
                for i, frame in enumerate(self.target_ids)
            ])
            return np.vstack([j, self.posture * np.eye(self.model.nq),
                              self.temporal * np.eye(self.model.nq)])

        result = least_squares(
            residual, np.clip(previous, self.lower + 1e-8, self.upper - 1e-8),
            jac=jacobian, bounds=(self.lower, self.upper),
            max_nfev=self.config["max_evaluations"],
            ftol=1e-6, xtol=1e-6, gtol=1e-6,
        )
        if not np.isfinite(result.x).all():
            raise ValueError("IK produced non-finite joint coordinates")
        error = np.linalg.norm(self.positions(result.x, self.names) - target, axis=1)
        return result.x, error, result.success


def retarget(input_path, output_dir, config_path, output_fps=30., scale=None, max_gap=0.25):
    if not np.isfinite(output_fps) or output_fps <= 0:
        raise ValueError("output_fps must be finite and positive")
    motion = load_motion(input_path)
    ids, fps, joints = motion["frame_ids"], motion["fps"], motion["joints_world"]
    if not np.isfinite(max_gap) or max_gap <= 0:
        raise ValueError("max_gap must be finite and positive")
    if np.max(np.diff(ids) / fps) > max_gap:
        raise ValueError("Tracking gap exceeds max_gap; split the sequence instead of interpolating a long gap")
    config = json.loads(Path(config_path).read_text())
    urdf = Path(config["urdf"])
    if not urdf.is_absolute():
        urdf = ROOT / urdf
    solver = PositionIK(urdf, config)
    if scale is None:
        robot_leg = solver.positions(solver.neutral, config["leg_frames"])
        robot_length = np.linalg.norm(np.diff(robot_leg, axis=0), axis=1).sum()
        human_length = (np.linalg.norm(joints[:, 4] - joints[:, 1], axis=1)
                        + np.linalg.norm(joints[:, 7] - joints[:, 4], axis=1))
        scale = robot_length / np.median(human_length)
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("Human-to-robot scale must be finite and positive")
    rotations = body_frames(joints)
    roots = (joints[:, 0] - joints[0, 0]) * scale
    local = np.einsum("fji,fkj->fki", rotations, joints - joints[:, :1]) * scale
    source_indices = [item["joint"] for item in config["targets"]]
    poses, previews, feet, errors, success = [], [], [], [], []
    previous = solver.neutral.copy()
    for i in range(len(ids)):
        q, error, ok = solver.solve(local[i, source_indices], previous)
        poses.append(q.copy())
        errors.append(error)
        success.append(bool(ok))
        previews.append(solver.positions(q, config["preview_frames"]) @ rotations[i].T + roots[i])
        feet.append(solver.positions(q, config["ground_frames"]) @ rotations[i].T + roots[i])
        previous = q
    poses, previews, feet = np.array(poses), np.array(previews), np.array(feet)
    # One global vertical shift preserves motion, including jumps.
    z_shift = config["ground_clearance"] - feet[:, :, 2].min()
    roots[:, 2] += z_shift
    previews[:, :, 2] += z_shift
    times = (ids - ids[0]) / fps
    grid = np.arange(int(np.floor(times[-1] * output_fps + 1e-8)) + 1) / output_fps
    if len(grid) < 2:
        raise ValueError("Clip is too short for the requested output rate")
    q_grid = np.column_stack([np.interp(grid, times, poses[:, j]) for j in range(solver.model.nq)])
    roots_grid = np.column_stack([np.interp(grid, times, roots[:, j]) for j in range(3)])
    quats = Slerp(times, Rotation.from_matrix(rotations))(grid).as_quat()  # XYZW
    indexes = []
    for name in config["joint_order"]:
        joint_id = solver.model.getJointId(name)
        if joint_id >= solver.model.njoints or solver.model.joints[joint_id].nq != 1:
            raise ValueError(f"Expected scalar robot joint: {name}")
        indexes.append(solver.model.joints[joint_id].idx_q)
    if len(set(indexes)) != solver.model.nq:
        raise ValueError("joint_order must contain every robot joint exactly once")
    csv = np.column_stack([roots_grid, quats, q_grid[:, indexes]])
    output_dir = Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError(f"Output is not empty: {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savetxt(output_dir / "motion.csv", csv, delimiter=",", fmt="%.8f")
    np.savez_compressed(output_dir / "preview.npz", frame_ids=ids, fps=np.array([fps]),
                        points=previews, edges=np.asarray(config["preview_edges"]),
                        source_sha256=np.array(file_sha256(input_path)))
    errors = np.array(errors)
    report = {
        "method": "joint-limited position IK", "source": str(Path(input_path).resolve()),
        "source_frames": len(ids), "output_frames": len(grid), "output_fps": output_fps,
        "scale": float(scale), "mean_target_error_m": float(errors.mean()),
        "max_target_error_m": float(errors.max()), "converged_frames": sum(success),
        "joint_limits_satisfied": bool(np.all(poses >= solver.lower) and np.all(poses <= solver.upper)),
        "joint_order": config["joint_order"],
        "limitations": ["Kinematic reference only; balance and collision constraints are not enforced.",
                        "End-effector orientations and contact/velocity limits are not enforced.",
                        "IK convergence does not establish tracking quality; inspect target errors and preview."],
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--config", type=Path, default=ROOT / "configs/retarget_g1.json")
    parser.add_argument("--output-fps", type=float, default=30.)
    parser.add_argument("--scale", type=float)
    parser.add_argument("--max-gap", type=float, default=0.25, help="Maximum interpolated tracking gap in seconds")
    args = parser.parse_args()
    report = retarget(args.input, args.output, args.config, args.output_fps, args.scale, args.max_gap)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
