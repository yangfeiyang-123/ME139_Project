"""CPU regression coverage for the video/SMPL/robot boundary."""
import json
from pathlib import Path
import pickle
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from export_wham import align_wham_root, world_parameters
from retarget_smpl import PositionIK, body_frames, retarget
from smpl_data import (Y_UP_TO_Z_UP, load_motion, load_numeric, select_person,
                       validate_frames)
from visualize_smpl import export
from wham import ASSETS, infer_command, missing_assets


URDF = """<robot name="two_link">
<link name="base"/><link name="arm"/><link name="forearm"/><link name="tip"/>
<joint name="a" type="revolute"><parent link="base"/><child link="arm"/>
<axis xyz="0 0 1"/><limit lower="-1" upper="1" effort="10" velocity="3"/></joint>
<joint name="b" type="revolute"><parent link="arm"/><child link="forearm"/>
<origin xyz="1 0 0"/><axis xyz="0 0 1"/>
<limit lower="-1" upper="1" effort="10" velocity="3"/></joint>
<joint name="end" type="fixed"><parent link="forearm"/><child link="tip"/>
<origin xyz="1 0 0"/></joint></robot>"""


def simple_config(urdf):
    return dict(
        urdf=str(urdf), joint_order=["a", "b"],
        targets=[dict(joint=20, frame="tip", weight=1.)],
        preview_frames=["base", "arm", "forearm", "tip"],
        preview_edges=[[0, 1], [1, 2], [2, 3]],
        leg_frames=["base", "forearm", "tip"], ground_frames=["tip"],
        posture_weight=1e-8, temporal_weight=1e-8,
        ground_clearance=.03, max_evaluations=100,
    )


def motion_data():
    joints = np.zeros((3, 24, 3))
    joints[:, 1, 1] = .1
    joints[:, 2, 1] = -.1
    joints[:, 12, 2] = 1.
    joints[:, 4] = [0, .1, -.4]
    joints[:, 7] = [0, .1, -.8]
    joints[:, 20] = [1.8, .3, 0]
    return dict(joints_world=joints, frame_ids=np.array([0, 1, 3]),
                fps=np.array([30.]), coordinate_system=np.array("world_z_up"))


class MotionPipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="me139 motion ")
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)

    def save_motion(self, **updates):
        data = motion_data()
        data.update(updates)
        path = self.root / "motion.npz"
        np.savez_compressed(path, **data)
        return path

    def test_world_transform_is_right_handed_and_maps_up(self):
        np.testing.assert_allclose(Y_UP_TO_Z_UP @ [0, 1, 0], [0, 0, 1])
        np.testing.assert_allclose(Y_UP_TO_Z_UP.T @ Y_UP_TO_Z_UP, np.eye(3))
        self.assertAlmostEqual(np.linalg.det(Y_UP_TO_Z_UP), 1.)
        r = body_frames(motion_data()["joints_world"])
        np.testing.assert_allclose(r, np.repeat(np.eye(3)[None], 3, axis=0))

    def test_invalid_frames_are_rejected(self):
        for ids in ([0, 0, 1], [-1, 0, 1], [0., 1., 2.], [0]):
            with self.subTest(ids=ids), self.assertRaises(ValueError):
                validate_frames(np.array(ids), 30.)
        with self.assertRaises(ValueError):
            validate_frames(np.array([0, 9]), 30., count=9)

    def test_multiple_people_require_selection(self):
        data = {0: {"pose_world": 1}, 7: {"pose_world": 2}}
        with self.assertRaisesRegex(ValueError, "Multiple"):
            select_person(data)
        self.assertEqual(select_person(data, "7"), ("7", data[7]))

    def test_world_parameters_never_use_camera_pose(self):
        data = dict(pose=np.ones((3, 72)), pose_world=np.zeros((3, 72)),
                    trans_world=np.zeros((3, 3)), betas=np.zeros(10),
                    frame_ids=np.array([0, 1, 3]))
        _, pose, trans, betas = world_parameters(data)
        self.assertEqual(pose.sum(), 0)
        self.assertEqual(betas.shape, (3, 10))
        data.pop("pose_world")
        with self.assertRaises(ValueError):
            world_parameters(data)

    def test_wham_hip_origin_matches_world_translation(self):
        vertices = np.array([[[8., 2., 1.], [10., 2., 1.], [9., 3., 1.]]])
        joints = np.repeat(vertices[:, :1], 24, axis=1)
        regressor = np.zeros((17, 3))
        regressor[11, 0] = regressor[12, 1] = 1.
        trans = np.array([[1., 2., 3.]])
        j, v = align_wham_root(joints, vertices, trans, regressor)
        np.testing.assert_allclose(v[:, :2].mean(axis=1), trans)
        np.testing.assert_allclose(j[:, 0] - v[:, 0], 0.)

    def test_numeric_reader_rejects_executable_globals(self):
        path = self.root / "bad.pkl"
        path.write_bytes(pickle.dumps(eval))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            load_numeric(path)

    def test_motion_format_rejects_wrong_space_and_nonfinite_values(self):
        with self.assertRaisesRegex(ValueError, "coordinate_system"):
            load_motion(self.save_motion(coordinate_system=np.array("camera")))
        bad = motion_data()["joints_world"]
        bad[0, 1, 2] = np.nan
        with self.assertRaisesRegex(ValueError, "finite"):
            load_motion(self.save_motion(joints_world=bad))

    def test_ik_matches_reachable_target_and_enforces_limits(self):
        urdf = self.root / "robot.urdf"
        urdf.write_text(URDF)
        solver = PositionIK(urdf, simple_config(urdf))
        target = solver.positions(np.array([.4, -.6]), ["tip"])
        q, error, ok = solver.solve(target, np.array([.1, -.1]))
        self.assertTrue(ok)
        self.assertLess(error.max(), 1e-3)
        q, error, _ = solver.solve(np.array([[10., 10., 0.]]), q)
        self.assertTrue(np.all(q >= solver.lower))
        self.assertTrue(np.all(q <= solver.upper))
        self.assertGreater(error.min(), 1.)

    def test_retarget_preserves_time_axis_and_csv_contract(self):
        urdf = self.root / "robot.urdf"
        urdf.write_text(URDF)
        config = self.root / "config.json"
        config.write_text(json.dumps(simple_config(urdf)))
        source = self.save_motion()
        report = retarget(source, self.root / "result", config, scale=1.)
        csv = np.loadtxt(self.root / "result/motion.csv", delimiter=",")
        self.assertEqual(csv.shape, (4, 9))
        np.testing.assert_allclose(np.linalg.norm(csv[:, 3:7], axis=1), 1.)
        self.assertEqual(report["joint_order"], ["a", "b"])
        self.assertTrue(report["joint_limits_satisfied"])
        with np.load(self.root / "result/preview.npz") as preview:
            np.testing.assert_array_equal(preview["frame_ids"], [0, 1, 3])
        with self.assertRaisesRegex(ValueError, "not empty"):
            retarget(source, self.root / "result", config, scale=1.)

    def test_long_tracking_gaps_are_not_silently_interpolated(self):
        source = self.save_motion(frame_ids=np.array([0, 1, 90]))
        with self.assertRaisesRegex(ValueError, "Tracking gap"):
            retarget(source, self.root / "result", self.root / "unused.json")

    def test_wham_download_manifest_and_argument_boundaries(self):
        missing = missing_assets(self.root, local_only=True)
        self.assertTrue(missing)
        self.assertFalse(any(x["group"] == "global" for x in missing))
        paths = [x["path"] for x in json.loads(ASSETS.read_text())["assets"]]
        self.assertEqual(len(paths), len(set(paths)))
        command = infer_command("/env with spaces/python", Path("video with spaces.mp4"),
                                self.root, local_only=False)
        self.assertIn("--save_pkl", command)
        self.assertEqual(command[0], "/env with spaces/python")
        self.assertEqual(command[command.index("--video")+1], str(Path("video with spaces.mp4").resolve()))
        self.assertNotIn("--estimate_local_only", command)

    def test_viewer_uses_encoded_video_path_and_checks_alignment(self):
        import cv2
        video = self.root / "video #1.avi"
        writer = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"MJPG"), 30., (32, 32))
        self.assertTrue(writer.isOpened())
        for _ in range(4):
            writer.write(np.zeros((32, 32, 3), dtype=np.uint8))
        writer.release()
        source = self.save_motion()
        page = export(source, video, self.root / "viewer")
        self.assertTrue(page.is_file())
        self.assertIn("video%20%231.avi", (page.parent / "motion.js").read_text())
        hashed_source = self.save_motion(source_video_sha256=np.array("0" * 64))
        with self.assertRaisesRegex(ValueError, "Video does not match"):
            export(hashed_source, video, self.root / "wrong-video")
        source = self.save_motion()
        preview = self.root / "mismatch.npz"
        np.savez(preview, frame_ids=np.array([0, 1, 2]))
        with self.assertRaisesRegex(ValueError, "frame_ids"):
            export(source, video, self.root / "another-viewer", preview)


if __name__ == "__main__":
    unittest.main()
