"""Kinematic visualization of a local BeyondMimic reference (not policy evaluation)."""
import argparse
from runtime import finish
from pathlib import Path
from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, type=Path)
parser.add_argument("--steps", type=int, default=500)
parser.add_argument("--video", type=Path, help="Save a headless-compatible MP4 and a PNG preview.")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.motion.is_file():
    parser.error(f"Missing motion: {args.motion}")
if args.steps < 1:
    parser.error("--steps must be positive")
if args.video:
    args.enable_cameras = True
launcher = AppLauncher(args, multi_gpu=False)
app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG

@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Camera", height=720, width=960,
        update_period=0.0, data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(focal_length=24.0, horizontal_aperture=24.0,
                                       clipping_range=(0.1, 100.0)),
    ) if args.video else None

def main():
    with np.load(args.motion, allow_pickle=False) as data:
        fps = float(data["fps"].reshape(-1)[0])
        motion = {k: torch.tensor(data[k], dtype=torch.float32, device=args.device)
                  for k in data.files if k != "fps"}
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=1/fps, device=args.device))
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    assert motion["joint_pos"].shape[1] == robot.num_joints
    assert motion["body_pos_w"].shape[1] == robot.num_bodies
    writer = None
    if args.video:
        import imageio.v2 as imageio
        args.video.parent.mkdir(parents=True, exist_ok=True)
        writer = imageio.get_writer(str(args.video), fps=fps, codec="libx264", quality=8,
                                   macro_block_size=16, output_params=["-movflags", "+faststart"])
    try:
        for step in range(args.steps):
            if not app.is_running():
                break
            i = step % len(motion["joint_pos"])
            state = robot.data.default_root_state.clone()
            state[0, :3] = motion["body_pos_w"][i, 0]
            state[0, 3:7] = motion["body_quat_w"][i, 0]
            state[0, 7:10] = motion["body_lin_vel_w"][i, 0]
            state[0, 10:13] = motion["body_ang_vel_w"][i, 0]
            robot.write_root_state_to_sim(state)
            robot.write_joint_state_to_sim(motion["joint_pos"][i:i+1], motion["joint_vel"][i:i+1])
            target = state[0, :3].cpu().numpy()
            target[2] = 0.7
            eye = target + np.array([2.3, 2.3, 0.8])
            sim.set_camera_view(eye, target)
            if args.video:
                scene["camera"].set_world_poses_from_view(
                    torch.tensor(eye[None], dtype=torch.float32, device=sim.device),
                    torch.tensor(target[None], dtype=torch.float32, device=sim.device),
                )
            # Warm up RTX without advancing the reference motion.
            for _ in range(12 if step == 0 and args.video else 1):
                sim.render()
            scene.update(1/fps)
            if writer is not None:
                frame = scene["camera"].data.output["rgb"][0, :, :, :3].cpu().numpy()
                writer.append_data(frame)
                if step == args.steps // 2:
                    imageio.imwrite(args.video.with_suffix(".png"), frame)
                if step % 100 == 0:
                    print(f"[VIDEO] {step}/{args.steps}")
    finally:
        if writer is not None:
            writer.close()
    if args.video:
        print(f"VIDEO_SAVED: {args.video.resolve()}")
    print("REFERENCE_REPLAY_PASSED")

try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    finish(1)
else:
    finish(0)
