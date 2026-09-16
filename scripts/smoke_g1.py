"""Load the actual BeyondMimic G1, check GPU physics, and create a static test reference."""
import argparse
from runtime import finish
import json
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--steps", type=int, default=200)
parser.add_argument("--output", default="data/motions/g1_stand_smoke.npz")
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
launcher = AppLauncher(args, multi_gpu=False)
app = launcher.app

import numpy as np
import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.utils import configclass
from whole_body_tracking.robots.g1 import G1_CYLINDER_CFG


@configclass
class SceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/Ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/Light", spawn=sim_utils.DomeLightCfg(intensity=1500))
    robot: ArticulationCfg = G1_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def main():
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(dt=0.005, device=args.device))
    sim.set_camera_view([2.5, 2.5, 1.8], [0, 0, 0.7])
    scene = InteractiveScene(SceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    robot = scene["robot"]
    assert robot.num_joints == 29, robot.joint_names
    root = robot.data.default_root_state.clone()
    robot.write_root_state_to_sim(root)
    robot.write_joint_state_to_sim(robot.data.default_joint_pos, robot.data.default_joint_vel)
    scene.reset()
    sim.render()
    scene.update(sim.get_physics_dt())
    # Synthetic static pose, only for plumbing / PPO smoke tests, not a learned skill.
    fields = {
        "joint_pos": robot.data.joint_pos,
        "joint_vel": robot.data.joint_vel,
        "body_pos_w": robot.data.body_pos_w,
        "body_quat_w": robot.data.body_quat_w,
        "body_lin_vel_w": robot.data.body_lin_vel_w,
        "body_ang_vel_w": robot.data.body_ang_vel_w,
    }
    motion = {k: np.repeat(v[0].cpu().numpy()[None], 250, axis=0) for k, v in fields.items()}
    output = Path(args.output).resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, fps=np.array([50]), **motion)
    min_height = float(root[0, 2])
    for _ in range(args.steps):
        robot.set_joint_position_target(robot.data.default_joint_pos)
        scene.write_data_to_sim()
        sim.step()
        scene.update(sim.get_physics_dt())
        assert torch.isfinite(robot.data.root_state_w).all(), "Non-finite root state"
        assert torch.isfinite(robot.data.joint_pos).all(), "Non-finite joints"
        min_height = min(min_height, float(robot.data.root_pos_w[0, 2]))
    assert min_height > 0.45, f"G1 fell during default-pose PD test: {min_height}"
    report = {"status": "passed", "device": str(sim.device), "steps": args.steps,
              "joint_count": robot.num_joints, "body_count": robot.num_bodies,
              "joint_names": robot.joint_names, "body_names": robot.body_names,
              "minimum_root_height_m": min_height, "motion": str(output)}
    Path("manifests").mkdir(exist_ok=True)
    Path("manifests/g1-smoke.json").write_text(json.dumps(report, indent=2) + "\n")
    print("G1_SMOKE_PASSED", json.dumps(report))


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    finish(1)
else:
    finish(0)
