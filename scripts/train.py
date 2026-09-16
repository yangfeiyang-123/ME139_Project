"""Local BeyondMimic training / checkpoint playback on Isaac Lab 2.3.2 + RSL-RL 3.1.2."""
import argparse
from runtime import finish
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--motion", required=True, type=Path, help="Local BeyondMimic motion NPZ (50 Hz)")
parser.add_argument("--task", default="Tracking-Flat-G1-v0")
parser.add_argument("--num_envs", type=int, default=256)
parser.add_argument("--max_iterations", type=int, default=1000)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--run_name", default="local")
parser.add_argument("--checkpoint", type=Path, help="Resume or play this local checkpoint")
parser.add_argument("--play", action="store_true")
parser.add_argument("--steps", type=int, default=500)
AppLauncher.add_app_launcher_args(parser)
args = parser.parse_args()
if not args.motion.is_file():
    parser.error(f"Motion file does not exist: {args.motion}")
if args.play and not args.checkpoint:
    parser.error("--play requires --checkpoint")
if args.num_envs < 1 or args.max_iterations < 1 or args.steps < 1:
    parser.error("Environment, iteration and step counts must be positive")
launcher = AppLauncher(args, multi_gpu=False)
app = launcher.app

import gymnasium as gym
import numpy as np
import torch
from isaaclab.utils.io import dump_yaml
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab_tasks.utils import load_cfg_from_registry
from rsl_rl.runners import OnPolicyRunner
import whole_body_tracking.tasks  # registers the upstream environments


def main():
    torch.set_num_threads(4)
    torch.manual_seed(args.seed)
    env_cfg = load_cfg_from_registry(args.task, "env_cfg_entry_point")
    agent_cfg = load_cfg_from_registry(args.task, "rsl_rl_cfg_entry_point")
    env_cfg.scene.num_envs = args.num_envs
    env_cfg.seed = agent_cfg.seed = args.seed
    env_cfg.sim.device = agent_cfg.device = args.device
    env_cfg.commands.motion.motion_file = str(args.motion.resolve())
    # Local rendering assets avoid cloud material downloads; physics is unchanged.
    env_cfg.scene.terrain.visual_material = None
    env_cfg.commands.motion.debug_vis = False
    env_cfg.scene.contact_forces.debug_vis = False
    agent_cfg.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
    agent_cfg.policy.actor_obs_normalization = True
    agent_cfg.policy.critic_obs_normalization = True
    agent_cfg.empirical_normalization = None
    agent_cfg.logger = "tensorboard"
    agent_cfg.max_iterations = args.max_iterations
    agent_cfg.run_name = args.run_name
    with np.load(args.motion, allow_pickle=False) as motion:
        expected_fps = 1.0 / (env_cfg.decimation * env_cfg.sim.dt)
        if not np.isclose(float(motion["fps"].reshape(-1)[0]), expected_fps):
            raise ValueError(f"Motion fps must match policy rate: {expected_fps}")
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if not np.isfinite(motion[key]).all():
                raise ValueError(f"Non-finite motion field: {key}")
        if motion["joint_pos"].shape[1] != 29:
            raise ValueError("Expected the BeyondMimic G1 29-joint motion layout")
    log_dir = Path("logs/rsl_rl") / agent_cfg.experiment_name / (datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + "_" + args.run_name)
    log_dir.mkdir(parents=True, exist_ok=False)
    dump_yaml(str(log_dir / "env.yaml"), env_cfg)
    dump_yaml(str(log_dir / "agent.yaml"), agent_cfg)
    env = gym.make(args.task, cfg=env_cfg)
    wrapped = RslRlVecEnvWrapper(env)
    try:
        obs = wrapped.get_observations()
        assert all(torch.isfinite(v).all() for v in obs.values()), "Non-finite observations"
        runner = OnPolicyRunner(wrapped, agent_cfg.to_dict(), log_dir=str(log_dir), device=args.device)
        if args.checkpoint:
            runner.load(str(args.checkpoint.resolve()), load_optimizer=not args.play)
        if args.play:
            policy = runner.get_inference_policy(device=args.device)
            with torch.inference_mode():
                for _ in range(args.steps):
                    if not app.is_running():
                        break
                    obs, rewards, dones, extras = wrapped.step(policy(obs))
                    assert torch.isfinite(rewards).all(), "Non-finite rewards"
                    assert all(torch.isfinite(v).all() for v in obs.values()), "Non-finite observations"
            print("PLAYBACK_PASSED")
        else:
            runner.learn(num_learning_iterations=args.max_iterations, init_at_random_ep_len=True)
            assert all(torch.isfinite(p).all() for p in runner.alg.policy.parameters()), "Non-finite policy parameters"
            checkpoint = log_dir / "model_final.pt"
            runner.save(str(checkpoint))
            result = {"status": "passed", "task": args.task, "num_envs": args.num_envs,
                      "iterations": args.max_iterations, "seed": args.seed, "device": args.device,
                      "observation_shapes": {k: list(v.shape) for k, v in obs.items()},
                      "action_count": wrapped.num_actions, "motion": str(args.motion.resolve()),
                      "motion_sha256": hashlib.sha256(args.motion.read_bytes()).hexdigest(),
                      "checkpoint": str(checkpoint.resolve())}
            (log_dir / "result.json").write_text(json.dumps(result, indent=2) + "\n")
            print("TRAINING_PASSED", json.dumps(result))
    finally:
        if "runner" in locals() and runner.writer is not None:
            runner.writer.flush()
            runner.writer.close()
        wrapped.close()


try:
    main()
except BaseException:
    import traceback
    traceback.print_exc()
    finish(1)
else:
    finish(0)
