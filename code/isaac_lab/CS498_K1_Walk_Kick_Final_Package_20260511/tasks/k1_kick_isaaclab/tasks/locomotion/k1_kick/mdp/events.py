"""Reset-time randomization for ball pose (curriculum) and velocities."""

from __future__ import annotations

import torch
from isaaclab.assets import RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import math as math_utils


def reset_ball_relative_to_robot(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    z_offset: float = 0.11,
    curriculum_steps_per_stage: int = 150_000,
):
    """Place the ball in front of the robot; difficulty from ``env.cfg`` fields.

    Reads ``kick_curriculum_stage`` (1..5), ``kick_curriculum_auto``, and optional
    ``kick_curriculum_steps_per_stage`` from the environment config.
    """
    robot = env.scene[robot_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    cfg = env.cfg
    auto = bool(getattr(cfg, "kick_curriculum_auto", False))
    step_stage = int(getattr(cfg, "kick_curriculum_steps_per_stage", curriculum_steps_per_stage))
    if auto:
        step = int(getattr(env, "common_step_counter", 0))
        stage = int(min(5, max(1, 1 + step // max(1, step_stage))))
    else:
        stage = int(getattr(cfg, "kick_curriculum_stage", 1))
        stage = max(1, min(5, stage))

    root_pos = robot.data.root_pos_w[env_ids].clone()
    root_quat = robot.data.root_quat_w[env_ids].clone()

    n = len(env_ids)
    dev = env.device
    if stage <= 1:
        rx = tuple(getattr(cfg, "kick_stage1_ball_x_range", (0.48, 0.62)))
        ry = tuple(getattr(cfg, "kick_stage1_ball_y_range", (-0.02, 0.02)))
    elif stage == 2:
        rx = (0.42, 0.68)
        ry = (-0.12, 0.12)
    elif stage == 3:
        rx = (0.38, 0.78)
        ry = (-0.18, 0.18)
    elif stage == 4:
        rx = (0.35, 0.85)
        ry = (-0.22, 0.22)
    else:
        rx = (0.32, 0.95)
        ry = (-0.28, 0.28)

    local = torch.zeros(n, 3, device=dev)
    local[:, 0] = math_utils.sample_uniform(rx[0], rx[1], (n,), dev)
    local[:, 1] = math_utils.sample_uniform(ry[0], ry[1], (n,), dev)
    local[:, 2] = 0.0

    offset_w = math_utils.quat_apply(root_quat, local)
    pos = root_pos + offset_w
    pos[:, 2] = env.scene.env_origins[env_ids, 2] + z_offset

    default_root = ball.data.default_root_state[env_ids].clone()
    quat = default_root[:, 3:7]
    pose = torch.cat([pos, quat], dim=-1)
    ball.write_root_pose_to_sim(pose, env_ids=env_ids)

    zero_vel = torch.zeros(n, 6, device=dev)
    ball.write_root_velocity_to_sim(zero_vel, env_ids=env_ids)
