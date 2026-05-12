from __future__ import annotations

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import euler_xyz_from_quat, quat_rotate_inverse


def _robot(env: ManagerBasedRLEnv) -> Articulation:
    return env.scene["robot"]


def survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def zero_term(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.zeros(env.num_envs, device=env.device)


def tracking_lin_vel_x(env: ManagerBasedRLEnv, sigma: float = 0.25) -> torch.Tensor:
    robot = _robot(env)
    cmd = torch.zeros(env.num_envs, device=env.device) + 0.05
    return torch.exp(-torch.square(cmd - robot.data.root_lin_vel_b[:, 0]) / sigma)


def tracking_lin_vel_y(env: ManagerBasedRLEnv, sigma: float = 0.25) -> torch.Tensor:
    robot = _robot(env)
    return torch.exp(-torch.square(robot.data.root_lin_vel_b[:, 1]) / sigma)


def tracking_ang_vel(env: ManagerBasedRLEnv, sigma: float = 0.25) -> torch.Tensor:
    robot = _robot(env)
    return torch.exp(-torch.square(robot.data.root_ang_vel_b[:, 2]) / sigma)


def base_height(env: ManagerBasedRLEnv, target: float = 0.52) -> torch.Tensor:
    # Bootstrap-safe: clip height error so fallen/flying states do not explode the value loss.
    z = _robot(env).data.root_pos_w[:, 2]
    err = torch.clamp(z - target, min=-0.8, max=0.8)
    return torch.square(err)


def orientation(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    roll, pitch, _ = euler_xyz_from_quat(robot.data.root_quat_w)
    return torch.square(roll) + torch.square(pitch)


def torques(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.applied_torque), dim=-1)


def torque_tiredness(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    lim = torch.clamp(robot.data.joint_effort_limits, min=1e-6)
    return torch.sum(torch.square((robot.data.applied_torque / lim).clip(max=1.0)), dim=-1)


def power(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    return torch.sum((robot.data.applied_torque * robot.data.joint_vel).clip(min=0.0), dim=-1)


def lin_vel_z(env: ManagerBasedRLEnv) -> torch.Tensor:
    # Bootstrap-safe: clip vertical velocity penalty.
    vz = torch.clamp(_robot(env).data.root_lin_vel_b[:, 2], min=-5.0, max=5.0)
    return torch.square(vz)


def ang_vel_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.root_ang_vel_b[:, :2]), dim=-1)


def dof_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(_robot(env).data.joint_vel), dim=-1)


def action_rate(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.sum(torch.square(env.action_manager.action - env.action_manager.prev_action), dim=-1)


def _match_foot_body_ids(robot: Articulation) -> tuple[int, int]:
    """Resolve (left_foot_id, right_foot_id) using robust name matching."""
    body_names = getattr(robot.data, "body_names", robot.body_names)
    names_lower = [name.lower() for name in body_names]

    left_candidates: list[int] = []
    right_candidates: list[int] = []
    for i, name in enumerate(names_lower):
        has_foot_token = ("foot" in name) or ("ankle" in name)
        if not has_foot_token:
            continue
        if "left" in name:
            left_candidates.append(i)
        if "right" in name:
            right_candidates.append(i)

    if not left_candidates or not right_candidates:
        raise RuntimeError(
            "Could not match left/right foot bodies. Expected names containing "
            "'left'/'right' with 'foot' or 'ankle'."
        )
    return left_candidates[0], right_candidates[0]


def step_length_reward(
    env: ManagerBasedRLEnv,
    target_min: float = 0.16,
    target_max: float = 0.34,
    min_vx: float = 0.20,
) -> torch.Tensor:
    """Reward forward left-right foot separation in the base heading frame."""
    robot = _robot(env)

    # Cache body indices once; robust to common naming variants.
    if not hasattr(env, "_k1_fullbody_step_foot_ids"):
        env._k1_fullbody_step_foot_ids = _match_foot_body_ids(robot)
    left_foot_id, right_foot_id = env._k1_fullbody_step_foot_ids

    left_pos_w = robot.data.body_pos_w[:, left_foot_id, :]
    right_pos_w = robot.data.body_pos_w[:, right_foot_id, :]
    feet_sep_w = left_pos_w - right_pos_w

    # Project world-frame foot separation into base frame; x is forward.
    feet_sep_b = quat_rotate_inverse(robot.data.root_quat_w, feet_sep_w)
    stride = torch.abs(feet_sep_b[:, 0])

    span = max(target_max - target_min, 1e-6)
    stride_reward = torch.clamp((stride - target_min) / span, min=0.0, max=1.0)

    # Gate by forward walking intent to avoid rewarding static wide stance.
    vx = robot.data.root_lin_vel_b[:, 0]
    forward_gate = (vx > min_vx).float()
    return stride_reward * forward_gate


def foot_width_reward(
    env: ManagerBasedRLEnv,
    target_min: float = 0.10,
    target_max: float = 0.22,
    ideal: float = 0.16,
) -> torch.Tensor:
    """Reward natural lateral foot spacing and suppress feet-touching gait."""
    robot = _robot(env)
    if not hasattr(env, "_k1_fullbody_step_foot_ids"):
        env._k1_fullbody_step_foot_ids = _match_foot_body_ids(robot)
    left_foot_id, right_foot_id = env._k1_fullbody_step_foot_ids

    left_pos_w = robot.data.body_pos_w[:, left_foot_id, :]
    right_pos_w = robot.data.body_pos_w[:, right_foot_id, :]
    feet_sep_w = left_pos_w - right_pos_w
    feet_sep_b = quat_rotate_inverse(robot.data.root_quat_w, feet_sep_w)
    width = torch.abs(feet_sep_b[:, 1])

    reward = torch.exp(-torch.square(width - ideal) / 0.01)
    in_range = (width >= target_min) & (width <= target_max)
    reward = reward * in_range.float()
    return torch.clamp(reward, min=0.0, max=1.0)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    min_clearance: float = 0.025,
    max_clearance: float = 0.09,
    min_vx: float = 0.20,
) -> torch.Tensor:
    """Conservative anti-shuffle proxy using left/right foot height difference."""
    robot = _robot(env)
    if not hasattr(env, "_k1_fullbody_step_foot_ids"):
        env._k1_fullbody_step_foot_ids = _match_foot_body_ids(robot)
    left_foot_id, right_foot_id = env._k1_fullbody_step_foot_ids

    left_z = robot.data.body_pos_w[:, left_foot_id, 2]
    right_z = robot.data.body_pos_w[:, right_foot_id, 2]
    clearance = torch.abs(left_z - right_z)

    span = max(max_clearance - min_clearance, 1e-6)
    reward = torch.clamp((clearance - min_clearance) / span, min=0.0, max=1.0)
    vx = robot.data.root_lin_vel_b[:, 0]
    moving = (vx > min_vx).float()
    return reward * moving


def base_height_reward(env: ManagerBasedRLEnv, target: float = 0.52) -> torch.Tensor:
    return base_height(env, target=target)


def orientation_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return orientation(env)


def lin_vel_z_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return lin_vel_z(env)


def ang_vel_xy_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return ang_vel_xy(env)


def torque_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torques(env)


def action_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return action_rate(env)


def hip_roll_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize excessive hip ab/adduction to reduce wide, asymmetric shuffling."""
    robot = _robot(env)
    names = robot.data.joint_names
    idx = {n: i for i, n in enumerate(names)}
    pen = torch.zeros(env.num_envs, device=env.device)
    for jn in ("Left_Hip_Roll", "Right_Hip_Roll"):
        if jn in idx:
            pen = pen + torch.square(robot.data.joint_pos[:, idx[jn]])
    return pen


def dof_vel_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Light L2 joint-velocity penalty (all DoF) to damp flailing."""
    return dof_vel(env)
