"""Shared kick-task geometry, phase labels, and per-step feature cache.

Rewards are computed before observations in ManagerBasedRLEnv; features must be
built inside reward/termination terms. We cache once per env step using
``common_step_counter`` to avoid redundant work across terms.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import quat_rotate_inverse

from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.mdp.rewards import (
    _match_foot_body_ids,
)


def _robot(env: ManagerBasedRLEnv) -> Articulation:
    return env.scene["robot"]


def _ball(env: ManagerBasedRLEnv) -> RigidObject:
    return env.scene["ball"]


@dataclass
class KickFeatures:
    """Vectorized quantities for kick MDP (num_envs, ...)."""

    ball_pos_b: torch.Tensor  # position of ball origin relative to robot base, body frame
    ball_vel_b: torch.Tensor
    ball_vel_w: torch.Tensor
    dist: torch.Tensor
    heading_sin: torch.Tensor  # sin(angle) to ball in base xy, angle from +x_b
    heading_cos: torch.Tensor
    lateral: torch.Tensor  # ball y in base frame (signed)
    forward_proj: torch.Tensor  # ball x in base frame
    passed_ball: torch.Tensor  # 1 if ball is behind robot (+x_b) in rough sense
    goal_dir_b: torch.Tensor  # unit xy vector in body frame toward world +X goal
    phase_id: torch.Tensor  # 0..4 long
    left_foot_pos_w: torch.Tensor
    right_foot_pos_w: torch.Tensor
    left_foot_vel_w: torch.Tensor
    right_foot_vel_w: torch.Tensor
    d_left_ball: torch.Tensor
    d_right_ball: torch.Tensor
    support_foot_speed_xy: torch.Tensor  # left foot = support for right-foot kick
    swing_foot_speed_xy: torch.Tensor
    ball_speed_xy: torch.Tensor
    ball_vx_w: torch.Tensor
    base_xy_w: torch.Tensor
    support_foot_xy_w: torch.Tensor


def _resolve_foot_cache(env: ManagerBasedRLEnv, robot: Articulation) -> tuple[int, int]:
    if not hasattr(env, "_k1_kick_foot_body_ids"):
        env._k1_kick_foot_body_ids = _match_foot_body_ids(robot)
    return env._k1_kick_foot_body_ids


def compute_kick_features(env: ManagerBasedRLEnv) -> KickFeatures:
    """Compute kick features for all environments (no caching)."""
    robot = _robot(env)
    ball = _ball(env)
    left_id, right_id = _resolve_foot_cache(env, robot)

    root_pos_w = robot.data.root_pos_w
    root_quat_w = robot.data.root_quat_w
    ball_pos_w = ball.data.root_pos_w
    ball_vel_w = ball.data.root_lin_vel_w

    ball_rel_w = ball_pos_w - root_pos_w
    ball_pos_b = quat_rotate_inverse(root_quat_w, ball_rel_w)
    ball_vel_b = quat_rotate_inverse(root_quat_w, ball_vel_w)

    dx = ball_pos_b[:, 0]
    dy = ball_pos_b[:, 1]
    dist = torch.linalg.norm(ball_pos_b[:, :2], dim=-1).clamp(min=1e-4)
    heading_sin = (dy / dist).clamp(-1.0, 1.0)
    heading_cos = (dx / dist).clamp(-1.0, 1.0)
    lateral = dy
    forward_proj = dx
    passed_ball = (dx < -0.06).float()

    # World +X is the scoring direction; express that direction in the robot base frame (xy).
    forward_w = torch.zeros(env.num_envs, 3, device=env.device)
    forward_w[:, 0] = 1.0
    goal_dir_b = quat_rotate_inverse(root_quat_w, forward_w)[:, :2]
    gnorm = torch.linalg.norm(goal_dir_b, dim=-1, keepdim=True).clamp(min=1e-4)
    goal_dir_b = goal_dir_b / gnorm

    left_pos = robot.data.body_pos_w[:, left_id, :]
    right_pos = robot.data.body_pos_w[:, right_id, :]
    left_vel = robot.data.body_lin_vel_w[:, left_id, :]
    right_vel = robot.data.body_lin_vel_w[:, right_id, :]

    d_left = torch.linalg.norm(left_pos - ball_pos_w, dim=-1)
    d_right = torch.linalg.norm(right_pos - ball_pos_w, dim=-1)
    support_speed_xy = torch.linalg.norm(left_vel[:, :2], dim=-1)
    swing_speed_xy = torch.linalg.norm(right_vel[:, :2], dim=-1)

    ball_speed_xy = torch.linalg.norm(ball_vel_w[:, :2], dim=-1)
    ball_vx_w = ball_vel_w[:, 0]

    heading_abs = torch.abs(torch.atan2(heading_sin, heading_cos))

    # Phase semantics:
    #   0 approach — default when no other mask fires (typically far: dist > 0.55 m, or closing without align gates)
    #   1 align — closing in but still correcting heading (narrow band in distance)
    #   2 plant — close, aligned, support settled
    #   3 kick — strike foot near ball
    #   4 recover — ball moving forward after contact
    #
    # Assignment order (later torch.where wins): recover > kick > plant > align > approach (0).
    recover = ball_vx_w > 0.05
    kick_near = (
        (d_right < 0.40)
        & (dist < 0.80)
    )
    plant = (
        (dist < 0.60)
        & (heading_abs < 0.40)
        & (support_speed_xy < 0.60)
        & (d_right < 0.70)
    )
    align = (dist < 1.00) & (heading_abs > 0.05)

    phase_id = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase_id = torch.where(align, torch.ones_like(phase_id), phase_id)
    phase_id = torch.where(plant, torch.full_like(phase_id, 2), phase_id)
    phase_id = torch.where(kick_near, torch.full_like(phase_id, 3), phase_id)
    phase_id = torch.where(recover, torch.full_like(phase_id, 4), phase_id)

    return KickFeatures(
        ball_pos_b=ball_pos_b,
        ball_vel_b=ball_vel_b,
        ball_vel_w=ball_vel_w,
        dist=dist,
        heading_sin=heading_sin,
        heading_cos=heading_cos,
        lateral=lateral,
        forward_proj=forward_proj,
        passed_ball=passed_ball,
        goal_dir_b=goal_dir_b,
        phase_id=phase_id,
        left_foot_pos_w=left_pos,
        right_foot_pos_w=right_pos,
        left_foot_vel_w=left_vel,
        right_foot_vel_w=right_vel,
        d_left_ball=d_left,
        d_right_ball=d_right,
        support_foot_speed_xy=support_speed_xy,
        swing_foot_speed_xy=swing_speed_xy,
        ball_speed_xy=ball_speed_xy,
        ball_vx_w=ball_vx_w,
        base_xy_w=root_pos_w[:, :2].clone(),
        support_foot_xy_w=left_pos[:, :2].clone(),
    )


def get_kick_features(env: ManagerBasedRLEnv) -> KickFeatures:
    """Cached kick features (one compute per env step)."""
    step = int(env.common_step_counter)
    tag = getattr(env, "_k1_kick_feat_step_tag", None)
    if tag != step:
        env._k1_kick_feat_cache = compute_kick_features(env)
        env._k1_kick_feat_step_tag = step
    return env._k1_kick_feat_cache


def phase_sin_cos(env: ManagerBasedRLEnv, num_phases: int = 5) -> torch.Tensor:
    f = get_kick_features(env)
    ang = 2.0 * torch.pi * f.phase_id.float() / float(num_phases)
    return torch.stack((torch.sin(ang), torch.cos(ang)), dim=-1)

