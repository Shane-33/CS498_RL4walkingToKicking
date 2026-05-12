"""
Reward functions for humanoid kicking policy — single stage training.

Design philosophy:
    One coherent reward set. No curriculum switching.
    Three naturally chained phases gated by proximity/ball-speed:

    Phase 1 — always active:
        approach_ball, yaw_alignment, smoothness penalties

    Phase 2 — gates on foot within 0.15m of ball:
        foot_velocity_toward_ball (swing signal)

    Phase 3 — gates on ball moving > 0.1 m/s:
        ball_velocity_toward_goal, post_kick_upright

    The robot naturally unlocks each phase as it succeeds at the previous one.
    No stage switching, no reward regression, no checkpoint management.
"""

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import quat_rotate_inverse


# ──────────────────────────────────────────────────────────────────────────────
# Posture penalties (always active)
# ──────────────────────────────────────────────────────────────────────────────

def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from standing height. Prevents crouch exploits."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)


def base_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical COM velocity. Discourages bouncing or collapsing."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


# ──────────────────────────────────────────────────────────────────────────────
# Phase 1 — Approach (always active)
# ──────────────────────────────────────────────────────────────────────────────

def distance_foot_to_ball(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Dense shaping: negative distance from right foot to ball.
    Always active — provides a gradient at any distance.
    Returns values <= 0; closer = less negative = better.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    foot_idx = asset.data.body_names.index(foot_body_name)
    foot_pos = asset.data.body_pos_w[:, foot_idx, :]
    ball_pos = ball.data.root_pos_w

    return -torch.norm(foot_pos - ball_pos, dim=-1)


def yaw_alignment_to_goal(
    env: ManagerBasedRLEnv,
    goal_position: tuple = (10.0, 0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Reward facing along the ball→goal axis. Always active.
    Prevents sideways approach — robot facing sideways scores ~0
    regardless of how close it is to the ball.
    Returns exp(-yaw_error^2 / 0.3), range (0, 1].
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    ball_pos = ball.data.root_pos_w
    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)

    ball_to_goal = goal - ball_pos
    desired_yaw = torch.atan2(ball_to_goal[:, 1], ball_to_goal[:, 0])

    quat = asset.data.root_quat_w                       # (N, 4) w,x,y,z
    w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
    current_yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    yaw_error = (desired_yaw - current_yaw + torch.pi) % (2 * torch.pi) - torch.pi
    return torch.exp(-torch.square(yaw_error) / 0.3)


# ──────────────────────────────────────────────────────────────────────────────
# Phase 2 — Swing (gated: foot within proximity_threshold of ball)
# ──────────────────────────────────────────────────────────────────────────────

def foot_velocity_toward_ball(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    proximity_threshold: float = 0.15,
) -> torch.Tensor:
    """
    Reward the right foot moving toward the ball, but ONLY when already
    within proximity_threshold (default 0.15m) of the ball.

    This creates a natural two-phase sequence:
      - Far: robot gets no reward here, only approach_ball pulls it in
      - Close: this activates and rewards the actual swing motion

    No explicit leg-lift constraints — the robot discovers the most
    efficient swing naturally. Just: get close, then move foot into ball.
    The reward scales with foot speed toward ball, capped to avoid lunging.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    foot_idx = asset.data.body_names.index(foot_body_name)
    foot_pos = asset.data.body_pos_w[:, foot_idx, :]
    foot_vel = asset.data.body_lin_vel_w[:, foot_idx, :]
    ball_pos = ball.data.root_pos_w

    # Direction from foot to ball
    foot_to_ball = ball_pos - foot_pos
    dist = torch.norm(foot_to_ball, dim=-1)
    foot_to_ball_norm = foot_to_ball / (dist.unsqueeze(-1) + 1e-6)

    # Foot velocity projected onto foot→ball direction
    vel_toward_ball = (foot_vel * foot_to_ball_norm).sum(dim=-1)

    # Gate: only active within proximity_threshold
    near_ball = (dist < proximity_threshold).float()

    # Clamp to avoid rewarding explosive lunges — natural swing is ~1-3 m/s
    vel_clamped = torch.clamp(vel_toward_ball, min=0.0, max=3.0)

    return vel_clamped * near_ball


# ──────────────────────────────────────────────────────────────────────────────
# Phase 3 — Ball quality (gated: ball moving > 0.1 m/s)
# ──────────────────────────────────────────────────────────────────────────────

def ball_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_position: tuple = (10.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    Continuous reward for ball velocity component pointing toward goal.
    Gated: only fires when ball is actually rolling (> 0.1 m/s).
    This is the primary direction quality signal — always dense once ball moves.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    ball_pos = ball.data.root_pos_w

    ball_speed = ball_vel.norm(dim=-1)
    moving = (ball_speed > 0.1).float()

    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)
    to_goal = goal - ball_pos
    to_goal_norm = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    toward = (ball_vel * to_goal_norm).sum(dim=-1)
    return torch.clamp(toward, min=0.0) * moving


def kick_quality(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_position: tuple = (10.0, 0.0, 0.0),
    min_speed_threshold: float = 1.0,
    max_reward: float = 10.0,
) -> torch.Tensor:
    """
    Sparse shot quality bonus: ball speed × direction accuracy.
    Only fires above min_speed_threshold — weak taps don't score.
    Complements ball_velocity_toward_goal with a stronger one-time signal.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    ball_pos = ball.data.root_pos_w
    ball_speed = ball_vel.norm(dim=-1)

    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)
    to_goal = goal - ball_pos
    to_goal_norm = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    direction_quality = torch.clamp(
        (ball_vel * to_goal_norm).sum(dim=-1) / (ball_speed + 1e-6),
        min=0.0, max=1.0,
    )

    above_threshold = (ball_speed > min_speed_threshold).float()
    reward = ball_speed * direction_quality * above_threshold
    return torch.clamp(reward, max=max_reward)


def post_kick_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ball_speed_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Reward staying upright AFTER the ball has been kicked.
    Silent during approach and swing — only activates post-contact.
    Teaches recovery without interfering with kick dynamics.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    ball_moving = (ball.data.root_lin_vel_w.norm(dim=-1) > ball_speed_threshold).float()
    projected_grav = asset.data.projected_gravity_b
    upright = torch.clamp(1.0 - projected_grav[:, :2].norm(dim=-1), min=0.0)

    return upright * ball_moving


# ──────────────────────────────────────────────────────────────────────────────
# Observation term
# ──────────────────────────────────────────────────────────────────────────────

def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Ball position in the robot's local frame. Used as an observation term.
    Returns (N, 3): x=forward, y=left, z=up relative to robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    diff = ball.data.root_pos_w - asset.data.root_pos_w
    return quat_rotate_inverse(asset.data.root_quat_w, diff)


# ──────────────────────────────────────────────────────────────────────────────
# Termination condition
# ──────────────────────────────────────────────────────────────────────────────

def root_height_out_of_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_height: float = 0.57,
    tolerance: float = 0.25,
) -> torch.Tensor:
    """
    Terminate if height deviates beyond ±25% of standing height.
    Loose enough for kick lean, tight enough to catch falls.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    h = asset.data.root_pos_w[:, 2]
    return (h < base_height * (1.0 - tolerance)) | (h > base_height * (1.0 + tolerance))