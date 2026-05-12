"""
Reward functions for humanoid kicking policy — single stage training.

Design philosophy:
    One coherent reward set. No curriculum switching.
    Three naturally chained phases gated by proximity/ball-speed:

    Phase 1 — always active:
        approach_ball, yaw_alignment, smoothness penalties, arm penalty

    Phase 2 — gates on foot within 0.25m of ball:
        inside_foot_approach (guides foot to behind-right of ball)

    Phase 3 — gates on ball moving > 0.1 m/s:
        ball_velocity_toward_goal, post_kick_upright
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


def arm_joints_deviation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize shoulder and elbow joints deviating from their default pose.
    Keeps arms relaxed by the sides rather than raised for balance.
    Soft penalty — allows some natural arm swing during kick, just
    discourages the sustained raised-arms posture.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    joint_names = asset.data.joint_names

    arm_indices = [
        i for i, n in enumerate(joint_names)
        if any(k in n for k in ["Shoulder", "Elbow"])
    ]

    if not arm_indices:
        return torch.zeros(env.num_envs, device=env.device)

    # Default pose offset is already baked into joint_pos_rel,
    # so we use joint_pos and compare against default_joint_pos
    joint_pos = asset.data.joint_pos[:, arm_indices]
    default_pos = asset.data.default_joint_pos[:, arm_indices]

    return torch.sum(torch.square(joint_pos - default_pos), dim=-1)


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
    Prevents sideways approach — robot facing sideways scores ~0.
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
# Phase 2 — Inside foot approach geometry
# (gated: foot within proximity_threshold of ball)
# ──────────────────────────────────────────────────────────────────────────────

def inside_foot_approach(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_position: tuple = (10.0, 0.0, 0.0),
    proximity_threshold: float = 0.25,
    behind_offset: float = 0.15,
    side_offset: float = 0.10,
) -> torch.Tensor:
    """
    Reward the right foot approaching the ball from the correct position
    for an inside-foot instep kick:

        ideal foot position = ball - (ball_to_goal * behind_offset)
                                   + (right_perpendicular * side_offset)

    i.e. behind the ball along the kick direction and slightly to the right.
    This geometry forces the inside of the right foot to face the ball
    at contact — the natural instep kick used for accurate passing.

    Gated by proximity_threshold (0.25m) so it only activates once the
    robot is already close from the approach phase. Uses a soft exponential
    reward so there's always a gradient — no hard binary switching.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    foot_idx = asset.data.body_names.index(foot_body_name)
    foot_pos = asset.data.body_pos_w[:, foot_idx, :]     # (N, 3)
    ball_pos = ball.data.root_pos_w                       # (N, 3)
    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)  # (1, 3)

    # Distance from foot to ball — used for gating
    foot_to_ball_dist = torch.norm(foot_pos - ball_pos, dim=-1)  # (N,)

    # Ball → goal unit vector (x-y plane only, z ignored)
    ball_to_goal = goal - ball_pos
    ball_to_goal_xy = ball_to_goal.clone()
    ball_to_goal_xy[:, 2] = 0.0
    ball_to_goal_norm = ball_to_goal_xy / (ball_to_goal_xy.norm(dim=-1, keepdim=True) + 1e-6)

    # Right perpendicular to ball→goal in x-y plane
    # Rotate ball_to_goal_norm by -90 degrees: (x,y) → (y,-x)
    right_perp = torch.zeros_like(ball_to_goal_norm)
    right_perp[:, 0] =  ball_to_goal_norm[:, 1]
    right_perp[:, 1] = -ball_to_goal_norm[:, 0]

    # Ideal foot position: behind ball and slightly to the right
    ideal_foot_pos = (
        ball_pos
        - ball_to_goal_norm * behind_offset
        + right_perp * side_offset
    )
    ideal_foot_pos[:, 2] = foot_pos[:, 2]  # match foot height, don't constrain z

    # Distance from current foot position to ideal position
    dist_to_ideal = torch.norm(foot_pos - ideal_foot_pos, dim=-1)

    # Exponential reward — peaks when foot is at ideal position
    position_reward = torch.exp(-dist_to_ideal / 0.1)

    # Gate: soft activation within proximity_threshold using sigmoid
    # Fully active at 0m, half active at proximity_threshold/2, near zero beyond
    gate = torch.sigmoid(
        (proximity_threshold - foot_to_ball_dist) / (proximity_threshold * 0.2)
    )

    return position_reward * gate


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