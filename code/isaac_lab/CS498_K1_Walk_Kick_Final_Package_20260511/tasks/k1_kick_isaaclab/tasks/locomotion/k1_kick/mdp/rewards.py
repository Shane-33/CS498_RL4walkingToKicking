"""Phase-aware kick shaping + locomotion preservation + regularization."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.mdp import rewards as walk_rew

from .kick_features import KickFeatures, get_kick_features


def _robot(env: ManagerBasedRLEnv):
    return env.scene["robot"]


def _phase_gate(f: KickFeatures, *allowed: int) -> torch.Tensor:
    t = torch.tensor(allowed, device=f.phase_id.device, dtype=torch.long)
    return torch.isin(f.phase_id, t).float()


def survival(env: ManagerBasedRLEnv) -> torch.Tensor:
    return torch.ones(env.num_envs, device=env.device)


def track_walk_command(env: ManagerBasedRLEnv, command_name: str = "base_velocity", std: float = 0.55) -> torch.Tensor:
    """Velocity tracking gated weakly off during late kick/recover."""
    import isaaclab.envs.mdp as mdp

    f = get_kick_features(env)
    gate = 1.0 - 0.35 * ((f.phase_id == 3) | (f.phase_id == 4)).float()
    return mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=std) * gate


def track_yaw_command(env: ManagerBasedRLEnv, command_name: str = "base_velocity", std: float = 0.55) -> torch.Tensor:
    import isaaclab.envs.mdp as mdp

    f = get_kick_features(env)
    gate = 1.0 - 0.4 * ((f.phase_id == 3) | (f.phase_id == 4)).float()
    return mdp.track_ang_vel_z_exp(env, command_name=command_name, std=std) * gate


def reward_approach_distance(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 0)
    return torch.exp(-f.dist / 0.75) * g


def reward_alignment(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 1)
    heading_err = torch.square(torch.atan2(f.heading_sin, f.heading_cos))
    return torch.exp(-heading_err / 0.35) * g


def penalty_excess_lateral_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    return torch.square(f.lateral / 0.6) * (f.dist < 0.85).float()


def penalty_passed_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    return f.passed_ball * (f.ball_vx_w < 0.2).float()


def reward_support_foot_stable(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 2, 3)
    return torch.exp(-torch.square(f.support_foot_speed_xy / 0.45)) * g


def reward_com_over_support_xy(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 2, 3)
    delta = f.base_xy_w - f.support_foot_xy_w
    dist_xy = torch.linalg.norm(delta, dim=-1)
    return torch.exp(-torch.square(dist_xy / 0.18)) * g


def reward_right_foot_to_ball(env: ManagerBasedRLEnv, sigma: float = 0.35) -> torch.Tensor:
    """Proximity of strike foot to ball — only while genuinely approaching, not camped on the ball.

    Active only in approach/align phases (0–1) and when still meaningfully far (dist > 0.25 m).
    Disables farming by standing still with the right foot near the ball.
    """
    f = get_kick_features(env)
    active = (f.phase_id <= 1) & (f.dist > 0.25)
    return torch.exp(-f.d_right_ball / sigma) * active.float()



def reward_ball_contact(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Dense shaping for actual foot-ball interaction."""

    f = get_kick_features(env)

    close = (f.d_right_ball < 0.25).float()

    forward_swing = torch.clamp(
        f.right_foot_vel_w[:, 0],
        min=0.0,
        max=2.0,
    )

    return close * forward_swing

def reward_kick_leg_forward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward right foot swinging forward near the ball."""

    f = get_kick_features(env)

    close = (f.d_right_ball < 0.60).float()

    forward_vel = torch.clamp(
        f.right_foot_vel_w[:, 0],
        min=0.0,
        max=3.0,
    )

    return close * forward_vel


def penalty_no_kick_progress(env: ManagerBasedRLEnv, episode_step_threshold: int = 100, kick_vx_thresh: float = 0.1) -> torch.Tensor:
    """Time pressure: penalize late episode steps if the ball still has negligible forward speed.

    Returns nonnegative magnitude; use negative weight in RewardsCfg (e.g. -2.0).
    ``episode_length_buf`` counts environment steps (post-decimation).
    """
    f = get_kick_features(env)
    episode_too_long = (env.episode_length_buf > episode_step_threshold).float()
    ball_not_kicked = (f.ball_vx_w < kick_vx_thresh).float()
    return episode_too_long * ball_not_kicked


def reward_phase_progress(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Sparse bonus for reaching plant / kick / recover phases (sequence completion)."""
    f = get_kick_features(env)
    phase_bonus = torch.zeros(env.num_envs, device=env.device)
    phase_bonus = torch.where(f.phase_id == 2, torch.full_like(phase_bonus, 0.5), phase_bonus)
    phase_bonus = torch.where(f.phase_id == 3, torch.full_like(phase_bonus, 1.5), phase_bonus)
    phase_bonus = torch.where(f.phase_id == 4, torch.full_like(phase_bonus, 1.0), phase_bonus)
    return phase_bonus


def reward_approach_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Reward forward world velocity when the ball is still ahead and not yet in strike range."""
    robot = _robot(env)
    f = get_kick_features(env)
    forward_vel = robot.data.root_lin_vel_w[:, 0]
    ball_ahead = (f.forward_proj > 0.2).float()
    far_enough = (f.dist > 0.3).float()
    return (forward_vel * ball_ahead).clamp(min=0.0) * far_enough


def penalty_left_foot_near_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    close = torch.exp(-f.d_left_ball / 0.18)
    g = (f.dist < 0.65).float()
    return close * g


def reward_ball_forward_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 3, 4)
    return torch.clamp(f.ball_vx_w, min=0.0, max=4.0) * (0.35 + 0.65 * g)


def penalty_ball_lateral_velocity(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    vy = torch.abs(f.ball_vel_w[:, 1])
    moving = (f.ball_speed_xy > 0.25).float()
    return torch.clamp(vy, max=3.0) * moving


def reward_forward_ball_speed_after_hit(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    hit = f.ball_speed_xy > 0.35
    fwd = torch.clamp(f.ball_vel_b[:, 0], min=0.0, max=3.0)
    return fwd * hit.float()


def penalty_backward_ball(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    return torch.clamp(-f.ball_vx_w, min=0.0, max=2.5)


def reward_post_kick_balance(env: ManagerBasedRLEnv) -> torch.Tensor:
    f = get_kick_features(env)
    g = _phase_gate(f, 4)
    up = walk_rew.orientation_penalty(env)
    height = walk_rew.base_height_reward(env, target=0.52)
    quality = torch.exp(-(0.8 * up + 0.35 * height))
    return quality * g


def reward_post_kick_walk_recovery(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    import isaaclab.envs.mdp as mdp

    f = get_kick_features(env)
    g = _phase_gate(f, 4)
    return mdp.track_lin_vel_xy_exp(env, command_name=command_name, std=0.65) * g


def arm_counterbalance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    f = get_kick_features(env)
    name_to_idx = {n: i for i, n in enumerate(robot.data.joint_names)}
    li = name_to_idx.get("ALeft_Shoulder_Pitch")
    ri = name_to_idx.get("ARight_Shoulder_Pitch")
    if li is None or ri is None:
        return torch.zeros(env.num_envs, device=env.device)
    q = robot.data.joint_pos
    lp = q[:, li]
    rp = q[:, ri]
    tgt_l = 0.55
    tgt_r = -0.35
    err = torch.square(lp - tgt_l) + torch.square(rp - tgt_r)
    shaped = torch.exp(-err / 0.8)
    g = _phase_gate(f, 3, 4)
    base_pose = torch.exp(-(torch.square(lp - 0.0) + torch.square(rp - 0.0)) / 1.5)
    return shaped * g + base_pose * (1.0 - g)


def hip_roll_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    names = robot.data.joint_names
    name_to_idx = {n: i for i, n in enumerate(names)}
    li = name_to_idx.get("Left_Hip_Roll")
    ri = name_to_idx.get("Right_Hip_Roll")
    cols = []
    if li is not None:
        cols.append(robot.data.joint_pos[:, li].unsqueeze(-1))
    if ri is not None:
        cols.append(robot.data.joint_pos[:, ri].unsqueeze(-1))
    if not cols:
        return torch.zeros(env.num_envs, device=env.device)
    q = torch.cat(cols, dim=-1)
    return torch.sum(torch.square(q), dim=-1)


def joint_speed_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    return walk_rew.dof_vel(env)


def yaw_rate_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    return torch.square(robot.data.root_ang_vel_b[:, 2])


def lateral_velocity_penalty(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot = _robot(env)
    return torch.square(robot.data.root_lin_vel_b[:, 1])
