"""Kick-task termination helpers."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .kick_features import get_kick_features


def terminate_robot_far_from_ball(env: ManagerBasedRLEnv, max_dist: float = 2.2) -> torch.Tensor:
    f = get_kick_features(env)
    return f.dist > max_dist


def terminate_ball_behind_robot(env: ManagerBasedRLEnv, behind_thresh: float = -0.15) -> torch.Tensor:
    """End episode if ball is clearly behind the torso without forward progress."""
    f = get_kick_features(env)
    slow = f.ball_vx_w < 0.15
    return (f.forward_proj < behind_thresh) & slow


def terminate_height(env: ManagerBasedRLEnv, threshold: float = 0.35) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < threshold
