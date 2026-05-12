"""Kick MDP primitives (Isaac Lab ``ManagerBasedRLEnv``)."""

from __future__ import annotations

from . import events, kick_features, observations, rewards, terminations
from .kick_features import KickFeatures, compute_kick_features, get_kick_features, phase_sin_cos
from .observations import kick_ball_features
from .terminations import (
    terminate_ball_behind_robot,
    terminate_height,
    terminate_robot_far_from_ball,
)


reset_ball_relative_to_robot = events.reset_ball_relative_to_robot


__all__ = [
    "events",
    "kick_features",
    "observations",
    "rewards",
    "terminations",
    "KickFeatures",
    "compute_kick_features",
    "get_kick_features",
    "phase_sin_cos",
    "kick_ball_features",
    "reset_ball_relative_to_robot",
    "terminate_ball_behind_robot",
    "terminate_height",
    "terminate_robot_far_from_ball",
]
