"""Kick-task observations (17-D kick block)."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv

from .kick_features import get_kick_features, phase_sin_cos


def kick_ball_features(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return **exactly** 17-D normalized observations (concatenated, base frame unless noted).

    Fixed layout::

        dims  name                          scaling / notes
        0–2   ball_pos_b (x,y,z)          divide by 2.0
        3–5   ball_vel_b (lin, body)      divide by 3.0
        6     distance_to_ball_xy         Euclidean in base xy / 1.5
        7–8   angle_to_ball               sin ψ, cos ψ (bearing from +x_b to ball)
        9–10  heading_to_goal              sin φ, cos φ (world +X projected in base xy unit vector)
        11    left_foot_ball_dist          Euclidean foot–ball world / 0.85
        12    right_foot_ball_dist         Euclidean foot–ball world / 0.85
        13–14 implicit phase embedding     phase_sin_cos (5-phase clock)
        15    passed_ball_scalar           binary {0,1}
        16    ball_speed_xy_mag            planar speed world / 3.0

    ``heading_to_goal`` matches ``(+sin φ,+cos φ)`` with φ = atan2(g_y,g_x), ``g = goal_dir_b`` unit.
    """
    f = get_kick_features(env)
    psi_s = f.heading_sin.unsqueeze(-1)
    psi_c = f.heading_cos.unsqueeze(-1)
    # Unit direction toward scoring axis in robot base xy: sin/cos of heading-from-+x_b
    gh_s = f.goal_dir_b[:, 1:2].clamp(-1.0, 1.0)
    gh_c = f.goal_dir_b[:, 0:1].clamp(-1.0, 1.0)
    d_l = (f.d_left_ball / 0.85).unsqueeze(-1)
    d_r = (f.d_right_ball / 0.85).unsqueeze(-1)
    pc = phase_sin_cos(env, num_phases=5)
    out = torch.cat(
        [
            f.ball_pos_b / 2.0,
            f.ball_vel_b / 3.0,
            (f.dist / 1.5).unsqueeze(-1),
            psi_s,
            psi_c,
            gh_s,
            gh_c,
            d_l,
            d_r,
            pc,
            f.passed_ball.unsqueeze(-1),
            (f.ball_speed_xy / 3.0).unsqueeze(-1),
        ],
        dim=-1,
    )
    return torch.clamp(out, -6.0, 6.0)
