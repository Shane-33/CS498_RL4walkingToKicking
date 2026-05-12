from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


def terminate_vel(env: ManagerBasedRLEnv, threshold: float = 50.0) -> torch.Tensor:
    vel = env.scene["robot"].data.root_state_w[:, 7:13]
    return vel.square().sum(dim=-1) > threshold


def terminate_height(env: ManagerBasedRLEnv, threshold: float = 0.35) -> torch.Tensor:
    return env.scene["robot"].data.root_pos_w[:, 2] < threshold
