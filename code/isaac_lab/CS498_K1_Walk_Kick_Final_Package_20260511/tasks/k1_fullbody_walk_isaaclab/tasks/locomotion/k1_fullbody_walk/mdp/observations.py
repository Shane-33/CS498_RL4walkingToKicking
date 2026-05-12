from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg

CONTROLLED_JOINT_NAMES = [
    "Left_Hip_Pitch",
    "Left_Hip_Roll",
    "Left_Hip_Yaw",
    "Left_Knee_Pitch",
    "Left_Ankle_Pitch",
    "Left_Ankle_Roll",
    "Right_Hip_Pitch",
    "Right_Hip_Roll",
    "Right_Hip_Yaw",
    "Right_Knee_Pitch",
    "Right_Ankle_Pitch",
    "Right_Ankle_Roll",
]


def _controlled_joint_ids(env: ManagerBasedRLEnv, robot: Articulation) -> torch.Tensor:
    if not hasattr(env, "_k1_controlled_joint_ids"):
        name_to_idx = {name: i for i, name in enumerate(robot.data.joint_names)}
        env._k1_controlled_joint_ids = torch.tensor(
            [name_to_idx[name] for name in CONTROLLED_JOINT_NAMES], device=env.device, dtype=torch.long
        )
    return env._k1_controlled_joint_ids


def projected_gravity(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return robot.data.projected_gravity_b


def base_ang_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    return robot.data.root_ang_vel_b


def gait_cos(env: ManagerBasedRLEnv) -> torch.Tensor:
    """HTWK source: envs/K1/parameter_walk.py lines with cos(2*pi*gait_process)."""
    if not hasattr(env, "_k1_gait_process"):
        env._k1_gait_process = torch.zeros(env.num_envs, device=env.device)
    return torch.cos(2.0 * torch.pi * env._k1_gait_process).unsqueeze(-1)


def gait_sin(env: ManagerBasedRLEnv) -> torch.Tensor:
    if not hasattr(env, "_k1_gait_process"):
        env._k1_gait_process = torch.zeros(env.num_envs, device=env.device)
    return torch.sin(2.0 * torch.pi * env._k1_gait_process).unsqueeze(-1)


def dof_pos_rel(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    joint_ids = _controlled_joint_ids(env, robot)
    return (robot.data.joint_pos - robot.data.default_joint_pos).index_select(dim=1, index=joint_ids)


def dof_vel(env: ManagerBasedRLEnv) -> torch.Tensor:
    robot: Articulation = env.scene["robot"]
    joint_ids = _controlled_joint_ids(env, robot)
    return robot.data.joint_vel.index_select(dim=1, index=joint_ids)
