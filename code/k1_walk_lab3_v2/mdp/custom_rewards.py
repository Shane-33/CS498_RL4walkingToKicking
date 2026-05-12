import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg

def base_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize vertical velocity of the base (helps reduce hopping)."""
    asset = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])

def bilateral_symmetry_l2(
    env: ManagerBasedRLEnv,
    left_joint_names: list[str],
    right_joint_names: list[str],
    mirror_signs: list[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize asymmetry between left and right limb joint positions."""
    asset = env.scene[asset_cfg.name]
    left_idx, _ = asset.find_joints(left_joint_names)
    right_idx, _ = asset.find_joints(right_joint_names)
    
    left_pos = asset.data.joint_pos[:, left_idx]
    right_pos = asset.data.joint_pos[:, right_idx]
    signs = torch.tensor(mirror_signs, device=env.device)
    
    return torch.sum(torch.square(left_pos - (right_pos * signs)), dim=-1)

def arm_natural_pose_soft_l2(
    env: ManagerBasedRLEnv,
    joint_names: list[str],
    deadzone: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward keeping arms near a natural pose with a deadzone."""
    asset = env.scene[asset_cfg.name]
    joint_idx, _ = asset.find_joints(joint_names)
    joint_pos = asset.data.joint_pos[:, joint_idx]
    # Default pose is usually 0.0 for these relative pos
    error = torch.abs(joint_pos) - deadzone
    return torch.sum(torch.square(torch.clamp(error, min=0.0)), dim=-1)

def base_height_reward(
    env: ManagerBasedRLEnv,
    target_height: float = 0.57,
    std: float = 0.06,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Gaussian reward centered at target_height — gives +1 when upright, ~0 when fallen."""
    asset = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2]
    error = height - target_height
    return torch.exp(-0.5 * (error / std) ** 2)


def base_ang_momentum_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base angular momentum to keep the trunk stable."""
    asset = env.scene[asset_cfg.name]
    # Approximation using base angular velocity
    return torch.sum(torch.square(asset.data.root_ang_vel_w), dim=-1)


def target_out_of_range_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize raw network outputs beyond ±3 (outside the Gaussian clip range)."""
    action_term = env.action_manager.get_term(action_name)
    excess = torch.clamp(torch.abs(action_term.raw_actions) - 3.0, min=0.0)
    return torch.sum(torch.square(excess), dim=-1)


def action_magnitude_l2(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
) -> torch.Tensor:
    """Penalize large raw action magnitudes to keep the policy head inside clip bounds."""
    action_term = env.action_manager.get_term(action_name)
    return torch.sum(torch.square(action_term.raw_actions), dim=-1)


def joint_pos_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize deviation from default joint positions (summed over joints)."""
    asset = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.joint_pos - asset.data.default_joint_pos), dim=-1)
