import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg


def base_lin_vel_z_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_lin_vel_w[:, 2])


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.square(asset.data.root_pos_w[:, 2] - target_height)

def distance_foot_to_ball(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Dense shaping — negative distance from kicking foot to ball.
    Always provides a gradient toward the ball even without contact.
    Closer = less negative = better. Never positive.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    body_names = asset.data.body_names
    foot_idx = body_names.index(foot_body_name)

    foot_pos = asset.data.body_pos_w[:, foot_idx, :]    # (N, 3)
    ball_pos = ball.data.root_pos_w                      # (N, 3)

    dist = torch.norm(foot_pos - ball_pos, dim=-1)
    return -dist


def ball_contact_reward(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Sparse binary reward — any foot contact force above threshold counts.
    Large weight in managers_cfg makes this dominate once contact happens.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)                      # (N, num_feet)
    any_contact = (force_mag > force_threshold).any(dim=-1).float()
    return any_contact


def ball_displacement(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    max_reward: float = 2.0,
) -> torch.Tensor:
    """
    Reward proportional to ball displacement from spawn position (0.35, -0.1).
    Capped at max_reward to prevent huge values from lucky bounces.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    spawn_xy = torch.tensor([0.35, -0.1], device=env.device)
    current_xy = ball.data.root_pos_w[:, :2]
    displacement = torch.norm(current_xy - spawn_xy, dim=-1)
    return torch.clamp(displacement, max=max_reward)

def plant_foot_stability(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    plant_foot_idx: int = 0,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """
    Reward keeping the plant foot (left, idx=0) on the ground.
    Prevents hopping or losing balance on stance leg during kick.
    Binary: 1.0 if in contact, 0.0 if not.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)
    plant_contact = (force_mag[:, plant_foot_idx] > force_threshold).float()
    return plant_contact

def ball_position_in_robot_frame(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
) -> torch.Tensor:
    """
    Ball position expressed in the robot root frame.
    Returns (N, 3) tensor — x forward, y left, z up relative to robot.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    # World positions
    robot_pos = asset.data.root_pos_w          # (N, 3)
    robot_quat = asset.data.root_quat_w        # (N, 4) w,x,y,z
    ball_pos = ball.data.root_pos_w            # (N, 3)

    # Vector from robot to ball in world frame
    diff = ball_pos - robot_pos                # (N, 3)

    # Rotate into robot frame using inverse of robot orientation
    from isaaclab.utils.math import quat_rotate_inverse
    ball_in_robot_frame = quat_rotate_inverse(robot_quat, diff)

    return ball_in_robot_frame


def ball_velocity_toward_goal(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    goal_position: tuple = (10.0, 0.0, 0.0),
) -> torch.Tensor:
    """
    Reward ball velocity component pointing toward goal.
    Only fires when ball is actually moving — zero otherwise.
    This is the primary shot quality signal.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos = ball.data.root_pos_w
    ball_vel = ball.data.root_lin_vel_w

    # Only reward when ball is moving
    ball_speed = ball_vel.norm(dim=-1)
    ball_moving = (ball_speed > 0.1).float()

    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)
    to_goal = goal - ball_pos
    to_goal_norm = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)

    toward = (ball_vel * to_goal_norm).sum(dim=-1)
    return torch.clamp(toward, min=0.0) * ball_moving


def ball_acceleration_reward(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    acceleration_scale: float = 10.0,
    max_reward: float = 5.0,
) -> torch.Tensor:
    """
    Reward the impulse quality of the kick — how explosively 
    the ball accelerated. Uses change in ball speed between steps.
    Only fires on the kick frame itself, not during rolling.
    This encourages a clean powerful strike rather than a nudge.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w          # current velocity
    
    # We need previous velocity — store it in env
    if not hasattr(env, "_prev_ball_vel"):
        env._prev_ball_vel = torch.zeros_like(ball_vel)

    # Speed change this step
    prev_speed = env._prev_ball_vel.norm(dim=-1)
    curr_speed = ball_vel.norm(dim=-1)
    delta_speed = curr_speed - prev_speed

    # Update stored velocity
    env._prev_ball_vel = ball_vel.clone()

    # Only reward positive acceleration (ball speeding up)
    # tanh keeps it bounded, acceleration_scale controls sensitivity
    reward = torch.tanh(
        torch.clamp(delta_speed, min=0.0) / acceleration_scale
    ) * max_reward

    return reward


def kick_quality(
    env: ManagerBasedRLEnv,
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    goal_position: tuple = (10.0, 0.0, 0.0),
    min_speed_threshold: float = 1.0,
    max_reward: float = 10.0,
) -> torch.Tensor:
    """
    Combined kick quality metric — fires once when ball reaches
    meaningful speed. Rewards both power (ball speed) and 
    accuracy (direction toward goal) together.
    High speed toward goal = high reward.
    High speed away from goal = zero reward.
    This is the terminal kick quality signal.
    """
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_vel = ball.data.root_lin_vel_w
    ball_pos = ball.data.root_pos_w

    ball_speed = ball_vel.norm(dim=-1)

    # Direction quality
    goal = torch.tensor(goal_position, device=env.device).unsqueeze(0)
    to_goal = goal - ball_pos
    to_goal_norm = to_goal / (to_goal.norm(dim=-1, keepdim=True) + 1e-6)
    direction_quality = torch.clamp(
        (ball_vel * to_goal_norm).sum(dim=-1) / (ball_speed + 1e-6),
        min=0.0, max=1.0
    )  # 1.0 = perfectly toward goal, 0.0 = perpendicular or away

    # Only fire when ball has meaningful speed
    above_threshold = (ball_speed > min_speed_threshold).float()

    # Combined: speed * direction quality
    # A fast kick in the wrong direction gets near zero
    # A slow kick in the right direction gets partial reward
    reward = ball_speed * direction_quality * above_threshold
    return torch.clamp(reward, max=max_reward)

def single_leg_balance_quality(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """
    During the kick, reward clean single leg stance on plant foot.
    Specifically rewards: plant foot contact + kicking foot in air.
    This is the biomechanically correct kick posture.
    """
    sensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)                      # (N, 2)

    plant_in_contact = (force_mag[:, 0] > force_threshold).float()   # left
    kick_in_air = (force_mag[:, 1] < force_threshold).float()        # right

    # Both conditions must be true simultaneously
    return plant_in_contact * kick_in_air

def com_backward_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Penalize backward COM movement.
    Natural kick has COM moving forward or neutral over plant foot.
    Backward drift = losing balance = bad kick posture.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_vel_x = asset.data.root_lin_vel_w[:, 0]
    backward_vel = torch.clamp(-com_vel_x, min=0.0)
    return torch.clamp(backward_vel, max=max_penalty)


def com_height_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    min_height: float = 0.45,
) -> torch.Tensor:
    """
    Penalize COM dropping too low — collapsing into the kick.
    Exponential penalty below threshold, zero above.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    com_z = asset.data.root_pos_w[:, 2]
    drop = torch.clamp(min_height - com_z, min=0.0)
    return torch.square(drop)


def post_kick_upright(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ball_cfg: SceneEntityCfg = SceneEntityCfg("ball"),
    ball_speed_threshold: float = 0.3,
) -> torch.Tensor:
    """
    Reward being upright AFTER ball has been kicked.
    Silent during approach and swing — only fires post contact.
    Teaches recovery without fighting kick motion itself.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]

    ball_speed = ball.data.root_lin_vel_w.norm(dim=-1)
    post_kick = (ball_speed > ball_speed_threshold).float()

    projected_grav = asset.data.projected_gravity_b
    upright = torch.clamp(
        1.0 - torch.norm(projected_grav[:, :2], dim=-1),
        min=0.0,
    )
    return upright * post_kick

def kick_foot_aerial_swing(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    min_height: float = 0.05,
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """
    Reward kicking foot being airborne AND moving forward simultaneously.
    Both conditions must be true — aerial + forward velocity + elevated.
    Forces actual leg lift rather than ground-skimming nudge.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    sensor = env.scene[sensor_cfg.name]

    body_names = asset.data.body_names
    foot_idx = body_names.index(foot_body_name)

    # Condition 1 — right foot not in contact (idx 1)
    forces = sensor.data.net_forces_w_history[:, 0, sensor_cfg.body_ids, :]
    force_mag = forces.norm(dim=-1)
    kick_foot_airborne = (force_mag[:, 1] < force_threshold).float()

    # Condition 2 — foot has forward velocity
    foot_vel = asset.data.body_lin_vel_w[:, foot_idx, :]
    forward_vel = torch.clamp(foot_vel[:, 0], min=0.0)

    # Condition 3 — foot elevated relative to robot
    foot_height = asset.data.body_pos_w[:, foot_idx, 2]
    robot_height = asset.data.root_pos_w[:, 2]
    relative_foot_height = foot_height - (robot_height - 0.57)
    foot_elevated = (relative_foot_height > min_height).float()

    return forward_vel * kick_foot_airborne * foot_elevated


def kick_foot_jitter_penalty(
    env: ManagerBasedRLEnv,
    foot_body_name: str = "right_foot_link",
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """
    Penalize rapid direction reversals in kicking foot velocity.
    Sliding = consistent direction = zero penalty.
    Jittering = rapid reversals = high penalty.
    Dot product between consecutive steps detects reversals.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    body_names = asset.data.body_names
    foot_idx = body_names.index(foot_body_name)

    curr_foot_vel = asset.data.body_lin_vel_w[:, foot_idx, :2]

    if not hasattr(env, "_prev_kick_foot_vel"):
        env._prev_kick_foot_vel = torch.zeros_like(curr_foot_vel)

    prev_foot_vel = env._prev_kick_foot_vel
    dot = (curr_foot_vel * prev_foot_vel).sum(dim=-1)
    reversal_magnitude = torch.clamp(-dot, min=0.0)
    env._prev_kick_foot_vel = curr_foot_vel.clone()

    return reversal_magnitude


def lateral_com_velocity_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_penalty: float = 1.0,
) -> torch.Tensor:
    """
    Penalize sideways COM movement specifically.
    Sideways drift = lateral fall = bad kick posture.
    Complements com_backward_vel which only covers x axis.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    lateral_vel = torch.abs(asset.data.root_lin_vel_w[:, 1])
    return torch.clamp(lateral_vel, max=max_penalty)


def root_height_out_of_range(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    base_height: float = 0.57,
    tolerance: float = 0.2,
) -> torch.Tensor:
    """
    Terminate if robot height deviates more than tolerance from base height.
    ±20% means acceptable range is [0.456m, 0.684m].
    Catches both collapsing and bouncing.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    current_height = asset.data.root_pos_w[:, 2]
    min_height = base_height * (1.0 - tolerance)
    max_height = base_height * (1.0 + tolerance)
    return (current_height < min_height) | (current_height > max_height)