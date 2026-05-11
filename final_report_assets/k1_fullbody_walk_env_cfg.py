
"""
K1 Full-Body Walk Task - 20 DoF (legs + arms).
Based on best practices from:
- Unitree H1/G1 IsaacLab locomotion
- XBot-L full-body locomotion (Gu et al. 2024)
- KSLC arm swing for yaw stability

Key differences from 12-DoF task:
1. Arms included in action space (20 DoF total)
2. Arm default pos penalty keeps arms natural
3. Yaw penalty moderate (not too strong)
4. Lateral vel penalty moderate
5. Random yaw reset for generalization
"""
from __future__ import annotations
import torch
from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp

import sys
sys.path.insert(0, "/home/team2/team2_project/booster_train/source/booster_train")
from booster_train.tasks.k1_parameter_walk_isaaclab.tasks.locomotion.k1_parameter_walk.k1_param_walk_env_cfg import (
    add_k1_foot_box_collisions,
)
from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.mdp import rewards as k1_rew
from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.mdp import terminations as k1_done
from booster_train.assets.robots.booster import BOOSTER_K1_CFG


# ── Custom rewards ──

ARM_TARGETS = {
    "ALeft_Shoulder_Pitch": 0.0,
    "ARight_Shoulder_Pitch": 0.0,
    "Left_Shoulder_Roll": -1.3,
    "Right_Shoulder_Roll": 1.3,
    "Left_Elbow_Pitch": 0.3,
    "Right_Elbow_Pitch": 0.3,
    "Left_Elbow_Yaw": 0.0,
    "Right_Elbow_Yaw": 0.0,
}
ACTION_JOINT_NAMES = [
    # Legs (12)
    "Left_Hip_Pitch", "Left_Hip_Roll", "Left_Hip_Yaw", "Left_Knee_Pitch",
    "Left_Ankle_Pitch", "Left_Ankle_Roll",
    "Right_Hip_Pitch", "Right_Hip_Roll", "Right_Hip_Yaw", "Right_Knee_Pitch",
    "Right_Ankle_Pitch", "Right_Ankle_Roll",
    # Arms (8)
    "ALeft_Shoulder_Pitch", "Left_Shoulder_Roll", "Left_Elbow_Pitch", "Left_Elbow_Yaw",
    "ARight_Shoulder_Pitch", "Right_Shoulder_Roll", "Right_Elbow_Pitch", "Right_Elbow_Yaw",
]


def no_backward_velocity_penalty(env, command_name: str = "base_velocity"):
    """Penalize backward velocity only when forward command is requested."""
    robot = env.scene["robot"]
    vx = robot.data.root_lin_vel_b[:, 0]
    cmd = env.command_manager.get_command(command_name)
    forward_cmd = (cmd[:, 0] > 0.05).float()
    return torch.clamp(-vx, min=0.0, max=1.0) * forward_cmd

def yaw_rate_penalty(env):
    """Penalize yaw rotation - keeps robot walking straight."""
    robot = env.scene["robot"]
    yaw_rate = robot.data.root_ang_vel_b[:, 2]
    return torch.square(yaw_rate)

def lateral_velocity_penalty(env):
    """Penalize sideways movement."""
    robot = env.scene["robot"]
    vy = robot.data.root_lin_vel_b[:, 1]
    return torch.square(vy)

def arm_default_pos_penalty(env):
    """Penalize arm joints deviating from desired natural hanging pose."""
    robot = env.scene["robot"]
    joint_pos = robot.data.joint_pos
    joint_names = robot.joint_names
    target = torch.zeros_like(joint_pos)
    used_indices = []
    for joint_name, joint_target in ARM_TARGETS.items():
        if joint_name in joint_names:
            idx = joint_names.index(joint_name)
            target[:, idx] = joint_target
            used_indices.append(idx)
    if not used_indices:
        return torch.zeros(env.num_envs, device=env.device)
    arm_pos = joint_pos[:, used_indices]
    arm_target = target[:, used_indices]
    return torch.sum(torch.square(arm_pos - arm_target), dim=-1)


def arm_action_penalty(env):
    """Small penalty for excessive arm action changes."""
    robot = env.scene["robot"]
    joint_names = robot.joint_names
    # JointPositionActionCfg preserves order; map arm joints into action slots.
    action_joint_names = ACTION_JOINT_NAMES
    arm_action_ids = [
        action_joint_names.index(name)
        for name in ARM_TARGETS
        if name in action_joint_names and name in joint_names
    ]
    if not arm_action_ids:
        return torch.zeros(env.num_envs, device=env.device)
    act_delta = env.action_manager.action[:, arm_action_ids] - env.action_manager.prev_action[:, arm_action_ids]
    return torch.sum(torch.square(act_delta), dim=-1)


def survival_reward(env):
    """Constant alive reward for every active environment."""
    return torch.ones(env.num_envs, device=env.device)


@configclass
class K1FullBodySceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
    )
    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )
    robot: ArticulationCfg = MISSING
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True
    )


@configclass
class ActionsCfg:
    """20 DoF: legs (12) + arms (8)."""
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=ACTION_JOINT_NAMES,
        scale=0.2,
        preserve_order=True,
        use_default_offset=False,
        offset={
            # Leg offsets (same as working 12-DoF task)
            "Left_Hip_Pitch": -0.2, "Left_Hip_Roll": 0.0, "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.4, "Left_Ankle_Pitch": -0.25, "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.2, "Right_Hip_Roll": 0.0, "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.4, "Right_Ankle_Pitch": -0.25, "Right_Ankle_Roll": 0.0,
            # Arm offsets: 0 = natural hanging position
            "ALeft_Shoulder_Pitch": 0.0, "Left_Shoulder_Roll": -1.3,
            "Left_Elbow_Pitch": 0.3, "Left_Elbow_Yaw": 0.0,
            "ARight_Shoulder_Pitch": 0.0, "Right_Shoulder_Roll": 1.3,
            "Right_Elbow_Pitch": 0.3, "Right_Elbow_Yaw": 0.0,
        },
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel)
        command           = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        dof_pos           = ObsTerm(func=mdp.joint_pos_rel)   # 20 dims
        dof_vel           = ObsTerm(func=mdp.joint_vel_rel)   # 20 dims
        actions           = ObsTerm(func=mdp.last_action)     # 20 dims
        concatenate_terms = True
        enable_corruption = False
    policy: PolicyCfg = PolicyCfg()
    # Total obs: 3+3+3+20+20+20 = 69 dims


@configclass
class CommandsCfg:
    # Training samples high-level velocity commands.
    # Deployment/play can override these commands externally (not low-level joint control).
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.05,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.35, 0.75),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.10, 0.10),
        ),
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            # Random yaw for generalization (restore after learning straight walk)
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (0.0, 0.0)},
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0),
                               "roll": (0.0, 0.0), "pitch": (0.0, 0.0), "yaw": (0.0, 0.0)},
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.01, 0.01),
            "velocity_range": (-0.01, 0.01),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )


@configclass
class RewardsCfg:
    # ── Core walking rewards ──
    survival         = RewTerm(func=survival_reward,          weight=0.10)
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=5.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z  = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=1.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )
    step_length      = RewTerm(
        func=k1_rew.step_length_reward,
        weight=0.8,
        params={"target_min": 0.16, "target_max": 0.34, "min_vx": 0.20},
    )
    foot_width       = RewTerm(
        func=k1_rew.foot_width_reward,
        weight=0.6,
        params={"target_min": 0.10, "target_max": 0.22, "ideal": 0.16},
    )
    foot_clearance   = RewTerm(
        func=k1_rew.foot_clearance_reward,
        weight=0.3,
        params={"min_clearance": 0.025, "max_clearance": 0.09, "min_vx": 0.20},
    )
    no_backward      = RewTerm(
        func=no_backward_velocity_penalty,
        weight=-6.0,
        params={"command_name": "base_velocity"},
    )

    # ── Stability ──
    base_height      = RewTerm(func=k1_rew.base_height_reward,       weight=-2.0)
    orientation      = RewTerm(func=k1_rew.orientation_penalty,      weight=-3.5)
    lin_vel_z        = RewTerm(func=k1_rew.lin_vel_z_penalty,        weight=-0.5)
    ang_vel_xy       = RewTerm(func=k1_rew.ang_vel_xy_penalty,       weight=-0.05)

    # ── Straightness (moderate, not strong enough to stop walking) ──
    yaw_rate         = RewTerm(func=yaw_rate_penalty,                weight=-1.0)
    lateral_vel      = RewTerm(func=lateral_velocity_penalty,        weight=-0.5)
    hip_roll         = RewTerm(func=k1_rew.hip_roll_penalty,         weight=-0.25)
    dof_vel_l2       = RewTerm(func=k1_rew.dof_vel_penalty,         weight=-8.0e-6)

    # ── Arm naturalness ──
    arm_default      = RewTerm(func=arm_default_pos_penalty,         weight=-0.8)
    arm_action       = RewTerm(func=arm_action_penalty,              weight=-0.02)

    # ── Efficiency ──
    torques          = RewTerm(func=k1_rew.torque_penalty,           weight=-0.0001)
    action_rate      = RewTerm(func=k1_rew.action_rate_penalty,      weight=-0.012)
    dof_pos_limits   = RewTerm(func=mdp.joint_pos_limits,            weight=-0.2,
                               params={"asset_cfg": SceneEntityCfg("robot")})


@configclass
class TerminationsCfg:
    time_out         = DoneTerm(func=mdp.time_out, time_out=True)
    terminate_height = DoneTerm(
        func=k1_done.terminate_height,
        params={"threshold": 0.35},
    )


@configclass
class K1FullBodyWalkEnvCfg(ManagerBasedRLEnvCfg):
    scene:        K1FullBodySceneCfg = K1FullBodySceneCfg(num_envs=64, env_spacing=2.0)
    commands:     CommandsCfg        = CommandsCfg()
    actions:      ActionsCfg         = ActionsCfg()
    observations: ObservationsCfg    = ObservationsCfg()
    events:       EventCfg           = EventCfg()
    rewards:      RewardsCfg         = RewardsCfg()
    terminations: TerminationsCfg    = TerminationsCfg()

    def __post_init__(self):
        self.decimation          = 10
        self.episode_length_s    = 20.0
        self.sim.dt              = 0.002
        self.sim.render_interval = self.decimation
        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.578)
        # Keep a natural arm pose target at reset for arm-down behavior.
        self.scene.robot.init_state.joint_pos.update({
            "ALeft_Shoulder_Pitch": ARM_TARGETS["ALeft_Shoulder_Pitch"],
            "ARight_Shoulder_Pitch": ARM_TARGETS["ARight_Shoulder_Pitch"],
            "Left_Shoulder_Roll": ARM_TARGETS["Left_Shoulder_Roll"],
            "Right_Shoulder_Roll": ARM_TARGETS["Right_Shoulder_Roll"],
            "Left_Elbow_Pitch": ARM_TARGETS["Left_Elbow_Pitch"],
            "Right_Elbow_Pitch": ARM_TARGETS["Right_Elbow_Pitch"],
            "Left_Elbow_Yaw": ARM_TARGETS["Left_Elbow_Yaw"],
            "Right_Elbow_Yaw": ARM_TARGETS["Right_Elbow_Yaw"],
        })
        # Parameter-walk helper is USD-only. Skip for URDF spawns to avoid startup crash.
        if hasattr(self.scene.robot.spawn, "usd_path"):
            self.scene.robot.spawn.func = add_k1_foot_box_collisions
        # Diagnostics for arm setup and action list.
        print("[K1FullBodyWalk] arm action joints:", [j for j in ACTION_JOINT_NAMES if ("Shoulder" in j or "Elbow" in j)])
        base_init_jpos = dict(self.scene.robot.init_state.joint_pos)
        arm_init_entries = {
            k: v for k, v in base_init_jpos.items() if ("Shoulder" in k or "Elbow" in k)
        }
        print("[K1FullBodyWalk] arm init-state joint_pos entries:", arm_init_entries)
        print("[K1FullBodyWalk] arm target pose:", ARM_TARGETS)


@configclass
class K1FullBodyWalkEnvCfg_PLAY(K1FullBodyWalkEnvCfg):
    """Play/eval config with deterministic forward command for big-step inspection."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.55, 0.55)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
