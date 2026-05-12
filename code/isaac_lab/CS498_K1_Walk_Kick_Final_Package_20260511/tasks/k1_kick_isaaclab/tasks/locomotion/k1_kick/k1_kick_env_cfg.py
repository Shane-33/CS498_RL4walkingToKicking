"""
K1 full-body soccer kick (Booster K1, 20 DoF) with phase-aware shaping.

Design goals
------------
* **Preserve locomotion:** velocity-command tracking stays active, gated softer only
  during swing/recover phases.
* **Structured kick:** implicit finite phases (approach → align → plant → kick → recover)
  drive reward gates; right foot is the designated strike foot.
* **Ball-centric observations:** relative pose/velocity, heading, curriculum-stable
  normalization (see ``mdp/observations.py``).
* **Curriculum:** staged ball placement via ``kick_curriculum_stage`` or auto-progress.

Observation layout
------------------
Proprioceptive block (**69**) — concatenation order in ``ObservationsCfg.PolicyCfg``::

    projected_gravity (3)
    root_angular_velocity_b (3)
    generated_commands ``base_velocity`` (3)   # vx, vy, ωz — no planar base_lin_vel term
    joint_pos_rel 20-actuated joints (20)
    joint_vel_rel (20)
    last_action / policy delay buffer (20)

Kick block (**17**) — ``mdp.observations.kick_ball_features``. See docstring there for indices.

**Total policy observation: 86** (prior walk-only checkpoints **54-D** legacy or **69-D** full-body are incompatible without ``expand_walk_checkpoint.py`` linear padding).

There is **no** height-scan or contact-patch observation in this environment; semantics match the Isaac Lab MDP terms configured above.
"""

from __future__ import annotations

from dataclasses import MISSING

import sys

import isaaclab.sim as sim_utils
import isaaclab.envs.mdp as mdp
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

sys.path.insert(0, "/home/team2/team2_project/booster_train/source/booster_train")

from booster_train.assets.robots.booster import BOOSTER_K1_CFG
from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.k1_fullbody_walk_env_cfg import (
    ACTION_JOINT_NAMES as FULLBODY_ACTION_JOINT_NAMES,
)
from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.k1_fullbody_walk_env_cfg import (
    add_k1_foot_box_collisions,
)
from booster_train.tasks.k1_fullbody_walk_isaaclab.tasks.locomotion.k1_fullbody_walk.mdp import rewards as walk_mdp_rew

from .mdp import events as kick_events
from .mdp import observations as kick_obs
from .mdp import rewards as kick_rew
from .mdp import terminations as kick_done


@configclass
class K1KickSceneCfg(InteractiveSceneCfg):
    terrain = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.8,
                dynamic_friction=0.8,
                restitution=0.2,
            ),
        ),
    )

    dome_light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(intensity=2000.0, color=(1.0, 1.0, 1.0)),
    )

    robot: ArticulationCfg = MISSING

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        history_length=3,
        track_air_time=True,
    )

    ball = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Ball",
        spawn=sim_utils.SphereCfg(
            radius=0.11,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                linear_damping=0.05,
                angular_damping=0.05,
                max_linear_velocity=10.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.45),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.05, 0.05, 0.05)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.6, 0.0, 0.11)),
    )


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(FULLBODY_ACTION_JOINT_NAMES),
        scale=0.2,
        preserve_order=True,
        use_default_offset=False,
        offset={
            "Left_Hip_Pitch": -0.2,
            "Left_Hip_Roll": 0.0,
            "Left_Hip_Yaw": 0.0,
            "Left_Knee_Pitch": 0.4,
            "Left_Ankle_Pitch": -0.25,
            "Left_Ankle_Roll": 0.0,
            "Right_Hip_Pitch": -0.2,
            "Right_Hip_Roll": 0.0,
            "Right_Hip_Yaw": 0.0,
            "Right_Knee_Pitch": 0.4,
            "Right_Ankle_Pitch": -0.25,
            "Right_Ankle_Roll": 0.0,
            "ALeft_Shoulder_Pitch": 0.0,
            "Left_Shoulder_Roll": -1.3,
            "Left_Elbow_Pitch": 0.3,
            "Left_Elbow_Yaw": 0.0,
            "ARight_Shoulder_Pitch": 0.0,
            "Right_Shoulder_Roll": 1.3,
            "Right_Elbow_Pitch": 0.3,
            "Right_Elbow_Yaw": 0.0,
        },
    )


_ROBOT_OBS_JOINTS = SceneEntityCfg("robot", joint_names=list(FULLBODY_ACTION_JOINT_NAMES))


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})
        dof_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _ROBOT_OBS_JOINTS})
        dof_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _ROBOT_OBS_JOINTS})
        actions = ObsTerm(func=mdp.last_action)
        kick_context = ObsTerm(func=kick_obs.kick_ball_features)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class CommandsCfg:
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(3.5, 6.0),
        rel_standing_envs=0.08,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.65),
            lin_vel_y=(-0.06, 0.06),
            ang_vel_z=(-0.18, 0.18),
        ),
    )


@configclass
class CommandsCfgStableApproach(CommandsCfg):
    """Conservative command distribution for stage-1 approach stability."""

    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.10,
        rel_heading_envs=0.0,
        heading_command=False,
        debug_vis=False,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.25, 0.40),
            lin_vel_y=(-0.03, 0.03),
            ang_vel_z=(-0.12, 0.12),
        ),
    )


@configclass
class EventCfg:
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.15, 0.15), "y": (-0.15, 0.15), "yaw": (-0.35, 0.35)},
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.02, 0.02),
            "velocity_range": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    reset_ball = EventTerm(
        func=kick_events.reset_ball_relative_to_robot,
        mode="reset",
        params={"z_offset": 0.11},
    )


@configclass
class RewardsCfg:
    # --- Alive / walk prior ---
    survival = RewTerm(func=kick_rew.survival, weight=0.08)
    track_lin_vel = RewTerm(
        func=kick_rew.track_walk_command,
        weight=4.5,
        params={"command_name": "base_velocity", "std": 0.55},
    )
    track_ang_vel = RewTerm(
        func=kick_rew.track_yaw_command,
        weight=0.9,
        params={"command_name": "base_velocity", "std": 0.55},
    )

    # --- Phase-shaped kick pipeline ---
    approach = RewTerm(func=kick_rew.reward_approach_distance, weight=1.2)
    alignment = RewTerm(func=kick_rew.reward_alignment, weight=0.4)
    phase_progress = RewTerm(func=kick_rew.reward_phase_progress, weight=2.0)
    approach_vel = RewTerm(func=kick_rew.reward_approach_velocity, weight=1.5)
    support_stable = RewTerm(func=kick_rew.reward_support_foot_stable, weight=0.5)
    com_over_support = RewTerm(func=kick_rew.reward_com_over_support_xy, weight=1.5)
    right_foot_approach = RewTerm(func=kick_rew.reward_right_foot_to_ball, weight=0.4)

    ball_contact = RewTerm(
        func=kick_rew.reward_ball_contact,
        weight=8.0,
    )
    kick_leg_forward = RewTerm(
        func=kick_rew.reward_kick_leg_forward,
        weight=6.0,
    )

    ball_forward = RewTerm(func=kick_rew.reward_ball_forward_velocity, weight=25.0)
    ball_fwd_body = RewTerm(func=kick_rew.reward_forward_ball_speed_after_hit, weight=8.0)

    recover_balance = RewTerm(func=kick_rew.reward_post_kick_balance, weight=1.2)
    recover_walk = RewTerm(func=kick_rew.reward_post_kick_walk_recovery, weight=2.0)

    arms_balance = RewTerm(func=kick_rew.arm_counterbalance_reward, weight=0.55)

    # --- Penalties (moderate scale, PPO-stable) ---
    no_kick_progress = RewTerm(
        func=kick_rew.penalty_no_kick_progress,
        weight=-4.0,
        params={"episode_step_threshold": 100},
    )
    lat_ball = RewTerm(func=kick_rew.penalty_excess_lateral_ball, weight=-0.9)
    passed_ball = RewTerm(func=kick_rew.penalty_passed_ball, weight=-2.5)
    left_foot_hack = RewTerm(func=kick_rew.penalty_left_foot_near_ball, weight=-1.8)
    ball_lat_vel = RewTerm(func=kick_rew.penalty_ball_lateral_velocity, weight=-1.2)
    ball_backward = RewTerm(func=kick_rew.penalty_backward_ball, weight=-2.8)

    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-1.8,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.52},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-2.8)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.45)
    ang_vel_xy = RewTerm(func=mdp.ang_vel_xy_l2, weight=-0.04)

    yaw_rate = RewTerm(func=kick_rew.yaw_rate_penalty, weight=-0.85)
    lateral_vel = RewTerm(func=kick_rew.lateral_velocity_penalty, weight=-0.55)
    hip_roll = RewTerm(func=kick_rew.hip_roll_penalty, weight=-0.35)
    dof_speed = RewTerm(func=kick_rew.joint_speed_penalty, weight=-4.0e-6)

    torques = RewTerm(func=walk_mdp_rew.torque_penalty, weight=-0.0001)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.015)
    dof_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.2,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )


@configclass
class RewardsCfgStableApproach(RewardsCfg):
    """Stage-1 reward profile: robust walk-to-approach before true kicking."""

    track_lin_vel = RewTerm(
        func=kick_rew.track_walk_command,
        weight=2.8,
        params={"command_name": "base_velocity", "std": 0.60},
    )
    track_ang_vel = RewTerm(
        func=kick_rew.track_yaw_command,
        weight=0.7,
        params={"command_name": "base_velocity", "std": 0.60},
    )

    approach = RewTerm(func=kick_rew.reward_approach_distance, weight=2.8)
    alignment = RewTerm(func=kick_rew.reward_alignment, weight=1.2)
    phase_progress = RewTerm(func=kick_rew.reward_phase_progress, weight=1.5)
    approach_vel = RewTerm(func=kick_rew.reward_approach_velocity, weight=1.0)
    support_stable = RewTerm(func=kick_rew.reward_support_foot_stable, weight=1.4)
    com_over_support = RewTerm(func=kick_rew.reward_com_over_support_xy, weight=1.8)
    right_foot_approach = RewTerm(func=kick_rew.reward_right_foot_to_ball, weight=1.6)

    # Contact shaping should exist in stage-1, but final ball-drive objectives stay weak.
    ball_contact = RewTerm(func=kick_rew.reward_ball_contact, weight=1.2)
    kick_leg_forward = RewTerm(func=kick_rew.reward_kick_leg_forward, weight=0.8)
    ball_forward = RewTerm(func=kick_rew.reward_ball_forward_velocity, weight=0.6)
    ball_fwd_body = RewTerm(func=kick_rew.reward_forward_ball_speed_after_hit, weight=0.0)

    no_kick_progress = RewTerm(
        func=kick_rew.penalty_no_kick_progress,
        weight=-1.5,
        params={"episode_step_threshold": 180},
    )

    # Stronger anti-fall regularization than kick-focused config.
    base_height = RewTerm(
        func=mdp.base_height_l2,
        weight=-2.4,
        params={"asset_cfg": SceneEntityCfg("robot"), "target_height": 0.52},
    )
    orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-4.2)
    lin_vel_z = RewTerm(func=mdp.lin_vel_z_l2, weight=-0.55)
    yaw_rate = RewTerm(func=kick_rew.yaw_rate_penalty, weight=-1.0)
    lateral_vel = RewTerm(func=kick_rew.lateral_velocity_penalty, weight=-0.70)
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.020)


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    terminate_height = DoneTerm(func=kick_done.terminate_height, params={"threshold": 0.35})
    far_ball = DoneTerm(func=kick_done.terminate_robot_far_from_ball, params={"max_dist": 2.3})
    ball_behind = DoneTerm(func=kick_done.terminate_ball_behind_robot, params={"behind_thresh": -0.22})


@configclass
class K1KickEnvCfg(ManagerBasedRLEnvCfg):
    """Kick environment configuration."""

    scene: K1KickSceneCfg = K1KickSceneCfg(num_envs=64, env_spacing=3.0)
    commands: CommandsCfg = CommandsCfg()
    actions: ActionsCfg = ActionsCfg()
    observations: ObservationsCfg = ObservationsCfg()
    events: EventCfg = EventCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    kick_curriculum_stage: int = 1
    """Manual curriculum stage in ``[1, 5]`` (ignored if ``kick_curriculum_auto``)."""

    kick_curriculum_auto: bool = False
    """If True, stage increases from global ``common_step_counter``."""

    kick_curriculum_steps_per_stage: int = 150_000
    """Simulation steps per stage when auto curriculum is enabled."""

    def __post_init__(self):
        self.decimation = 10
        self.episode_length_s = 12.0
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        self.scene.robot = BOOSTER_K1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.pos = (0.0, 0.0, 0.578)
        if hasattr(self.scene.robot.spawn, "usd_path"):
            self.scene.robot.spawn.func = add_k1_foot_box_collisions


@configclass
class K1KickEnvCfg_PLAY(K1KickEnvCfg):
    """Narrow commands for inspection / video capture."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.45, 0.45)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.kick_curriculum_stage = 1


@configclass
class K1KickStableApproachV2EnvCfg(K1KickEnvCfg):
    """Conservative stage-1 curriculum for stable approach before full kicking."""

    commands: CommandsCfgStableApproach = CommandsCfgStableApproach()
    rewards: RewardsCfgStableApproach = RewardsCfgStableApproach()

    def __post_init__(self):
        super().__post_init__()
        self.kick_curriculum_auto = False
        self.kick_curriculum_stage = 1
        # Easier centered ball placement for stage-1.
        self.kick_stage1_ball_x_range = (0.40, 0.52)
        self.kick_stage1_ball_y_range = (-0.01, 0.01)


@configclass
class K1KickStableApproachV2EnvCfg_PLAY(K1KickStableApproachV2EnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.30, 0.30)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
