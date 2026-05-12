"""
Manager configs for humanoid kicking policy — finetune stage.

Single stage, three naturally chained phases:
  Phase 1 (always):        approach ball, face goal, arms relaxed
  Phase 2 (foot < 0.25m): inside foot geometry — approach from behind-right
  Phase 3 (ball moving):  ball rolls toward goal, robot recovers

Changes from previous single-stage:
  - foot_velocity_toward_ball replaced with inside_foot_approach
  - arm_joints_deviation added to keep arms down
  - joint_vel reduced globally, leg joints penalized separately
    to allow more fluid natural motion
"""

from isaaclab.utils import configclass
from isaaclab.utils.noise import GaussianNoiseCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
import isaaclab.envs.mdp as mdp

from . import mdp as custom_mdp

# ──────────────────────────────────────────────────────────────────────────────
# Actions
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class CommandsCfg:
    pass


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
            ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
        ],
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Observations
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.1),
        )
        base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            params={"asset_cfg": SceneEntityCfg("robot")},
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        dof_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                        ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
                    ],
                )
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.01),
        )
        dof_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    joint_names=[
                        ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                        ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
                    ],
                )
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.05),
        )
        actions = ObsTerm(func=mdp.last_action)
        ball_position = ObsTerm(
            func=custom_mdp.ball_position_in_robot_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# ──────────────────────────────────────────────────────────────────────────────
# Rewards — single stage, three phases
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class RewardsCfg:

    # ── Survival ──────────────────────────────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=2.0)
    terminated = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # ── Posture ───────────────────────────────────────────────────────────────
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-0.1)
    base_height = RewTerm(
        func=custom_mdp.base_height_l2,
        weight=-0.2,                    # reduced from -0.3 for more natural lean
        params={"target_height": 0.57, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_lin_vel_z = RewTerm(
        func=custom_mdp.base_lin_vel_z_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Smoothness — leg joints only, allows fluid motion ─────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    # Reduced from -5e-3 to -2e-3 — less stiff, allows more fluid motion
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-2e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # ── Arms — keep relaxed by sides ──────────────────────────────────────────
    # Soft penalty — allows natural arm swing, discourages sustained raising
    arm_pose = RewTerm(
        func=custom_mdp.arm_joints_deviation,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Phase 1: approach (always active) ─────────────────────────────────────
    approach_ball = RewTerm(
        func=custom_mdp.distance_foot_to_ball,
        weight=2.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    yaw_alignment = RewTerm(
        func=custom_mdp.yaw_alignment_to_goal,
        weight=2.0,
        params={
            "goal_position": (10.0, 0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # ── Phase 2: inside foot geometry (gated at 0.25m) ────────────────────────
    # Guides foot to behind-right of ball — produces natural instep contact
    # Replaces foot_velocity_toward_ball which caused outside-foot kicks
    inside_foot = RewTerm(
        func=custom_mdp.inside_foot_approach,
        weight=5.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_position": (10.0, 0.0, 0.0),
            "proximity_threshold": 0.25,
            "behind_offset": 0.15,      # foot starts 15cm behind ball
            "side_offset": 0.10,        # foot starts 10cm to the right
        },
    )

    # ── Phase 3: ball quality (gated — ball moving > 0.1 m/s) ────────────────
    ball_toward_goal = RewTerm(
        func=custom_mdp.ball_velocity_toward_goal,
        weight=4.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_position": (10.0, 0.0, 0.0),
        },
    )
    kick_quality = RewTerm(
        func=custom_mdp.kick_quality,
        weight=5.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_position": (10.0, 0.0, 0.0),
            "min_speed_threshold": 1.0,
            "max_reward": 10.0,
        },
    )
    post_kick_upright = RewTerm(
        func=custom_mdp.post_kick_upright,
        weight=4.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "ball_speed_threshold": 0.3,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Terminations
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.6},
    )
    height_out_of_range = DoneTerm(
        func=custom_mdp.root_height_out_of_range,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "base_height": 0.57,
            "tolerance": 0.25,
        },
    )


# ──────────────────────────────────────────────────────────────────────────────
# Domain randomization
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class EventsCfg:
    robot_init_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "pose_range": {
                "x": (-0.1, 0.1),
                "y": (-0.1, 0.1),
                "yaw": (-0.3, 0.3),
            },
            "velocity_range": {
                "x": (-0.05, 0.05),
                "y": (-0.05, 0.05),
            },
        },
    )
    ball_init_pose = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("ball"),
            "pose_range": {
                "x": (0.25, 0.45),
                "y": (-0.2, 0.0),
            },
            "velocity_range": {},
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(3.0, 6.0),
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "velocity_range": {
                "x": (-0.3, 0.3),
                "y": (-0.3, 0.3),
                "yaw": (-0.2, 0.2),
            },
        },
    )