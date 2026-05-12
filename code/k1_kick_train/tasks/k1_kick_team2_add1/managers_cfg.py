"""
Manager configs for humanoid kicking policy — single stage.

No curriculum switching. One reward set that naturally chains three phases:
  Phase 1 (always):        approach ball, face goal direction
  Phase 2 (foot < 0.15m): swing foot into ball
  Phase 3 (ball moving):  ball rolls toward goal, robot recovers

Train from Stage 2 checkpoint for best results — it already knows
how to lift the foot, so Phase 2 activates quickly.
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
# Single stage rewards
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
        weight=-0.3,
        params={"target_height": 0.57, "asset_cfg": SceneEntityCfg("robot")},
    )
    base_lin_vel_z = RewTerm(
        func=custom_mdp.base_lin_vel_z_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── Smoothness — suppress jitter and explosive motion ─────────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2e-4,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )

    # ── Phase 1: approach (always active) ─────────────────────────────────────
    # Pulls right foot toward ball continuously at any distance
    approach_ball = RewTerm(
        func=custom_mdp.distance_foot_to_ball,
        weight=2.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )
    # Robot must face the ball→goal axis throughout
    yaw_alignment = RewTerm(
        func=custom_mdp.yaw_alignment_to_goal,
        weight=2.0,
        params={
            "goal_position": (10.0, 0.0, 0.0),
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # ── Phase 2: swing (gated — only fires within 0.15m of ball) ──────────────
    # Robot gets no reward here until it's already close from Phase 1.
    # Once close, rewards foot velocity toward ball — the natural swing motion.
    # Capped at 3 m/s to prevent explosive lunging.
    foot_toward_ball = RewTerm(
        func=custom_mdp.foot_velocity_toward_ball,
        weight=5.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "proximity_threshold": 0.15,
        },
    )

    # ── Phase 3: ball quality (gated — only fires when ball is moving) ─────────
    # Continuous signal: ball rolling toward goal
    ball_toward_goal = RewTerm(
        func=custom_mdp.ball_velocity_toward_goal,
        weight=4.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_position": (10.0, 0.0, 0.0),
        },
    )
    # Sparse bonus: speed × direction accuracy above 1 m/s
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
    # Recovery: upright after ball moves — silent during kick itself
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