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


@configclass
class CommandsCfg:
    pass


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[
            ".*_Hip_Pitch",
            ".*_Hip_Roll",
            ".*_Hip_Yaw",
            ".*_Knee_Pitch",
            ".*_Ankle_Pitch",
            ".*_Ankle_Roll",
        ],
        scale=1.0,
        preserve_order=True,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        # ── IMU equivalent — gravity and angular velocity ─────────────────────
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
        # ── Joint encoders ────────────────────────────────────────────────────
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
        # ── Ball position — perception/vision noise ───────────────────────────
        ball_position = ObsTerm(
            func=custom_mdp.ball_position_in_robot_frame,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "ball_cfg": SceneEntityCfg("ball"),
            },
            noise=GaussianNoiseCfg(mean=0.0, std=0.02),
        )

        def __post_init__(self):
            self.enable_corruption = True    # noise active during training
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    # ── Survival ──────────────────────────────────────────────────────────────
    alive = RewTerm(func=mdp.is_alive, weight=2.0)

    # ── Termination — tightened to outweigh ball_contact ─────────────────────
    termination = RewTerm(func=mdp.is_terminated, weight=-10.0)

    # ── Posture — soft nudges, allow dynamic lean during kick ─────────────────
    flat_orientation = RewTerm(
        func=mdp.flat_orientation_l2,
        weight=-0.1,
    )
    base_height = RewTerm(
        func=custom_mdp.base_height_l2,
        weight=-0.3,
        params={
            "target_height": 0.57,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    base_lin_vel_z = RewTerm(
        func=custom_mdp.base_lin_vel_z_l2,
        weight=-0.1,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # ── COM stability — penalize backward lean and collapse ───────────────────
    com_backward_vel = RewTerm(
        func=custom_mdp.com_backward_velocity_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_penalty": 1.0,
        },
    )
    com_height = RewTerm(
        func=custom_mdp.com_height_penalty,
        weight=-5.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "min_height": 0.45,
        },
    )

    # ── Smoothness — very loose, allow explosive kick motion ──────────────────
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1.0e-4)
    joint_torques = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*"])},
    )
    joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-0.5,
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=[
                    ".*_Hip_Pitch", ".*_Hip_Roll", ".*_Hip_Yaw",
                    ".*_Knee_Pitch", ".*_Ankle_Pitch", ".*_Ankle_Roll",
                ],
            )
        },
    )

    # ── Single leg balance — plant foot down, kick foot up ────────────────────
    single_leg_balance = RewTerm(
        func=custom_mdp.single_leg_balance_quality,
        weight=5.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_foot_link"],
            ),
            "asset_cfg": SceneEntityCfg("robot"),
            "force_threshold": 10.0,
        },
    )

    # ── Kick motion primitive — reward forward swing of right foot ────────────
    kick_swing = RewTerm(
        func=custom_mdp.kick_foot_aerial_swing,
        weight=5.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_foot_link"],
            ),
            "min_height": 0.05,
            "force_threshold": 1.0,
        },
    )

    foot_jitter = RewTerm(
        func=custom_mdp.kick_foot_jitter_penalty,
        weight=-2.0,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )

    lateral_com_vel = RewTerm(
        func=custom_mdp.lateral_com_velocity_penalty,
        weight=-3.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "max_penalty": 1.0,
        },
    )

    # ── Ball approach — dense shaping, gradient always exists ─────────────────
    approach_ball = RewTerm(
        func=custom_mdp.distance_foot_to_ball,
        weight=1.5,
        params={
            "foot_body_name": "right_foot_link",
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
        },
    )

    # ── Ball contact — sparse but dominant ───────────────────────────────────
    ball_contact = RewTerm(
        func=custom_mdp.ball_contact_reward,
        weight=10.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["right_foot_link"],
            ),
            "force_threshold": 1.0,
        },
    )

    # ── Ball moved — confirms contact was meaningful ──────────────────────────
    ball_moved = RewTerm(
        func=custom_mdp.ball_displacement,
        weight=3.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "max_reward": 2.0,
        },
    )

    # ── Plant foot stability — left foot stays grounded ───────────────────────
    plant_stability = RewTerm(
        func=custom_mdp.plant_foot_stability,
        weight=1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[".*_foot_link"],
            ),
            "plant_foot_idx": 0,
            "force_threshold": 10.0,
        },
    )

    # ── Shot quality ──────────────────────────────────────────────────────────
    ball_toward_goal = RewTerm(
        func=custom_mdp.ball_velocity_toward_goal,
        weight=3.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "goal_position": (10.0, 0.0, 0.0),
        },
    )
    ball_acceleration = RewTerm(
        func=custom_mdp.ball_acceleration_reward,
        weight=2.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "acceleration_scale": 10.0,
            "max_reward": 5.0,
        },
    )
    kick_quality = RewTerm(
        func=custom_mdp.kick_quality,
        weight=5.0,
        params={
            "ball_cfg": SceneEntityCfg("ball"),
            "asset_cfg": SceneEntityCfg("robot"),
            "goal_position": (10.0, 0.0, 0.0),
            "min_speed_threshold": 1.0,
            "max_reward": 10.0,
        },
    )

    # ── Post kick recovery — reward staying upright after ball moves ──────────
    post_kick_upright = RewTerm(
        func=custom_mdp.post_kick_upright,
        weight=8.0,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "ball_cfg": SceneEntityCfg("ball"),
            "ball_speed_threshold": 0.3,
        },
    )


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    bad_orientation = DoneTerm(
        func=mdp.bad_orientation,
        params={"limit_angle": 0.6},    # tightened from 1.57
    )
    height_out_of_range = DoneTerm(
        func=custom_mdp.root_height_out_of_range,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "base_height": 0.57,
            "tolerance": 0.2,
        },
    )

@configclass
class EventsCfg:
    """Domain randomization for sim2real transfer."""

    # ── Per episode pose randomization ────────────────────────────────────────

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

    # ── Push disturbances ─────────────────────────────────────────────────────

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