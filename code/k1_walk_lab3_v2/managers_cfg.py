from isaaclab.utils import configclass
import isaaclab.envs.mdp as mdp
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from .mdp.custom_rewards import (
    base_lin_vel_z_l2,
    bilateral_symmetry_l2,
    arm_natural_pose_soft_l2,
    base_ang_momentum_l2,
    target_out_of_range_l2,
    action_magnitude_l2,
    base_height_reward,
    joint_pos_l2,
)

# Left/right joint symmetry pairs — mirror signs enforce anti-symmetric gait
_SYMMETRY_LEFT  = ("Left_Hip_Roll",  "Left_Hip_Yaw",  "Left_Ankle_Roll",  "Left_Shoulder_Roll",  "Left_Elbow_Yaw")
_SYMMETRY_RIGHT = ("Right_Hip_Roll", "Right_Hip_Yaw", "Right_Ankle_Roll", "Right_Shoulder_Roll", "Right_Elbow_Yaw")
_SYMMETRY_SIGNS = (-1.0, -1.0, -1.0, -1.0, -1.0)


# =============================================================================
# Commands
# =============================================================================

@configclass
class CommandsCfg:
    """Velocity command with heading control.
    
    In V2, we enable turning commands to ensure the robot can maneuver while staying upright.
    """
    base_velocity = mdp.UniformVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(5.0, 10.0),
        rel_standing_envs=0.0,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.75,
        debug_vis=True,
        ranges=mdp.UniformVelocityCommandCfg.Ranges(
            lin_vel_x=(0.0, 0.6),  # Include zero for standing stability
            lin_vel_y=(-0.1, 0.1),
            ang_vel_z=(-0.6, 0.6), # Increased turning range
            heading=(-3.14159, 3.14159),
        ),
    )


# =============================================================================
# Observations  (78-dim, invariant layout)
# =============================================================================

@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_lin_vel      = ObsTerm(func=mdp.base_lin_vel,      params={"asset_cfg": SceneEntityCfg("robot")})
        base_ang_vel      = ObsTerm(func=mdp.base_ang_vel,      params={"asset_cfg": SceneEntityCfg("robot")})
        projected_gravity = ObsTerm(func=mdp.projected_gravity, params={"asset_cfg": SceneEntityCfg("robot")})
        joint_pos         = ObsTerm(func=mdp.joint_pos_rel,     params={"asset_cfg": SceneEntityCfg("robot")})
        joint_vel         = ObsTerm(func=mdp.joint_vel_rel,     params={"asset_cfg": SceneEntityCfg("robot")})
        actions           = ObsTerm(func=mdp.last_action)
        velocity_command  = ObsTerm(func=mdp.generated_commands, params={"command_name": "base_velocity"})

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


# =============================================================================
# Actions
# =============================================================================

@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        scale=1.0,
        use_default_offset=True,
    )


# =============================================================================
# Rewards
# =============================================================================

@configclass
class RewardsCfg:
    # --- velocity tracking ---
    track_lin_vel_xy = RewTerm(
        func=mdp.track_lin_vel_xy_exp,
        weight=20.0,  # Lowered from 50 to allow posture to matter more
        params={"command_name": "base_velocity", "std": 0.5},
    )
    track_ang_vel_z = RewTerm(
        func=mdp.track_ang_vel_z_exp,
        weight=15.0,
        params={"command_name": "base_velocity", "std": 0.5},
    )

    # --- posture: THE "PROUD WALKER" FIXES ---
    base_height = RewTerm(
        func=base_height_reward,
        weight=15.0, # High weight to keep body up
        params={"target_height": 0.57, "std": 0.05, "asset_cfg": SceneEntityCfg("robot")},
    )
    # --- stability: WIDER STANCE ---
    foot_distance = RewTerm(
        func=joint_pos_l2, # Using our custom joint_pos_l2
        weight=-2.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Roll"]),
        },
    )
    flat_orientation = RewTerm(func=mdp.flat_orientation_l2, weight=-25.0) # Penalty for leaning
    
    joint_pos_rel = RewTerm(
        func=joint_pos_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot")},
    )

    # --- episode bookkeeping ---
    alive      = RewTerm(func=mdp.is_alive,     weight=5.0)
    termination = RewTerm(func=mdp.is_terminated, weight=-1000.0)

    # --- motion quality ---
    joint_vel    = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_.*", ".*_Knee_.*", ".*_Ankle_.*"])},
    )
    action_rate  = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    base_lin_vel_z = RewTerm(func=base_lin_vel_z_l2, weight=-2.0)
    arm_flail_penalty = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-0.1, # Significantly increased
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"])},
    )

    # --- action bounds ---
    target_out_of_range = RewTerm(
        func=target_out_of_range_l2,
        weight=-10.0,
        params={"action_name": "joint_pos"},
    )
    action_magnitude = RewTerm(
        func=action_magnitude_l2,
        weight=-1.0e-2,
        params={"action_name": "joint_pos"},
    )

    # --- gait shaping ---
    bilateral_symmetry = RewTerm(
        func=bilateral_symmetry_l2,
        weight=-0.5,
        params={
            "asset_cfg":        SceneEntityCfg("robot"),
            "left_joint_names": _SYMMETRY_LEFT,
            "right_joint_names": _SYMMETRY_RIGHT,
            "mirror_signs":     _SYMMETRY_SIGNS,
        },
    )
    arm_natural_pose = RewTerm(
        func=arm_natural_pose_soft_l2,
        weight=-10.0, # Massive weight to stop T-posing
        params={
            "joint_names": [".*_Shoulder_.*", ".*_Elbow_.*"],
            "deadzone":    0.05, # Extremely tight
            "asset_cfg":   SceneEntityCfg("robot"),
        },
    )
    ang_momentum = RewTerm(func=base_ang_momentum_l2, weight=-0.2)


# =============================================================================
# Events (domain randomisation)
# =============================================================================

@configclass
class EventCfg:
    reset_robot_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range":     {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-3.14159, 3.14159)},
            "velocity_range": {"x": (-0.3, 0.3), "y": (-0.3, 0.3)},
            "asset_cfg":      SceneEntityCfg("robot"),
        },
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1,  0.1),
            "asset_cfg":      SceneEntityCfg("robot"),
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={
            "velocity_range": {"x": (-0.4, 0.4), "y": (-0.4, 0.4)},
            "asset_cfg":      SceneEntityCfg("robot"),
        },
    )
    physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg":             SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range":  (0.4, 1.2),
            "dynamic_friction_range": (0.3, 1.0),
            "restitution_range":      (0.0, 0.0),
            "num_buckets":            64,
        },
    )


# =============================================================================
# Terminations
# =============================================================================

@configclass
class TerminationsCfg:
    time_out         = DoneTerm(func=mdp.time_out,         time_out=True)
    bad_orientation  = DoneTerm(func=mdp.bad_orientation,  params={"limit_angle": 0.75})
    root_too_low     = DoneTerm(
        func=mdp.root_height_below_minimum,
        params={"minimum_height": 0.35, "asset_cfg": SceneEntityCfg("robot")},
    )
