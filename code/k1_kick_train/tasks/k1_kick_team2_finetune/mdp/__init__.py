from .custom_rewards import (
    # observation
    ball_position_in_robot_frame,
    # posture
    base_height_l2,
    base_lin_vel_z_l2,
    arm_joints_deviation,       # new
    # phase 1
    distance_foot_to_ball,
    yaw_alignment_to_goal,
    # phase 2
    inside_foot_approach,       # replaces foot_velocity_toward_ball
    # phase 3
    ball_velocity_toward_goal,
    kick_quality,
    post_kick_upright,
    # termination
    root_height_out_of_range,
)