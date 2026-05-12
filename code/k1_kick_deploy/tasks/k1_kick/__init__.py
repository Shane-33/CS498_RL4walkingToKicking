from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.registry import register_task
from .kick import K1KickControllerCfg
import torch

@configclass
class K1KickTeam2ControllerCfg(K1KickControllerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.policy.checkpoint_path = "models/policy.pt"
        self.mujoco.init_pos = [0.0, 0.0, 0.58]
        self.robot.joint_stiffness = [
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            40., 40., 40., 40., 15., 15.,
            40., 40., 40., 40., 15., 15.,
        ]
        self.robot.joint_damping = [
            1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
            4.0, 4.0, 4.0, 4.0, 2.0, 2.0,
        ]
        self.policy.action_scale = 0.3
        self.policy.ball_pos_in_robot_frame = torch.tensor([0.6, -0.1, 0.11], dtype=torch.float32)

register_task("k1_kick_team2", K1KickTeam2ControllerCfg())