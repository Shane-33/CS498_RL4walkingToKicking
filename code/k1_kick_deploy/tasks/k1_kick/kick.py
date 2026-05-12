from __future__ import annotations
from dataclasses import MISSING
import os
import torch

from booster_deploy.controllers.base_controller import BaseController, Policy
from booster_deploy.controllers.controller_cfg import ControllerCfg, PolicyCfg
from booster_deploy.robots.booster import K1_CFG
from booster_deploy.utils.isaaclab.configclass import configclass
from booster_deploy.utils.isaaclab import math as lab_math

# Path to our custom scene XML with ball
TASK_DIR = os.path.dirname(os.path.abspath(__file__))
KICK_SCENE_XML = os.path.join(TASK_DIR, "K1_kick_scene.xml")


class KickPolicy(Policy):
    def __init__(self, cfg: KickPolicyCfg, controller: BaseController):
        super().__init__(cfg, controller)
        self.cfg = cfg
        self.robot = controller.robot

        policy_path = self.cfg.checkpoint_path
        if not os.path.isabs(policy_path):
            policy_path = os.path.join(self.task_path, self.cfg.checkpoint_path)

        self._model: torch.jit.ScriptModule = torch.jit.load(
            policy_path, map_location="cpu")
        self._model.eval()

        self.action_scale = cfg.action_scale

        self.policy_joint_names = cfg.policy_joint_names
        self.real2sim_joint_map = torch.tensor([
            self.robot.cfg.joint_names.index(name)
            for name in self.policy_joint_names
        ], dtype=torch.long)

        self.last_action = torch.zeros(
            len(self.policy_joint_names), dtype=torch.float32)

        # ball joint index in mujoco qpos: 7 (robot) + 7 (ball free joint)
        self._ball_qpos_start = 7 + self.robot.num_joints + 7  # noqa

    def reset(self) -> None:
        self.last_action = torch.zeros(
            len(self.policy_joint_names), dtype=torch.float32)

    def _get_ball_pos_in_robot_frame(self) -> torch.Tensor:
        """Get ball position in robot body frame from MuJoCo data."""
        try:
            mj_data = self.controller.mj_data
            # ball free joint qpos starts after robot qpos (7+22=29)
            ball_pos_w = torch.tensor(
                mj_data.qpos[29:32].copy(), dtype=torch.float32)
            robot_pos_w = self.robot.data.root_pos_w
            robot_quat_w = self.robot.data.root_quat_w
            # translate to robot frame
            ball_relative = ball_pos_w - robot_pos_w
            ball_in_body = lab_math.quat_apply_inverse(
                robot_quat_w, ball_relative)
            return ball_in_body
        except Exception:
            return self.cfg.ball_pos_in_robot_frame

    def compute_observation(self) -> torch.Tensor:
        dof_pos = self.robot.data.joint_pos
        dof_vel = self.robot.data.joint_vel
        base_quat = self.robot.data.root_quat_w
        base_ang_vel = self.robot.data.root_ang_vel_b

        # root_lin_vel_b from mujoco is actually world frame — rotate to body
        base_lin_vel_w = self.robot.data.root_lin_vel_b
        base_lin_vel = lab_math.quat_apply_inverse(base_quat, base_lin_vel_w)

        gravity_w = torch.tensor([0.0, 0.0, -1.0], dtype=torch.float32)
        projected_gravity = lab_math.quat_apply_inverse(base_quat, gravity_w)

        default_joint_pos = self.robot.default_joint_pos
        mapped_default_pos = default_joint_pos[self.real2sim_joint_map]
        mapped_dof_pos = dof_pos[self.real2sim_joint_map]
        mapped_dof_vel = dof_vel[self.real2sim_joint_map]

        ball_pos = self._get_ball_pos_in_robot_frame()

        # Exactly matches training ObservationsCfg order:
        # projected_gravity(3), base_ang_vel(3), base_lin_vel(3),
        # dof_pos(12), dof_vel(12), last_action(12), ball_pos(3) = 48
        obs = torch.cat([
            projected_gravity,                      # 3
            base_ang_vel,                           # 3
            base_lin_vel,                           # 3
            (mapped_dof_pos - mapped_default_pos),  # 12
            mapped_dof_vel,                         # 12
            self.last_action,                       # 12
            ball_pos,                               # 3
        ], dim=0)  # total: 48

        return obs

    def inference(self) -> torch.Tensor:
        obs = self.compute_observation()

        with torch.no_grad():
            action = self._model(obs.unsqueeze(0)).squeeze(0)
            action = torch.clamp(action, -10.0, 10.0)  # clip first

        self.last_action = action.clone()  # store clipped action

        default_joint_pos = self.robot.default_joint_pos
        dof_targets = default_joint_pos.clone()
        dof_targets[self.real2sim_joint_map] = (
            default_joint_pos[self.real2sim_joint_map]
            + action * self.action_scale
        )

        # clip to joint limits
        dof_targets = torch.clamp(dof_targets,
            torch.tensor([-3.14]*22),
            torch.tensor([3.14]*22))

        return dof_targets


@configclass
class KickPolicyCfg(PolicyCfg):
    constructor = KickPolicy
    checkpoint_path: str = MISSING
    action_scale: float = 1.0
    policy_joint_names: list = MISSING
    ball_pos_in_robot_frame: torch.Tensor = torch.tensor(
        [0.35, -0.1, 0.11], dtype=torch.float32)


@configclass
class K1KickControllerCfg(ControllerCfg):
    robot = K1_CFG.replace(
        mjcf_path=KICK_SCENE_XML,
        default_joint_pos=[
            0, 0,                               # head
            0.0, -1.3, 0, 0.0,                 # left arm
            0.0,  1.3, 0, 0.0,                 # right arm
            -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # left leg
            -0.2, 0.0, 0.0, 0.4, -0.25, 0.0,  # right leg
        ],
        joint_stiffness=[
            4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            4.0, 4.0, 4.0, 4.0,
            80., 80., 80., 80., 30., 30.,
            80., 80., 80., 80., 30., 30.,
        ],
        joint_damping=[
            1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0, 1.0, 1.0,
        ],
    )
    enable_velocity_commands = False
    policy: KickPolicyCfg = KickPolicyCfg(
        policy_joint_names=[
            "Left_Hip_Pitch",
            "Left_Hip_Roll",
            "Left_Hip_Yaw",
            "Left_Knee_Pitch",
            "Left_Ankle_Pitch",
            "Left_Ankle_Roll",
            "Right_Hip_Pitch",
            "Right_Hip_Roll",
            "Right_Hip_Yaw",
            "Right_Knee_Pitch",
            "Right_Ankle_Pitch",
            "Right_Ankle_Roll",
        ],
    )